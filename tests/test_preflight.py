"""scripts/preflight.py against two fake replicas behind an in-process gateway (the Cluster helper
from test_multi_replica): passes when idle, fails for the right reason otherwise."""

from __future__ import annotations

import asyncio
import contextlib
import importlib.util
import sys
from pathlib import Path

import httpx
import pytest

from slotgw.config import AdmissionSettings

from .test_multi_replica import MSG, TWO_SEATS, Cluster

REPO = Path(__file__).resolve().parents[1]
FAKE_HF_MODEL = "Qwen/Qwen2.5-3B-Instruct-AWQ"  # config/replicas.yaml `models.small`, which Cluster keeps


def _load():
    spec = importlib.util.spec_from_file_location("preflight", REPO / "scripts" / "preflight.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # dataclasses resolve string annotations through sys.modules
    spec.loader.exec_module(mod)
    return mod


preflight = _load()


@pytest.fixture
def cluster(tmp_path):
    c = Cluster([TWO_SEATS] * 2, AdmissionSettings(), queue_s=5.0, meter_path=tmp_path / "m.sqlite")
    yield c
    c.stop()


def _failed(checks):
    return {c.name: c.detail for c in checks if not c.ok}


def _run(gw, tmp_path, **kw):
    kw.setdefault("expect_replicas", 2)
    kw.setdefault("expect_model", FAKE_HF_MODEL)
    kw.setdefault("runs_dir", tmp_path)
    kw.setdefault("min_free_gb", 0)
    return preflight.run_checks(gw, **kw)


async def test_idle_cluster_passes(cluster, tmp_path):
    checks, run_id = await asyncio.to_thread(_run, cluster.gateway.url, tmp_path, expect_calibration="fake-0")
    assert run_id == "cluster"
    failed = _failed(checks)
    # calibration[fake-1] is the only expected miss: its inline profile is named after fake-1
    assert set(failed) == {"calibration[fake-1]"}, failed
    names = {c.name for c in checks}
    assert {"healthz", "replica_count", "replica_idle[fake-0]", "ledger_idle[fake-1]", "model[fake-0]",
            "queues_empty", "tokenizer", "disk_free"} <= names
    checks, _ = await asyncio.to_thread(_run, cluster.gateway.url, tmp_path, expect_calibration=":inline")
    assert _failed(checks) == {}


async def test_wrong_replica_count_model_and_stale_run_id_fail(cluster, tmp_path):
    checks, _ = await asyncio.to_thread(_run, cluster.gateway.url, tmp_path, expect_replicas=3)
    failed = _failed(checks)
    assert set(failed) == {"replica_count"} and "2 replica(s), expected 3" in failed["replica_count"]

    checks, _ = await asyncio.to_thread(_run, cluster.gateway.url, tmp_path, not_run_id="cluster")
    failed = _failed(checks)
    assert set(failed) == {"run_id_fresh"} and "previous" in failed["run_id_fresh"]

    checks, _ = await asyncio.to_thread(_run, cluster.gateway.url, tmp_path, expect_model="Qwen/Qwen2.5-7B-Instruct-AWQ")
    assert set(_failed(checks)) == {"model[fake-0]", "model[fake-1]"}

    checks, _ = await asyncio.to_thread(_run, cluster.gateway.url, tmp_path, min_free_gb=1e9)
    assert set(_failed(checks)) == {"disk_free"}


async def test_in_flight_request_fails_idle_checks(cluster, tmp_path):
    gw = cluster.gateway.url
    body = {"model": "small.8k.15tps", "messages": MSG, "max_tokens": 60, "ignore_eos": True, "stream": True}
    started = asyncio.Event()

    async def hold():
        async with httpx.AsyncClient(base_url=gw, timeout=60) as c, c.stream(
            "POST", "/v1/chat/completions", json=body, headers={"Authorization": "Bearer sk-local-chat"}
        ) as r:
            assert r.status_code == 200
            async for _ in r.aiter_lines():
                started.set()

    task = asyncio.create_task(hold())
    await asyncio.wait_for(started.wait(), 10)
    try:
        checks, _ = await asyncio.to_thread(_run, gw, tmp_path)
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError, httpx.HTTPError):
            await task
    failed = _failed(checks)
    busy = [n for n in failed if n.startswith(("replica_idle[", "ledger_idle["))]
    assert len(busy) == 2, failed  # the one replica running the stream fails both its idle checks
    assert any("running=1" in failed[n] for n in busy) and any("engine_active=1" in failed[n] for n in busy)
    assert not any(n.startswith(("model", "replica_count", "tokenizer")) for n in failed)


async def test_gateway_down_is_one_clear_failure(tmp_path):
    checks, run_id = await asyncio.to_thread(_run, "http://127.0.0.1:9", tmp_path)
    assert run_id is None
    failed = _failed(checks)
    assert "healthz" in failed and "/healthz" in failed["healthz"]


def test_cli_prints_ok_line_and_exit_code(cluster, tmp_path, capsys):
    rc = preflight.main(["--gateway", cluster.gateway.url, "--expect-replicas", "2", "--expect-model", FAKE_HF_MODEL,
                         "--runs-dir", str(tmp_path), "--min-free-gb", "0"])
    assert rc == 0 and capsys.readouterr().out.strip() == "PREFLIGHT OK run_id=cluster"
    rc = preflight.main(["--gateway", cluster.gateway.url, "--expect-replicas", "3", "--expect-model", FAKE_HF_MODEL,
                         "--runs-dir", str(tmp_path), "--min-free-gb", "0", "--not-run-id", "cluster"])
    out = capsys.readouterr().out
    assert rc == 1 and "PREFLIGHT FAIL replica_count:" in out and "PREFLIGHT FAIL run_id_fresh:" in out
    rc = preflight.main(["--gateway", cluster.gateway.url, "--expect-replicas", "2", "--expect-model", FAKE_HF_MODEL,
                         "--runs-dir", str(tmp_path), "--min-free-gb", "0", "--json"])
    assert rc == 0 and '"ok": true' in capsys.readouterr().out
