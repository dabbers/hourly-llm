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


async def test_kv_budget_checks_against_the_live_cluster(cluster, tmp_path):
    """--expect-kv-total passes when every replica's ledger kv_total is within the tolerance and fails for
    every replica otherwise; --kv-spread-max passes on two equal fakes."""
    async with httpx.AsyncClient(base_url=cluster.gateway.url, timeout=10) as c:
        snap = (await c.get("/admin/replicas")).json()
    kv = [r["ledger"]["kv_total"] for r in snap["replicas"]]
    assert len(kv) == 2 and kv[0] == kv[1] > 0
    checks, _ = await asyncio.to_thread(_run, cluster.gateway.url, tmp_path, expect_calibration=":inline",
                                        expect_kv_total=kv[0], kv_spread_max=0.01)
    assert _failed(checks) == {}
    assert {"kv_budget[fake-0]", "kv_budget[fake-1]", "kv_uniform"} <= {c.name for c in checks}
    checks, _ = await asyncio.to_thread(_run, cluster.gateway.url, tmp_path, expect_calibration=":inline",
                                        expect_kv_total=int(kv[0] * 1.066))  # the phase-3 deficit, inverted
    failed = _failed(checks)
    assert set(failed) == {"kv_budget[fake-0]", "kv_budget[fake-1]"} and "within 1%" in failed["kv_budget[fake-0]"]


def test_kv_spread_flags_one_short_engine(tmp_path):
    """A canned /admin/replicas snapshot with one engine 6 % short (the campaign pod, 2026-11-09): kv_uniform
    fails and names the spread, the absolute check fails for that replica only."""
    def ledger(kv_total):
        return {"kv_total": kv_total, "kv_usable": int(kv_total * 0.95), "engine_active": 0, "running": 0,
                "reserved_slots": 0, "profile": "canned"}
    health = {"ok": True, "running": 0, "waiting": 0, "kv_usage": 0.0}
    snap = {"replicas": [{"id": "a", "hf_model": "m", "health": health, "ledger": ledger(271152)},
                         {"id": "b", "hf_model": "m", "health": health, "ledger": ledger(289056)}],
            "scheduler": {"queues": {"interactive": 0}}}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/healthz":
            return httpx.Response(200, json={"ok": True, "run_id": "canned"})
        if request.url.path == "/admin/replicas":
            return httpx.Response(200, json=snap)
        return httpx.Response(404)

    client = httpx.Client(base_url="http://canned", transport=httpx.MockTransport(handler))
    checks, run_id = preflight.run_checks("http://canned", expect_replicas=2, expect_model="m", runs_dir=tmp_path,
                                          min_free_gb=0, client=client, expect_kv_total=289056, kv_spread_max=0.01)
    failed = _failed(checks)
    assert run_id == "canned"
    assert set(failed) == {"kv_budget[a]", "kv_uniform", "tokenizer"}, failed  # no /admin/tokenize in the canned gateway
    assert "kv_total=271152" in failed["kv_budget[a]"] and "spread 6.19%" in failed["kv_uniform"]


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
