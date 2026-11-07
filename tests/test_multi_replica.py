"""Milestone 3 acceptance through the real HTTP path: several fake replicas behind one gateway."""

from __future__ import annotations

import asyncio
from pathlib import Path

import httpx
import pytest

from slotgw.app import create_app
from slotgw.config import (
    AdmissionSettings,
    BurstSettings,
    GatewaySettings,
    PacingSettings,
    ReplicaConfig,
    Shape,
    load_config,
)
from slotgw.meter import Meter
from slotgw.replicas.fake_server import create_fake_app

from .servers import ServerHandle

REPO = Path(__file__).resolve().parents[1]
TWO_SEATS = "1:100,2:40,3:20"  # seats: 3 at 15 tps, 2 at 30 tps, 1 at 60 tps
ONE_SLOW_SEAT = "1:20,2:12,3:8"  # seats: 1 at 15 tps; engine slow enough to see preemption
MSG = [{"role": "user", "content": "Tell me about ledgers and seats."}]


def _shapes(queue_s: float, lazy_queue_s: float = 60.0) -> list[Shape]:
    return [
        Shape(name="small.8k.15tps", model="small", max_context=8192, tps=15, price_per_second=9e-6, max_queue_s=queue_s),
        Shape(name="small.8k.30tps", model="small", max_context=8192, tps=30, price_per_second=1.8e-5, max_queue_s=queue_s),
        Shape(name="small.8k.60tps", model="small", max_context=8192, tps=60, price_per_second=3.6e-5, max_queue_s=queue_s),
        Shape(name="small.8k.15tps.lazy", model="small", max_context=8192, tps=15, price_per_second=4.5e-6, lazy=True,
              max_queue_s=lazy_queue_s),
    ]


class Cluster:
    def __init__(
        self,
        curves: list[str],
        admission: AdmissionSettings,
        queue_s: float,
        meter_path: Path,
        burst: BurstSettings | None = None,
        kv_tokens: int | None = None,
    ):
        self.fakes = [ServerHandle(create_fake_app(curve=c, prefill_tps=50_000.0)).start() for c in curves]
        config = load_config(REPO / "config")
        config = config.model_copy(
            update={
                "shapes": _shapes(queue_s),
                "replicas": [
                    ReplicaConfig(id=f"fake-{i}", kind="fake", url=f.url, model="small", curve=c, kv_tokens=kv_tokens)
                    for i, (f, c) in enumerate(zip(self.fakes, curves, strict=True))
                ],
                "gateway": GatewaySettings(
                    pacing=PacingSettings(enabled=True, bucket_depth_s=1.0), admission=admission,
                    burst=burst or BurstSettings(enabled=False),
                ),
            }
        )
        self.meter_path = meter_path
        self.meter = Meter(meter_path, run_id="cluster")
        self.gateway = ServerHandle(create_app(config, self.meter)).start()

    def stop(self) -> None:
        self.gateway.stop()
        for f in self.fakes:
            f.stop()

    def rows(self, table: str = "requests"):
        import sqlite3

        conn = sqlite3.connect(str(self.meter_path))
        conn.row_factory = sqlite3.Row
        rows = [dict(r) for r in conn.execute(f"SELECT * FROM {table}").fetchall()]
        conn.close()
        return rows


@pytest.fixture
def three_replicas(tmp_path):
    c = Cluster([TWO_SEATS] * 3, AdmissionSettings(lazy_max_share=1.0), queue_s=2.0, meter_path=tmp_path / "m.sqlite")
    yield c
    c.stop()


async def _post(url, body, key="sk-local-chat"):
    async with httpx.AsyncClient(base_url=url, timeout=120) as c:
        return await c.post("/v1/chat/completions", json=body, headers={"Authorization": f"Bearer {key}"})


async def _admin(url):
    async with httpx.AsyncClient(base_url=url, timeout=10) as c:
        return (await c.get("/admin/replicas")).json()


async def test_admission_never_exceeds_seats_and_overflow_is_refused(three_replicas):
    gw = three_replicas.gateway.url
    # 240 tokens at 40 tok/s per stream (two streams per replica) hold a seat ~6 s, longer than the
    # 2 s queue window, so the four requests beyond the six seats time out
    body = {"model": "small.8k.30tps", "messages": MSG, "max_tokens": 240, "ignore_eos": True}
    over_seats = []

    async def watch():
        for _ in range(60):
            snap = await _admin(gw)
            for rep in snap["replicas"]:
                lg = rep["ledger"]
                if lg["running"] > lg["seats_now"]:
                    over_seats.append((rep["id"], lg["running"], lg["seats_now"]))
            await asyncio.sleep(0.1)

    watcher = asyncio.create_task(watch())
    results = await asyncio.gather(*(_post(gw, body) for _ in range(10)))
    await watcher
    codes = sorted(r.status_code for r in results)
    assert codes == [200] * 6 + [503] * 4, codes  # 3 replicas x 2 seats at 30 tps; the rest time out in 2 s
    assert not over_seats, over_seats
    refused = [r for r in results if r.status_code == 503]
    assert all(r.json()["error"]["code"] == "capacity" and r.headers.get("retry-after") for r in refused)
    placed = sorted(r.json()["slotgw"]["replica"] for r in results if r.status_code == 200)
    assert placed == ["fake-0", "fake-0", "fake-1", "fake-1", "fake-2", "fake-2"]  # spread by load
    rows = three_replicas.rows()
    assert sum(r["status"] == "refused" and r["abort_reason"] == "capacity:seat" for r in rows) == 4
    assert all(r["queued_ms"] >= 1900 for r in rows if r["status"] == "refused")
    snap = await _admin(gw)
    assert all(rep["ledger"]["running"] == 0 for rep in snap["replicas"])
    assert snap["scheduler"]["timeouts"] == 4


async def test_fast_shape_avoids_replica_running_lazy_work(three_replicas):
    gw = three_replicas.gateway.url
    lazy_body = {"model": "small.8k.15tps.lazy", "messages": MSG, "max_tokens": 200, "ignore_eos": True}
    lazy_task = asyncio.create_task(_post(gw, lazy_body, key="sk-local-batch"))
    await asyncio.sleep(0.5)
    snap = await _admin(gw)
    lazy_on = [rep["id"] for rep in snap["replicas"] if rep["ledger"]["lazy_running"] == 1]
    assert len(lazy_on) == 1, snap
    r = await _post(gw, {"model": "small.8k.60tps", "messages": MSG, "max_tokens": 30, "ignore_eos": True}, key="sk-local-agent")
    assert r.status_code == 200, r.text
    assert r.json()["slotgw"]["replica"] != lazy_on[0]  # a 60 tps seat there would cost the lazy replica 2 seats
    lazy = await lazy_task
    assert lazy.status_code == 200


async def test_lazy_waits_while_interactive_is_queued(tmp_path):
    """With every 30 tps seat taken, an interactive request queues; a lazy request arriving after it
    must not be admitted first when a seat frees."""
    cluster = Cluster([TWO_SEATS] * 3, AdmissionSettings(lazy_max_share=1.0), queue_s=20.0, meter_path=tmp_path / "m.sqlite")
    gw = cluster.gateway.url
    busy = {"model": "small.8k.30tps", "messages": MSG, "max_tokens": 150, "ignore_eos": True}
    holders = [asyncio.create_task(_post(gw, busy)) for _ in range(6)]
    await asyncio.sleep(0.4)
    interactive = asyncio.create_task(_post(gw, {"model": "small.8k.30tps", "messages": MSG, "max_tokens": 10, "ignore_eos": True}))
    await asyncio.sleep(0.1)
    lazy = asyncio.create_task(_post(gw, {"model": "small.8k.15tps.lazy", "messages": MSG, "max_tokens": 10, "ignore_eos": True}, key="sk-local-batch"))
    await asyncio.sleep(0.2)
    snap = await _admin(gw)
    assert snap["scheduler"]["queues"] == {"interactive": 1, "lazy": 1, "burst": 0}
    ri = await interactive
    rl = await lazy
    assert ri.status_code == 200 and rl.status_code == 200
    rows = {r["request_id"]: r for r in cluster.rows()}
    t_i = rows[ri.json()["slotgw"]["request_id"]]["t_admitted"]
    t_l = rows[rl.json()["slotgw"]["request_id"]]["t_admitted"]
    assert t_i < t_l
    await asyncio.gather(*holders)
    cluster.stop()


async def test_requeue_preempts_lazy_run_and_it_completes_later(tmp_path):
    settings = AdmissionSettings(lazy_preempt="requeue", lazy_max_share=1.0)
    cluster = Cluster([ONE_SLOW_SEAT], settings, queue_s=5.0, meter_path=tmp_path / "m.sqlite")
    try:
        gw = cluster.gateway.url
        lazy = asyncio.create_task(
            _post(gw, {"model": "small.8k.15tps.lazy", "messages": MSG, "max_tokens": 120, "ignore_eos": True}, key="sk-local-batch")
        )
        await asyncio.sleep(1.0)  # the lazy run holds the only seat and has produced ~20 tokens
        r = await _post(gw, {"model": "small.8k.15tps", "messages": MSG, "max_tokens": 30, "ignore_eos": True})
        assert r.status_code == 200, r.text
        assert r.json()["slotgw"]["queued_ms"] < 1000  # it did not wait for the lazy run to finish
        rl = await lazy
        assert rl.status_code == 200, rl.text
        slot = rl.json()["slotgw"]
        assert slot["preempted_runs"] == 1
        assert rl.json()["usage"]["completion_tokens"] == 120
        row = next(x for x in cluster.rows() if x["request_id"] == slot["request_id"])
        assert row["preempted_runs"] == 1 and row["wasted_tokens"] > 0 and row["delivered_tokens"] == 120
        assert row["status"] == "ok" and row["output_tokens"] == 120
        snap = await _admin(gw)
        assert snap["scheduler"]["preemptions"] == 1 and snap["replicas"][0]["ledger"]["running"] == 0
    finally:
        cluster.stop()


async def test_unsellable_shape_is_refused_immediately(tmp_path):
    cluster = Cluster([ONE_SLOW_SEAT], AdmissionSettings(), queue_s=5.0, meter_path=tmp_path / "m.sqlite")
    try:
        r = await _post(cluster.gateway.url, {"model": "small.8k.60tps", "messages": MSG, "max_tokens": 5}, key="sk-local-agent")
        assert r.status_code == 503 and r.json()["error"]["code"] == "shape_unsellable"
        assert any(x["abort_reason"] == "shape_unsellable" for x in cluster.rows())
    finally:
        cluster.stop()
