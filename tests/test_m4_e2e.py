"""Milestone 4 through the real HTTP path: reservations, tps burst, context burst and revocation."""

from __future__ import annotations

import asyncio
import time

import httpx

from loadgen.content import words
from slotgw.config import AdmissionSettings, BurstSettings

from .test_multi_replica import MSG, TWO_SEATS, Cluster, _admin, _post

FAST = "1:200,2:100,3:60,4:40"  # the engine far ahead of the shapes: room to lend


async def _reserve(url: str, body: dict) -> httpx.Response:
    async with httpx.AsyncClient(base_url=url, timeout=10) as c:
        return await c.post("/admin/reservations", json=body)


async def _unreserve(url: str, rid: str) -> httpx.Response:
    async with httpx.AsyncClient(base_url=url, timeout=10) as c:
        return await c.delete(f"/admin/reservations/{rid}")


async def test_reserved_tenant_is_served_at_once_while_on_demand_waits(tmp_path):
    cluster = Cluster([TWO_SEATS], AdmissionSettings(), queue_s=1.0, meter_path=tmp_path / "m.sqlite")
    try:
        gw = cluster.gateway.url
        r = await _reserve(gw, {"tenant": "chat", "shape": "small.8k.30tps", "slots": 1, "load_run": "lr1", "seed": 7})
        assert r.status_code == 201, r.text
        rid = r.json()["id"]
        assert r.json()["replicas"] == ["fake-0"] and r.json()["load_run"] == "lr1"
        snap = await _admin(gw)
        lg = snap["replicas"][0]["ledger"]
        assert lg["running"] == 1 and lg["reserved_slots"] == 1 and lg["reserved_idle"] == 1 and lg["engine_active"] == 0
        # one seat left for on-demand: the first agent request takes it, the second times out;
        # the chat tenant's request goes straight into its reserved slot
        body = {"model": "small.8k.30tps", "messages": MSG, "max_tokens": 120, "ignore_eos": True}
        d1 = asyncio.create_task(_post(gw, body, key="sk-local-agent"))
        await asyncio.sleep(0.3)
        d2 = asyncio.create_task(_post(gw, body, key="sk-local-agent"))
        c1 = asyncio.create_task(_post(gw, body, key="sk-local-chat"))
        await asyncio.sleep(0.3)
        snap = await _admin(gw)
        lg = snap["replicas"][0]["ledger"]
        assert lg["running"] == 2 and lg["reserved_idle"] == 0 and lg["engine_active"] == 2
        assert snap["scheduler"]["reservations"][0]["busy"] == 1
        c2 = asyncio.create_task(_post(gw, body, key="sk-local-chat"))  # its only slot is busy: waits on it
        rd1, rd2, rc1, rc2 = await asyncio.gather(d1, d2, c1, c2)
        assert rd1.status_code == 200 and rd2.status_code == 503 and rd2.json()["error"]["code"] == "capacity"
        assert rc1.status_code == 200, rc1.text
        slot = rc1.json()["slotgw"]
        assert slot["reservation_id"] == rid and slot["billing"]["reserved"] is True and slot["queued_ms"] < 200
        assert rc2.status_code == 503 and "reservation_busy" in rc2.json()["error"]["message"]
        r = await _unreserve(gw, rid)
        assert r.status_code == 200 and r.json()["served"] == 1 and r.json()["timeouts"] == 1
        assert (await _unreserve(gw, rid)).status_code == 404
        rows = {x["request_id"]: x for x in cluster.rows()}
        served = rows[slot["request_id"]]
        assert served["reservation_id"] == rid and served["status"] == "ok"
        refused = [x for x in rows.values() if x["status"] == "refused"]
        assert sorted(x["abort_reason"] for x in refused) == ["capacity:reservation_busy", "capacity:seat"]
        res_rows = cluster.rows("reservations")
        assert len(res_rows) == 1 and res_rows[0]["t_end"] and res_rows[0]["slots"] == 1 and res_rows[0]["served"] == 1
        assert res_rows[0]["load_run"] == "lr1" and res_rows[0]["seed"] == 7 and res_rows[0]["price_per_second"] > 0
        snap = await _admin(gw)
        assert snap["replicas"][0]["ledger"]["running"] == 0 and snap["scheduler"]["reservations"] == []
    finally:
        cluster.stop()


async def test_reservation_that_does_not_fit_is_refused_whole(tmp_path):
    cluster = Cluster([TWO_SEATS], AdmissionSettings(), queue_s=1.0, meter_path=tmp_path / "m.sqlite")
    try:
        gw = cluster.gateway.url
        r = await _reserve(gw, {"tenant": "chat", "shape": "small.8k.30tps", "slots": 3})
        assert r.status_code == 409 and r.json()["error"]["code"] == "reservation_capacity"
        assert (await _admin(gw))["replicas"][0]["ledger"]["running"] == 0
        assert (await _reserve(gw, {"tenant": "nobody", "slots": 1})).status_code == 404
    finally:
        cluster.stop()


async def test_tps_burst_delivers_above_the_shape_rate_and_counts_burst_tokens(tmp_path):
    cluster = Cluster([FAST], AdmissionSettings(), queue_s=5.0, meter_path=tmp_path / "m.sqlite", burst=BurstSettings())
    try:
        gw = cluster.gateway.url
        # 15 tps shape, 90 tokens: paced alone that is 1 s head start + 75 tokens / 15 = 6 s; the idle
        # engine runs at 200 tok/s and the burst manager lends 200 x 0.9 - 15 = 165 tok/s
        body = {"model": "small.8k.15tps", "messages": MSG, "max_tokens": 90, "ignore_eos": True, "stream": True}
        t0 = time.perf_counter()
        async with httpx.AsyncClient(base_url=gw, timeout=60) as c, c.stream(
            "POST", "/v1/chat/completions", json=body, headers={"Authorization": "Bearer sk-local-chat"}
        ) as r:
            final = None
            async for line in r.aiter_lines():
                if line.startswith("data:") and '"slotgw"' in line and '"billing"' in line:
                    import json

                    final = json.loads(line[5:])
        elapsed = time.perf_counter() - t0
        assert elapsed < 2.5, elapsed
        slot = final["slotgw"]
        assert slot["burst_tokens"] >= 60 and slot["burst_rate_max"] > 100
        assert slot["billing"]["billed_s"] < slot["billing"]["bound_s"]  # the burst discount
        row = next(x for x in cluster.rows() if x["request_id"] == slot["request_id"])
        assert row["output_tokens_burst"] == slot["burst_tokens"] and row["output_tokens_guaranteed"] + row["output_tokens_burst"] == 90
        assert (await _admin(gw))["scheduler"]["burst"]["enabled"] is True
    finally:
        cluster.stop()


async def test_context_burst_is_admitted_and_revoked_for_a_guaranteed_request(tmp_path):
    async def run(policy: str):
        cluster = Cluster(
            [TWO_SEATS], AdmissionSettings(), queue_s=5.0, meter_path=tmp_path / f"{policy}.sqlite",
            burst=BurstSettings(revoke_policy=policy, grace_s=0.3, max_context_tokens=16384), kv_tokens=16_000,
        )
        try:
            gw = cluster.gateway.url
            big = [{"role": "user", "content": words(1, 9_000)}]  # ~9k tokens > the 8k shape: a context burst
            burst = asyncio.create_task(
                _post(gw, {"model": "small.8k.30tps", "messages": big, "max_tokens": 200, "ignore_eos": True}, key="sk-local-chat")
            )
            await asyncio.sleep(0.5)
            snap = await _admin(gw)
            assert snap["replicas"][0]["ledger"]["burst_running"] == 1 and snap["scheduler"]["bursts_admitted"] == 1
            t0 = time.perf_counter()
            guaranteed = await _post(gw, {"model": "small.8k.30tps", "messages": MSG, "max_tokens": 10, "ignore_eos": True}, key="sk-local-agent")
            waited = time.perf_counter() - t0
            assert guaranteed.status_code == 200, guaranteed.text
            rb = await burst
            snap = await _admin(gw)
            rows = cluster.rows()
            brow = next(x for x in rows if x["class_ctx"] == "burst")
            return rb, waited, snap["scheduler"], brow
        finally:
            cluster.stop()

    rb, waited, sched, brow = await run("abort")
    assert rb.status_code == 503 and rb.json()["error"]["code"] == "burst_revoked"
    assert waited < 1.5 and sched["revocations"] == 1
    assert brow["status"] == "revoked" and brow["revoked"] == 1 and brow["wasted_tokens"] > 0

    rb, waited, sched, brow = await run("finish")
    assert rb.status_code == 200 and rb.json()["slotgw"]["class_ctx"] == "burst"
    assert waited > 1.0 and sched["revocations"] == 0  # the guaranteed request waited for the burst (200 tokens at 100 tok/s)
    assert brow["status"] == "ok" and brow["revoked"] == 0
