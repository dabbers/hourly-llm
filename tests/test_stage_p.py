"""Stage P: reserved slots without an output ceiling (pacing off), the prefill gate for reserved requests
with the fair order across tenants, and scheduled turns in the load generator."""

from __future__ import annotations

import asyncio
import json
import time

import httpx
import pytest

from loadgen.plan import Plan, Session, Step, TenantPlan
from loadgen.runner import LoadRunner
from slotgw.admission import QueueTimeout
from slotgw.config import AdmissionSettings

from .test_admission import S8_30
from .test_affinity import sched
from .test_loadgen import tiny_scenario
from .test_multi_replica import Cluster

FAST = "1:200,2:100,3:60,4:40"


async def settle():
    for _ in range(5):
        await asyncio.sleep(0)


# ---- the prefill gate -----------------------------------------------------------------------------


async def test_a_reserved_request_takes_a_free_slot_at_once_with_the_gate_off():
    sc = sched(n=1, max_prefill_tokens_in_flight=2048, ingest_fair_share=True)
    sc.reserve("a", S8_30, 2)
    h1 = await sc.admit("a1", S8_30, 1500, tenant="a")
    h2 = await sc.admit("a2", S8_30, 1500, tenant="a")  # 3,000 tokens prefilling: the cap does not apply to slots
    assert h1.reserved and h2.reserved and h1 is not h2
    assert sc.gate_holds == 0 and sc.snapshot()["ingest"]["reserved_prefill_gate"] is False


async def test_the_gate_holds_a_reserved_request_until_the_prefill_cap_has_room():
    sc = sched(n=1, max_prefill_tokens_in_flight=2048, reserved_prefill_gate=True)
    sc.reserve("a", S8_30, 2)
    h1 = await sc.admit("a1", S8_30, 1500, tenant="a")
    t2 = asyncio.ensure_future(sc.admit("a2", S8_30, 1500, tenant="a"))
    await settle()
    assert not t2.done() and sc.gate_holds == 1  # a slot is free; 1,500 + 1,500 is over the cap
    sc.mark_prefill_done(h1)
    await settle()
    assert t2.done() and t2.result().reserved and t2.result() is not h1


async def test_the_gate_lets_a_request_through_alone_whatever_its_size_and_is_idle_with_no_cap():
    sc = sched(n=1, max_prefill_tokens_in_flight=2048, reserved_prefill_gate=True)
    sc.reserve("a", S8_30, 1)
    assert (await sc.admit("a1", S8_30, 6000, tenant="a")).reserved  # nothing else prefilling: never starved
    sc = sched(n=1, reserved_prefill_gate=True)
    sc.reserve("a", S8_30, 2)
    await sc.admit("a1", S8_30, 6000, tenant="a")
    assert (await sc.admit("a2", S8_30, 6000, tenant="a")).reserved  # no cap: only the slot count limits


async def _order(fair: bool) -> list[str]:
    sc = sched(n=1, max_prefill_tokens_in_flight=2048, reserved_prefill_gate=True, ingest_fair_share=fair)
    sc.reserve("heavy", S8_30, 3)
    sc.reserve("light", S8_30, 1)
    h0 = await sc.admit("h0", S8_30, 2000, tenant="heavy")
    tasks = {}
    for rid, tokens, tenant in (("h1", 2000, "heavy"), ("h2", 2000, "heavy"), ("l1", 100, "light")):
        tasks[rid] = asyncio.ensure_future(sc.admit(rid, S8_30, tokens, tenant=tenant))
        await settle()
    assert not any(t.done() for t in tasks.values())  # all three have a free slot and are held by the cap
    order, running = [], [h0]
    for _ in range(3):
        sc.mark_prefill_done(running[-1])
        await settle()
        for rid, t in tasks.items():
            if t.done() and rid not in order:
                order.append(rid)
                running.append(t.result())
    return order


async def test_held_requests_are_released_first_come_without_fair_share():
    assert await _order(fair=False) == ["h1", "h2", "l1"]


async def test_held_requests_are_released_in_the_tenants_fair_order_with_fair_share():
    assert (await _order(fair=True))[0] == "l1"  # the light tenant's prompt goes ahead of the heavy backlog


async def test_a_request_held_by_the_gate_past_the_queue_limit_is_refused_for_prefill():
    sc = sched(n=1, max_prefill_tokens_in_flight=2048, reserved_prefill_gate=True)
    sc.reserve("a", S8_30, 2)
    await sc.admit("a1", S8_30, 1500, tenant="a")
    with pytest.raises(QueueTimeout) as e:
        await sc.admit("a2", S8_30, 1500, tenant="a", max_wait_s=0.05)
    assert e.value.reason == "prefill"


# ---- pacing off on a reserved slot ------------------------------------------------------------------


async def _stream(url: str, max_tokens: int) -> dict:
    final = {}
    async with httpx.AsyncClient(base_url=url, timeout=30) as c:
        body = {"model": "small.8k.30tps", "messages": [{"role": "user", "content": "Count the seats."}],
                "max_tokens": max_tokens, "ignore_eos": True, "stream": True}
        async with c.stream("POST", "/v1/chat/completions", json=body, headers={"Authorization": "Bearer sk-local-chat"}) as r:
            assert r.status_code == 200
            async for line in r.aiter_lines():
                if line.startswith("data:") and line[5:].strip() != "[DONE]":
                    obj = json.loads(line[5:])
                    if "slotgw" in obj and "usage" in obj:
                        final = obj
    return final


async def test_a_reserved_request_with_pacing_off_is_delivered_at_engine_speed(tmp_path):
    cluster = Cluster([FAST], AdmissionSettings(), queue_s=5.0, meter_path=tmp_path / "m.sqlite", pacing=False)
    try:
        gw = cluster.gateway.url
        async with httpx.AsyncClient(base_url=gw, timeout=10) as c:
            r = await c.post("/admin/reservations", json={"tenant": "chat", "shape": "small.8k.30tps", "slots": 1})
            assert r.status_code == 201, r.text
        first = await _stream(gw, 150)
        second = await _stream(gw, 20)  # the only slot was vacated when the engine finished
        assert first["slotgw"]["pacing"] is False and second["slotgw"]["queued_ms"] < 200
        rows = sorted(cluster.rows(), key=lambda r: r["t_arrival"])
    finally:
        cluster.stop()
    row = rows[0]
    assert row["status"] == "ok" and row["reservation_id"] and row["throttled_ms"] == 0
    assert row["delivered_tokens"] == row["output_tokens"] == 150
    rate = (row["output_tokens"] - 1) / (row["t_last_token"] - row["t_first_token"])
    assert rate > 100  # the engine makes 200 tokens a second alone; the 30 tps shape would have paced it to about 37
    assert abs(row["t_last_token"] - row["t_engine_done"]) < 0.2


# ---- scheduled turns ----------------------------------------------------------------------------------


async def _run_two_turns(tmp_path, timing: str, answer_s: float) -> tuple[float, list[float | None]]:
    scenario = tiny_scenario(["chat"])
    steps = [Step(i=0, user_tokens=10, max_tokens=30, stream=True, pause_before_s=0.0, content_seed=1),
             Step(i=1, user_tokens=10, max_tokens=30, stream=True, pause_before_s=0.1, content_seed=2)]
    tp = TenantPlan(name="chat", api_key="k", shape="small.8k.300tps", profile_kind="chat", sessions=[Session("s0", 0.0, None, steps)])
    plan = Plan(scenario="tiny", seed=1, duration_s=8.0, hour_length_s=6.0, tenants=[tp])
    runner = LoadRunner(scenario, plan, "http://localhost:1", tmp_path, turn_timing=timing)
    runner._t_end = time.perf_counter() + 60
    sent: list[float] = []
    late: list[float | None] = []

    async def fake_step(client, tp_, session, step, history, turn_late_s=None):
        sent.append(time.perf_counter())
        late.append(turn_late_s)
        await asyncio.sleep(answer_s)
        return True, "q", "a"

    runner._run_step = fake_step  # type: ignore[method-assign]
    await runner._run_session(None, tp, tp.sessions[0])  # type: ignore[arg-type]
    return sent[1] - sent[0], late


async def test_closed_loop_sends_the_next_turn_a_think_time_after_the_answer(tmp_path):
    gap, late = await _run_two_turns(tmp_path, "closed", answer_s=0.05)
    assert 0.1 < gap < 0.5 and late == [None, None]


async def test_scheduled_turns_do_not_come_sooner_after_a_fast_answer(tmp_path):
    # planned gap: think 0.1 s + nominal answer 1 s + 30 tokens / 300 tps = 1.2 s, whatever the answer took
    gap, late = await _run_two_turns(tmp_path, "scheduled", answer_s=0.05)
    assert gap == pytest.approx(1.2, abs=0.15)
    assert late[0] is None and late[1] == 0.0


async def test_a_scheduled_turn_waits_for_a_late_reply_and_records_how_late_it_was(tmp_path):
    gap, late = await _run_two_turns(tmp_path, "scheduled", answer_s=1.5)
    assert gap == pytest.approx(1.5, abs=0.15)
    assert late[1] == pytest.approx(0.3, abs=0.15)
