"""Milestone 4 units: the two-bucket pacer, reservations as standing holds, the burst manager's
lending and revocation (docs/DESIGN.md section 7)."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest

from slotgw.admission import QueueTimeout, ReplicaLedger, ReservationCapacity, Scheduler
from slotgw.calibration import ThroughputProfile
from slotgw.config import AdmissionSettings, BurstSettings, Shape
from slotgw.pacing import Pacer

S8_15 = Shape(name="small.8k.15tps", model="small", max_context=8192, tps=15, max_queue_s=1)
S8_30 = Shape(name="small.8k.30tps", model="small", max_context=8192, tps=30, max_queue_s=1)
S8_60 = Shape(name="small.8k.60tps", model="small", max_context=8192, tps=60, max_queue_s=1)
THREE_AT_30 = "1:100,2:60,3:40,4:30"  # seats: 3 at 30 tps (40 x 0.9 = 36), 4 at 15 tps


@dataclass
class Dummy:
    id: str
    model_alias: str = "small"


def ledger(id="r0", curve=THREE_AT_30, kv_total=181_936, **settings) -> ReplicaLedger:
    return ReplicaLedger(Dummy(id), ThroughputProfile.from_curve_string(curve), kv_total, 16, AdmissionSettings(**settings))


def test_pacer_lent_rate_speeds_delivery_and_classifies_tokens():
    p = Pacer(tps=10, bucket_depth_s=0.0)
    assert p.schedule(20, 0.0) == pytest.approx(2.0)  # 20 tokens at 10/s
    assert p.guaranteed_tokens == 20 and p.burst_tokens == 0
    p.set_burst_rate(30, 2.0)  # 40 tok/s combined
    assert p.schedule(40, 2.0) == pytest.approx(3.0)
    assert p.burst_tokens == 30 and p.guaranteed_tokens == 30  # the burst bucket paid its share
    p.set_burst_rate(0, 3.0)  # revoked: back to the shape's rate
    assert p.schedule(10, 3.0) == pytest.approx(4.0)
    assert p.burst_tokens == 30 and p.guaranteed_tokens == 40
    assert p.burst_rate_max == 30 and p.throttled_s == pytest.approx(4.0)


def test_pacer_burst_credit_is_one_bucket_depth_and_never_banked():
    p = Pacer(tps=10, bucket_depth_s=1.0)
    p.set_burst_rate(50, 0.0)
    assert p.schedule(60, 0.0) == 0.0  # 10 guaranteed + 50 burst of depth at once
    assert p.burst_tokens == 50 and p.guaranteed_tokens == 10
    p.set_burst_rate(50, 100.0)  # same rate: no change, no new credit
    assert p.schedule(60, 100.0) == pytest.approx(100.0)  # both buckets refilled over 100 s of idling
    p.set_burst_rate(20, 100.0)  # a new rate re-opens the burst bucket with 20 x 1 s of credit
    t = p.schedule(60, 100.0)  # guaranteed bucket is empty (just drained), burst gives 20, 40 more at 30/s
    assert t == pytest.approx(100.0 + 40 / 30)


def test_ledger_lends_step_rate_above_each_guarantee_and_ignores_idle_slots():
    led = ledger(curve="1:100,2:60,4:40")
    a = led.hold("a", S8_30, led.kv_for(S8_30), 100)
    a.pacer = Pacer(30)
    assert led.update_burst(0.0, lend=True) == pytest.approx(60)  # 100 x 0.9 - 30
    assert a.pacer.burst_rate == pytest.approx(60)
    b = led.hold("b", S8_15, led.kv_for(S8_15), 100)
    b.pacer = Pacer(15)
    led.update_burst(1.0, lend=True)  # two streams: 60 x 0.9 = 54 each
    assert a.pacer.burst_rate == pytest.approx(24) and b.pacer.burst_rate == pytest.approx(39)
    led.reserve("res", 0, S8_30)  # an idle reserved slot is a seat, not an engine stream
    led.update_burst(2.0, lend=True)
    assert led.n_running == 3 and led.n_engine == 2 and a.pacer.burst_rate == pytest.approx(24)
    led.update_burst(3.0, lend=False)
    assert a.pacer.burst_rate == 0 and b.pacer.burst_rate == 0 and led.burst_rate_lent == 0


async def test_reservation_slots_are_seats_and_the_tenant_takes_them_first():
    led = ledger()
    sched = Scheduler([led], led.settings)
    res = sched.reserve("chat", S8_30, 2)
    assert led.n_running == 2 and led.n_engine == 0 and led.reserved_idle == 2 and res.busy == 0
    on_demand = await sched.admit("d0", S8_30, 100)  # the third seat
    with pytest.raises(QueueTimeout) as e:
        await sched.admit("d1", S8_30, 100, max_wait_s=0.05)
    assert e.value.reason == "seat"
    r1 = await sched.admit("c0", S8_30, 100, tenant="chat")
    r2 = await sched.admit("c1", S8_30, 100, tenant="chat")
    assert r1.reserved and r1.occupant == "c0" and r1.reservation_id == res.id and r2.occupant == "c1"
    assert led.n_engine == 3 and res.busy == 2 and res.served == 2
    with pytest.raises(QueueTimeout) as e:  # all of the tenant's slots busy: it waits on its own slots
        await sched.admit("c2", S8_30, 100, tenant="chat", max_wait_s=0.05)
    assert e.value.reason == "reservation_busy" and res.timeouts == 1
    waiter = asyncio.create_task(sched.admit("c3", S8_30, 100, tenant="chat", max_wait_s=1.0))
    await asyncio.sleep(0.01)
    assert len(res.queue) == 1
    sched.release(r1)  # vacated, handed to the waiter
    got = await waiter
    assert got is r1 and got.occupant == "c3" and res.served == 3 and led.n_running == 3
    # nesting: a smaller shape of the same tenant uses the reservation; a faster one or another tenant does not
    assert sched.reservation_for("chat", S8_15) is res
    assert sched.reservation_for("chat", S8_60) is None and sched.reservation_for("agent", S8_30) is None
    sched.release(r2)
    ended = sched.unreserve(res.id)
    assert ended is res and res.t_end is not None
    assert led.n_running == 2  # the idle slot went at once; the occupied one stays until its request ends
    sched.release(r1)
    sched.release(on_demand)
    assert led.n_running == 0 and sched.unreserve(res.id) is None


def test_reservation_is_all_or_nothing():
    led = ledger()
    sched = Scheduler([led], led.settings)
    with pytest.raises(ReservationCapacity) as e:
        sched.reserve("chat", S8_30, 4)
    assert e.value.placed == 3 and e.value.reason == "seat" and led.n_running == 0
    res = sched.reserve("chat", S8_30, 3)
    assert len(res.slots) == 3 and led.seats_now() == 3


async def test_context_burst_yields_to_a_guaranteed_request_by_policy():
    async def scenario(policy: str, grace: float = 0.05):
        led = ledger(kv_total=16_000)  # usable 15,200: a 9,104-token burst and an 8,192 shape do not both fit
        sched = Scheduler([led], led.settings, BurstSettings(revoke_policy=policy, grace_s=grace))
        b = await sched.admit("b0", S8_30, 100, kv_tokens=9_100, burst_ctx=True)
        assert b.burst_ctx and b.kv_tokens == 9_104 and sched.bursts_admitted == 1
        g = asyncio.create_task(sched.admit("g0", S8_30, 100, max_wait_s=1.0))
        await asyncio.sleep(0.01)
        return led, sched, b, g

    led, sched, b, g = await scenario("abort")
    assert b.revoke_at is not None
    sched.tick()
    assert b.revoked and b.released and sched.revocations == 1
    hg = await g
    assert not hg.burst_ctx and led.n_running == 1

    led, sched, b, g = await scenario("grace")
    sched.tick()
    assert not b.revoked  # grace not over
    await asyncio.sleep(0.06)
    sched.tick()
    assert b.revoked and (await g).request_id == "g0"

    led, sched, b, g = await scenario("finish")
    await asyncio.sleep(0.06)
    sched.tick()
    assert b.revoke_at is None and not b.revoked
    sched.release(b)  # the burst finishes on its own; the guaranteed request gets in
    assert (await g).request_id == "g0" and sched.revocations == 0


async def test_context_burst_is_admitted_only_when_nothing_else_waits():
    led = ledger()
    sched = Scheduler([led], led.settings, BurstSettings())
    holds = [await sched.admit(f"c{i}", S8_30, 100) for i in range(3)]  # all seats
    interactive = asyncio.create_task(sched.admit("i0", S8_30, 100, max_wait_s=1.0))
    await asyncio.sleep(0.01)
    burst = asyncio.create_task(sched.admit("b0", S8_30, 100, kv_tokens=9_000, burst_ctx=True, max_wait_s=1.0))
    await asyncio.sleep(0.01)
    assert sched.queue_lengths() == {"interactive": 1, "lazy": 0, "burst": 1}
    sched.release(holds[0])  # one seat: the interactive waiter takes it, the burst keeps waiting
    hi = await interactive
    assert hi.request_id == "i0" and sched.queue_lengths()["burst"] == 1
    sched.release(holds[1])
    hb = await burst
    assert hb.burst_ctx and sched.bursts_admitted == 1
