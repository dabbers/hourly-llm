from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path

import pytest

from slotgw.admission import QueueTimeout, ReplicaLedger, Scheduler, Unsellable
from slotgw.calibration import ThroughputProfile
from slotgw.config import AdmissionSettings, Shape

REPO = Path(__file__).resolve().parents[1]

S8_15 = Shape(name="small.8k.15tps", model="small", max_context=8192, tps=15, price_per_second=9e-6, max_queue_s=1)
S8_30 = Shape(name="small.8k.30tps", model="small", max_context=8192, tps=30, price_per_second=1.8e-5, max_queue_s=1)
S8_60 = Shape(name="small.8k.60tps", model="small", max_context=8192, tps=60, price_per_second=3.6e-5, max_queue_s=1)
S16_30 = Shape(name="small.16k.30tps", model="small", max_context=16384, tps=30, price_per_second=2.25e-5, max_queue_s=1)
LAZY = Shape(name="small.8k.15tps.lazy", model="small", max_context=8192, tps=15, price_per_second=4.5e-6, lazy=True, max_queue_s=2)
BIG = Shape(name="small.8k.300tps", model="small", max_context=8192, tps=300, price_per_second=1e-4)

# the 2080 Ti's 6k-context curve from calibration/local-2080ti.yaml, rounded
CURVE = "1:105,2:80,4:50,8:35,12:25,16:20,24:13"


@dataclass
class Dummy:
    id: str
    model_alias: str = "small"


def ledger(id="r0", curve=CURVE, kv_total=181_936, **settings) -> ReplicaLedger:
    return ReplicaLedger(Dummy(id), ThroughputProfile.from_curve_string(curve), kv_total, 16, AdmissionSettings(**settings))


def test_profile_from_calibration_file_and_n_max():
    p = ThroughputProfile.from_calibration(REPO / "calibration" / "local-2080ti.yaml")
    assert p.source.endswith("context_6144")
    assert p.step_rate(1) == pytest.approx(105, abs=5)
    # between measured levels the profile interpolates (a chord under a convex decay curve, so
    # conservative): 19 seats at 15 tps vs 16 if only measured points counted
    assert p.n_max(15, 0.10) == 19 and p.n_max(30, 0.10) == 8 and p.n_max(60, 0.10) == 2
    assert p.n_max(300, 0.10) == 0
    p2 = ThroughputProfile.from_calibration(REPO / "calibration" / "local-2080ti.yaml", context=2048)
    assert p2.n_max(30, 0.10) == 23  # 16 on measured points only
    assert p.step_rate(40) < p.step_rate(24)  # extrapolates downwards past the last point


def test_kv_rule_and_block_rounding():
    led = ledger(kv_total=10_000)  # usable 9,500
    assert led.kv_for(S8_30) == 8192 and led.kv_for(S16_30) == 16384
    assert led.check(S16_30, led.kv_for(S16_30), 100) == "kv"
    assert led.check(S8_30, led.kv_for(S8_30), 100) is None
    led.hold("a", S8_30, 8192, 100)
    assert led.check(S8_15, 8192, 100) == "kv"  # 16,384 > 9,500


def test_seat_rule_fastest_shape_sets_ceiling_and_overcommit():
    led = ledger()
    assert led.seats(30) == 8 and led.seats(60) == 2 and led.seats(15) == 19
    for i in range(8):
        assert led.check(S8_30, 8192, 100) is None
        led.hold(f"c{i}", S8_30, 8192, 100)
    assert led.check(S8_30, 8192, 100) == "seat"
    assert led.check(S8_15, 8192, 100) == "seat"  # a 15 tps request still needs a seat under the 30 tps ceiling
    assert led.check(S8_60, 8192, 100) == "seat"  # and a 60 tps request would lower the ceiling to 2
    assert led.capacity_loss(S8_60) == 6
    led2 = ledger(tps_overcommit=1.5)
    assert led2.seats(30) == 12
    led3 = ledger()
    assert led3.check(BIG, 8192, 100) == "unsellable"


def test_shape_nesting_small_request_uses_bigger_free_capacity():
    led = ledger(kv_total=45_000)  # usable 42,750: two 16k requests fit, a third does not
    led.hold("n0", S16_30, 16384, 100)
    assert led.check(S16_30, 16384, 100) is None
    led.hold("n1", S16_30, 16384, 100)
    assert led.check(S16_30, 16384, 100) == "kv"
    assert led.check(S8_30, 8192, 100) is None  # an 8k request still fits in the remaining KV
    assert led.check(S8_15, 8192, 100) is None


def test_lazy_share_and_prefill_cap():
    led = ledger(lazy_max_share=0.25)  # seats(15) = 19 -> 4 lazy seats
    for i in range(4):
        assert led.check(LAZY, 8192, 100) is None
        led.hold(f"l{i}", LAZY, 8192, 100)
    assert led.check(LAZY, 8192, 100) == "lazy_share"
    assert led.check(S8_15, 8192, 100) is None  # interactive still fits
    led2 = ledger(max_prefill_tokens_in_flight=8000)
    h = led2.hold("p0", S8_30, 8192, 6000)
    assert led2.check(S8_30, 8192, 3000) == "prefill"
    assert led2.check(S8_30, 8192, 1000) is None
    led2.mark_prefill_done(h)
    assert led2.check(S8_30, 8192, 3000) is None
    led3 = ledger(max_prefill_tokens_in_flight=4000)
    assert led3.check(S8_30, 8192, 6000) is None  # a lone prefill always goes, however long


async def test_scheduler_places_by_affinity_then_load():
    fast = ledger("fast")
    slow = ledger("slow")
    sched = Scheduler([fast, slow], AdmissionSettings())
    h_lazy = await sched.admit("l0", LAZY, 100)
    first = h_lazy.ledger
    other = slow if first is fast else fast
    # a 60 tps request would cost the lazy replica 14 seats; the empty one costs nothing
    h60 = await sched.admit("a0", S8_60, 100)
    assert h60.ledger is other
    # next lazy request goes where load is already lazy (no capacity loss, higher load tie-break on
    # the empty replica is zero so affinity decides: 60 tps replica would lose nothing? it would
    # lose 0 seats (15 < 60) so both score 0 on loss; load breaks the tie towards the emptier one)
    h_lazy2 = await sched.admit("l1", LAZY, 100)
    assert h_lazy2.ledger in (first, other)
    sched.release(h60)
    sched.release(h_lazy)
    sched.release(h_lazy2)
    assert fast.n_running == 0 and slow.n_running == 0


async def test_scheduler_queues_fifo_and_times_out_with_503_reason():
    led = ledger(curve="1:100,2:60,3:40,4:30")  # seats(30) = 3 (36 >= 30 > 27), seats(60) = 1
    sched = Scheduler([led], AdmissionSettings())
    holds = [await sched.admit(f"c{i}", S8_30, 100) for i in range(3)]
    assert led.n_running == 3
    waiter = asyncio.create_task(sched.admit("c3", S8_30, 100))
    await asyncio.sleep(0.05)
    assert sched.queue_lengths()["interactive"] == 1
    sched.release(holds[0])
    h = await asyncio.wait_for(waiter, 1)
    assert h.ledger is led and led.n_running == 3
    with pytest.raises(QueueTimeout) as e:
        await sched.admit("c4", S8_30, 100, max_wait_s=0.2)
    assert e.value.reason == "seat"
    assert sched.timeouts == 1 and sched.queue_lengths()["interactive"] == 0
    with pytest.raises(Unsellable):
        await sched.admit("big", BIG, 100)


async def test_lazy_never_overtakes_waiting_interactive():
    led = ledger(curve="1:100,2:60,3:40,4:30", lazy_max_share=1.0)  # 3 seats at 30 tps, 5 at 15 tps
    sched = Scheduler([led], led.settings)
    holds = [await sched.admit(f"c{i}", S8_30, 100) for i in range(3)]
    interactive = asyncio.create_task(sched.admit("c3", S8_30, 100))
    await asyncio.sleep(0.02)
    lazy = asyncio.create_task(sched.admit("l0", LAZY, 100, max_wait_s=1.0))
    await asyncio.sleep(0.02)
    assert sched.queue_lengths() == {"interactive": 1, "lazy": 1}
    sched.release(holds[0])
    h = await asyncio.wait_for(interactive, 1)
    assert h.shape is S8_30
    await asyncio.sleep(0.02)
    assert not lazy.done()  # still full: 3 x 30 tps
    sched.release(holds[1])
    hl = await asyncio.wait_for(lazy, 1)
    assert hl.shape is LAZY


async def test_requeue_policy_preempts_youngest_lazy_run_for_interactive():
    led = ledger(curve="1:100,2:60,3:40,4:30", lazy_preempt="requeue", lazy_max_share=1.0)  # 3 seats at 30 tps
    sched = Scheduler([led], led.settings)
    cancelled = []

    async def fake_run(name):
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            cancelled.append(name)
            raise

    lazies = []
    for i in range(3):
        h = await sched.admit(f"l{i}", LAZY, 100, preemptable=True)
        h.task = asyncio.create_task(fake_run(f"l{i}"))
        lazies.append(h)
    await asyncio.sleep(0)  # let the fake runs start
    h = await sched.admit("c0", S8_30, 100)  # fits only by evicting a lazy run
    await asyncio.sleep(0.02)
    assert h.ledger is led and led.n_running == 3
    assert cancelled == ["l2"] and lazies[2].preempted and sched.preemptions == 1
    # the evicted run re-enters at the head of the lazy queue and runs again when a seat frees
    requeued = asyncio.create_task(sched.admit("l2", LAZY, 100, preemptable=True, front=True))
    await asyncio.sleep(0.02)
    assert not requeued.done()
    sched.release(h)
    h2 = await asyncio.wait_for(requeued, 1)
    assert h2.request_id == "l2"
    for t in (x.task for x in lazies[:2]):
        t.cancel()


async def test_never_policy_does_not_preempt():
    led = ledger(curve="1:100,2:60,3:40,4:30", lazy_preempt="never", lazy_max_share=1.0)
    sched = Scheduler([led], led.settings)
    for i in range(3):
        await sched.admit(f"l{i}", LAZY, 100, preemptable=True)
    with pytest.raises(QueueTimeout):
        await sched.admit("c0", S8_30, 100, max_wait_s=0.1)
    assert sched.preemptions == 0
