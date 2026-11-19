"""Stage N: weighted fair share of prompt reading in the interactive queue (start-time fair queuing over
tenants, weight = the shape's tps) and the new-token count of a turn's prompt."""

from __future__ import annotations

import asyncio

from .test_admission import S8_15, S8_30
from .test_affinity import sched


def full(sc):
    """Fill every seat of the only replica; returns the holds (release one to admit one waiter)."""
    led = sc.ledgers[0]
    return [led.hold(f"fill{i}", S8_30, 8192, 100) for i in range(led.seats(30))]


async def enqueue(sc, specs):
    tasks = []
    for rid, shape, tokens, tenant in specs:
        tasks.append(asyncio.ensure_future(sc.admit(rid, shape, tokens, tenant=tenant)))
        await asyncio.sleep(0)
    return tasks


async def serve(sc, fill, tasks, n, seen=None):
    """Free one seat at a time; return the request ids in the order they were admitted."""
    order, seen = [], set() if seen is None else seen
    for _ in range(n):
        sc.release(fill.pop())
        for _ in range(5):  # let the admitted request's coroutine return
            await asyncio.sleep(0)
        for t in tasks:
            if t.done() and id(t) not in seen:
                seen.add(id(t))
                h = t.result()
                order.append(h.request_id)
                fill.append(h)
    return order


async def test_queue_is_first_in_first_out_with_fair_share_off():
    sc = sched(n=1)
    fill = full(sc)
    tasks = await enqueue(sc, [("h1", S8_30, 6000, "heavy"), ("h2", S8_30, 6000, "heavy"), ("h3", S8_30, 6000, "heavy"),
                               ("l1", S8_30, 300, "light")])
    assert await serve(sc, fill, tasks, 4) == ["h1", "h2", "h3", "l1"]
    assert sc.fq_reorders == 0


async def test_a_light_tenant_goes_ahead_of_a_heavy_tenants_backlog():
    sc = sched(n=1, ingest_fair_share=True)
    fill = full(sc)
    tasks = await enqueue(sc, [("h1", S8_30, 6000, "heavy"), ("h2", S8_30, 6000, "heavy"), ("h3", S8_30, 6000, "heavy"),
                               ("l1", S8_30, 300, "light"), ("l2", S8_30, 300, "light")])
    # heavy's requests start at 0, 200 and 400 (6000 tokens / 30); light's at 0 and 10
    assert await serve(sc, fill, tasks, 5) == ["h1", "l1", "l2", "h2", "h3"]
    assert sc.fq_reorders == 2 and sc.snapshot()["ingest"]["reorders"] == 2


async def test_a_faster_shape_gets_a_proportionally_larger_share():
    sc = sched(n=1, ingest_fair_share=True)
    fill = full(sc)
    specs = []
    for i in range(12):
        specs += [(f"pro{i}", S8_30, 1000, "pro"), (f"std{i}", S8_15, 1000, "std")]
    tasks = await enqueue(sc, specs)
    order = await serve(sc, fill, tasks, 12)
    pro = sum(1 for r in order if r.startswith("pro"))
    assert 7 <= pro <= 9  # two thirds of the reading goes to the 30 tps tenant, one third to the 15 tps one
    # within a tenant the order is still first come, first served
    assert [r for r in order if r.startswith("std")] == [f"std{i}" for i in range(12 - pro)]


async def test_an_idle_tenant_does_not_bank_credit():
    sc = sched(n=1, ingest_fair_share=True)
    fill = full(sc)
    tasks = await enqueue(sc, [(f"a{i}", S8_30, 3000, "a") for i in range(4)])
    seen = set()
    assert await serve(sc, fill, tasks, 3, seen) == ["a0", "a1", "a2"]  # virtual time is now a2's start, 200
    late = await enqueue(sc, [("b0", S8_30, 3000, "b"), ("b1", S8_30, 3000, "b"), ("b2", S8_30, 3000, "b")])
    # b starts at the virtual time, not at zero: b0 (200) goes before a3 (300), then a3 before b1 (300, later) and b2
    assert await serve(sc, fill, tasks + late, 4, seen) == ["b0", "a3", "b1", "b2"]


async def test_new_token_count_of_a_conversations_next_turn():
    sc = sched(n=1, ingest_new_tokens=True, max_prefill_tokens_in_flight=2048)
    h0 = await sc.admit("t0", S8_30, 1500, conversation="c1")
    assert h0.prompt_tokens == 1500  # a first turn is read in full
    sc.mark_prefill_done(h0)
    h1 = await sc.admit("t1", S8_30, 2100, conversation="c1")
    assert h1.prompt_tokens == 600  # 2100 - 1500: the rest is in the prefix cache
    # 600 in flight + 1000 new tokens fits under the cap of 2048, although the whole prompt (3900) would not
    h3 = await sc.admit("t3", S8_30, 1000, conversation="c3")
    assert h3.prompt_tokens == 1000
    h4 = await sc.admit("t4", S8_30, 2105, conversation="c1")
    assert h4.prompt_tokens == 16  # never below one KV block
    for h in (h0, h1, h3, h4):
        sc.release(h)


async def test_whole_prompt_is_counted_with_the_new_token_count_off():
    sc = sched(n=1)
    h0 = await sc.admit("t0", S8_30, 1500, conversation="c1")
    h1 = await sc.admit("t1", S8_30, 2100, conversation="c1")
    assert (h0.prompt_tokens, h1.prompt_tokens) == (1500, 2100)
    sc.release(h0)
    sc.release(h1)
