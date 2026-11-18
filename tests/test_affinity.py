"""Stage K: soft prefix affinity in Scheduler.place() (a conversation's next turn returns to the replica
that served its last one when that replica fits it now; it never waits for it)."""

from __future__ import annotations

import asyncio
import random

import pytest

from slotgw import admission
from slotgw.admission import Scheduler
from slotgw.config import AdmissionSettings

from .test_admission import LAZY, S8_15, S8_30, S8_60, S16_30, ledger


def sched(n=3, kv_total=181_936, **settings) -> Scheduler:
    s = AdmissionSettings(**settings)
    ledgers = [ledger(f"r{i}", kv_total=kv_total, **settings) for i in range(n)]
    return Scheduler(ledgers, s)


async def test_affinity_returns_to_the_remembered_replica_when_it_fits():
    sc = sched(prefix_affinity=True)
    h0 = await sc.admit("t0", S8_30, 500, conversation="c1")
    home = h0.ledger
    assert h0.affinity is None  # a first turn has nothing remembered
    sc.release(h0)
    # load the home replica so that (capacity_loss, load) alone would choose another one
    extra = [home.hold(f"busy{i}", S8_30, 8192, 100) for i in range(2)]
    assert sc.place(S8_30, 500) is not home
    h1 = await sc.admit("t1", S8_30, 900, conversation="c1")
    assert h1.ledger is home and h1.affinity == "hit"
    assert sc.affinity_hits == 1 and sc.affinity_fallbacks == {}
    assert sc.snapshot()["prefix_affinity"]["hits"] == 1
    for h in (*extra, h1):
        sc.release(h)


async def test_fallback_when_the_remembered_replica_has_no_seat_and_no_added_wait():
    sc = sched(n=2, prefix_affinity=True)
    h0 = await sc.admit("t0", S8_30, 500, conversation="c1")
    home = h0.ledger
    sc.release(h0)
    fill = [home.hold(f"f{i}", S8_30, 8192, 100) for i in range(home.seats(30))]
    assert home.check(S8_30, 8192, 500) == "seat"
    h1 = await sc.admit("t1", S8_30, 900, conversation="c1")  # returns at once: nothing queued
    assert h1.ledger is not home and h1.affinity == "seat"
    assert sc.queue_lengths() == {"interactive": 0, "lazy": 0, "burst": 0}
    assert sc.affinity_fallbacks == {"seat": 1} and sc.affinity_hits == 0
    # the conversation now lives on the replica that took it
    assert sc.remembered("c1") == h1.ledger.id
    sc.release(h1)
    for h in fill:
        home.release(h)
    h2 = await sc.admit("t2", S8_30, 1300, conversation="c1")
    assert h2.ledger is h1.ledger and h2.affinity == "hit"


async def test_fallback_reasons_kv_and_unhealthy():
    sc = sched(n=2, kv_total=30_000, prefix_affinity=True)  # usable 28,500: three 8k holds, not a 16k beside two
    h0 = await sc.admit("t0", S8_30, 500, conversation="c1")
    home = h0.ledger
    sc.release(h0)
    fill = [home.hold(f"f{i}", S8_30, 8192, 100) for i in range(2)]
    assert home.check(S16_30, home.kv_for(S16_30), 500) == "kv"
    h1 = await sc.admit("t1", S16_30, 900, conversation="c1")
    assert h1.ledger is not home and h1.affinity == "kv"
    sc.release(h1)
    for h in fill:
        home.release(h)
    h2 = await sc.admit("u0", S8_30, 500, conversation="c2")
    h2.ledger.healthy = False
    sc.release(h2)
    h3 = await sc.admit("u1", S8_30, 900, conversation="c2")
    assert h3.ledger is not h2.ledger and h3.affinity == "unhealthy"
    assert sc.affinity_fallbacks == {"kv": 1, "unhealthy": 1}


async def test_queued_request_takes_the_first_replica_that_frees_not_the_remembered_one():
    sc = sched(n=2, prefix_affinity=True)
    r0, r1 = sc.ledgers
    h0 = await sc.admit("t0", S8_30, 500, conversation="c1")
    home, other = h0.ledger, (r1 if h0.ledger is r0 else r0)
    sc.release(h0)
    fill_home = [home.hold(f"h{i}", S8_30, 8192, 100) for i in range(home.seats(30))]
    fill_other = [other.hold(f"o{i}", S8_30, 8192, 100) for i in range(other.seats(30))]
    task = asyncio.ensure_future(sc.admit("t1", S8_30, 900, conversation="c1", max_wait_s=5))
    await asyncio.sleep(0.01)
    assert not task.done() and sc.queue_lengths()["interactive"] == 1
    sc.release(fill_other[0])  # a seat frees on the replica it does not remember: it takes it, no waiting for home
    h1 = await asyncio.wait_for(task, 1)
    assert h1.ledger is other and h1.affinity == "seat"
    for h in (*fill_home, *fill_other[1:], h1):
        sc.release(h)


async def test_idle_conversation_is_forgotten_after_the_ttl(monkeypatch):
    t = [1000.0]
    monkeypatch.setattr(admission, "now", lambda: t[0])
    sc = sched(n=2, prefix_affinity=True, prefix_affinity_ttl_s=120)
    h0 = await sc.admit("t0", S8_30, 500, conversation="c1")
    t[0] += 60
    sc.release(h0)  # idle time counts from the end of the turn
    t[0] += 119
    assert sc.remembered("c1") == h0.ledger.id
    t[0] += 2
    assert sc.remembered("c1") is None
    h1 = await sc.admit("t1", S8_30, 900, conversation="c1")
    assert h1.affinity is None


async def test_map_is_bounded():
    sc = sched(n=2, prefix_affinity=True, prefix_affinity_max_entries=5)
    for i in range(20):
        sc.release(await sc.admit(f"t{i}", S8_30, 100, conversation=f"c{i}"))
    assert len(sc._affinity) == 5 and sc.remembered("c19") is not None and sc.remembered("c0") is None


@pytest.mark.parametrize("seed", [1, 2, 3])
async def test_flag_off_reproduces_the_baseline_placement_exactly(seed):
    """The same random stream of admissions and releases, once without conversation ids (the phase-3 code
    path) and once with them and the flag off: every request lands on the same replica."""
    shapes = [S8_15, S8_30, S8_30, S8_60, S16_30, LAZY]

    async def run(with_conversations: bool, flag: bool) -> list[str]:
        rng = random.Random(seed)
        sc = sched(n=3, prefix_affinity=flag)
        live, placed = [], []
        for i in range(400):
            if live and rng.random() < 0.45:
                sc.release(live.pop(rng.randrange(len(live))))
                continue
            shape, conv = rng.choice(shapes), f"c{rng.randrange(12)}"
            if sc.place(shape, 500) is None:  # full: skip, as both variants do
                placed.append("-")
                continue
            h = await sc.admit(f"q{i}", shape, 500, conversation=conv if with_conversations else None)
            live.append(h)
            placed.append(h.ledger.id)
        return placed

    plain = await run(False, False)
    off = await run(True, False)
    on = await run(True, True)
    assert off == plain
    assert on != plain  # and the flag does change placement on this stream


async def test_flag_off_still_reports_the_natural_hit_rate():
    sc = sched(n=2, prefix_affinity=False)
    h0 = await sc.admit("t0", S8_30, 500, conversation="c1")
    home = h0.ledger
    sc.release(h0)
    extra = home.hold("busy0", S8_30, 8192, 100)  # home is now the more loaded one: load placement leaves it
    h1 = await sc.admit("t1", S8_30, 900, conversation="c1")
    assert h1.ledger is not home and h1.affinity == "score"
    sc.release(h1)
    h2 = await sc.admit("t2", S8_30, 900, conversation="c1")
    assert h2.affinity in ("hit", "score")
    home.release(extra)


# ---- through the real HTTP path: three fake replicas behind one gateway ------------------------


async def _turns(cluster, n: int, conversation: str | None) -> list[str]:
    """n sequential turns of one conversation while a second stream keeps one other replica busy."""
    import httpx

    headers = {"Authorization": "Bearer sk-local-chat"}
    if conversation:
        headers["X-Slotgw-Conversation"] = conversation
    ids = []
    async with httpx.AsyncClient(base_url=cluster.gateway.url, timeout=30) as c:
        for i in range(n):
            # a concurrent request shifts the load so that placement by load alone moves the conversation
            other = asyncio.ensure_future(c.post(
                "/v1/chat/completions", headers={"Authorization": "Bearer sk-local-chat"},
                json={"model": "small.8k.30tps", "messages": [{"role": "user", "content": f"filler {i}"}], "max_tokens": 20},
            ))
            await asyncio.sleep(0.05)
            r = await c.post("/v1/chat/completions", headers=headers, json={
                "model": "small.8k.30tps", "messages": [{"role": "user", "content": f"turn {i}"}], "max_tokens": 5,
            })
            assert r.status_code == 200, r.text
            ids.append(r.json()["id"])
            await other
    return ids


@pytest.mark.parametrize("flag", [True, False])
async def test_gateway_reads_the_conversation_header_and_meters_affinity(tmp_path, flag):
    import sqlite3

    from .test_multi_replica import TWO_SEATS, Cluster

    cluster = Cluster([TWO_SEATS] * 3, AdmissionSettings(prefix_affinity=flag), 5.0, tmp_path / "meter.sqlite")
    try:
        ids = await _turns(cluster, 6, "run1:chat-7")
    finally:
        cluster.stop()
    conn = sqlite3.connect(str(tmp_path / "meter.sqlite"))
    rows = {r[0]: r[1:] for r in conn.execute("SELECT request_id, replica, conversation_id, affinity FROM requests")}
    conn.close()
    mine = [rows[i] for i in ids]
    assert all(conv == "run1:chat-7" for _, conv, _ in mine)
    assert mine[0][2] is None  # the first turn has no remembered replica
    assert all(a is not None for _, _, a in mine[1:])
    assert all(conv is None and a is None for rid, (_, conv, a) in rows.items() if rid not in ids)
    if flag:
        assert len({replica for replica, _, _ in mine}) == 1 and all(a == "hit" for _, _, a in mine[1:])
    else:
        assert len({replica for replica, _, _ in mine}) > 1 and any(a == "score" for _, _, a in mine[1:])


# ---- stage L: a bounded wait for the remembered replica -------------------------------------------


def _fill(sc: Scheduler, home, shape=S8_60):
    """Hold seats on `home` until it refuses `shape` for want of a seat."""
    held = []
    while home.check(shape, home.kv_for(shape), 100) is None:
        held.append(home.hold(f"fill{len(held)}", shape, home.kv_for(shape), 100))
    assert home.check(shape, home.kv_for(shape), 100) == "seat"
    return held


async def test_wait_takes_the_remembered_replica_when_a_seat_frees_in_time():
    sc = sched(n=2, prefix_affinity=True, prefix_affinity_wait_ms=300)
    h0 = await sc.admit("t0", S8_60, 500, conversation="c1")
    home = h0.ledger
    sc.release(h0)
    held = _fill(sc, home)
    task = asyncio.create_task(sc.admit("t1", S8_60, 900, conversation="c1"))
    await asyncio.sleep(0.05)
    assert not task.done()  # another replica fits, and it waits anyway
    sc.release(held.pop())
    h1 = await asyncio.wait_for(task, 1.0)
    assert h1.ledger is home and h1.affinity == "hit"
    assert h1.affinity_wait_ms is not None and 30 <= h1.affinity_wait_ms < 300
    snap = sc.snapshot()["prefix_affinity"]
    assert snap["waits"] == 1 and snap["wait_hits"] == 1 and snap["wait_ms"] == 300
    for h in (*held, h1):
        sc.release(h)


async def test_wait_runs_out_and_the_request_runs_elsewhere_with_the_reason():
    sc = sched(n=2, prefix_affinity=True, prefix_affinity_wait_ms=60)
    h0 = await sc.admit("t0", S8_60, 500, conversation="c1")
    home = h0.ledger
    sc.release(h0)
    held = _fill(sc, home)
    h1 = await asyncio.wait_for(sc.admit("t1", S8_60, 900, conversation="c1"), 1.0)
    assert h1.ledger is not home and h1.affinity == "seat"
    assert h1.affinity_wait_ms is not None and h1.affinity_wait_ms >= 55
    assert sc.affinity_waits == 1 and sc.affinity_wait_hits == 0 and sc.affinity_fallbacks == {"seat": 1}
    for h in (*held, h1):
        sc.release(h)


async def test_no_wait_when_off_unremembered_or_home_fits():
    for kw in ({"prefix_affinity": True}, {"prefix_affinity": False, "prefix_affinity_wait_ms": 300}):
        sc = sched(n=2, **kw)
        h0 = await sc.admit("t0", S8_60, 500, conversation="c1")
        home = h0.ledger
        sc.release(h0)
        held = _fill(sc, home)
        h1 = await asyncio.wait_for(sc.admit("t1", S8_60, 900, conversation="c1"), 0.05)  # at once
        assert h1.ledger is not home and h1.affinity_wait_ms is None and sc.affinity_waits == 0
        for h in (*held, h1):
            sc.release(h)
    sc = sched(n=2, prefix_affinity=True, prefix_affinity_wait_ms=300)
    h0 = await asyncio.wait_for(sc.admit("t0", S8_60, 500, conversation="c1"), 0.05)  # nothing remembered
    sc.release(h0)
    h1 = await asyncio.wait_for(sc.admit("t1", S8_60, 500, conversation="c1"), 0.05)  # home fits: a plain hit
    assert h1.affinity == "hit" and h1.affinity_wait_ms is None and sc.affinity_waits == 0
    sc.release(h1)


async def test_wait_that_ends_with_nothing_free_queues_the_request():
    sc = sched(n=2, prefix_affinity=True, prefix_affinity_wait_ms=60)
    h0 = await sc.admit("t0", S8_60, 500, conversation="c1")
    home = h0.ledger
    other = next(l for l in sc.ledgers if l is not home)
    sc.release(h0)
    held = _fill(sc, home)
    task = asyncio.create_task(sc.admit("t1", S8_60, 900, conversation="c1"))
    await asyncio.sleep(0.02)
    held_other = _fill(sc, other)  # the other replica fills up while it waits
    await asyncio.sleep(0.1)
    assert not task.done() and sc.queue_lengths()["interactive"] == 1
    sc.release(held_other.pop())
    h1 = await asyncio.wait_for(task, 1.0)
    assert h1.ledger is other
    for h in (*held, *held_other, h1):
        sc.release(h)
