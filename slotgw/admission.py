"""Admission control: per-replica ledgers (KV, seats, lazy share, prefill), priority queues, the
shape-affinity scheduler, reservations and the burst manager (docs/DESIGN.md sections 5.2-5.4,
7 and 4.1 step 5).

A request holds `kv_for(shape)` tokens of KV and one seat from admission until the engine finishes
(delivery may continue afterwards, paced). Seats are bounded by the calibrated step-rate curve: with
n streams running, every stream gets step_rate(n) tokens/s, so the replica may run at most
n_max(t) streams while a shape of t tokens/s is among them. The fastest running shape sets the
ceiling; `tps_overcommit` relaxes it. Lazy shapes fill idle capacity only: limited to a share of the
seats and admitted only when no interactive request is waiting.

Milestone 4: a **reservation** is N standing holds of a shape (slots) that count as seats whether a
request occupies them or not; the tenant's requests take a free slot at once and queue only on the
tenant's own slots. The **burst manager** lends the step rate the engine has beyond the running
requests' guarantees to their pacers (tps burst, revoked by recomputation on every tick), and
admits over-context requests into free KV (context burst) with a revocation policy when a
guaranteed admission needs that capacity.
"""

from __future__ import annotations

import asyncio
import logging
import math
import uuid
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Protocol

from .calibration import ThroughputProfile
from .clock import now
from .config import AdmissionSettings, BurstSettings, Shape
from .pacing import Pacer

log = logging.getLogger("slotgw.admission")

QUEUE_ORDER = ("interactive", "lazy", "burst")


class ReplicaLike(Protocol):
    id: str
    model_alias: str


class QueueTimeout(Exception):
    def __init__(self, reason: str, waited_s: float):
        super().__init__(f"no capacity after {waited_s:.1f}s ({reason})")
        self.reason = reason
        self.waited_s = waited_s


class Unsellable(Exception):
    """No replica can honour this shape's rate even for a single stream."""


class ReservationCapacity(Exception):
    """The reservation's slots do not fit on the replicas now."""

    def __init__(self, reason: str, placed: int):
        super().__init__(f"reservation does not fit ({reason}, {placed} slot(s) placed)")
        self.reason = reason
        self.placed = placed


@dataclass
class Hold:
    request_id: str  # the request, or "<reservation>#<i>" for a reserved slot
    shape: Shape
    kv_tokens: int
    tps: float
    lazy: bool
    prompt_tokens: int
    t_admitted: float
    ledger: ReplicaLedger
    preemptable: bool = False
    prefill_done: bool = False
    preempted: bool = False
    released: bool = False
    task: asyncio.Task | None = None  # the engine run to cancel on preemption or revocation
    # milestone 4
    reserved: bool = False  # a standing reserved slot: a seat whether occupied or not
    reservation_id: str | None = None
    occupant: str | None = None  # request running in this reserved slot
    burst_ctx: bool = False  # context burst: KV beyond the shape, revocable
    revoke_at: float | None = None  # when a waiting guaranteed admission may abort this burst
    revoked: bool = False
    pacer: Pacer | None = None  # the running request's pacer, for burst-rate updates
    engine_active: bool = False  # an engine run is in progress (an idle reserved slot is not)


class ReplicaLedger:
    def __init__(
        self,
        replica: ReplicaLike,
        profile: ThroughputProfile,
        kv_total: int,
        block_size: int,
        settings: AdmissionSettings,
        slot_plan: dict[str, int] | None = None,
    ):
        self.replica = replica
        self.id = replica.id
        self.model_alias = replica.model_alias
        self.profile = profile
        self.kv_total = kv_total
        self.block_size = max(1, block_size)
        self.settings = settings
        self.slot_plan = dict(slot_plan or {})
        self.healthy = True
        self.holds: dict[str, Hold] = {}
        self.admitted_total = 0
        self.burst_rate_lent = 0.0  # tokens/s currently lent to running requests (last tick)

    # ---- derived state -----------------------------------------------------------------------

    @property
    def kv_usable(self) -> int:
        return int(self.kv_total * (1.0 - self.settings.kv_margin))

    @property
    def kv_held(self) -> int:
        return sum(h.kv_tokens for h in self.holds.values())

    @property
    def n_running(self) -> int:
        return len(self.holds)

    @property
    def n_engine(self) -> int:
        """Requests the engine is actually running (idle reserved slots excluded)."""
        return sum(1 for h in self.holds.values() if h.engine_active)

    @property
    def lazy_running(self) -> int:
        return sum(1 for h in self.holds.values() if h.lazy)

    @property
    def reserved_slots(self) -> int:
        return sum(1 for h in self.holds.values() if h.reserved)

    @property
    def reserved_idle(self) -> int:
        return sum(1 for h in self.holds.values() if h.reserved and h.occupant is None)

    @property
    def burst_running(self) -> int:
        return sum(1 for h in self.holds.values() if h.burst_ctx)

    @property
    def max_tps_running(self) -> float:
        return max((h.tps for h in self.holds.values()), default=0.0)

    @property
    def prefill_tokens_in_flight(self) -> int:
        return sum(h.prompt_tokens for h in self.holds.values() if not h.prefill_done)

    @property
    def load(self) -> float:
        return self.kv_held / self.kv_usable if self.kv_usable else 1.0

    def kv_for(self, shape: Shape) -> int:
        """A1: KV is held for the shape's whole context, rounded up to whole blocks."""
        return self.kv_blocks(shape.max_context)

    def kv_blocks(self, tokens: int) -> int:
        return math.ceil(tokens / self.block_size) * self.block_size

    def seats(self, tps: float) -> int:
        """Seats available while a shape of `tps` is among the running streams."""
        n = self.profile.n_max(tps, self.settings.tps_margin)
        return math.floor(n * self.settings.tps_overcommit)

    def seats_now(self) -> int:
        return self.seats(self.max_tps_running) if self.holds else self.seats(0.0)

    def step_rate_now(self) -> float:
        """Per-stream engine rate at the current engine concurrency (the burst ceiling)."""
        return self.profile.step_rate(max(1, self.n_engine))

    # ---- admission checks ---------------------------------------------------------------------

    def check(
        self, shape: Shape, kv_needed: int, prompt_tokens: int, exclude: Hold | Iterable[Hold] | None = None
    ) -> str | None:
        """None if the request fits now, else the first failing rule."""
        if not self.healthy:
            return "unhealthy"
        if self.seats(shape.tps) == 0:
            return "unsellable"
        excluded = {id(exclude)} if isinstance(exclude, Hold) else {id(h) for h in (exclude or ())}
        holds = [h for h in self.holds.values() if id(h) not in excluded]
        kv_held = sum(h.kv_tokens for h in holds)
        if kv_held + kv_needed > self.kv_usable:
            return "kv"
        max_tps = max([shape.tps, *(h.tps for h in holds)])
        seats = self.seats(max_tps)
        if len(holds) + 1 > seats:
            return "seat"
        if shape.lazy:
            lazy_seats = max(1, math.floor(self.settings.lazy_max_share * seats))
            if sum(1 for h in holds if h.lazy) + 1 > lazy_seats:
                return "lazy_share"
        cap = self.settings.max_prefill_tokens_in_flight
        if cap > 0:
            in_flight = sum(h.prompt_tokens for h in holds if not h.prefill_done)
            if in_flight > 0 and in_flight + prompt_tokens > cap:
                return "prefill"
        return None

    def capacity_loss(self, shape: Shape) -> int:
        """Seats this replica would lose by running `shape` (a faster shape lowers the ceiling)."""
        if not self.holds:
            return 0
        before = self.seats(self.max_tps_running)
        after = self.seats(max(self.max_tps_running, shape.tps))
        return max(0, before - after)

    # ---- state changes ------------------------------------------------------------------------

    def hold(
        self,
        request_id: str,
        shape: Shape,
        kv_needed: int,
        prompt_tokens: int,
        preemptable: bool = False,
        burst_ctx: bool = False,
    ) -> Hold:
        h = Hold(
            request_id=request_id,
            shape=shape,
            kv_tokens=kv_needed,
            tps=shape.tps,
            lazy=shape.lazy,
            prompt_tokens=prompt_tokens,
            t_admitted=now(),
            ledger=self,
            preemptable=preemptable,
            burst_ctx=burst_ctx,
            engine_active=True,
        )
        self.holds[request_id] = h
        self.admitted_total += 1
        return h

    def reserve(self, reservation_id: str, index: int, shape: Shape) -> Hold:
        """A standing slot: a seat and the shape's KV, held idle until a request occupies it."""
        h = Hold(
            request_id=f"{reservation_id}#{index}",
            shape=shape,
            kv_tokens=self.kv_for(shape),
            tps=shape.tps,
            lazy=shape.lazy,
            prompt_tokens=0,
            t_admitted=now(),
            ledger=self,
            prefill_done=True,
            reserved=True,
            reservation_id=reservation_id,
        )
        self.holds[h.request_id] = h
        return h

    def occupy(self, slot: Hold, request_id: str, prompt_tokens: int) -> None:
        slot.occupant = request_id
        slot.prompt_tokens = prompt_tokens
        slot.prefill_done = False
        slot.engine_active = True
        slot.task = None
        slot.pacer = None
        self.admitted_total += 1

    def vacate(self, slot: Hold) -> None:
        slot.occupant = None
        slot.prompt_tokens = 0
        slot.prefill_done = True
        slot.engine_active = False
        slot.task = None
        slot.pacer = None

    def release(self, h: Hold) -> None:
        if not h.released:
            h.released = True
            h.engine_active = False
            self.holds.pop(h.request_id, None)

    def mark_prefill_done(self, h: Hold) -> None:
        h.prefill_done = True

    def update_burst(self, t_now: float, lend: bool) -> float:
        """Lend each running request the step rate above its guarantee (or revoke it all)."""
        rate = self.step_rate_now() * (1.0 - self.settings.tps_margin)
        total = 0.0
        for h in self.holds.values():
            if h.pacer is None or not h.engine_active:
                continue
            lent = max(0.0, rate - h.tps) if lend else 0.0
            h.pacer.set_burst_rate(lent, t_now)
            total += lent
        self.burst_rate_lent = total
        return total

    def snapshot(self) -> dict[str, Any]:
        return {
            "healthy": self.healthy,
            "kv_total": self.kv_total,
            "kv_usable": self.kv_usable,
            "kv_held": self.kv_held,
            "block_size": self.block_size,
            "running": self.n_running,
            "engine_active": self.n_engine,
            "lazy_running": self.lazy_running,
            "reserved_slots": self.reserved_slots,
            "reserved_idle": self.reserved_idle,
            "burst_running": self.burst_running,
            "burst_rate_lent": round(self.burst_rate_lent, 2),
            "step_rate_now": round(self.step_rate_now(), 2),
            "max_tps_running": self.max_tps_running,
            "seats_now": self.seats_now(),
            "prefill_tokens_in_flight": self.prefill_tokens_in_flight,
            "admitted_total": self.admitted_total,
            "profile": self.profile.source,
            "slot_plan": self.slot_plan,
        }


@dataclass
class Waiter:
    request_id: str
    shape: Shape
    prompt_tokens: int
    kv_needed: int | None
    preemptable: bool
    burst_ctx: bool
    t_enqueued: float
    future: asyncio.Future[Hold]
    last_reason: str = "capacity"
    reasons: dict[str, int] = field(default_factory=dict)


@dataclass
class _Probe:
    """Collects the rules a placement failed on, like a Waiter does."""

    last_reason: str = "capacity"
    reasons: dict[str, int] = field(default_factory=dict)


@dataclass
class Reservation:
    id: str
    tenant: str
    shape: Shape
    slots: list[Hold]
    t_start: float
    t_end: float | None = None
    tags: dict[str, Any] = field(default_factory=dict)
    queue: deque[Waiter] = field(default_factory=deque)
    served: int = 0
    timeouts: int = 0

    def free_slot(self) -> Hold | None:
        return next((s for s in self.slots if s.occupant is None and not s.released), None)

    @property
    def busy(self) -> int:
        return sum(1 for s in self.slots if s.occupant is not None)

    def covers(self, tenant: str, shape: Shape) -> bool:
        return (
            self.t_end is None
            and self.tenant == tenant
            and self.shape.model == shape.model
            and self.shape.lazy == shape.lazy
            and shape.max_context <= self.shape.max_context
            and shape.tps <= self.shape.tps
        )

    def snapshot(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "tenant": self.tenant,
            "shape": self.shape.name,
            "slots": len(self.slots),
            "busy": self.busy,
            "queued": len(self.queue),
            "served": self.served,
            "timeouts": self.timeouts,
            "replicas": sorted({s.ledger.id for s in self.slots}),
            "t_start": self.t_start,
            "t_end": self.t_end,
            **self.tags,
        }


class Scheduler:
    """Places requests on replicas, queues them when nothing fits, and times them out."""

    def __init__(self, ledgers: list[ReplicaLedger], settings: AdmissionSettings, burst: BurstSettings | None = None):
        self.ledgers = ledgers
        self.settings = settings
        self.burst = burst or BurstSettings()
        self.queues: dict[str, deque[Waiter]] = {k: deque() for k in QUEUE_ORDER}
        self.reservations: dict[str, Reservation] = {}
        self.preemptions = 0
        self.timeouts = 0
        self.revocations = 0
        self.bursts_admitted = 0

    # ---- placement ----------------------------------------------------------------------------

    def candidates(self, shape: Shape) -> list[ReplicaLedger]:
        return [ledg for ledg in self.ledgers if ledg.model_alias == shape.model]

    def place(
        self, shape: Shape, prompt_tokens: int, record: Waiter | _Probe | None = None, kv_needed: int | None = None
    ) -> ReplicaLedger | None:
        best: tuple[tuple[int, float], ReplicaLedger] | None = None
        for ledg in self.candidates(shape):
            reason = ledg.check(shape, ledg.kv_blocks(kv_needed) if kv_needed is not None else ledg.kv_for(shape), prompt_tokens)
            if reason is not None:
                if record is not None:
                    record.reasons[reason] = record.reasons.get(reason, 0) + 1
                    record.last_reason = reason
                continue
            score = (ledg.capacity_loss(shape), ledg.load)
            if best is None or score < best[0]:
                best = (score, ledg)
        return best[1] if best else None

    def sellable(self, shape: Shape) -> bool:
        return any(ledg.seats(shape.tps) > 0 for ledg in self.candidates(shape))

    def _kv_needed(self, ledg: ReplicaLedger, w: Waiter) -> int:
        return ledg.kv_blocks(w.kv_needed) if w.kv_needed is not None else ledg.kv_for(w.shape)

    def _preempt_for(self, shape: Shape, prompt_tokens: int) -> ReplicaLedger | None:
        """Policy `requeue`: evict one preemptable lazy run if that alone makes the request fit."""
        if self.settings.lazy_preempt != "requeue":
            return None
        for ledg in self.candidates(shape):
            victims = sorted(
                (h for h in ledg.holds.values() if h.preemptable and not h.preempted),
                key=lambda h: h.t_admitted,
                reverse=True,  # the youngest run has the least work to throw away
            )
            for v in victims:
                if ledg.check(shape, ledg.kv_for(shape), prompt_tokens, exclude=v) is None:
                    v.preempted = True
                    ledg.release(v)
                    if v.task is not None and not v.task.done():
                        v.task.cancel()
                    self.preemptions += 1
                    log.info("preempted lazy %s on %s for %s", v.request_id, ledg.id, shape.name)
                    return ledg
        return None

    def _revoke_for(self, w: Waiter) -> None:
        """A guaranteed request waits: by policy, schedule the youngest context bursts whose
        capacity it needs for abort (`abort`: now, `grace`: after grace_s, `finish`: never)."""
        if self.burst.revoke_policy == "finish":
            return
        grace = 0.0 if self.burst.revoke_policy == "abort" else self.burst.grace_s
        for ledg in self.candidates(w.shape):
            bursts = sorted(
                (h for h in ledg.holds.values() if h.burst_ctx and not h.revoked),
                key=lambda h: h.t_admitted,
                reverse=True,
            )
            chosen: list[Hold] = []
            for h in bursts:
                chosen.append(h)
                if ledg.check(w.shape, self._kv_needed(ledg, w), w.prompt_tokens, exclude=chosen) is None:
                    for v in chosen:
                        if v.revoke_at is None:
                            v.revoke_at = now() + grace
                            log.info("burst %s on %s scheduled for revocation in %.0fs", v.request_id, ledg.id, grace)
                    return

    def _dispatch_queue(self, cls: str) -> bool:
        """Admit the head of one queue while it fits; True if the queue is empty afterwards."""
        q = self.queues[cls]
        while q:
            w = q[0]
            if w.future.done():
                q.popleft()
                continue
            ledg = self.place(w.shape, w.prompt_tokens, record=w, kv_needed=w.kv_needed)
            if ledg is None and cls == "interactive":
                ledg = self._preempt_for(w.shape, w.prompt_tokens)
            if ledg is None:
                if cls != "burst":
                    self._revoke_for(w)
                return False
            q.popleft()
            w.future.set_result(
                ledg.hold(w.request_id, w.shape, self._kv_needed(ledg, w), w.prompt_tokens, w.preemptable, w.burst_ctx)
            )
            if w.burst_ctx:
                self.bursts_admitted += 1
        return True

    def dispatch(self) -> None:
        """Admit queued waiters in priority order: interactive FIFO first, lazy only when no
        interactive request is waiting, context bursts only when nothing else waits. Strict FIFO
        within a class (no overtaking)."""
        for cls in QUEUE_ORDER:
            if not self._dispatch_queue(cls):
                break
        self._refresh_burst()

    def _refresh_burst(self) -> None:
        t = now()
        lend = self.burst.enabled and self.burst.tps
        for ledg in self.ledgers:
            ledg.update_burst(t, lend)

    def tick(self) -> None:
        """Periodic: abort context bursts whose revocation time has come, re-lend burst rate."""
        t = now()
        for ledg in self.ledgers:
            for h in list(ledg.holds.values()):
                if h.burst_ctx and h.revoke_at is not None and t >= h.revoke_at and not h.revoked:
                    h.revoked = True
                    self.revocations += 1
                    ledg.release(h)
                    if h.task is not None and not h.task.done():
                        h.task.cancel()
                    log.info("revoked context burst %s on %s", h.request_id, ledg.id)
        self.dispatch()

    # ---- reservations -------------------------------------------------------------------------

    def reservation_for(self, tenant: str | None, shape: Shape) -> Reservation | None:
        if tenant is None:
            return None
        return next((r for r in self.reservations.values() if r.covers(tenant, shape)), None)

    def reserve(
        self, tenant: str, shape: Shape, slots: int, reservation_id: str | None = None, tags: dict[str, Any] | None = None
    ) -> Reservation:
        """Place `slots` standing holds of `shape` (spread by the usual scoring); all or nothing."""
        if slots < 1:
            raise ValueError("slots must be at least 1")
        if not self.sellable(shape):
            raise Unsellable(shape.name)
        rid = reservation_id or "res-" + uuid.uuid4().hex[:10]
        placed: list[Hold] = []
        probe = _Probe()
        for i in range(slots):
            ledg = self.place(shape, 0, record=probe)
            if ledg is None:
                for h in placed:
                    h.ledger.release(h)
                self.dispatch()
                raise ReservationCapacity(probe.last_reason, len(placed))
            placed.append(ledg.reserve(rid, i, shape))
        res = Reservation(id=rid, tenant=tenant, shape=shape, slots=placed, t_start=now(), tags=dict(tags or {}))
        self.reservations[rid] = res
        log.info("reservation %s: %d x %s for %s on %s", rid, slots, shape.name, tenant, sorted({h.ledger.id for h in placed}))
        self._refresh_burst()
        return res

    def unreserve(self, reservation_id: str) -> Reservation | None:
        """End a reservation: idle slots are freed now, occupied ones when their request finishes;
        its waiters move to the ordinary queue."""
        res = self.reservations.get(reservation_id)
        if res is None or res.t_end is not None:
            return None
        res.t_end = now()
        for s in res.slots:
            if s.occupant is None:
                s.ledger.release(s)
        while res.queue:
            w = res.queue.popleft()
            if not w.future.done():
                self.queues["lazy" if w.shape.lazy else "interactive"].append(w)
        self.dispatch()
        return res

    def _dispatch_reservation(self, res: Reservation) -> None:
        while res.queue:
            w = res.queue[0]
            if w.future.done():
                res.queue.popleft()
                continue
            slot = res.free_slot()
            if slot is None:
                return
            res.queue.popleft()
            slot.ledger.occupy(slot, w.request_id, w.prompt_tokens)
            res.served += 1
            w.future.set_result(slot)

    async def _wait_reservation(self, res: Reservation, request_id: str, shape: Shape, prompt_tokens: int, timeout: float) -> Hold:
        loop = asyncio.get_running_loop()
        w = Waiter(request_id, shape, prompt_tokens, None, False, False, now(), loop.create_future())
        res.queue.append(w)
        await asyncio.wait([w.future], timeout=timeout)
        if w.future.done():
            return w.future.result()
        try:
            res.queue.remove(w)
        except ValueError:
            pass
        if w.future.done():
            return w.future.result()
        w.future.cancel()
        res.timeouts += 1
        self.timeouts += 1
        raise QueueTimeout("reservation_busy", now() - w.t_enqueued)

    # ---- public API ---------------------------------------------------------------------------

    async def admit(
        self,
        request_id: str,
        shape: Shape,
        prompt_tokens: int,
        *,
        tenant: str | None = None,
        preemptable: bool = False,
        front: bool = False,
        max_wait_s: float | None = None,
        kv_tokens: int | None = None,
        burst_ctx: bool = False,
    ) -> Hold:
        """Return a Hold on some replica, waiting up to the shape's max_queue_s. A tenant with a
        reservation covering the shape takes one of its slots (or waits for one); a context burst
        (`burst_ctx`, `kv_tokens` = its real need) is admitted only when nothing else waits."""
        timeout = shape.max_queue_s if max_wait_s is None else max_wait_s
        if not burst_ctx:
            res = self.reservation_for(tenant, shape)
            if res is not None:
                slot = res.free_slot()
                if slot is not None:
                    slot.ledger.occupy(slot, request_id, prompt_tokens)
                    res.served += 1
                    self._refresh_burst()
                    return slot
                if not self.settings.reservation_spill:
                    return await self._wait_reservation(res, request_id, shape, prompt_tokens, timeout)
        if not self.sellable(shape):
            raise Unsellable(shape.name)
        cls = "burst" if burst_ctx else ("lazy" if shape.lazy else "interactive")
        queue = self.queues[cls]
        ahead = [self.queues[c] for c in QUEUE_ORDER[: QUEUE_ORDER.index(cls) + 1]]
        others_waiting = any(bool(q) for q in ahead)
        if not others_waiting and not front:
            ledg = self.place(shape, prompt_tokens, kv_needed=kv_tokens)
            if ledg is not None:
                kv = ledg.kv_blocks(kv_tokens) if kv_tokens is not None else ledg.kv_for(shape)
                h = ledg.hold(request_id, shape, kv, prompt_tokens, preemptable, burst_ctx)
                if burst_ctx:
                    self.bursts_admitted += 1
                self._refresh_burst()
                return h
        loop = asyncio.get_running_loop()
        w = Waiter(request_id, shape, prompt_tokens, kv_tokens, preemptable, burst_ctx, now(), loop.create_future())
        if front:
            queue.appendleft(w)
        else:
            queue.append(w)
        self.dispatch()  # may resolve immediately (e.g. requeue preemption)
        await asyncio.wait([w.future], timeout=timeout)
        if w.future.done():
            return w.future.result()
        try:
            queue.remove(w)
        except ValueError:
            pass
        if w.future.done():  # resolved while we were removing it
            return w.future.result()
        w.future.cancel()
        self.timeouts += 1
        self.place(w.shape, w.prompt_tokens, record=w, kv_needed=w.kv_needed)  # the reason it still does not fit
        self.dispatch()
        raise QueueTimeout(w.last_reason, now() - w.t_enqueued)

    def release(self, h: Hold) -> None:
        if h.reserved:
            res = self.reservations.get(h.reservation_id or "")
            h.ledger.vacate(h)
            if res is None or res.t_end is not None:
                h.ledger.release(h)
            else:
                self._dispatch_reservation(res)
        else:
            h.ledger.release(h)
        self.dispatch()

    def mark_prefill_done(self, h: Hold) -> None:
        h.ledger.mark_prefill_done(h)
        self.dispatch()

    def queue_lengths(self) -> dict[str, int]:
        return {k: len(v) for k, v in self.queues.items()}

    def snapshot(self) -> dict[str, Any]:
        return {
            "queues": self.queue_lengths(),
            "preemptions": self.preemptions,
            "timeouts": self.timeouts,
            "revocations": self.revocations,
            "bursts_admitted": self.bursts_admitted,
            "burst": self.burst.model_dump(),
            "reservations": [r.snapshot() for r in self.reservations.values() if r.t_end is None],
            "ledgers": {ledg.id: ledg.snapshot() for ledg in self.ledgers},
        }
