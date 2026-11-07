"""Admission control: per-replica ledgers (KV, seats, lazy share, prefill), priority queues and the
shape-affinity scheduler (docs/DESIGN.md sections 5.2-5.4 and 4.1 step 5).

A request holds `kv_for(shape)` tokens of KV and one seat from admission until the engine finishes
(delivery may continue afterwards, paced). Seats are bounded by the calibrated step-rate curve: with
n streams running, every stream gets step_rate(n) tokens/s, so the replica may run at most
n_max(t) streams while a shape of t tokens/s is among them. The fastest running shape sets the
ceiling; `tps_overcommit` relaxes it. Lazy shapes fill idle capacity only: limited to a share of the
seats and admitted only when no interactive request is waiting.
"""

from __future__ import annotations

import asyncio
import logging
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Protocol

from .calibration import ThroughputProfile
from .clock import now
from .config import AdmissionSettings, Shape

log = logging.getLogger("slotgw.admission")


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


@dataclass
class Hold:
    request_id: str
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
    task: asyncio.Task | None = None  # the engine run to cancel on preemption


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
    def lazy_running(self) -> int:
        return sum(1 for h in self.holds.values() if h.lazy)

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
        return math.ceil(shape.max_context / self.block_size) * self.block_size

    def seats(self, tps: float) -> int:
        """Seats available while a shape of `tps` is among the running streams."""
        n = self.profile.n_max(tps, self.settings.tps_margin)
        return math.floor(n * self.settings.tps_overcommit)

    def seats_now(self) -> int:
        return self.seats(self.max_tps_running) if self.holds else self.seats(0.0)

    # ---- admission checks ---------------------------------------------------------------------

    def check(self, shape: Shape, kv_needed: int, prompt_tokens: int, exclude: Hold | None = None) -> str | None:
        """None if the request fits now, else the first failing rule."""
        if not self.healthy:
            return "unhealthy"
        if self.seats(shape.tps) == 0:
            return "unsellable"
        holds = [h for h in self.holds.values() if h is not exclude]
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

    def hold(self, request_id: str, shape: Shape, kv_needed: int, prompt_tokens: int, preemptable: bool = False) -> Hold:
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
        )
        self.holds[request_id] = h
        self.admitted_total += 1
        return h

    def release(self, h: Hold) -> None:
        if not h.released:
            h.released = True
            self.holds.pop(h.request_id, None)

    def mark_prefill_done(self, h: Hold) -> None:
        h.prefill_done = True

    def snapshot(self) -> dict[str, Any]:
        return {
            "healthy": self.healthy,
            "kv_total": self.kv_total,
            "kv_usable": self.kv_usable,
            "kv_held": self.kv_held,
            "block_size": self.block_size,
            "running": self.n_running,
            "lazy_running": self.lazy_running,
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
    preemptable: bool
    t_enqueued: float
    future: asyncio.Future[Hold]
    last_reason: str = "capacity"
    reasons: dict[str, int] = field(default_factory=dict)


class Scheduler:
    """Places requests on replicas, queues them when nothing fits, and times them out."""

    def __init__(self, ledgers: list[ReplicaLedger], settings: AdmissionSettings):
        self.ledgers = ledgers
        self.settings = settings
        self.queues: dict[str, deque[Waiter]] = {"interactive": deque(), "lazy": deque()}
        self.preemptions = 0
        self.timeouts = 0

    # ---- placement ----------------------------------------------------------------------------

    def candidates(self, shape: Shape) -> list[ReplicaLedger]:
        return [ledg for ledg in self.ledgers if ledg.model_alias == shape.model]

    def place(self, shape: Shape, prompt_tokens: int, record: Waiter | None = None) -> ReplicaLedger | None:
        best: tuple[tuple[int, float], ReplicaLedger] | None = None
        for ledg in self.candidates(shape):
            reason = ledg.check(shape, ledg.kv_for(shape), prompt_tokens)
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

    def dispatch(self) -> None:
        """Admit queued waiters in priority order: interactive FIFO first, lazy only when no
        interactive request is waiting. Strict FIFO within a class (no overtaking)."""
        q = self.queues["interactive"]
        while q:
            w = q[0]
            if w.future.done():
                q.popleft()
                continue
            ledg = self.place(w.shape, w.prompt_tokens, record=w) or self._preempt_for(w.shape, w.prompt_tokens)
            if ledg is None:
                break
            q.popleft()
            w.future.set_result(ledg.hold(w.request_id, w.shape, ledg.kv_for(w.shape), w.prompt_tokens, w.preemptable))
        if q:
            return  # interactive work waiting: lazy filler stays queued
        q = self.queues["lazy"]
        while q:
            w = q[0]
            if w.future.done():
                q.popleft()
                continue
            ledg = self.place(w.shape, w.prompt_tokens, record=w)
            if ledg is None:
                break
            q.popleft()
            w.future.set_result(ledg.hold(w.request_id, w.shape, ledg.kv_for(w.shape), w.prompt_tokens, w.preemptable))

    # ---- public API ---------------------------------------------------------------------------

    async def admit(
        self,
        request_id: str,
        shape: Shape,
        prompt_tokens: int,
        *,
        preemptable: bool = False,
        front: bool = False,
        max_wait_s: float | None = None,
    ) -> Hold:
        """Return a Hold on some replica, waiting up to the shape's max_queue_s."""
        if not self.sellable(shape):
            raise Unsellable(shape.name)
        cls = "lazy" if shape.lazy else "interactive"
        queue = self.queues[cls]
        others_waiting = bool(self.queues["interactive"]) or (cls == "lazy" and bool(queue))
        if not others_waiting and not front:
            ledg = self.place(shape, prompt_tokens)
            if ledg is not None:
                return ledg.hold(request_id, shape, ledg.kv_for(shape), prompt_tokens, preemptable)
        loop = asyncio.get_running_loop()
        w = Waiter(request_id, shape, prompt_tokens, preemptable, now(), loop.create_future())
        if front:
            queue.appendleft(w)
        else:
            queue.append(w)
        self.dispatch()  # may resolve immediately (e.g. requeue preemption)
        timeout = shape.max_queue_s if max_wait_s is None else max_wait_s
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
        self.place(w.shape, w.prompt_tokens, record=w)  # the reason it still does not fit
        self.dispatch()
        raise QueueTimeout(w.last_reason, now() - w.t_enqueued)

    def release(self, h: Hold) -> None:
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
            "ledgers": {ledg.id: ledg.snapshot() for ledg in self.ledgers},
        }
