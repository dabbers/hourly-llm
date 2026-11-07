"""Per-request delivery pacing: a token bucket at the shape's rate plus a revocable burst bucket
(docs/DESIGN.md sections 6 and 7).

The buckets decide *when* each chunk of tokens may be delivered to the client; they never slow the
engine. `schedule()` is pure arithmetic over timestamps, so streamed delivery (sleep until the
scheduled time, then send) and complete delivery (release the whole response at the time the paced
stream would have sent its last token) share one model and one definition of throttled time:
the wall-clock time delivery was held back by the buckets, not by the engine.

Every token draws from the guaranteed bucket first; what the guaranteed bucket cannot cover is
burst, paid by the burst bucket when the burst manager has lent a rate (`set_burst_rate`), or
waited for otherwise. Revoking the burst (rate 0) takes effect on the next chunk.
"""

from __future__ import annotations


class TokenBucket:
    """Continuous-time token bucket. Starts full so a fresh request can deliver `depth` tokens
    at once (default: one second of the shape's rate)."""

    def __init__(self, rate: float, depth: float, t0: float):
        if rate < 0:
            raise ValueError("rate must not be negative")
        self.rate = rate
        self.depth = max(0.0, depth)
        self.level = self.depth
        self.t = t0  # time at which `level` is valid

    def refill(self, t: float) -> None:
        if t > self.t:
            self.level = min(self.depth, self.level + self.rate * (t - self.t))
            self.t = t

    def take(self, n: int) -> int:
        """Take up to n whole tokens from the current level; returns how many were taken."""
        k = min(n, int(self.level))
        self.level -= k
        return k

    def schedule(self, n: int, t_now: float) -> tuple[float, float]:
        """Consume n tokens that became available at t_now. Returns (t_deliver, wait) where
        wait is the time delivery is held by the bucket beyond max(t_now, previous delivery)."""
        t = max(t_now, self.t)
        self.refill(t)
        if self.level >= n:
            self.level -= n
            return t, 0.0
        if self.rate <= 0:
            raise ValueError("bucket has no rate")
        wait = (n - self.level) / self.rate
        self.level = 0.0
        self.t = t + wait
        return self.t, wait


class Pacer:
    """Pacing state for one request: a guaranteed bucket at the shape's tps and a burst bucket at
    whatever rate the burst manager currently lends. Counts tokens by which bucket paid."""

    def __init__(self, tps: float, bucket_depth_s: float = 1.0, enabled: bool = True):
        if tps <= 0:
            raise ValueError("tps must be positive")
        self.tps = tps
        self.bucket_depth_s = bucket_depth_s
        self.enabled = enabled
        self._guaranteed: TokenBucket | None = None
        self._burst: TokenBucket | None = None
        self.burst_rate = 0.0
        self.throttled_s = 0.0
        self.scheduled_tokens = 0
        self.guaranteed_tokens = 0
        self._burst_tokens = 0.0
        self.t_last_scheduled: float | None = None
        self.burst_rate_max = 0.0  # highest rate lent during the request (for the meter)

    @property
    def burst_tokens(self) -> int:
        return round(self._burst_tokens)

    def set_burst_rate(self, rate: float, t_now: float) -> None:
        """Lend (or revoke, rate 0) burst rate from now on. Lent rate is not banked: a change
        re-opens the burst bucket at the new rate with one bucket depth of credit."""
        rate = max(0.0, rate)
        if rate == self.burst_rate:
            return
        self.burst_rate = rate
        self.burst_rate_max = max(self.burst_rate_max, rate)
        self._burst = TokenBucket(rate, rate * self.bucket_depth_s, t_now) if rate > 0 else None

    def schedule(self, n: int, t_now: float) -> float:
        """Earliest time n tokens available at t_now may be delivered."""
        if n <= 0:
            return t_now
        self.scheduled_tokens += n
        if not self.enabled:
            self.guaranteed_tokens += n
            self.t_last_scheduled = t_now
            return t_now
        if self._guaranteed is None:
            self._guaranteed = TokenBucket(self.tps, self.tps * self.bucket_depth_s, t_now)
        t = max(t_now, self.t_last_scheduled or t_now)
        g = self._guaranteed
        g.refill(t)
        n_g = g.take(n)
        rest = n - n_g
        t_deliver = t
        n_b = 0.0
        if rest > 0:
            b = self._burst
            if b is not None:
                b.refill(t)
                taken = b.take(rest)
                n_b += taken
                rest -= taken
            if rest > 0:
                rate = g.rate + (b.rate if b is not None else 0.0)
                wait = rest / rate
                t_deliver = t + wait
                self.throttled_s += wait
                g.level = 0.0
                g.t = t_deliver
                if b is not None:
                    b.level = 0.0
                    b.t = t_deliver
                    n_b += rest * (b.rate / rate)  # the burst bucket's share of the wait
        self.guaranteed_tokens += n - round(n_b) if n_b else n
        self._burst_tokens += n_b
        self.t_last_scheduled = t_deliver
        return t_deliver

    @property
    def throttled_ms(self) -> float:
        return self.throttled_s * 1000.0
