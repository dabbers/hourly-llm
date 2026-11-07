"""Per-request delivery pacing: a token bucket at the shape's rate (docs/DESIGN.md section 6).

The bucket decides *when* each chunk of tokens may be delivered to the client; it never slows the
engine. `schedule()` is pure arithmetic over timestamps, so streamed delivery (sleep until the
scheduled time, then send) and complete delivery (release the whole response at the time the paced
stream would have sent its last token) share one model and one definition of throttled time:
the wall-clock time delivery was held back by the bucket, not by the engine.
"""

from __future__ import annotations


class TokenBucket:
    """Continuous-time token bucket. Starts full so a fresh request can deliver `depth` tokens
    at once (default: one second of the shape's rate)."""

    def __init__(self, rate: float, depth: float, t0: float):
        if rate <= 0:
            raise ValueError("rate must be positive")
        self.rate = rate
        self.depth = max(0.0, depth)
        self.level = self.depth
        self.t = t0  # time at which `level` is valid; also the last scheduled delivery time

    def schedule(self, n: int, t_now: float) -> tuple[float, float]:
        """Consume n tokens that became available at t_now. Returns (t_deliver, wait) where
        wait is the time delivery is held by the bucket beyond max(t_now, previous delivery)."""
        t = max(t_now, self.t)
        self.level = min(self.depth, self.level + self.rate * (t - self.t))
        if self.level >= n:
            self.level -= n
            self.t = t
            return t, 0.0
        wait = (n - self.level) / self.rate
        self.level = 0.0
        self.t = t + wait
        return self.t, wait


class Pacer:
    """Pacing state for one request. Milestone 2: one guaranteed bucket at the shape's tps.
    Milestone 4 adds a revocable burst bucket on top and classifies tokens by which bucket paid."""

    def __init__(self, tps: float, bucket_depth_s: float = 1.0, enabled: bool = True):
        self.tps = tps
        self.bucket_depth_s = bucket_depth_s
        self.enabled = enabled
        self._bucket: TokenBucket | None = None
        self.throttled_s = 0.0
        self.scheduled_tokens = 0
        self.t_last_scheduled: float | None = None

    def schedule(self, n: int, t_now: float) -> float:
        """Earliest time n tokens available at t_now may be delivered."""
        if n <= 0:
            return t_now
        self.scheduled_tokens += n
        if not self.enabled:
            self.t_last_scheduled = t_now
            return t_now
        if self._bucket is None:
            self._bucket = TokenBucket(self.tps, self.tps * self.bucket_depth_s, t_now)
        t_deliver, wait = self._bucket.schedule(n, t_now)
        self.throttled_s += wait
        self.t_last_scheduled = t_deliver
        return t_deliver

    @property
    def throttled_ms(self) -> float:
        return self.throttled_s * 1000.0
