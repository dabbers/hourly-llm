"""High-resolution, monotonic epoch clock for request timestamps.

`time.time()` is coarse on some platforms (about 15.6 ms on Windows with Python < 3.13) and can
step backwards. Billing by wall-clock runtime needs sub-millisecond, monotonic timestamps that are
still comparable across processes, so this module anchors the epoch once and advances it with
`perf_counter()`.
"""

from __future__ import annotations

import time

_WALL0 = time.time()
_MONO0 = time.perf_counter()


def now() -> float:
    """Seconds since the Unix epoch, monotonic within this process, sub-millisecond resolution."""
    return _WALL0 + (time.perf_counter() - _MONO0)
