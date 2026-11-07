"""Arrival processes (docs/DESIGN.md section 9). Each returns sorted arrival offsets in seconds
within [0, duration_s), drawn from the given generator so a tenant's arrivals depend only on its
child seed."""

from __future__ import annotations

import numpy as np

from .scenario import (
    Arrivals,
    BurstyArrivals,
    DiurnalArrivals,
    HourmarkArrivals,
    PoissonArrivals,
    Scenario,
)


def poisson_times(rng: np.random.Generator, rate_per_s: float, t0: float, t1: float) -> list[float]:
    """Homogeneous Poisson arrivals in [t0, t1)."""
    if rate_per_s <= 0 or t1 <= t0:
        return []
    out: list[float] = []
    t = t0
    while True:
        t += rng.exponential(1.0 / rate_per_s)
        if t >= t1:
            return out
        out.append(t)


def diurnal_times(rng: np.random.Generator, scenario: Scenario, rate_per_s: float) -> list[float]:
    """Non-homogeneous Poisson by thinning: rate(t) = rate_per_s x phase rate_scale."""
    lam_max = rate_per_s * scenario.max_rate_scale
    if lam_max <= 0:
        return []
    out: list[float] = []
    t = 0.0
    duration = scenario.duration_s
    while True:
        t += rng.exponential(1.0 / lam_max)
        if t >= duration:
            return out
        if rng.uniform() * lam_max < rate_per_s * scenario.rate_scale_at(t):
            out.append(t)


def bursty_times(rng: np.random.Generator, spec: BurstyArrivals, duration: float) -> list[float]:
    """Markov-modulated Poisson: alternate exponential on/off periods, Poisson within each."""
    out: list[float] = []
    t = 0.0
    on = rng.uniform() < spec.mean_on_s / (spec.mean_on_s + spec.mean_off_s)
    while t < duration:
        length = rng.exponential(spec.mean_on_s if on else spec.mean_off_s)
        rate = (spec.rate_on_per_min if on else spec.rate_off_per_min) / 60.0
        out.extend(poisson_times(rng, rate, t, min(duration, t + length)))
        t += length
        on = not on
    return out


def hourmark_times(
    rng: np.random.Generator, spec: HourmarkArrivals, hour_length_s: float, duration: float
) -> list[float]:
    out: list[float] = []
    k = 1
    while True:
        mark = k * hour_length_s + spec.offset_s
        if mark > duration:
            break
        n = int(rng.poisson(spec.per_mark))
        for _ in range(n):
            t = mark + rng.normal(0.0, spec.sigma_s) if spec.sigma_s > 0 else mark
            if 0.0 <= t < duration:
                out.append(float(t))
        k += 1
    out.extend(poisson_times(rng, spec.background_per_min / 60.0, 0.0, duration))
    out.sort()
    return out


def arrival_times(rng: np.random.Generator, spec: Arrivals, scenario: Scenario) -> list[float]:
    duration = scenario.duration_s
    if isinstance(spec, PoissonArrivals):
        return poisson_times(rng, spec.rate_per_min / 60.0, 0.0, duration)
    if isinstance(spec, DiurnalArrivals):
        return diurnal_times(rng, scenario, spec.rate_per_min / 60.0)
    if isinstance(spec, BurstyArrivals):
        return bursty_times(rng, spec, duration)
    if isinstance(spec, HourmarkArrivals):
        return hourmark_times(rng, spec, scenario.hour_length_s, duration)
    raise ValueError(f"closed-loop tenants have no arrival times ({spec.kind})")
