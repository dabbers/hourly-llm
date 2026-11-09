"""Pure statistics helpers for the campaign report (docs/PHASE3.md section 4).

Everything here takes plain sequences of numbers and returns plain Python values, so it is easy
to test against scipy directly and reuse outside pandas. NaN inputs are dropped before anything
is computed; a paired comparison drops the pair when either side is NaN.

- `mean_ci`: mean, sample sd and a two-sided Student-t interval for the mean.
- `paired`: a seed-paired comparison of a setting against its baseline: mean difference with its
  t-interval, paired t-test and Wilcoxon signed-rank p-values, Hedges' g, relative difference.
- `holm`: Holm–Bonferroni adjusted p-values for a family of tests.
- `linear_fit`: least-squares line with r² and the x where the line crosses a level (break-even).
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from scipy import stats as sps

NAN = float("nan")


def _clean(values: Iterable[float]) -> np.ndarray:
    a = np.asarray(list(values), dtype=float)
    return a[~np.isnan(a)]


def mean_ci(values: Iterable[float], conf: float = 0.95) -> tuple[float, float, int, float, float]:
    """(mean, sd, n, lo, hi): the sample mean, the sample standard deviation (ddof = 1) and the
    two-sided Student-t confidence interval for the mean at level `conf`. NaN values are dropped;
    n = 0 gives all-NaN, n = 1 gives the mean with NaN sd and bounds (one run says nothing about
    the spread)."""
    a = _clean(values)
    n = len(a)
    if n == 0:
        return NAN, NAN, 0, NAN, NAN
    mean = float(a.mean())
    if n < 2:
        return mean, NAN, 1, NAN, NAN
    sd = float(a.std(ddof=1))
    half = float(sps.t.ppf((1.0 + conf) / 2.0, n - 1)) * sd / math.sqrt(n)
    return mean, sd, n, mean - half, mean + half


def paired(a: Iterable[float], b: Iterable[float], conf: float = 0.95) -> dict[str, float]:
    """Compare `b` (the setting) against `a` (the baseline) on pairs matched by position (one pair
    per seed). The difference is d = b − a, so a positive `diff` means the setting is higher.

    Returns a dict with: `n` pairs (after dropping pairs with a NaN side), `mean_a`, `mean_b`,
    `diff` = mean(d), `sd_diff`, `diff_lo`/`diff_hi` (Student-t interval for mean(d) at `conf`),
    `rel_diff` = mean(d) ÷ mean(a) (NaN when mean(a) is 0), `p_t` (two-sided paired t-test,
    scipy.stats.ttest_rel; NaN when n < 2 or every difference is identical, where the statistic
    is undefined), `p_wilcoxon` (two-sided Wilcoxon signed-rank test, scipy.stats.wilcoxon, exact
    when scipy can be: no ties and n ≤ 50; NaN when n < 2 or all differences are zero) and
    `hedges_g` = mean(d) ÷ sd(d) × J with the small-sample correction J = 1 − 3 ÷ (4n − 5), the
    standardised size of the paired effect (NaN when n < 2 or sd(d) = 0)."""
    x = np.asarray(list(a), dtype=float)
    y = np.asarray(list(b), dtype=float)
    if len(x) != len(y):
        raise ValueError(f"paired arrays differ in length: {len(x)} vs {len(y)}")
    keep = ~(np.isnan(x) | np.isnan(y))
    x, y = x[keep], y[keep]
    n = len(x)
    out = {
        "n": n, "mean_a": NAN, "mean_b": NAN, "diff": NAN, "sd_diff": NAN, "diff_lo": NAN, "diff_hi": NAN,
        "rel_diff": NAN, "p_t": NAN, "p_wilcoxon": NAN, "hedges_g": NAN,
    }
    if n == 0:
        return out
    d = y - x
    mean_a, mean_b = float(x.mean()), float(y.mean())
    out.update(mean_a=mean_a, mean_b=mean_b, diff=float(d.mean()))
    out["rel_diff"] = float(d.mean() / mean_a) if mean_a != 0 else NAN
    if n < 2:
        return out
    sd = float(d.std(ddof=1))
    out["sd_diff"] = sd
    half = float(sps.t.ppf((1.0 + conf) / 2.0, n - 1)) * sd / math.sqrt(n)
    out["diff_lo"], out["diff_hi"] = float(d.mean() - half), float(d.mean() + half)
    if sd > 0:
        out["p_t"] = float(sps.ttest_rel(y, x).pvalue)
        j = 1.0 - 3.0 / (4.0 * n - 5.0)
        out["hedges_g"] = float(d.mean() / sd * j)
    if np.any(d != 0):
        try:
            out["p_wilcoxon"] = float(sps.wilcoxon(d, zero_method="wilcox", method="auto").pvalue)
        except ValueError:  # e.g. every difference zero after the zero rule
            out["p_wilcoxon"] = NAN
    return out


def holm(pvalues: Iterable[float]) -> list[float]:
    """Holm–Bonferroni step-down adjustment: the i-th smallest p (of m non-NaN ones) is multiplied
    by (m − i + 1), the running maximum is enforced so adjusted values stay monotone, and the
    result is capped at 1. NaN entries are left NaN and do not count towards m. The output keeps
    the input order."""
    p = np.asarray(list(pvalues), dtype=float)
    out = np.full(p.shape, NAN)
    idx = np.flatnonzero(~np.isnan(p))
    m = len(idx)
    if m == 0:
        return out.tolist()
    order = idx[np.argsort(p[idx], kind="stable")]
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, float((m - rank) * p[i]))
        out[i] = min(1.0, running)
    return out.tolist()


@dataclass(frozen=True)
class LinearFit:
    """y = slope × x + intercept, fitted by least squares on n points; r² is the share of the
    variance of y the line explains (NaN when y is constant). `x_at(level)` is the x where the
    line reaches `level` (NaN for a flat line); `y_at(x)` evaluates the line."""

    slope: float
    intercept: float
    r2: float
    n: int

    def x_at(self, level: float) -> float:
        if not math.isfinite(self.slope) or self.slope == 0:
            return NAN
        return (level - self.intercept) / self.slope

    def y_at(self, x: float) -> float:
        return self.slope * x + self.intercept


def linear_fit(x: Iterable[float], y: Iterable[float]) -> LinearFit:
    """Least-squares line through (x, y) pairs; pairs with a NaN side are dropped. Fewer than two
    distinct x values give a NaN fit; a constant y gives slope 0 and NaN r²."""
    xa = np.asarray(list(x), dtype=float)
    ya = np.asarray(list(y), dtype=float)
    if len(xa) != len(ya):
        raise ValueError(f"x and y differ in length: {len(xa)} vs {len(ya)}")
    keep = ~(np.isnan(xa) | np.isnan(ya))
    xa, ya = xa[keep], ya[keep]
    n = len(xa)
    if n < 2 or np.ptp(xa) == 0:
        return LinearFit(NAN, NAN, NAN, n)
    if np.ptp(ya) == 0:  # constant y: an exactly flat line, so x_at() is NaN rather than 1e16
        return LinearFit(0.0, float(ya[0]), NAN, n)
    slope, intercept = np.polyfit(xa, ya, 1)
    resid = ya - (slope * xa + intercept)
    ss_tot = float(((ya - ya.mean()) ** 2).sum())
    r2 = 1.0 - float((resid**2).sum()) / ss_tot if ss_tot > 0 else NAN
    return LinearFit(float(slope), float(intercept), r2, n)
