"""Throughput profiles (docs/DESIGN.md section 5.2): per-stream decode rate as a function of the
number of concurrent streams, from a calibration file written by scripts/calibrate.py or from an
inline curve string such as "1:120,2:100,4:70,8:50" (the fake replica's format)."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

import yaml

from .config import ReplicaConfig

MAX_SEATS_SEARCH = 1024


@dataclass(frozen=True)
class ThroughputProfile:
    curve: tuple[tuple[int, float], ...]  # (n concurrent streams, per-stream tokens/s), ascending n
    source: str = ""

    def step_rate(self, n: int) -> float:
        """Per-stream decode tokens/s at n concurrent streams; linear between points, linear
        extrapolation past the last point (never below 0.1)."""
        n = max(1, n)
        c = self.curve
        if n <= c[0][0]:
            return c[0][1]
        for (n0, r0), (n1, r1) in pairwise(c):
            if n0 <= n <= n1:
                return r1 if n1 == n0 else r0 + (r1 - r0) * (n - n0) / (n1 - n0)
        if len(c) >= 2:
            (n0, r0), (n1, r1) = c[-2], c[-1]
            slope = (r1 - r0) / (n1 - n0) if n1 != n0 else 0.0
            return max(0.1, r1 + slope * (n - n1))
        return c[-1][1]

    def n_max(self, tps: float, margin: float = 0.0) -> int:
        """Largest n such that step_rate(n) x (1 - margin) >= tps; 0 if even one stream cannot."""
        best = 0
        for n in range(1, MAX_SEATS_SEARCH + 1):
            if self.step_rate(n) * (1.0 - margin) >= tps:
                best = n
            else:
                break
        return best

    @classmethod
    def from_curve_string(cls, text: str, source: str = "inline") -> ThroughputProfile:
        pts = []
        for part in text.split(","):
            n, rate = part.split(":")
            pts.append((int(n), float(rate)))
        if not pts:
            raise ValueError("empty curve")
        return cls(curve=tuple(sorted(pts)), source=source)

    @classmethod
    def from_calibration(cls, path: str | Path, context: int | None = None) -> ThroughputProfile:
        """Use the `step_rate_pure_mean` series of one calibration level. `context` picks the level
        (prompt tokens it was measured at); default: the longest context measured (conservative)."""
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        levels: dict[str, dict] = data.get("levels") or {}
        if not levels:
            raise ValueError(f"{path}: no levels")
        by_ctx = {int(k.split("_", 1)[1]): v for k, v in levels.items() if k.startswith("context_")}
        if context is None:
            context = max(by_ctx)
        if context not in by_ctx:
            raise ValueError(f"{path}: no level for context {context}; have {sorted(by_ctx)}")
        pts = [
            (int(v["n"]), float(v["step_rate_pure_mean"]))
            for v in by_ctx[context].values()
            if v.get("step_rate_pure_mean") is not None
        ]
        if not pts:
            raise ValueError(f"{path}: level context_{context} has no step rates")
        return cls(curve=tuple(sorted(pts)), source=f"{Path(path).name}:context_{context}")


def profile_for(cfg: ReplicaConfig, config_dir: Path | None = None) -> ThroughputProfile:
    """The replica's profile: an inline `curve:` or its calibration file, resolved relative to the
    working directory or to the config directory's parent (the repository root)."""
    if cfg.curve:
        return ThroughputProfile.from_curve_string(cfg.curve, source=f"{cfg.id}:inline")
    if cfg.calibration:
        path = Path(cfg.calibration)
        if not path.is_file() and config_dir is not None:
            path = Path(config_dir).parent / cfg.calibration
        return ThroughputProfile.from_calibration(path, cfg.curve_context)
    raise RuntimeError(f"replica {cfg.id}: admission needs `calibration` or `curve` in replicas.yaml")
