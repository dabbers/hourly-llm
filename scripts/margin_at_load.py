"""Price needed for a target gross margin when only part of the paid GPU time is busy (RESULTS Q17).

An extrapolation, not a measurement. It takes the revenue per GPU-hour that the runs earned at the
reference prices, assumes the cluster earns that for a share of the hours it pays for and nothing in the
rest, and solves for the price multiple that leaves the target margin:

    margin = 1 - gpu_usd_per_hour / (revenue_per_gpu_hour * busy_share * price_multiple)

Usage: python scripts/margin_at_load.py docs/reports/stage-m --out docs/reports/stage-m/margin_at_load.csv
"""
from __future__ import annotations

import argparse
import csv
import glob
import os

SHARES = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("campaign_dir")
    ap.add_argument("--margin", type=float, default=0.25)
    ap.add_argument("--gpus", type=int, default=3)
    ap.add_argument("--output-usd-per-million", type=float, default=0.175, help="reference output price")
    ap.add_argument("--seat-tps", type=float, default=30.0)
    ap.add_argument("--runs-glob", default="runs/m-*", help="run directories, for seat fill (optional)")
    ap.add_argument("--out")
    a = ap.parse_args()

    runs = list(csv.DictReader(open(os.path.join(a.campaign_dir, "runs.csv"), encoding="utf-8")))
    print("per run, at the reference prices")
    rev_h, cost_h = [], []
    for r in runs:
        gpu_hours = float(r["duration_s"]) / 3600 * a.gpus
        rs, rt, c = float(r["tenant_usd"]) / gpu_hours, float(r["per_token_usd"]) / gpu_hours, float(r["gpu_cost_usd"]) / gpu_hours
        rev_h.append(rs)
        cost_h.append(c)
        need = c / (1 - a.margin)
        print(f"  {r['label']:<11} s{r['seed']}  slots ${rs:.3f}/GPU-h  tokens ${rt:.3f}/GPU-h  cost ${c:.3f}  "
              f"util {float(r['utilisation']):.0%}  price multiple for {a.margin:.0%}: slots {need / rs:.2f}  tokens {need / rt:.2f}")
    rev, cost = sum(rev_h) / len(rev_h), sum(cost_h) / len(cost_h)
    need = cost / (1 - a.margin)
    print(f"\nmean revenue ${rev:.3f} per GPU-hour, cost ${cost:.3f}; {a.margin:.0%} margin needs ${need:.3f}")
    print(f"busy share for {a.margin:.0%} at the reference prices: {need / rev:.0%}; break-even busy share: {cost / rev:.0%}\n")

    seat = a.seat_tps * 3600 / 1e6 * a.output_usd_per_million
    rows = []
    for s in SHARES:
        mult = need / (rev * s)
        rows.append({"busy_share": s, "price_multiple": round(mult, 3), "seat_usd_per_hour": round(seat * mult, 4),
                     "output_usd_per_million": round(a.output_usd_per_million * mult, 4),
                     "margin_at_reference_prices": round(1 - cost / (rev * s), 3)})
        print(f"  busy {s:>4.0%}  price x{mult:.2f}  {a.seat_tps:.0f} tps seat ${seat * mult:.4f}/h  "
              f"output ${a.output_usd_per_million * mult:.3f}/M  margin at reference prices {1 - cost / (rev * s):+.0%}")
    if a.out:
        with open(a.out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

    fills, utils = [], []
    for p in sorted(glob.glob(os.path.join(a.runs_glob, "report", "per_replica.csv"))):
        for r in csv.DictReader(open(p, encoding="utf-8")):
            fills.append(float(r["fill_rate"]))
            utils.append(float(r["utilization"]))
    if fills:
        print(f"\nper replica over {len(fills)} replica-runs: seat fill {min(fills):.0%} to {max(fills):.0%}, "
              f"token utilisation {min(utils):.0%} to {max(utils):.0%}")


if __name__ == "__main__":
    main()
