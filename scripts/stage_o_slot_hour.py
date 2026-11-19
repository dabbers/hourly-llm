"""Stage O, offline (RESULTS Q19): re-price finished runs under slot-hour billing. No GPU time.

Slot-hour billing: a tenant pays a flat price per slot per hour for its tier, used or not; no prompt charge,
no runtime seconds, no cache-read price. The price of a tier is the GPU cost of one seat of that tier
(the report's `breakeven_slot_usd_per_hour`, GPU $/h x the shape's capacity weight / seats) divided by the
share of seats the provider expects to sell (`--sold`, 0.60): at that share the provider breaks even.

Each run's report already has a what-if table (`report/reservations.csv`): the slots a tenant would need
to cover its measured concurrency at p50, p95 and max. This script takes the p95 row as the tenant's
purchase for that "day" (a seed) and compares the slot-hour bill with the per-token and per-second bills of
the same requests.

Usage: python scripts/stage_o_slot_hour.py runs/k-base-s* runs/k-affinity-s* runs/m-* --out docs/reports/stage-o/slot_hour.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import re
from collections import defaultdict

import numpy as np


def cv(xs: list[float]) -> float:
    return float(np.std(xs, ddof=1) / np.mean(xs)) if len(xs) > 1 and np.mean(xs) else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--sold", type=float, default=0.60, help="share of seats sold at which the tier price breaks even")
    ap.add_argument("--sizing", default="p95")
    ap.add_argument("--out")
    a = ap.parse_args()

    rows, tiers = [], {}
    for d in sorted(a.run_dirs):
        path = os.path.join(d, "report", "reservations.csv")
        summ = os.path.join(d, "report", "summary.csv")
        if not (os.path.exists(path) and os.path.exists(summ)):
            continue
        name = os.path.basename(os.path.normpath(d))
        m = re.match(r"(.+)-s(\d+)-\d{8}-\d{6}$", name)
        arm, seed = m.group(1), int(m.group(2))
        s = next(csv.DictReader(open(summ, encoding="utf-8")))
        gpu_cost, hours = float(s["gpu_cost_usd"]), float(s["duration_s"]) / 3600
        for r in csv.DictReader(open(path, encoding="utf-8")):
            if r["sizing"] != a.sizing:
                continue
            price = float(r["breakeven_slot_usd_per_hour"]) / a.sold
            tiers[r["shape"]] = (float(r["breakeven_slot_usd_per_hour"]), price)
            slot_hours = float(r["reserved_slot_hours"])
            rows.append({
                "arm": arm, "seed": seed, "tenant": r["tenant"], "shape": r["shape"], "slots": int(r["slots"]),
                "slot_hours": slot_hours, "used_fill": float(r["used_fill"]), "would_queue_share": float(r["would_queue_share"]),
                "slot_usd_per_hour": price, "slot_hour_usd": slot_hours * price, "per_token_usd": float(r["per_token_usd"]),
                "per_second_usd": float(r["shape_usd"]), "gpu_share_usd": float(r["cost_share_usd"]),
                "run_gpu_cost_usd": gpu_cost, "run_hours": hours,
            })
    if a.out:
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        with open(a.out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

    print(f"tier prices: GPU cost of one seat, and the slot-hour price that breaks even with {a.sold:.0%} of seats sold")
    for shape, (be, price) in sorted(tiers.items()):
        print(f"  {shape:<22} seat cost ${be:.4f}/h   slot-hour price ${price:.4f}   (${price * 720:.2f} a month)")

    by_run = defaultdict(list)
    for r in rows:
        by_run[(r["arm"], r["seed"])].append(r)
    print(f"\nper run ({a.sizing} sizing): revenue under each scheme, share of GPU capacity sold, provider margin")
    per_arm = defaultdict(list)
    for (arm, seed), rs in sorted(by_run.items()):
        cost = rs[0]["run_gpu_cost_usd"]
        slot, tok, sec = (sum(r[k] for r in rs) for k in ("slot_hour_usd", "per_token_usd", "per_second_usd"))
        sold = sum(r["gpu_share_usd"] for r in rs) / cost
        per_arm[arm].append({"slot": slot, "tok": tok, "sec": sec, "sold": sold, "cost": cost, "slots": sum(r["slots"] for r in rs)})
        print(f"  {arm:<11} s{seed}  slots {sum(r['slots'] for r in rs):>3}  sold {sold:>4.0%}  slot-hour ${slot:.3f} (margin {1 - cost / slot:+.0%})"
              f"  per token ${tok:.3f} ({1 - cost / tok:+.0%})  per second ${sec:.3f} ({1 - cost / sec:+.0%})")
    print("\nper arm, mean over seeds")
    means = {}
    for arm, xs in sorted(per_arm.items()):
        mean = {k: float(np.mean([x[k] for x in xs])) for k in xs[0]}
        means[arm] = mean
        print(f"  {arm:<11} slots {mean['slots']:.1f}  sold {mean['sold']:.0%}  slot-hour ${mean['slot']:.3f} ({1 - mean['cost'] / mean['slot']:+.0%})"
              f"  per token ${mean['tok']:.3f} ({1 - mean['cost'] / mean['tok']:+.0%})  per second ${mean['sec']:.3f} ({1 - mean['cost'] / mean['sec']:+.0%})")
    for base, arm in (("k-base", "k-affinity"), ("m-affinity", "m-queue")):
        if base in means and arm in means:
            b, x = means[base], means[arm]
            print(f"\n  {arm} against {base}: slots {x['slots'] / b['slots'] - 1:+.1%}, slot-hour revenue {x['slot'] / b['slot'] - 1:+.1%}, "
                  f"per-token revenue {x['tok'] / b['tok'] - 1:+.1%}, per-second revenue {x['sec'] / b['sec'] - 1:+.1%}")

    print("\nday-to-day variation of each tenant's bill (coefficient of variation over seeds), and the slot-hour bill against per token")
    by_tenant = defaultdict(list)
    for r in rows:
        by_tenant[(r["arm"], r["tenant"])].append(r)
    for (arm, tenant), rs in sorted(by_tenant.items()):
        print(f"  {arm:<11} {tenant:<10} slots {[r['slots'] for r in rs]}  CV slot-hour {cv([r['slot_hour_usd'] for r in rs]):.1%}"
              f"  per token {cv([r['per_token_usd'] for r in rs]):.1%}  per second {cv([r['per_second_usd'] for r in rs]):.1%}"
              f"  slot-hour / per token {np.mean([r['slot_hour_usd'] for r in rs]) / np.mean([r['per_token_usd'] for r in rs]):.2f}x"
              f"  used {np.mean([r['used_fill'] for r in rs]):.0%}  would queue {np.mean([r['would_queue_share'] for r in rs]):.1%}")


if __name__ == "__main__":
    main()
