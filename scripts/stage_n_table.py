"""Stage N table (RESULTS Q18): per run and tenant, waits, queue time, delivered rate and prompt reading.

Usage: python scripts/stage_n_table.py runs/n-* --out docs/reports/stage-n/stage_n_table.csv

"Wait" is arrival to first token. "Queue" is arrival to admission. A stream is "slow" when it delivered
under 95% of its shape's rate from first to last token. "New prompt tokens" are prompt tokens the engine
did not serve from its prefix cache.
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sqlite3
from collections import defaultdict

import numpy as np

TPS = re.compile(r"\.(\d+)tps")


def load(run_dir: str) -> list[dict]:
    c = sqlite3.connect(f"file:{run_dir}/meter.sqlite?mode=ro", uri=True)
    c.row_factory = sqlite3.Row
    return [dict(r) for r in c.execute("select * from requests")]


def stats(rows: list[dict]) -> dict:
    ok = [r for r in rows if r["status"] == "ok" and r["t_first_token"] is not None]
    out = {"requests": len(rows), "ok": len(ok), "refused": sum(r["status"] == "refused" for r in rows),
           "other": sum(r["status"] not in ("ok", "refused") for r in rows)}
    if not ok:
        return out
    wait = np.array([r["t_first_token"] - r["t_arrival"] for r in ok])
    queue = np.array([r["t_admitted"] - r["t_arrival"] for r in ok])
    read = np.array([r["t_first_token"] - r["t_admitted"] for r in ok])
    e2e = np.array([r["t_last_token"] - r["t_arrival"] for r in ok])
    slow = []
    for r in ok:
        m = TPS.search(r["shape"])
        dur = r["t_last_token"] - r["t_first_token"]
        if m and dur > 1.0 and (r["output_tokens"] or 0) > 20:
            slow.append((r["output_tokens"] - 1) / dur < 0.95 * float(m.group(1)))
    new = sum((r["prompt_tokens"] or 0) - (r["cached_tokens"] or 0) for r in ok)
    out.update({
        "wait_mean_s": wait.mean(), "wait_p50_s": np.percentile(wait, 50), "wait_p95_s": np.percentile(wait, 95),
        "queue_mean_s": queue.mean(), "queue_p95_s": np.percentile(queue, 95), "read_mean_s": read.mean(),
        "e2e_mean_s": e2e.mean(), "slow_stream_share": float(np.mean(slow)) if slow else float("nan"),
        "output_tokens": sum(r["output_tokens"] or 0 for r in ok), "prompt_tokens": sum(r["prompt_tokens"] or 0 for r in ok),
        "new_prompt_tokens": new, "cached_share": 1 - new / max(1, sum(r["prompt_tokens"] or 0 for r in ok)),
    })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--out")
    a = ap.parse_args()
    table = []
    for d in sorted(a.run_dirs):
        if not os.path.exists(os.path.join(d, "meter.sqlite")):
            continue
        name = os.path.basename(os.path.normpath(d))
        m = re.match(r"(.+)-s(\d+)-\d{8}-\d{6}$", name)
        arm, seed = (m.group(1), int(m.group(2))) if m else (name, 0)
        rows = load(d)
        groups = defaultdict(list)
        for r in rows:
            groups[r["tenant"]].append(r)
            groups["all"].append(r)
            groups[r["tenant"].split("-")[0] + "-*"].append(r)
        for tenant in sorted(groups):
            table.append({"arm": arm, "seed": seed, "tenant": tenant, **stats(groups[tenant])})
    cols = list(dict.fromkeys(k for row in table for k in row))
    if a.out:
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        with open(a.out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(table)
    show = ["ok", "refused", "wait_mean_s", "wait_p95_s", "queue_mean_s", "read_mean_s", "slow_stream_share", "output_tokens", "new_prompt_tokens", "cached_share"]
    print(f"{'arm':<8}{'seed':>4} {'tenant':<10}" + "".join(f"{c[:13]:>14}" for c in show))
    for row in table:
        cells = "".join(f"{row.get(c, float('nan')):>14.2f}" if isinstance(row.get(c), float) else f"{row.get(c, ''):>14}" for c in show)
        print(f"{row['arm']:<8}{row['seed']:>4} {row['tenant']:<10}{cells}")


if __name__ == "__main__":
    main()
