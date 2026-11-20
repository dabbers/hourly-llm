"""Stage O table (RESULTS Q19 part B): per run and tenant, waits, delivered output rate and input-bucket waits.

Usage: python scripts/stage_o_table.py runs/n-fifo-* runs/o-* --out docs/reports/stage-o/stage_o_table.csv

"Wait" is arrival to first token. "Queue" is arrival to admission and includes the input-bucket wait.
"Read" is admission to first token. The delivered rate of a stream is its output tokens over the time from
first to last token, as a share of its tier's rate (streams over 1 s and 20 tokens).
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
ARMS = ["n-fifo", "o-slots", "o-bucket", "o-prio", "o-over"]


def load(run_dir: str) -> list[dict]:
    c = sqlite3.connect(f"file:{run_dir}/meter.sqlite?mode=ro", uri=True)
    c.row_factory = sqlite3.Row
    return [dict(r) for r in c.execute("select * from requests")]


def stats(rows: list[dict]) -> dict:
    ok = [r for r in rows if r["status"] == "ok" and r["t_first_token"] is not None]
    refused = [r for r in rows if r["status"] == "refused"]
    out = {"requests": len(rows), "ok": len(ok), "refused": len(refused),
           "refused_ingest_rate": sum("ingest_rate" in (r["abort_reason"] or "") for r in refused),
           "refused_slots_busy": sum("reservation_busy" in (r["abort_reason"] or "") for r in refused),
           "other": len(rows) - len(ok) - len(refused)}
    if not ok:
        return out
    wait = np.array([r["t_first_token"] - r["t_arrival"] for r in ok])
    queue = np.array([r["t_admitted"] - r["t_arrival"] for r in ok])
    read = np.array([r["t_first_token"] - r["t_admitted"] for r in ok])
    rate = []
    for r in ok:
        m = TPS.search(r["shape"])
        dur = r["t_last_token"] - r["t_first_token"]
        if m and dur > 1.0 and (r["output_tokens"] or 0) > 20:
            rate.append((r["output_tokens"] - 1) / dur / float(m.group(1)))
    bucket = np.array([r["ingest_wait_ms"] / 1000 for r in ok if r.get("ingest_wait_ms")])
    out.update({
        "wait_mean_s": wait.mean(), "wait_p50_s": np.percentile(wait, 50), "wait_p95_s": np.percentile(wait, 95),
        "queue_mean_s": queue.mean(), "queue_p95_s": np.percentile(queue, 95),
        "read_mean_s": read.mean(), "read_p95_s": np.percentile(read, 95),
        "rate_p5": float(np.percentile(rate, 5)) if rate else float("nan"),
        "rate_p50": float(np.percentile(rate, 50)) if rate else float("nan"),
        "bucket_waits": len(bucket), "bucket_wait_mean_s": float(bucket.mean()) if len(bucket) else 0.0,
        "bucket_wait_all_mean_s": float(bucket.sum() / len(ok)),
        "output_tokens": sum(r["output_tokens"] or 0 for r in ok),
        "prompt_tokens": sum(r["prompt_tokens"] or 0 for r in ok),
    })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--out")
    a = ap.parse_args()
    table, cell = [], {}
    for d in sorted(a.run_dirs):
        if not os.path.exists(os.path.join(d, "meter.sqlite")):
            continue
        name = os.path.basename(os.path.normpath(d))
        m = re.match(r"(.+)-s(\d+)-\d{8}-\d{6}$", name)
        arm, seed = m.group(1), int(m.group(2))
        groups = defaultdict(list)
        for r in load(d):
            kind, tier = r["tenant"].split("-")
            for g in (r["tenant"], "all", kind + "-*", "*-" + tier):
                groups[g].append(r)
        for tenant in sorted(groups):
            s = stats(groups[tenant])
            cell[(arm, seed, tenant)] = s
            table.append({"arm": arm, "seed": seed, "tenant": tenant, **s})
    if a.out:
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        cols = list(dict.fromkeys(k for row in table for k in row))
        with open(a.out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(table)

    seeds = sorted({k[1] for k in cell})
    arms = [x for x in ARMS if any(k[0] == x for k in cell)]

    def line(title: str, tenant: str, key: str, fmt: str = "{:.2f}") -> None:
        print(f"  {title:<34}" + "".join(
            f"{arm + ' ':>11}" + " / ".join(fmt.format(cell[(arm, s, tenant)].get(key, float('nan'))) for s in seeds) for arm in arms))

    print(f"per seed ({' / '.join(map(str, seeds))})")
    for tenant in ("all", "chat-*", "docs-*", "*-std", "*-pro", "chat-std", "chat-pro", "docs-std", "docs-pro"):
        print(tenant)
        line("ok", tenant, "ok", "{}")
        line("refused", tenant, "refused", "{}")
        line("wait mean s", tenant, "wait_mean_s")
        line("wait p95 s", tenant, "wait_p95_s", "{:.1f}")
        line("queue mean s", tenant, "queue_mean_s")
        line("read mean s (admission to token)", tenant, "read_mean_s")
        line("delivered rate p50 (of tier)", tenant, "rate_p50")
        line("delivered rate p5", tenant, "rate_p5")
        line("bucket waits", tenant, "bucket_waits", "{}")
        line("bucket wait, mean over waiters s", tenant, "bucket_wait_mean_s", "{:.1f}")
        line("output tokens", tenant, "output_tokens", "{}")


if __name__ == "__main__":
    main()
