"""Stage P table (RESULTS Q20): per run and tenant, what the host spends and what the customer gets.

Usage: python scripts/stage_p_table.py runs/o-*-2026112[1-9]* runs/p-* --out docs/reports/stage-p/stage_p_table.csv

GPU side. Seat time is admission to engine done. Slot busy share is the sum of seat time over slots x the
load's duration. Slots needed is the p95, sampled every second, of a tenant's requests between admission and
engine done. Engine decode rate is output tokens over first token to engine done, as a share of the tier rate.
Customer side. Wait is arrival to first token; end-to-end is arrival to last token; delivered rate is output
tokens over first to last token, as a share of the tier rate; a conversation runs from its first arrival to
the last token of its last turn.
Bills. Slot-hour: slots x hours x the shape's price per second x 3600 (flat). Per token and delivery-second:
the gateway's own figures per request (client.csv). Engine-second: seat time x the shape's price per second.
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import sqlite3
from collections import defaultdict

import numpy as np

TPS = re.compile(r"\.(\d+)tps")
SLOTS = {"slots": {"chat-std": 2, "chat-pro": 3, "docs-std": 1, "docs-pro": 2},
         "over": {"chat-std": 5, "chat-pro": 6, "docs-std": 2, "docs-pro": 3}}
DURATION_S = 480.0
ARMS = ["n-fifo", "p-plain", "o-slots", "p-slots", "p-fair", "o-slots-open", "p-slots-open", "p-prio", "o-over", "p-over"]


def load(run_dir: str) -> tuple[list[dict], dict[str, dict]]:
    c = sqlite3.connect(f"file:{run_dir}/meter.sqlite?mode=ro", uri=True)
    c.row_factory = sqlite3.Row
    rows = [dict(r) for r in c.execute("select * from requests")]
    client = {}
    for path in glob.glob(os.path.join(run_dir, "loads", "*", "client.csv")):
        for r in csv.DictReader(open(path, encoding="utf-8")):
            if r.get("request_id"):
                client[r["request_id"]] = r
    return rows, client


def num(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def pct(xs, q) -> float:
    return float(np.percentile(xs, q)) if len(xs) else float("nan")


def stats(rows: list[dict], client: dict[str, dict], slots: int | None) -> dict:
    ok = [r for r in rows if r["status"] == "ok" and r["t_first_token"] is not None and r["t_engine_done"] is not None]
    refused = [r for r in rows if r["status"] == "refused"]
    out = {"requests": len(rows), "ok": len(ok), "refused": len(refused), "other": len(rows) - len(ok) - len(refused), "slots": slots or 0}
    if not ok:
        return out
    seat = np.array([r["t_engine_done"] - r["t_admitted"] for r in ok])
    wait = np.array([r["t_first_token"] - r["t_arrival"] for r in ok])
    ttft = np.array([r["t_first_token"] - r["t_admitted"] for r in ok])
    e2e = np.array([r["t_last_token"] - r["t_arrival"] for r in ok if r["t_last_token"]])
    decode, deliver = [], []
    for r in ok:
        m = TPS.search(r["shape"])
        n = r["output_tokens"] or 0
        if not m or n <= 20:
            continue
        tps = float(m.group(1))
        d_engine = r["t_engine_done"] - r["t_first_token"]
        d_deliver = (r["t_last_token"] or 0) - r["t_first_token"]
        if d_engine > 0.2:
            decode.append((n - 1) / d_engine / tps)
        if d_deliver > 0.2:
            deliver.append((n - 1) / d_deliver / tps)
    t0 = min(r["t_arrival"] for r in rows)
    grid = np.arange(t0, t0 + DURATION_S, 1.0)
    starts = np.sort([r["t_admitted"] for r in ok])
    ends = np.sort([r["t_engine_done"] for r in ok])
    occupancy = np.searchsorted(starts, grid, side="right") - np.searchsorted(ends, grid, side="right")
    arrivals = np.sort([r["t_arrival"] for r in ok])
    wanted = np.searchsorted(arrivals, grid, side="right") - np.searchsorted(ends, grid, side="right")
    conv = defaultdict(list)
    for r in ok:
        if r["conversation_id"] and r["t_last_token"]:
            conv[r["conversation_id"]].append(r)
    conv_s = [max(x["t_last_token"] for x in rs) - min(x["t_arrival"] for x in rs) for rs in conv.values() if len(rs) > 1]
    cl = [client[r["request_id"]] for r in ok if r["request_id"] in client]
    price = [num(c["shape_usd"]) / num(c["billed_s"]) for c in cl if num(c["billed_s"]) > 0 and num(c["shape_usd"]) > 0]
    price_s = float(np.median(price)) if price else float("nan")
    late = [num(c.get("turn_late_s")) for c in cl if c.get("turn_late_s") not in (None, "")]
    out.update({
        "seat_mean_s": seat.mean(), "seat_sum_s": seat.sum(),
        "busy_share": seat.sum() / (slots * DURATION_S) if slots else float("nan"),
        "slots_needed_p95": pct(occupancy, 95), "slots_needed_max": float(occupancy.max()),
        "slots_wanted_p95": pct(wanted, 95),  # arrival to engine done: includes requests waiting for a slot
        "decode_rate_p50": pct(decode, 50), "decode_rate_p5": pct(decode, 5),
        "wait_mean_s": wait.mean(), "wait_p50_s": pct(wait, 50), "wait_p95_s": pct(wait, 95),
        "ttft_mean_s": ttft.mean(), "ttft_p50_s": pct(ttft, 50),
        "e2e_p50_s": pct(e2e, 50), "e2e_mean_s": float(e2e.mean()) if len(e2e) else float("nan"),
        "rate_p50": pct(deliver, 50), "rate_p5": pct(deliver, 5),
        "slow_stream_share": float(np.mean([x < 0.95 for x in deliver])) if deliver else float("nan"),
        "conversations": len(conv_s), "conversation_p50_s": pct(conv_s, 50),
        "output_tokens": sum(r["output_tokens"] or 0 for r in ok), "prompt_tokens": sum(r["prompt_tokens"] or 0 for r in ok),
        "per_token_usd": float(np.nansum([num(c["per_token_usd"]) for c in cl])),
        "delivery_second_usd": float(np.nansum([num(c["shape_usd"]) for c in cl])),
        "engine_second_usd": float(seat.sum() * price_s),
        "slot_hour_usd": float(slots * DURATION_S * price_s) if slots else float("nan"),
        "turns_scheduled": len(late), "turns_late_share": float(np.mean([x > 0.25 for x in late])) if late else float("nan"),
        "turn_late_p95_s": pct(late, 95),
    })
    if slots:
        out["output_per_slot_hour"] = out["output_tokens"] / (slots * DURATION_S / 3600)
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
        slot_map = None if arm in ("n-fifo", "p-plain") else SLOTS["over" if arm.endswith("-over") else "slots"]
        rows, client = load(d)
        groups = defaultdict(list)
        for r in rows:
            for g in (r["tenant"], "all", r["tenant"].split("-")[0] + "-*"):
                groups[g].append(r)
        for tenant in sorted(groups):
            if slot_map is None:
                slots = None
            else:
                slots = sum(v for k, v in slot_map.items() if tenant == "all" or k == tenant or (tenant.endswith("*") and k.startswith(tenant[:-1])))
            s = stats(groups[tenant], client, slots)
            cell[(arm, seed, tenant)] = s
            table.append({"arm": arm, "seed": seed, "tenant": tenant, **s})
        # the sum of the tenants' own p95 slot needs (the figure P8 and the paid-run rule use)
        cell[(arm, seed, "all")]["slots_needed_sum_p95"] = sum(cell[(arm, seed, t)].get("slots_needed_p95", 0) for t in SLOTS["slots"] if (arm, seed, t) in cell)
        next(r for r in table if r["arm"] == arm and r["seed"] == seed and r["tenant"] == "all")["slots_needed_sum_p95"] = cell[(arm, seed, "all")]["slots_needed_sum_p95"]
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
        cells = []
        for arm in arms:
            vals = [cell.get((arm, s, tenant), {}).get(key) for s in seeds]
            cells.append(f"{arm}: " + " / ".join("-" if v is None else fmt.format(v) for v in vals))
        print(f"  {title:<30}" + "   ".join(cells))

    print(f"per seed ({' / '.join(map(str, seeds))})")
    for tenant in ("all", "chat-*", "docs-*", "chat-std", "chat-pro", "docs-std", "docs-pro"):
        print(tenant)
        for title, key, fmt in (
            ("ok", "ok", "{}"), ("refused", "refused", "{}"), ("other", "other", "{}"),
            ("seat time mean s", "seat_mean_s", "{:.2f}"), ("slot busy share", "busy_share", "{:.2f}"),
            ("slots needed p95", "slots_needed_p95", "{:.0f}"), ("sum of tenants' p95 slots", "slots_needed_sum_p95", "{:.0f}"),
            ("slots wanted p95 (with queue)", "slots_wanted_p95", "{:.0f}"),
            ("engine decode rate p50", "decode_rate_p50", "{:.2f}"), ("engine decode rate p5", "decode_rate_p5", "{:.2f}"),
            ("wait mean s", "wait_mean_s", "{:.2f}"), ("wait p95 s", "wait_p95_s", "{:.1f}"),
            ("admission to first token p50 s", "ttft_p50_s", "{:.2f}"), ("admission to first token mean", "ttft_mean_s", "{:.2f}"),
            ("end-to-end p50 s", "e2e_p50_s", "{:.2f}"),
            ("delivered rate p50", "rate_p50", "{:.2f}"), ("delivered rate p5", "rate_p5", "{:.2f}"),
            ("streams under 95% of tier", "slow_stream_share", "{:.2f}"),
            ("conversation p50 s", "conversation_p50_s", "{:.1f}"),
            ("output tokens", "output_tokens", "{}"), ("output per slot-hour", "output_per_slot_hour", "{:.0f}"),
            ("slot-hour bill $", "slot_hour_usd", "{:.4f}"), ("per-token bill $", "per_token_usd", "{:.4f}"),
            ("delivery-second bill $", "delivery_second_usd", "{:.4f}"), ("engine-second bill $", "engine_second_usd", "{:.4f}"),
            ("late turns share", "turns_late_share", "{:.2f}"), ("turn lateness p95 s", "turn_late_p95_s", "{:.1f}"),
        ):
            if any(key in cell.get((arm, s, tenant), {}) for arm in arms for s in seeds):
                line(title, tenant, key, fmt)


if __name__ == "__main__":
    main()
