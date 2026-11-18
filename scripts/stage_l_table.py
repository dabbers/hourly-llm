"""Stage L and stage M table: one row per arm and seed from the campaign's runs.csv plus each run's report/per_request.csv
(TTFT p95 and floor misses by tenant, affinity waits) and its meter (stage M: queue jumps and passes), and the paired differences by seed for chosen pairs.

usage: python scripts/stage_l_table.py <campaign dir> [--pairs l-fp8:l-fp16 ...] [--out <csv>]
"""
import argparse
import sqlite3
import sys

import numpy as np
import pandas as pd

TENANTS = ("chat", "agent", "notetaker", "batch")


def per_run(run_dir: str) -> dict:
    df = pd.read_csv(f"{run_dir}/report/per_request.csv", low_memory=False)
    ok = df[df["ok"].astype(str).str.lower() == "true"]
    out: dict = {}
    for t in TENANTS:
        g = ok[ok["tenant"] == t]
        out[f"ttft_p95_ms_{t}"] = g["ttft_ms"].quantile(0.95) if len(g) else np.nan
        judged = g["delivered_tps"].notna()
        fv = g["floor_violation"].astype(str).str.lower() == "true"
        out[f"floor_miss_{t}"] = fv.sum() / judged.sum() if judged.sum() else np.nan
    out["ttft_p95_ms_all"] = ok["ttft_ms"].quantile(0.95)
    w = ok["affinity_wait_ms"].dropna() if "affinity_wait_ms" in ok else pd.Series(dtype=float)
    hit = ok.loc[w.index, "affinity"] == "hit" if len(w) else pd.Series(dtype=bool)
    out.update(waits=len(w), wait_hits=int(hit.sum()), wait_hit_rate=hit.mean() if len(w) else np.nan,
               wait_ms_median=w.median() if len(w) else np.nan, wait_ms_hit_median=w[hit].median() if len(w) and hit.any() else np.nan)
    aff = ok["affinity"].dropna()
    g = ok.loc[aff.index]
    out["cached_share_hits"] = g.loc[aff == "hit", "cached_tokens"].sum() / max(1, g.loc[aff == "hit", "prompt_tokens"].sum())
    out["cached_share_misses"] = g.loc[aff != "hit", "cached_tokens"].sum() / max(1, g.loc[aff != "hit", "prompt_tokens"].sum())
    out.update(queue_stats(run_dir))
    return out


def queue_stats(run_dir: str) -> dict:
    """Stage M, from the meter: jumps onto the remembered replica, and what being passed cost the others."""
    m = pd.read_sql("select * from requests where status = 'ok'", sqlite3.connect(f"{run_dir}/meter.sqlite"))
    free = m[m["reservation_id"].isna() & (m["lazy"] == 0)]
    out = {"queued_ms_p50": free["queued_ms"].median(), "queued_ms_p95": free["queued_ms"].quantile(0.95),
           "queued_ms_p99": free["queued_ms"].quantile(0.99), "queued_ms_max": free["queued_ms"].max()}
    if "queue_jumped" not in m:
        return out
    j, p = m[m["queue_jumped"].notna()], m[m["queue_passed"].notna()]
    out.update(queue_jumps=len(j), jump_prompt_tokens=int(j["prompt_tokens"].sum()),
               jump_cached_share=j["cached_tokens"].sum() / max(1, j["prompt_tokens"].sum()),
               jump_positions_mean=j["queue_jumped"].mean() if len(j) else np.nan,
               passed_requests=len(p), passed_times_max=p["queue_passed"].max() if len(p) else np.nan,
               passed_ms_p50=p["queue_passed_ms"].median(), passed_ms_p95=p["queue_passed_ms"].quantile(0.95),
               passed_ms_max=p["queue_passed_ms"].max(),
               passed_within_1s=(p["queue_passed_ms"] <= 1000).mean() if len(p) else np.nan,
               passed_hit_rate=(p["affinity"] == "hit").sum() / max(1, p["affinity"].notna().sum()))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("campaign")
    ap.add_argument("--pairs", nargs="*", default=[])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    runs = pd.read_csv(f"{a.campaign}/runs.csv")
    extra = pd.DataFrame([per_run(str(d).replace("\\", "/")) for d in runs["run_dir"]])
    t = pd.concat([runs, extra], axis=1)
    keep = ["label", "seed", "kv_total_min", "affinity_hit_rate", "affinity_fallback_seat", "affinity_fallback_kv",
            "affinity_fallback_unhealthy", "affinity_fallback_reasons", "waits", "wait_hits", "wait_hit_rate", "wait_ms_median",
            "wait_ms_hit_median", "cached_share", "cached_share_hits", "ttft_p95_ms_chat", "ttft_p95_ms_agent", "ttft_p95_ms_all",
            "start_p95_s", "floor_violation_rate", *(f"floor_miss_{x}" for x in TENANTS), "gross_margin_token", "gross_margin_shape",
            "output_tokens_per_gpu_hour", "replica_prompt_share_max", "replica_prompt_shares", "utilisation", "requests", "refused",
            "tenant_usd", "per_token_usd", "cached_share_misses", "queued_ms_p50", "queued_ms_p95", "queued_ms_p99", "queued_ms_max",
            "queue_jumps", "jump_prompt_tokens", "jump_cached_share", "jump_positions_mean", "passed_requests", "passed_times_max",
            "passed_ms_p50", "passed_ms_p95", "passed_ms_max", "passed_within_1s", "passed_hit_rate"]
    t = t[[k for k in keep if k in t]].sort_values(["label", "seed"])
    out = a.out or f"{a.campaign}/stage_l_table.csv"
    t.to_csv(out, index=False)
    pd.set_option("display.width", 300, "display.max_columns", 60, "display.float_format", "{:.3f}".format)
    print(t.drop(columns=["replica_prompt_shares", "affinity_fallback_reasons"]).T.to_string())
    num = t.select_dtypes("number").columns.drop("seed")
    rows = []
    for pair in a.pairs:
        x, y = pair.split(":")
        d = t[t.label == x].set_index("seed")[num] - t[t.label == y].set_index("seed")[num]
        d.insert(0, "pair", f"{x} - {y}")
        rows.append(d.reset_index())
        print(f"\n{x} - {y}"); print(d.drop(columns="pair").T.to_string())
    if rows:
        pd.concat(rows).to_csv(out.replace(".csv", "_paired.csv"), index=False)


if __name__ == "__main__":
    sys.exit(main())
