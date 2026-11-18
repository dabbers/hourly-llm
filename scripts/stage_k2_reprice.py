"""Stage K2: re-price the stage K runs offline (no new runs). For every seed and arm it splits the
per-token bill (basket-7b: fresh input, cache read, output) and the per-second shape bill (seconds,
prompt charge) from each run's report/per_request.csv, and prints margins, revenue per million output
tokens and the paired differences. A third column prices cache hits as fresh input (no cache discount).

usage: python scripts/stage_k2_reprice.py runs/k-base-s* runs/k-affinity-s* [--out docs/reports/stage-k/k2_reprice.csv]
"""
import argparse
import csv
import sys

import pandas as pd

IN, OUT, CACHE, BATCH = 0.08, 0.175, 0.015, 0.5  # basket-7b, config/phase2/price_references.yaml and pricing.yaml


def one(run_dir: str, gpu_cost: dict) -> dict:
    df = pd.read_csv(f"{run_dir}/report/per_request.csv", low_memory=False)
    ok = df[df["ok"].astype(str).str.lower() == "true"].copy()
    disc = ok["lazy"].astype(str).str.lower().eq("true").map({True: BATCH, False: 1.0})
    cached = ok[["cached_tokens", "prompt_tokens"]].min(axis=1).clip(lower=0)
    fresh = ok["prompt_tokens"] - cached
    t_in, t_cache, t_out = (fresh * IN * disc).sum() / 1e6, (cached * CACHE * disc).sum() / 1e6, (ok["output_tokens"] * OUT * disc).sum() / 1e6
    token = t_in + t_cache + t_out
    assert abs(token - ok["per_token_usd"].sum()) < 1e-6 * max(1.0, token), (run_dir, token, ok["per_token_usd"].sum())
    token_nodisc = ((fresh + cached) * IN * disc).sum() / 1e6 + t_out
    shape, s_prompt = ok["shape_usd"].sum(), ok["prompt_usd"].sum()
    label, seed = str(df["scenario"].iloc[0]).split("+")[-1], int(df["seed"].iloc[0])
    cost, out_m = gpu_cost[(label, seed)], ok["output_tokens"].sum() / 1e6
    return {
        "arm": label, "seed": seed, "gpu_cost_usd": cost, "output_m": out_m,
        "prompt_m": ok["prompt_tokens"].sum() / 1e6, "cached_share": cached.sum() / ok["prompt_tokens"].sum(),
        "billed_s": ok["billed_s"].sum(),
        "token_usd": token, "token_input_usd": t_in, "token_cache_usd": t_cache, "token_output_usd": t_out,
        "shape_usd": shape, "shape_seconds_usd": shape - s_prompt, "shape_prompt_usd": s_prompt,
        "token_nodiscount_usd": token_nodisc,
        "margin_token": 1 - cost / token, "margin_shape": 1 - cost / shape, "margin_token_nodiscount": 1 - cost / token_nodisc,
        "token_usd_per_m_output": token / out_m, "shape_usd_per_m_output": shape / out_m,
        "token_nodiscount_usd_per_m_output": token_nodisc / out_m,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("run_dirs", nargs="+")
    p.add_argument("--runs-csv", default="docs/reports/stage-k/runs.csv", help="campaign runs.csv (GPU cost per run)")
    p.add_argument("--out", default="docs/reports/stage-k/k2_reprice.csv")
    p.add_argument("--base", default="k-base")
    p.add_argument("--arm", default="k-affinity")
    a = p.parse_args()
    gpu = {(r["label"], int(r["seed"])): float(r["gpu_cost_usd"]) for r in csv.DictReader(open(a.runs_csv, encoding="utf-8"))}
    t = pd.DataFrame([one(d.rstrip("/\\"), gpu) for d in a.run_dirs]).sort_values(["seed", "arm"])
    t.to_csv(a.out, index=False)
    pd.set_option("display.width", 250, "display.max_columns", 40, "display.float_format", "{:.4f}".format)
    print(t.to_string(index=False))
    b, k = (t[t.arm == x].set_index("seed") for x in (a.base, a.arm))
    d = (k.drop(columns="arm") - b.drop(columns="arm"))
    print(f"\npaired difference {a.arm} - {a.base}")
    print(d[["margin_token", "margin_shape", "margin_token_nodiscount", "token_usd", "shape_usd", "token_nodiscount_usd",
             "token_usd_per_m_output", "shape_usd_per_m_output", "billed_s", "output_m"]].to_string())
    print("\nmeans"); print(t.groupby("arm").mean(numeric_only=True).T.to_string())
    d.to_csv(a.out.replace(".csv", "_paired.csv"))


if __name__ == "__main__":
    sys.exit(main())
