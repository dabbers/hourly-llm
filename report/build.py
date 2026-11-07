"""Entry point: `slotgw-report runs/<run_id> [runs/<other_run_id> ...] [--out DIR]`.

Reads each run's meter and load-generator side files, writes per_request.csv, per_tenant.csv,
summary.csv, across_seeds.csv, interference.csv and a self-contained report.html
(docs/DESIGN.md section 10; milestone 2 scope: bills, variance across seeds, latency, throttling,
timeline, interference, basic provider economics)."""

from __future__ import annotations

import argparse
import os
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd

from . import charts, html
from .metrics import (
    LONG_PREFILL_TOKENS,
    LoadRun,
    across_seeds,
    enrich,
    interference,
    load_run_dir,
    per_run,
    per_tenant,
    timeline,
)


def _bills_section(pt: pd.DataFrame, xs: pd.DataFrame) -> str:
    parts = []
    for scenario, sub in xs.groupby("scenario", sort=False):
        sub = sub.sort_values("tenant")
        tenants = sub["tenant"].tolist()
        shape_of = pt[pt["scenario"] == scenario].groupby("tenant")["shape"].first()
        sub = sub.assign(shape=sub["tenant"].map(shape_of))
        # effective price per million output tokens under each scheme (means over seeds)
        tok = pt[pt["scenario"] == scenario].groupby(["seed", "tenant"]).agg(
            per_token_usd=("per_token_usd", "sum"), shape_usd=("shape_usd", "sum"),
            output_tokens=("output_tokens", "sum"), delivered_tokens=("delivered_tokens", "sum"),
        ).reset_index()
        with np.errstate(divide="ignore", invalid="ignore"):
            tok["eff_token"] = tok["per_token_usd"] / tok["output_tokens"] * 1e6
            tok["eff_shape"] = tok["shape_usd"] / tok["delivered_tokens"] * 1e6
        eff = tok.groupby("tenant")[["eff_token", "eff_shape"]].mean()
        sub = sub.assign(eff_token=sub["tenant"].map(eff["eff_token"]), eff_shape=sub["tenant"].map(eff["eff_shape"]))
        sub = sub.assign(burst_discount=sub["bound_usd_mean"] - sub["shape_usd_mean"])
        chart = charts.bar_chart(
            f"Bill per tenant, mean over {int(sub['seeds'].max())} seed(s) – {scenario}",
            tenants,
            {
                "per-token": sub["per_token_usd_mean"].tolist(),
                "shape, billed": sub["shape_usd_mean"].tolist(),
                "shape, bound": sub["bound_usd_mean"].tolist(),
            },
            subtitle="billed = wall-clock capped at the shape's promise; bound = the cap itself",
            y_label="USD per run",
            fmt=charts.fmt_usd,
        )
        tbl = html.table(
            sub,
            [
                ("tenant", "tenant", html.fmt_text),
                ("shape", "shape", html.fmt_text),
                ("seeds", "seeds", html.fmt_int),
                ("requests_mean", "requests", html.fmt_f(1)),
                ("output_tokens_mean", "output tokens", html.fmt_int),
                ("per_token_usd_mean", "per-token USD", html.fmt_usd),
                ("shape_usd_mean", "shape billed USD", html.fmt_usd),
                ("bound_usd_mean", "shape bound USD", html.fmt_usd),
                ("burst_discount", "burst discount", html.fmt_usd),
                ("shape_vs_token_ratio", "shape ÷ per-token", html.fmt_f(2)),
                ("eff_token", "$/1M out (per-token)", html.fmt_f(3)),
                ("eff_shape", "$/1M out (shape)", html.fmt_f(3)),
            ],
            left=2,
        )
        parts.append(f'<div class="card">{chart}{tbl}</div>')
    return "".join(parts)


def _variance_section(xs: pd.DataFrame) -> str:
    parts = []
    for scenario, sub in xs.groupby("scenario", sort=False):
        sub = sub.sort_values("tenant")
        if int(sub["seeds"].max()) < 2:
            parts.append(f'<p class="muted">{escape(str(scenario))}: one seed only; run more seeds to measure variance.</p>')
            continue
        chart = charts.bar_chart(
            f"Coefficient of variation of the bill across seeds – {scenario}",
            sub["tenant"].tolist(),
            {"per-token": (sub["per_token_usd_cv"] * 100).tolist(), "shape, billed": (sub["shape_usd_cv"] * 100).tolist()},
            subtitle="std ÷ mean of each tenant's bill over the seeds (\"days\"); lower = more predictable",
            y_label="CV, %",
            fmt=lambda v: f"{v:.1f}%",
        )
        tbl = html.table(
            sub,
            [
                ("tenant", "tenant", html.fmt_text),
                ("seeds", "seeds", html.fmt_int),
                ("per_token_usd_mean", "per-token mean", html.fmt_usd),
                ("per_token_usd_cv", "per-token CV", html.fmt_pct),
                ("shape_usd_mean", "shape mean", html.fmt_usd),
                ("shape_usd_cv", "shape CV", html.fmt_pct),
                ("output_tokens_mean", "output tokens mean", html.fmt_int),
                ("output_tokens_cv", "output tokens CV", html.fmt_pct),
                ("refused_mean", "refused / run", html.fmt_f(1)),
            ],
        )
        parts.append(f'<div class="card">{chart}{tbl}</div>')
    return "".join(parts)


def _experience_section(pt: pd.DataFrame) -> str:
    parts = []
    for scenario, sub in pt.groupby("scenario", sort=False):
        agg = sub.groupby(["tenant", "shape", "delivery"], dropna=False).agg(
            requests=("requests", "sum"), ok=("ok", "sum"), refused=("refused", "sum"), errors=("errors", "sum"),
            tps=("tps", "first"), ttft_p50_ms=("ttft_p50_ms", "mean"), ttft_p95_ms=("ttft_p95_ms", "mean"),
            e2e_p50_ms=("e2e_p50_ms", "mean"), e2e_p95_ms=("e2e_p95_ms", "mean"),
            stream_s=("stream_s", "sum"), throttled_s=("throttled_s", "sum"),
            delivered_tps_mean=("delivered_tps_mean", "mean"), engine_tps_mean=("engine_tps_mean", "mean"),
            floor_judged=("floor_judged", "sum"), floor_violations=("floor_violations", "sum"),
            violation_stream_s=("violation_stream_s", "sum"), tokens_per_billed_s=("tokens_per_billed_s", "mean"),
        ).reset_index().sort_values(["tenant", "delivery"])
        with np.errstate(divide="ignore", invalid="ignore"):
            agg["throttled_share"] = np.where(agg["stream_s"] > 0, agg["throttled_s"] / agg["stream_s"], np.nan)
            agg["floor_violation_share"] = np.where(agg["stream_s"] > 0, agg["violation_stream_s"] / agg["stream_s"], np.nan)
        by_tenant = agg.groupby("tenant").agg(ttft_p50_ms=("ttft_p50_ms", "mean"), ttft_p95_ms=("ttft_p95_ms", "mean"),
                                              delivered=("delivered_tps_mean", "mean"), tps=("tps", "first")).reset_index()
        c1 = charts.bar_chart(
            f"Time to first token – {scenario}", by_tenant["tenant"].tolist(),
            {"p50": by_tenant["ttft_p50_ms"].tolist(), "p95": by_tenant["ttft_p95_ms"].tolist()},
            subtitle="from admission to the first delivered token (queue time excluded); mean of per-run percentiles",
            y_label="ms",
        )
        c2 = charts.bar_chart(
            f"Delivered rate vs the shape's promise – {scenario}", by_tenant["tenant"].tolist(),
            {"delivered": by_tenant["delivered"].tolist(), "shape tps": by_tenant["tps"].tolist()},
            subtitle="delivered tokens ÷ streaming time, head start included; below 95% of the promise = floor violation",
            y_label="tokens / s",
        )
        tbl = html.table(
            agg,
            [
                ("tenant", "tenant", html.fmt_text), ("shape", "shape", html.fmt_text), ("delivery", "delivery", html.fmt_text),
                ("requests", "requests", html.fmt_int), ("ok", "ok", html.fmt_int), ("refused", "refused", html.fmt_int),
                ("errors", "errors", html.fmt_int),
                ("ttft_p50_ms", "TTFT p50 ms", html.fmt_int), ("ttft_p95_ms", "TTFT p95 ms", html.fmt_int),
                ("e2e_p50_ms", "e2e p50 ms", html.fmt_int), ("e2e_p95_ms", "e2e p95 ms", html.fmt_int),
                ("throttled_s", "throttled s", html.fmt_f(1)), ("throttled_share", "throttled %", html.fmt_pct),
                ("tps", "shape tps", html.fmt_int), ("delivered_tps_mean", "delivered tps", html.fmt_f(1)),
                ("engine_tps_mean", "engine tps", html.fmt_f(1)), ("tokens_per_billed_s", "tokens / billed s", html.fmt_f(1)),
                ("floor_violations", "floor viol.", html.fmt_int), ("floor_judged", "judged", html.fmt_int),
                ("floor_violation_share", "viol. % of stream time", html.fmt_pct),
            ],
            left=3,
        )
        parts.append(f'<div class="card"><div class="grid">{c1}{c2}</div>{tbl}</div>')
    return "".join(parts)


def _timeline_section(tl: dict[str, dict]) -> str:
    parts = []
    for load_run, e in tl.items():
        series = {"in flight (meter)": (e["t"], e["inflight"])}
        if e.get("running"):
            series["running (engine, sampled)"] = (e["sample_t"], e["running"])
        c1 = charts.line_chart(f"Requests in flight – {load_run}", series, x_label="seconds since first arrival", y_label="requests")
        c2 = ""
        if e.get("kv_usage"):
            c2 = charts.line_chart(
                f"KV cache usage (engine, sampled) – {load_run}", {"KV used": (e["sample_t"], e["kv_usage"])},
                x_label="seconds since first arrival", y_label="% of KV cache", y_max=100.0,
            )
        parts.append(f'<div class="card"><div class="grid">{c1}{c2}</div></div>')
    return "".join(parts) or '<p class="muted">no completed requests</p>'


def _interference_section(inter: pd.DataFrame) -> str:
    if inter.empty:
        return '<p class="muted">no streamed requests to judge</p>'
    tbl = html.table(
        inter.assign(during=inter["during_long_prefill"].map({True: "yes", False: "no"})),
        [
            ("scenario", "scenario", html.fmt_text), ("load_run", "load run", html.fmt_text), ("tenant", "tenant", html.fmt_text),
            ("during", f"long prefill (≥{LONG_PREFILL_TOKENS}) in flight", html.fmt_text), ("n", "streams", html.fmt_int),
            ("engine_tps_mean", "engine tps mean", html.fmt_f(1)), ("engine_tps_p10", "engine tps p10", html.fmt_f(1)),
            ("delivered_tps_mean", "delivered tps", html.fmt_f(1)), ("floor_violation_share", "floor violation share", html.fmt_pct),
        ],
        left=4,
    )
    return f'<div class="card">{tbl}</div>'


def _economics_section(pr: pd.DataFrame) -> str:
    chart = charts.bar_chart(
        "Revenue per run vs GPU cost", pr["load_run"].tolist(),
        {"per-token": pr["per_token_usd"].tolist(), "shape, billed": pr["shape_usd"].tolist(), "GPU cost": pr["gpu_cost_usd"].tolist()},
        subtitle="GPU cost = gpu_usd_per_hour × run duration; break-even fill rate comes in milestone 3",
        y_label="USD", fmt=charts.fmt_usd,
    )
    tbl = html.table(
        pr,
        [
            ("load_run", "load run", html.fmt_text), ("duration_s", "duration s", html.fmt_int), ("output_tokens", "output tokens", html.fmt_int),
            ("per_token_usd", "per-token USD", html.fmt_usd), ("shape_usd", "shape billed USD", html.fmt_usd),
            ("bound_usd", "shape bound USD", html.fmt_usd), ("gpu_cost_usd", "GPU cost USD", html.fmt_usd),
            ("output_tokens_per_gpu_hour", "output tokens / GPU-hour", html.fmt_int),
        ],
    )
    return f'<div class="card">{chart}{tbl}</div>'


def _overview_section(pr: pd.DataFrame, run_ids: list[str]) -> str:
    tbl = html.table(
        pr,
        [
            ("load_run", "load run", html.fmt_text), ("scenario", "scenario", html.fmt_text), ("seed", "seed", html.fmt_int),
            ("pacing", "pacing", html.fmt_text), ("duration_s", "duration s", html.fmt_int),
            ("requests", "requests", html.fmt_int), ("ok", "ok", html.fmt_int), ("refused", "refused", html.fmt_int), ("errors", "errors", html.fmt_int),
            ("prompt_tokens", "prompt tokens", html.fmt_int), ("output_tokens", "output tokens", html.fmt_int),
            ("concurrency_mean", "in flight mean", html.fmt_f(1)), ("concurrency_p95", "in flight p95", html.fmt_f(1)),
            ("engine_running_mean", "engine running mean", html.fmt_f(1)),
            ("kv_usage_mean", "KV mean", html.fmt_pct), ("kv_usage_p95", "KV p95", html.fmt_pct), ("preemptions", "preemptions", html.fmt_int),
            ("throttled_s", "throttled s", html.fmt_f(1)), ("floor_violations", "floor viol.", html.fmt_int),
        ],
        left=2,
    )
    return f'<p>Gateway run(s): {", ".join(escape(r) for r in run_ids)}.</p><div class="card">{tbl}</div>'


DEFINITIONS = """
<dl>
<dt>per-token</dt><dd>prompt and output tokens at the reference list prices; lazy shapes get the batch discount (config/pricing.yaml).</dd>
<dt>shape, billed</dt><dd>seconds from admission to the last delivered token (queue excluded), capped at the shape's promise, × the shape's price per second.</dd>
<dt>shape, bound</dt><dd>the cap: time to first token + delivered tokens ÷ shape tps. The customer never pays more than this.</dd>
<dt>burst discount</dt><dd>bound − billed: what faster-than-promised delivery (bursts) took off the bill.</dd>
<dt>throttled</dt><dd>wall-clock time delivery was held back by the per-request token bucket, not by the engine; for complete responses, the time the finished response was held.</dd>
<dt>delivered tps</dt><dd>delivered tokens ÷ time from first to last delivered token, head start included (so short answers read above the shape's rate); judged on requests with at least 30 tokens over 0.5 s.</dd>
<dt>floor violation</dt><dd>delivered tps below 95% of the shape's tps: the engine fell behind the promise by more than the head start could cover (admission control and prefill interference, milestone 3).</dd>
<dt>engine tps</dt><dd>output tokens ÷ (engine done − first token): the engine's own decode rate for that request.</dd>
<dt>in flight</dt><dd>requests between admission and last delivered token, from the meter; "running" is vLLM's own count sampled through the gateway.</dd>
</dl>
"""


def build_report(run_dirs: list[str | Path], out_dir: str | Path | None = None, config_dir: str | Path | None = None) -> Path:
    datas = [load_run_dir(d, config_dir) for d in run_dirs]
    config = datas[0].config
    frames = [d.requests.assign(gateway_run=d.run_id) for d in datas if not d.requests.empty]
    if not frames:
        raise SystemExit("no requests in the given runs")
    df = enrich(pd.concat(frames, ignore_index=True), config)
    loads: list[LoadRun] = [lr for d in datas for lr in d.loads]
    pt = per_tenant(df)
    pr = per_run(df, loads, config)
    xs = across_seeds(pt)
    inter = interference(df)
    tl = timeline(df, loads)

    out = Path(out_dir) if out_dir else datas[0].run_dir / "report"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "per_request.csv", index=False)
    pt.to_csv(out / "per_tenant.csv", index=False)
    pr.to_csv(out / "summary.csv", index=False)
    xs.to_csv(out / "across_seeds.csv", index=False)
    inter.to_csv(out / "interference.csv", index=False)

    n_seeds = int(xs["seeds"].max()) if not xs.empty else 0
    intro = (
        f'<p class="lead">{len(pr)} load run(s), {int(pr["requests"].sum()):,} requests, '
        f'{int(pr["output_tokens"].sum()):,} output tokens, {n_seeds} seed(s). '
        f'Shapes and prices from the run\'s config snapshot; definitions at the end.</p>'
    )
    sections = [
        ("overview", "Runs", _overview_section(pr, [d.run_id for d in datas])),
        ("bills", "Bills: per-token vs shape", _bills_section(pt, xs)),
        ("variance", "Predictability across seeds", _variance_section(xs)),
        ("experience", "Latency, throttling and floor violations", _experience_section(pt)),
        ("timeline", "Timeline", _timeline_section(tl)),
        ("interference", "Prefill interference", _interference_section(inter)),
        ("economics", "Provider economics", _economics_section(pr)),
        ("definitions", "Definitions", DEFINITIONS),
    ]
    (out / "report.html").write_text(html.page("slotgw report", intro, sections), encoding="utf-8")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="build the slotgw report from one or more run directories")
    p.add_argument("run_dirs", nargs="+", help="runs/<run_id> directories (each with meter.sqlite)")
    p.add_argument("--out", default=None, help="output directory (default: <first run>/report)")
    p.add_argument("--config-dir", default=os.environ.get("SLOTGW_CONFIG_DIR", "config"),
                   help="fallback config when the run has no config snapshot")
    args = p.parse_args()
    out = build_report(args.run_dirs, args.out, args.config_dir)
    print(f"report written to {out / 'report.html'}")


if __name__ == "__main__":
    main()
