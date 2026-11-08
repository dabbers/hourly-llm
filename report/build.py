"""Entry point: `slotgw-report runs/<run_id> [runs/<other_run_id> ...] [--out DIR]`.

Reads each run's meter and load-generator side files, writes per_request.csv, per_tenant.csv,
summary.csv, across_seeds.csv, interference.csv, per_replica.csv, refusals.csv, market.csv, list_prices.csv,
reservations.csv, reservation_bills.csv and a self-contained report.html
(docs/DESIGN.md section 10; milestone 2 scope: bills, variance across seeds, latency, throttling,
timeline, interference, basic provider economics)."""

from __future__ import annotations

import argparse
import json
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
    list_prices,
    load_run_dir,
    market,
    per_replica,
    per_run,
    per_tenant,
    refusals,
    reservation_bills,
    reservations,
    samples,
    timeline,
)

SAMPLE_PROMPT_CHARS = 6000  # a transcript-sized prompt is folded in the page; the full text is in samples.csv


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
                ("prompt_usd_mean", "of which prefill", html.fmt_usd),
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
            requests=("requests", "sum"), ok=("ok", "sum"), refused=("refused", "sum"),
            refused_context=("refused_context", "sum"), refused_capacity=("refused_capacity", "sum"),
            errors=("errors", "sum"), queued_p50_ms=("queued_p50_ms", "mean"), queued_p95_ms=("queued_p95_ms", "mean"),
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
                ("requests", "requests", html.fmt_int), ("ok", "ok", html.fmt_int),
                ("refused_context", "429 ctx", html.fmt_int), ("refused_capacity", "503 cap", html.fmt_int),
                ("errors", "errors", html.fmt_int),
                ("queued_p50_ms", "queue p50 ms", html.fmt_int), ("queued_p95_ms", "queue p95 ms", html.fmt_int),
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
        subtitle="GPU cost = gpu_usd_per_hour × replicas used × run duration; fill and break-even in the replicas section, market prices in the next",
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


def _replicas_section(prp: pd.DataFrame, ref: pd.DataFrame) -> str:
    if prp.empty:
        return '<p class="muted">no completed requests</p>'
    labels = [f"{r.load_run} · {r.replica}" for r in prp.itertuples()]
    c1 = charts.bar_chart(
        "Fill rate vs break-even", labels,
        {"fill rate": (prp["fill_rate"] * 100).tolist(), "break-even fill": (prp["breakeven_fill_rate"] * 100).tolist(),
         "utilisation": (prp["utilization"] * 100).tolist()},
        subtitle="fill = seat-seconds held ÷ (slot plan × run time); break-even = GPU $/h ÷ plan revenue at 100% fill",
        y_label="%", fmt=lambda v: f"{v:.0f}%",
    )
    c2 = charts.bar_chart(
        "Revenue per hour vs GPU cost", labels,
        {"shape, billed": prp["shape_usd_per_hour"].tolist(), "per-token": prp["per_token_usd_per_hour"].tolist(),
         "GPU cost": [prp["gpu_cost_usd"].iloc[i] / prp["duration_s"].iloc[i] * 3600 if prp["duration_s"].iloc[i] else 0 for i in range(len(prp))]},
        y_label="USD / hour", fmt=charts.fmt_usd,
    )
    tbl = html.table(
        prp,
        [
            ("load_run", "load run", html.fmt_text), ("replica", "replica", html.fmt_text), ("slot_plan", "slot plan", html.fmt_text),
            ("ok", "ok", html.fmt_int), ("seat_seconds", "seat-seconds", html.fmt_int), ("fill_rate", "fill", html.fmt_pct),
            ("utilization", "utilisation", html.fmt_pct), ("breakeven_fill_rate", "break-even fill", html.fmt_pct),
            ("shape_usd_per_hour", "shape $/h", html.fmt_usd), ("per_token_usd_per_hour", "per-token $/h", html.fmt_usd),
            ("plan_usd_per_hour_full", "plan $/h at 100%", html.fmt_usd),
            ("ledger_running_mean", "ledger running", html.fmt_f(1)), ("seats_now_mean", "seats", html.fmt_f(1)),
            ("ledger_kv_held_share_mean", "KV held", html.fmt_pct), ("engine_kv_usage_mean", "KV used", html.fmt_pct),
            ("engine_running_mean", "engine running", html.fmt_f(1)),
            ("queue_interactive_mean", "queue mean", html.fmt_f(2)), ("queue_interactive_max", "queue max", html.fmt_int),
            ("gw_timeouts", "503s", html.fmt_int), ("gw_preemptions", "preemptions", html.fmt_int),
        ],
        left=3,
    )
    rtbl = html.table(
        ref,
        [
            ("load_run", "load run", html.fmt_text), ("tenant", "tenant", html.fmt_text), ("reason", "reason", html.fmt_text),
            ("n", "n", html.fmt_int), ("queued_p50_ms", "queued p50 ms", html.fmt_int),
        ],
        left=3,
    )
    return f'<div class="card"><div class="grid">{c1}{c2}</div>{tbl}<h3>Refusals by reason</h3>{rtbl}</div>'


def _market_section(mk: pd.DataFrame, lp: pd.DataFrame, config, fallback) -> str:
    if mk.empty:
        return '<p class="muted">no completed requests</p>'
    pr = config.pricing
    refs = pr if pr.references else (fallback.pricing if fallback and fallback.pricing.references else pr)
    src = f"{refs.references_source}, fetched {refs.references_fetched}" if refs.references_source else "the run's token prices only"
    if refs is not pr:
        src += " (the run's snapshot predates price references; the current config's are used)"
    cached = mk.drop_duplicates("load_run")["cached_share"].mean() if "cached_share" in mk else np.nan
    cached_txt = f"{cached:.1%}" if not np.isnan(cached) else "unmetered"
    intro = (
        f"<p>Per-token price references: {escape(src)}. Active reference <b>{escape(pr.token.reference)}</b> "
        f"(${pr.token.input_usd_per_million:.4f} in / ${pr.token.output_usd_per_million:.4f} out per 1M tokens) sets the "
        f"per-token bills and the shape prices derived from them. Cost per token is GPU cost ÷ tokens served, at the "
        f"measured utilisation and as if the slot plan ran at {pr.provider.target_utilization:.0%}. Prompt tokens vLLM "
        f"served from its prefix cache ({cached_txt} of prompt tokens here) are priced at each reference's cache-read "
        f"price: its listed one, else {pr.token.cache_read_ratio:.0%} of its input price (pricing.yaml "
        f"<code>cache_read_ratio</code>); \"uncached\" is the revenue with cache hits charged in full.</p>"
    )
    runs = mk.drop_duplicates("load_run")
    agg = mk.groupby("reference", sort=False)["revenue_usd"].mean()
    chart = charts.bar_chart(
        "Revenue per run at each reference price",
        [*agg.index.tolist(), "shape (snapshot prices)", "GPU cost"],
        {"USD per run": [*agg.tolist(), float(runs["shape_usd"].mean()), float(runs["gpu_cost_usd"].mean())]},
        subtitle=f"mean over {len(runs)} load run(s); prompt and output tokens at the reference's prices, lazy requests at the batch discount",
        y_label="USD", fmt=charts.fmt_usd,
    )
    t1 = html.table(
        mk.assign(active=mk["active"].map({True: "yes", False: ""})),
        [
            ("load_run", "load run", html.fmt_text), ("reference", "reference", html.fmt_text), ("model", "model", html.fmt_text),
            ("active", "active", html.fmt_text),
            ("input_usd_per_million", "$/1M in", html.fmt_f(4)), ("output_usd_per_million", "$/1M out", html.fmt_f(4)),
            ("cache_read_usd_per_million", "$/1M cached", html.fmt_f(4)), ("cached_share", "cached", html.fmt_pct),
            ("revenue_usd_uncached", "revenue uncached", html.fmt_usd),
            ("revenue_usd", "revenue USD", html.fmt_usd), ("revenue_vs_gpu", "× GPU cost", html.fmt_f(2)),
            ("revenue_vs_shape", "× shape billed", html.fmt_f(2)), ("gross_margin", "gross margin", html.fmt_pct),
            ("utilization", "utilisation", html.fmt_pct), ("gross_margin_at_target", "margin at target util.", html.fmt_pct),
            ("breakeven_gpu_usd_per_hour", "break-even GPU $/h", html.fmt_usd),
            ("breakeven_gpu_usd_per_hour_at_target", "at target util.", html.fmt_usd),
        ],
        left=4,
    )
    t2 = html.table(
        lp,
        [
            ("load_run", "load run", html.fmt_text), ("margin_target", "gross margin", html.fmt_pct),
            ("utilization", "utilisation", html.fmt_pct),
            ("input_usd_per_million", "$/1M in", html.fmt_f(4)), ("output_usd_per_million", "$/1M out", html.fmt_f(4)),
            ("vs_reference", "× active reference", html.fmt_f(2)),
            ("input_usd_per_million_at_target", "$/1M in at target util.", html.fmt_f(4)),
            ("output_usd_per_million_at_target", "$/1M out at target util.", html.fmt_f(4)),
            ("vs_reference_at_target", "× reference at target", html.fmt_f(2)),
        ],
    )
    return (
        f'<div class="card">{intro}{chart}{t1}'
        f'<h3>Prices this provider would have to list</h3>'
        f'<p class="muted">GPU cost ÷ (1 − margin) ÷ output-equivalent tokens; prompt tokens weighted by the market '
        f'reference\'s input:output price ratio ({escape(str(lp["reference"].iloc[0])) if not lp.empty else "–"}), '
        f'lazy tokens by the batch discount.</p>{t2}</div>'
    )


def _reservations_section(rs: pd.DataFrame) -> str:
    if rs.empty:
        return '<p class="muted">no completed requests</p>'
    intro = (
        "<p>What if each tenant reserved N slots of its shape for the whole run and paid slot-seconds whether used or "
        "not? N is sized from the tenant's own queue-free occupancy (each request holds a slot from arrival for the seat "
        "time it measured; capacity refusals at the median seat time). The reservation bill is slot time at the shape's "
        "price per second with no prefill charge, since the slot covers prefill; the cost share is the reserved slice of "
        "the card, with the fastest shapes weighted by how many plan slots a seat of theirs displaces "
        f"({escape(str(rs['weight_basis'].iloc[0]))}).</p>"
    )
    p95 = rs[rs["sizing"] == "p95"].groupby("tenant", sort=False)
    mx = rs[rs["sizing"] == "max"].groupby("tenant", sort=False)
    tenants = list(p95.groups)
    chart = charts.bar_chart(
        "Bill per tenant: per-token, per-request shape, reservation at p95 and at peak occupancy",
        tenants,
        {
            "per-token": p95["per_token_usd"].mean().tolist(),
            "shape, per request": p95["shape_usd"].mean().tolist(),
            "reservation p95": p95["reservation_usd"].mean().tolist(),
            "reservation peak": mx["reservation_usd"].mean().reindex(tenants).tolist(),
        },
        subtitle="mean over load runs; the reservation lasts the whole run",
        y_label="USD per run", fmt=charts.fmt_usd,
    )
    tbl = html.table(
        rs,
        [
            ("load_run", "load run", html.fmt_text), ("tenant", "tenant", html.fmt_text), ("shape", "shape", html.fmt_text),
            ("sizing", "sizing", html.fmt_text), ("slots", "slots", html.fmt_int),
            ("occupancy_p50", "occ. p50", html.fmt_f(1)), ("occupancy_p95", "occ. p95", html.fmt_f(1)), ("occupancy_max", "occ. max", html.fmt_f(0)),
            ("would_queue_share", "would queue", html.fmt_pct), ("used_fill", "used fill", html.fmt_pct),
            ("idle_slot_hours", "idle slot-h", html.fmt_f(2)),
            ("reservation_usd", "reservation USD", html.fmt_usd), ("shape_usd", "shape per-request USD", html.fmt_usd),
            ("per_token_usd", "per-token USD", html.fmt_usd),
            ("reservation_vs_per_token", "res. ÷ per-token", html.fmt_f(2)),
            ("capacity_weight", "cap. weight", html.fmt_f(2)), ("cost_share_usd", "GPU cost share", html.fmt_usd),
            ("reservation_vs_cost", "res. ÷ cost", html.fmt_f(2)),
            ("slot_usd_per_hour", "slot $/h", html.fmt_usd), ("breakeven_slot_usd_per_hour", "break-even slot $/h", html.fmt_usd),
            ("revenue_parity_slot_usd_per_hour", "per-token-parity slot $/h", html.fmt_usd),
        ],
        left=4,
    )
    return f'<div class="card">{intro}{chart}{tbl}</div>'




def _bursting_section(pr: pd.DataFrame, pt: pd.DataFrame) -> str:
    chart = charts.bar_chart(
        "Shape revenue, the burst discount and per-token revenue per run", pr["load_run"].tolist(),
        {"shape, billed": pr["shape_usd"].tolist(), "burst discount": pr["burst_discount_usd"].tolist(),
         "per-token": pr["per_token_usd"].tolist()},
        subtitle="burst discount = bound - billed: runtime the provider gave away by delivering faster than the shape (tps burst)",
        y_label="USD", fmt=charts.fmt_usd,
    )
    runs = html.table(
        pr,
        [
            ("load_run", "load run", html.fmt_text), ("ok", "ok", html.fmt_int), ("context_bursts", "context bursts", html.fmt_int),
            ("revoked", "revoked", html.fmt_int), ("output_tokens_burst", "burst tokens", html.fmt_int),
            ("burst_share", "burst share", html.fmt_pct), ("burst_rate_lent_mean", "lent tok/s (sampled mean)", html.fmt_f(1)),
            ("burst_discount_usd", "burst discount USD", html.fmt_usd), ("shape_usd", "shape billed USD", html.fmt_usd),
            ("floor_violations", "floor viol.", html.fmt_int), ("output_tokens_per_gpu_hour", "output tokens / GPU-hour", html.fmt_int),
        ],
    )
    cols = [
        ("load_run", "load run", html.fmt_text), ("ok", "ok", html.fmt_int), ("context_bursts", "ctx bursts", html.fmt_int),
        ("revoked", "revoked", html.fmt_int), ("burst_share", "burst share", html.fmt_pct),
        ("queued_p50_ms", "queue p50 ms", html.fmt_int), ("ttft_p50_ms", "TTFT p50 ms", html.fmt_int), ("ttft_p95_ms", "TTFT p95 ms", html.fmt_int),
        ("e2e_p50_ms", "e2e p50 ms", html.fmt_int), ("e2e_p95_ms", "e2e p95 ms", html.fmt_int),
        ("delivered_tps_mean", "delivered tps", html.fmt_f(1)), ("floor_violation_share", "viol. % of stream time", html.fmt_pct),
        ("shape_usd", "shape USD", html.fmt_usd), ("burst_discount_usd", "burst discount", html.fmt_usd),
    ]
    parts = [f'<div class="card">{chart}{runs}</div>']
    agg = pt.groupby(["tenant", "load_run"], sort=False).agg(
        ok=("ok", "sum"), context_bursts=("context_bursts", "sum"), revoked=("revoked", "sum"),
        burst_tokens=("burst_tokens", "sum"), output_tokens=("output_tokens", "sum"),
        queued_p50_ms=("queued_p50_ms", "mean"), ttft_p50_ms=("ttft_p50_ms", "mean"), ttft_p95_ms=("ttft_p95_ms", "mean"),
        e2e_p50_ms=("e2e_p50_ms", "mean"), e2e_p95_ms=("e2e_p95_ms", "mean"), delivered_tps_mean=("delivered_tps_mean", "mean"),
        stream_s=("stream_s", "sum"), violation_stream_s=("violation_stream_s", "sum"),
        shape_usd=("shape_usd", "sum"), burst_discount_usd=("burst_discount_usd", "sum"),
    ).reset_index()
    with np.errstate(divide="ignore", invalid="ignore"):
        agg["burst_share"] = np.where(agg["output_tokens"] > 0, agg["burst_tokens"] / agg["output_tokens"], np.nan)
        agg["floor_violation_share"] = np.where(agg["stream_s"] > 0, agg["violation_stream_s"] / agg["stream_s"], np.nan)
    for tenant, sub in agg.groupby("tenant", sort=False):
        parts.append(f'<div class="card"><h3>{escape(str(tenant))}</h3>{html.table(sub, cols)}</div>')
    return "".join(parts)


def _reservations_actual_section(rb: pd.DataFrame) -> str:
    if rb.empty:
        return '<p class="muted">no reservations in these runs (scenario `reservation:` specs, or --no-reservations)</p>'
    by_tenant = rb.groupby("tenant", sort=False).agg(
        tenant_usd=("tenant_usd", "mean"), reservation_usd=("reservation_usd", "mean"),
        shape_usd=("shape_usd", "mean"), per_token_usd=("per_token_usd", "mean"),
    ).reset_index()
    chart = charts.bar_chart(
        "Bill per tenant: with its reservation, per-request shape, per-token", by_tenant["tenant"].tolist(),
        {"reservation + on-demand": by_tenant["tenant_usd"].tolist(), "of which reservation": by_tenant["reservation_usd"].tolist(),
         "shape, per request": by_tenant["shape_usd"].tolist(), "per-token": by_tenant["per_token_usd"].tolist()},
        subtitle="mean over the load runs that had a reservation for the tenant", y_label="USD per run", fmt=charts.fmt_usd,
    )
    tbl = html.table(
        rb,
        [
            ("load_run", "load run", html.fmt_text), ("tenant", "tenant", html.fmt_text), ("shape", "shape", html.fmt_text),
            ("replicas", "replicas", html.fmt_text), ("slots", "slots", html.fmt_int), ("duration_s", "duration s", html.fmt_int),
            ("slot_usd_per_hour", "slot $/h", html.fmt_usd), ("served", "served", html.fmt_int), ("timeouts", "timeouts", html.fmt_int),
            ("queued_p50_ms", "queue p50 ms", html.fmt_int), ("queued_p95_ms", "queue p95 ms", html.fmt_int),
            ("used_fill", "used fill", html.fmt_pct), ("idle_slot_hours", "idle slot-h", html.fmt_f(2)),
            ("reservation_usd", "reservation USD", html.fmt_usd), ("on_demand_usd", "on-demand USD", html.fmt_usd),
            ("tenant_usd", "tenant total", html.fmt_usd), ("shape_usd", "per-request shape USD", html.fmt_usd),
            ("per_token_usd", "per-token USD", html.fmt_usd), ("tenant_vs_shape", "total ÷ shape", html.fmt_f(2)),
            ("tenant_vs_per_token", "total ÷ per-token", html.fmt_f(2)),
        ],
        left=4,
    )
    return f'<div class="card">{chart}{tbl}</div>'


def _sample_card(r) -> str:
    """One showcase request: what it ran as, the prompt (last user message open, the rest folded) and the completion."""
    def num(v, fmt):
        return fmt(v) if v is not None else "–"

    status = str(r.status)
    if status != "ok" and r.abort_reason:
        status += f" ({r.abort_reason})"
    cls = " · context burst" if r.class_ctx == "burst" else ""
    billed = f" billed as {escape(str(r.billed_shape))}" if r.billed_shape and r.billed_shape != r.shape else ""
    meta = (
        f"<b>{escape(str(r.tenant))}</b> · {escape(str(r.prompt_id))} turn {int(r.turn) + 1} · {escape(str(r.shape))}{billed}{cls} · "
        f"{escape(status)} · t+{num(r.t_rel, html.fmt_int)} s · prompt {num(r.prompt_tokens, html.fmt_int)} tokens "
        f"({num(r.cached_tokens, html.fmt_int)} cached) · output {num(r.output_tokens, html.fmt_int)} · "
        f"queue {num(r.queued_ms, html.fmt_int)} ms · TTFT {num(r.ttft_ms, html.fmt_int)} ms · "
        f"delivered {num(r.delivered_tps, html.fmt_f(1))} tok/s on a {num(r.tps, html.fmt_int)} tps shape · "
        f"throttled {num(r.throttled_s, html.fmt_f(1))} s · burst {num(r.burst_share, html.fmt_pct)}"
        f"{' · <b>floor violation</b>' if bool(r.floor_violation) else ''} · shape {num(r.shape_usd, html.fmt_usd)} / "
        f"per-token {num(r.per_token_usd, html.fmt_usd)}"
    )
    prompt_html = '<p class="muted">prompt text not stored (gateway store_text was off)</p>'
    if isinstance(r.prompt_json, str) and r.prompt_json:
        try:
            msgs = json.loads(r.prompt_json)
        except json.JSONDecodeError:
            msgs = []
        if msgs:
            last = msgs[-1]
            earlier = msgs[:-1]
            last_text = str(last.get("content", ""))
            folded = last_text[:SAMPLE_PROMPT_CHARS]
            more = f'<p class="muted">… {len(last_text) - SAMPLE_PROMPT_CHARS:,} more characters in samples.csv</p>' if len(last_text) > SAMPLE_PROMPT_CHARS else ""
            prompt_html = f'<h4>{escape(str(last.get("role", "user")))}</h4><pre class="sample">{escape(folded)}</pre>{more}'
            if earlier:
                items = "".join(
                    f'<h4>{escape(str(m.get("role", "")))}</h4><pre class="sample">{escape(str(m.get("content", ""))[:SAMPLE_PROMPT_CHARS])}</pre>'
                    for m in earlier
                )
                prompt_html = f"<details><summary>{len(earlier)} earlier message(s) in the context</summary>{items}</details>" + prompt_html
    output = r.output_text if isinstance(r.output_text, str) else None
    out_html = f'<pre class="sample out">{escape(output)}</pre>' if output else '<p class="muted">no completion stored</p>'
    return f'<div class="card sample"><p>{meta}</p>{prompt_html}<h4>completion</h4>{out_html}</div>'


def _samples_section(sm: pd.DataFrame) -> str:
    if sm.empty:
        return (
            '<p class="muted">no showcase requests in these runs (give a tenant a <code>showcase:</code> spec in the '
            "scenario and run the gateway with store_text on)</p>"
        )
    intro = (
        "<p>Real, hand-written prompts sent at a low fixed rate through the same shapes, admission, pacing and billing as "
        "the synthetic traffic (docs/PHASE2.md section 6). Each card shows the constraints the request ran under and the "
        "completion the client received; pacing changes when tokens arrive, not which tokens, so the text is the model's "
        "quality at the size that fits the card, and the status shows what a tenant sees when the context limit, a burst "
        "revocation or a refusal bites. Everything here is also in samples.csv and samples.md.</p>"
    )
    parts = [intro]
    for load_run, sub in sm.groupby("load_run", sort=False):
        parts.append(f"<h3>{escape(str(load_run))}</h3>")
        parts.extend(_sample_card(r) for r in sub.itertuples(index=False))
    return "".join(parts)


def samples_markdown(sm: pd.DataFrame) -> str:
    """samples.md: the same cards as Markdown, for curation into docs/SAMPLES.md."""
    if sm.empty:
        return "# Output samples\n\nNo showcase requests in these runs.\n"
    out = ["# Output samples", "", "One section per showcase request, in arrival order. Metrics are the gateway's; text is what the client received.", ""]
    for load_run, sub in sm.groupby("load_run", sort=False):
        out.append(f"## {load_run}")
        out.append("")
        for r in sub.itertuples(index=False):
            status = str(r.status) + (f" ({r.abort_reason})" if r.status != "ok" and r.abort_reason else "")
            billed = f", billed as {r.billed_shape}" if r.billed_shape and r.billed_shape != r.shape else ""
            out.append(f"### {r.tenant} · {r.prompt_id} · turn {int(r.turn) + 1}")
            out.append("")
            out.append(
                f"shape {r.shape}{billed}{' (context burst)' if r.class_ctx == 'burst' else ''} · {status} · "
                f"prompt {int(r.prompt_tokens)} tokens ({int(r.cached_tokens)} cached) · output {int(r.output_tokens)} · "
                f"queue {html.fmt_int(r.queued_ms)} ms · TTFT {html.fmt_int(r.ttft_ms)} ms · "
                f"delivered {html.fmt_f(1)(r.delivered_tps)} tok/s on {html.fmt_int(r.tps)} tps · throttled {html.fmt_f(1)(r.throttled_s)} s · "
                f"burst {html.fmt_pct(r.burst_share)}{' · floor violation' if bool(r.floor_violation) else ''} · "
                f"shape {html.fmt_usd(r.shape_usd)} / per-token {html.fmt_usd(r.per_token_usd)}"
            )
            out.append("")
            if isinstance(r.prompt_json, str) and r.prompt_json:
                try:
                    msgs = json.loads(r.prompt_json)
                except json.JSONDecodeError:
                    msgs = []
                for m in msgs:
                    text = str(m.get("content", ""))
                    if len(text) > SAMPLE_PROMPT_CHARS:
                        text = text[:SAMPLE_PROMPT_CHARS] + f"\n… ({len(text) - SAMPLE_PROMPT_CHARS:,} more characters)"
                    out.append(f"**{m.get('role', '')}**")
                    out.append("")
                    out.append("```text")
                    out.append(text)
                    out.append("```")
                    out.append("")
            out.append("**completion**")
            out.append("")
            out.append("```text")
            out.append(str(r.output_text) if isinstance(r.output_text, str) and r.output_text else "(no completion stored)")
            out.append("```")
            out.append("")
    return "\n".join(out)


def _overview_section(pr: pd.DataFrame, run_ids: list[str]) -> str:
    tbl = html.table(
        pr,
        [
            ("load_run", "load run", html.fmt_text), ("scenario", "scenario", html.fmt_text), ("seed", "seed", html.fmt_int),
            ("pacing", "pacing", html.fmt_text), ("duration_s", "duration s", html.fmt_int),
            ("requests", "requests", html.fmt_int), ("ok", "ok", html.fmt_int), ("refused", "refused", html.fmt_int), ("errors", "errors", html.fmt_int),
            ("revoked", "revoked", html.fmt_int), ("context_bursts", "ctx bursts", html.fmt_int), ("reserved_requests", "reserved req.", html.fmt_int),
            ("prompt_tokens", "prompt tokens", html.fmt_int), ("cached_share", "cached", html.fmt_pct), ("output_tokens", "output tokens", html.fmt_int),
            ("burst_share", "burst share", html.fmt_pct), ("reservation_usd", "reservation USD", html.fmt_usd),
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
<dt>shape, billed</dt><dd>seconds from admission to the last delivered token (queue excluded), capped at the shape's promise, × the shape's price per second, plus the prefill charge (prompt tokens × pricing.shape.prompt_usd_per_million) when one is configured.</dd>
<dt>shape, bound</dt><dd>the cap: time to first token + delivered tokens ÷ shape tps. The customer never pays more than this.</dd>
<dt>burst discount</dt><dd>bound − billed: what faster-than-promised delivery (bursts) took off the bill.</dd>
<dt>throttled</dt><dd>wall-clock time delivery was held back by the per-request token bucket, not by the engine; for complete responses, the time the finished response was held.</dd>
<dt>delivered tps</dt><dd>delivered tokens ÷ time from first to last delivered token, head start included (so short answers read above the shape's rate); judged on requests with at least 30 tokens over 0.5 s.</dd>
<dt>floor violation</dt><dd>delivered tps below 95% of the shape's tps: the engine fell behind the promise by more than the head start could cover (admission control and prefill interference, milestone 3).</dd>
<dt>engine tps</dt><dd>output tokens ÷ (engine done − first token): the engine's own decode rate for that request.</dd>
<dt>fill rate</dt><dd>seat-seconds held (admission to engine done) ÷ (slots in the replica's slot plan × run time): how much of the sellable capacity was occupied.</dd>
<dt>utilisation</dt><dd>output tokens ÷ (slot plan's total tps × run time): how much of the plan's token capacity was delivered.</dd>
<dt>break-even fill</dt><dd>GPU cost per hour ÷ what the slot plan earns per hour at 100% fill; fill above it is profit under shape billing.</dd>
<dt>price reference</dt><dd>a per-token market price point from config/price_references.yaml (an OpenRouter snapshot, slotgw-fetch-prices); the active one sets the per-token bills and the shape prices derived from them at parity (price per second ÷ tps = the reference output price per token).</dd>
<dt>revenue at a reference</dt><dd>the run's prompt and output tokens at that reference's prices, lazy requests at the batch discount; prefix-cache hits are charged in full, so an upper bound.</dd>
<dt>gross margin</dt><dd>1 − GPU cost ÷ revenue. "At target utilisation": the same GPU-hour serving tokens in proportion to target ÷ measured utilisation.</dd>
<dt>list price at a margin</dt><dd>GPU cost ÷ (1 − margin) ÷ output-equivalent tokens, where prompt tokens count at the market reference's input:output price ratio and lazy tokens at the batch discount; the input price follows at that ratio. The market reference is the active one, or the current config's for a run snapshotted before price references existed.</dd>
<dt>burst tokens / burst share</dt><dd>tokens the burst bucket paid for: delivered above the shape's tps because the burst manager lent idle step rate; share = burst tokens ÷ output tokens.</dd>
<dt>context burst</dt><dd>a request over its shape's context admitted into free KV with low engine priority; "revoked" counts the ones aborted (status revoked, unbilled) when a guaranteed admission needed the capacity under the revoke policy.</dd>
<dt>reservation bill</dt><dd>slots × reservation duration × the shape's price per second (× pricing.shape.reservation_price_factor), whether used or not; no prefill charge. The tenant's total is that plus on-demand shape bills for its requests not served from the reservation; "used fill" is seat time of the served requests ÷ reserved slot-seconds.</dd>
<dt>reservation what-if</dt><dd>N slots of the tenant's shape for the whole run, sized at the p50, p95 or max of the tenant's queue-free occupancy; bill = N × run time × the shape's price per second, no prefill charge; "would queue" = requests arriving while all N slots are busy; used fill = occupied slot-seconds within the reservation ÷ reserved; cost share = GPU cost × N × capacity weight ÷ plan slots, where the weight is seats(plan shape) ÷ seats(this shape) on the calibration curve; the parity slot price is the hourly slot price at which the reservation equals the tenant's per-token bill.</dd>
<dt>in flight</dt><dd>requests between admission and last delivered token, from the meter; "running" is vLLM's own count sampled through the gateway.</dd>
<dt>cached tokens</dt><dd>prompt tokens vLLM served from its prefix cache (usage.prompt_tokens_details.cached_tokens); priced at the cache-read price under both schemes: the reference's listed one, else pricing.token.cache_read_ratio × the input price. Unmetered before phase 2 (counted as 0).</dd>
<dt>billed shape</dt><dd>the shape whose tps and price applied to a request: its own, or for a context burst the burst class (the model's slowest non-lazy shape by default, post-M4): that shape's tps takes the seat, paces the stream and sets the floor and the runtime price; the tenant's shape stays on the record.</dd>
<dt>showcase / output samples</dt><dd>real prompts sent at a low fixed rate (scenario `showcase:`); the gateway keeps their prompt and completion when store_text is on, and the Output samples section shows each with the constraints it ran under.</dd>
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
    all_res = pd.concat([d.reservations for d in datas if len(d.reservations)], ignore_index=True) if any(len(d.reservations) for d in datas) else None
    pt = per_tenant(df)
    pr = per_run(df, loads, config, all_res)
    rb = reservation_bills(df, all_res, config)
    xs = across_seeds(pt)
    inter = interference(df)
    tl = timeline(df, loads)
    prp = per_replica(df, loads, config, datas[0].fallback_config)
    ref = refusals(df)
    mk = market(df, config, datas[0].fallback_config)
    lp = list_prices(df, config, datas[0].fallback_config)
    rs = reservations(df, config, datas[0].fallback_config, Path(config_dir) if config_dir else Path("config"))
    all_texts = pd.concat([d.texts for d in datas if len(d.texts)], ignore_index=True) if any(len(d.texts) for d in datas) else None
    sm = samples(df, all_texts)

    out = Path(out_dir) if out_dir else datas[0].run_dir / "report"
    out.mkdir(parents=True, exist_ok=True)
    sm.to_csv(out / "samples.csv", index=False)
    (out / "samples.md").write_text(samples_markdown(sm), encoding="utf-8", newline="\n")
    df.to_csv(out / "per_request.csv", index=False)
    pt.to_csv(out / "per_tenant.csv", index=False)
    pr.to_csv(out / "summary.csv", index=False)
    xs.to_csv(out / "across_seeds.csv", index=False)
    inter.to_csv(out / "interference.csv", index=False)
    prp.to_csv(out / "per_replica.csv", index=False)
    ref.to_csv(out / "refusals.csv", index=False)
    mk.to_csv(out / "market.csv", index=False)
    lp.to_csv(out / "list_prices.csv", index=False)
    rs.to_csv(out / "reservations.csv", index=False)
    rb.to_csv(out / "reservation_bills.csv", index=False)

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
        ("bursting", "Bursting: burst tokens, discount, revocations", _bursting_section(pr, pt)),
        ("reservations-actual", "Reservations: standing slots and their bills", _reservations_actual_section(rb)),
        ("replicas", "Replicas: fill rate, utilisation, break-even", _replicas_section(prp, ref)),
        ("economics", "Provider economics", _economics_section(pr)),
        ("market", "Market reference prices and margin", _market_section(mk, lp, config, datas[0].fallback_config)),
        ("reservations", "Reservation what-if", _reservations_section(rs)),
        ("samples", "Output samples", _samples_section(sm)),
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
