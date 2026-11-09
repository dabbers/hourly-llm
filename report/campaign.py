"""Campaign report: a statistics layer over many load runs (docs/PHASE3.md section 4).

    slotgw-campaign runs/<a> runs/<b> ... --manifest config/phase3/campaign.yaml --out DIR [--config-dir DIR]

Every run directory is loaded with the phase-2 report's own loaders (report.metrics), so the
per-run and per-tenant numbers are the ones report.html shows. This module only groups them by
*label* (the part after "+" in the scenario name: "mixed-4090+burst" -> "burst"), pairs labels by
seed and puts confidence intervals and tests around the comparisons the manifest asks for.

Outputs in --out: settings.csv (per label: n, mean, sd, 95% CI of every run-level metric),
tenants.csv (the same per label and tenant), paired.csv (seed-paired differences with t and
Wilcoxon p-values, Hedges' g, Holm-corrected p for the headline rows), predictability.csv (CV of
each tenant's bill across seeds per scheme, and the per-request bill ÷ quote distribution),
sweep.csv (the manifest's families: mean and CI per x), margin_curve.csv (revenue ÷ GPU cost
against utilisation: raw points and the fitted line with its break-even utilisation) and
campaign.html, one page over all of it.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from . import charts, html
from .metrics import (
    _run_window,
    _seat_weights,
    _slot_plans,
    enrich,
    load_run_dir,
    per_run,
    per_tenant,
    reservation_bills,
)
from .stats import LinearFit, holm, linear_fit, mean_ci, paired

DEFAULT_SEATS = 32  # seats a replica offers when its slot plan is unknown (the phase-2 4090 plan)
UNIT_TPS = 30.0  # the slot plan's unit shape; a shape without a seat weight counts tps ÷ 30 seats
SWEEP_TENANT = "chat"  # the tenant whose queue p95 the sweep table carries
SIGNIFICANCE = 0.05  # two-sided; rows below it are bold in the paired tables
AT_QUOTE_TOL = 1e-3  # a request billed within 0.1% of its quote counts as billed the full quote

# Run-level metrics: one value per load run (= per label and seed). name -> (meaning, formatter).
RUN_METRICS: dict[str, tuple[str, Callable]] = {
    "requests": ("requests the load generator sent in the run", html.fmt_int),
    "ok": ("requests completed (status ok)", html.fmt_int),
    "refused": ("requests refused: 429 over the shape's context, 503 no capacity", html.fmt_int),
    "revoked": ("context bursts aborted because a guaranteed admission needed their capacity", html.fmt_int),
    "context_bursts": ("requests over their shape's context admitted as bursts into free KV", html.fmt_int),
    "output_tokens": ("output tokens of completed requests", html.fmt_int),
    "cached_share": ("prompt tokens vLLM served from its prefix cache ÷ prompt tokens", html.fmt_pct),
    "per_token_usd": ("revenue under per-token billing: reference list prices, lazy at the batch discount", html.fmt_usd),
    "tenant_usd": ("revenue under shape billing: on-demand shape bills plus reservation fees", html.fmt_usd),
    "reservation_usd": ("of which reservation fees: slots × duration × the shape's price per second", html.fmt_usd),
    "gpu_cost_usd": ("GPU cost: gpu_usd_per_hour × replicas used × run duration", html.fmt_usd),
    "shape_rev_over_cost": ("tenant_usd ÷ gpu_cost_usd; 1 is break-even", html.fmt_f(2)),
    "token_rev_over_cost": ("per_token_usd ÷ gpu_cost_usd; 1 is break-even", html.fmt_f(2)),
    "gross_margin_shape": ("1 − gpu_cost_usd ÷ tenant_usd", html.fmt_pct),
    "gross_margin_token": ("1 − gpu_cost_usd ÷ per_token_usd", html.fmt_pct),
    "output_tokens_per_gpu_hour": ("output tokens ÷ (replicas used × run hours)", html.fmt_int),
    "floor_violations": ("completed requests delivered below 95% of their shape's tps", html.fmt_int),
    "floor_violation_rate": ("floor_violations ÷ ok (the decode floor: first to last token, queue and TTFT excluded)", html.fmt_pct),
    "start_violations": ("completed interactive requests whose first token came more than START_BOUND_S (2 s) after arrival: queue + TTFT", html.fmt_int),
    "start_violation_rate": ("start_violations ÷ completed interactive requests (the start bound)", html.fmt_pct),
    "start_p95_s": ("95th percentile of queue + TTFT over completed interactive requests, seconds", html.fmt_f(2)),
    "refusal_rate": ("refused ÷ requests", html.fmt_pct),
    "burst_discount_usd": ("shape bound − shape billed: runtime given away by faster-than-promised delivery", html.fmt_usd),
    "concurrency_p95": ("95th percentile of requests in flight, one-second bins", html.fmt_f(1)),
    "preemptions": ("vLLM preemptions during the run (sampled counter, last − first)", html.fmt_int),
    "utilisation": (
        ("seat-seconds sold ÷ seat-seconds offered: Σ over completed requests of billed_s × the seat weight of the "
         "shape that took the seat, ÷ (slot-plan seats of the replicas the run used × run duration)"),
        html.fmt_pct,
    ),
}

# Tenant-level metrics: one value per load run and tenant.
TENANT_METRICS: dict[str, tuple[str, Callable]] = {
    "requests": ("requests the tenant sent", html.fmt_int),
    "refused": ("the tenant's refusals (429 context + 503 capacity)", html.fmt_int),
    "refused_capacity": ("of which 503 capacity refusals", html.fmt_int),
    "revoked": ("the tenant's context bursts that were revoked", html.fmt_int),
    "context_bursts": ("the tenant's requests admitted as context bursts", html.fmt_int),
    "output_tokens": ("output tokens of the tenant's completed requests", html.fmt_int),
    "per_token_usd": ("the tenant's bill under per-token billing", html.fmt_usd),
    "tenant_usd": (
        "the tenant's bill under shape billing: shape bills of its unreserved completed requests plus its reservation fee",
        html.fmt_usd,
    ),
    "queued_p50_ms": ("median queue wait, mean of the tenant's per-shape × delivery rows", html.fmt_int),
    "queued_p95_ms": ("95th-percentile queue wait, mean of the tenant's per-shape × delivery rows", html.fmt_int),
    "ttft_p50_ms": ("median time to first token after admission, mean of the per-shape × delivery rows", html.fmt_int),
    "ttft_p95_ms": ("95th-percentile time to first token, mean of the per-shape × delivery rows", html.fmt_int),
    "floor_violation_share": ("floor violations ÷ requests judged (≥ 30 tokens over ≥ 0.5 s): the decode floor", html.fmt_pct),
    "start_violation_share": ("start-bound violations (queue + TTFT > 2 s) ÷ the tenant's completed interactive requests", html.fmt_pct),
    "start_p95_s": ("95th percentile of queue + TTFT, seconds, mean of the tenant's per-shape × delivery rows", html.fmt_f(2)),
    "delivered_tps_mean": ("mean delivered tokens per second over the judged requests", html.fmt_f(1)),
}

# Plan-load covariates: what the seed's plan scheduled for the open-loop tenants (chat, agent, note-taker;
# the batch tenant runs closed loop, so its planned steps are a cap, not demand). They describe the
# offered load, which varies by seed: the agent tenant's arrivals are an on/off ("bursty") process with
# only a few cycles per 18-minute run. Listed per run and per setting, never paired: within a seed they
# are identical by construction.
PLAN_METRICS: dict[str, tuple[str, Callable]] = {
    "plan_open_sessions": ("sessions the seed's plan schedules for the open-loop tenants", html.fmt_int),
    "plan_open_steps": ("requests (steps) the seed's plan schedules for the open-loop tenants: the offered load", html.fmt_int),
    "plan_agent_steps": ("of which the agent tenant's (bursty arrivals: the seed-to-seed swing)", html.fmt_int),
}

HEADLINE_METRICS = ["tenant_usd", "per_token_usd", "output_tokens_per_gpu_hour", "gross_margin_shape",
                    "floor_violation_rate", "refusal_rate"]
SWEEP_METRICS = ["tenant_usd", "per_token_usd", "gpu_cost_usd", "shape_rev_over_cost", "token_rev_over_cost",
                 "output_tokens_per_gpu_hour", "floor_violation_rate", "refusal_rate", "utilisation"]
SWEEP_TENANT_METRIC = f"{SWEEP_TENANT}_queued_p95_ms"
SCHEMES = (("per_token", "per_token_usd"), ("shape", "tenant_usd"))
SCHEME_LABELS = {"per_token": "per-token", "shape": "shape"}


# --------------------------------------------------------------------------------------- manifest


@dataclass
class Family:
    """One sweep: `series` maps a series name to its (x, label) points along `x_name`."""

    name: str
    x_name: str
    series: dict[str, list[tuple[float, str]]]


@dataclass
class Manifest:
    settings: dict[str, str]  # label -> description
    pairs: list[tuple[str, str]]  # (setting, baseline)
    families: dict[str, Family]
    headline: list[tuple[str, str]]  # the pairs whose headline metrics get Holm-corrected p-values


def load_manifest(path: str | Path) -> Manifest:
    """Read config/phase3/campaign.yaml; every label named in pairs, families or headline must be a
    setting, so a typo fails here rather than as a silent empty comparison."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    settings = {str(k): str((v or {}).get("desc", "")) for k, v in (raw.get("settings") or {}).items()}
    pairs = [(str(a), str(b)) for a, b in raw.get("pairs") or []]
    families: dict[str, Family] = {}
    for name, fam in (raw.get("families") or {}).items():
        series = {str(s): [(float(x), str(lbl)) for x, lbl in pts] for s, pts in (fam.get("series") or {}).items()}
        families[str(name)] = Family(str(name), str(fam.get("x_name") or name), series)
    headline = [(str(a), str(b)) for a, b in raw.get("headline") or []]
    named = {lbl for p in pairs + headline for lbl in p} | {
        lbl for f in families.values() for pts in f.series.values() for _, lbl in pts
    }
    unknown = sorted(named - set(settings))
    if unknown:
        raise ValueError(f"{path}: labels used but not defined under settings: {unknown}")
    if not set(headline) <= set(pairs):
        raise ValueError(f"{path}: headline pairs must also be listed under pairs")
    return Manifest(settings, pairs, families, headline)


def label_of(scenario) -> str | None:
    """The setting label of a run: the part after the last "+" in its scenario name."""
    if not isinstance(scenario, str) or "+" not in scenario:
        return None
    return scenario.rsplit("+", 1)[1].strip() or None


# ---------------------------------------------------------------------------------------- loading


@dataclass
class Campaign:
    """Everything the statistics need, already reduced to one row per load run (`runs`) and per
    load run × tenant (`tenants`); `requests` keeps the completed requests' shape bill and quote
    for the quote-ratio distribution. `unassigned` are runs whose label is not in the manifest;
    `duplicates` the (label, seed) repeats, with `kept` marking the one with the latest start."""

    manifest: Manifest
    runs: pd.DataFrame
    tenants: pd.DataFrame
    requests: pd.DataFrame
    unassigned: pd.DataFrame = field(default_factory=pd.DataFrame)
    duplicates: pd.DataFrame = field(default_factory=pd.DataFrame)
    seat_basis: str = "none"


def _utilisation(sub: pd.DataFrame, weights: dict[str, float], plans: dict[str, dict[str, int]], shapes: dict) -> float:
    """Seat-seconds sold ÷ seat-seconds offered for one load run. Sold: Σ over completed requests
    of billed_s × the seat weight of the shape whose tps took the seat (`billed_shape`: the
    request's own, or a context burst's class), weights from the calibration curve in units of the
    plan's unit shape (tps ÷ 30 for a shape without one). Offered: the slot-plan seats of every
    replica the run used (32 per replica without a plan) × the run's duration, first arrival to
    last delivered token."""
    t0, t1 = _run_window(sub)
    duration = max(0.0, t1 - t0)
    ok = sub[sub["ok"]]
    replicas = ok["replica"].dropna().unique() if "replica" in ok.columns else []
    seats = sum((sum(plans.get(r, {}).values()) or DEFAULT_SEATS) for r in replicas) or DEFAULT_SEATS

    def weight(name) -> float:
        w = weights.get(name, np.nan)
        if not np.isnan(w):
            return float(w)
        return shapes[name].tps / UNIT_TPS if name in shapes else np.nan

    w = ok["billed_shape"].map(weight).astype(float)
    sold = float((ok["billed_s"] * w).sum())
    return sold / (seats * duration) if seats and duration > 0 else np.nan


def _plan_load(run_dir: Path, load_run: str) -> dict[str, float]:
    """The plan-load covariates of one load run from its plan.json; NaN when the plan is not there."""
    out: dict[str, float] = {m: np.nan for m in PLAN_METRICS}
    path = run_dir / "loads" / load_run / "plan.json"
    if not path.is_file():
        return out
    plan = json.loads(path.read_text(encoding="utf-8"))
    sessions = steps = agent = 0
    for t in plan.get("tenants", []):
        if t.get("profile_kind") == "batch":
            continue
        n_steps = sum(len(s.get("steps", [])) for s in t.get("sessions", []))
        sessions += len(t.get("sessions", []))
        steps += n_steps
        if t.get("profile_kind") == "agent":
            agent += n_steps
    return {"plan_open_sessions": float(sessions), "plan_open_steps": float(steps), "plan_agent_steps": float(agent)}


def _load_one(run_dir: Path, config_dir: Path | None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    """One run directory -> (run rows, tenant rows, completed-request rows, seat-weight basis),
    each row tagged with label, seed, load_run and run_dir."""
    data = load_run_dir(run_dir, config_dir)
    if data.requests.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), "none"
    df = enrich(data.requests, data.config)
    res = data.reservations if len(data.reservations) else None
    pr = per_run(df, data.loads, data.config, res)
    pt = per_tenant(df)
    rb = reservation_bills(df, res, data.config)
    weights, basis = _seat_weights(data.config, None, config_dir)
    plans = _slot_plans(data.config, data.fallback_config)
    shapes = {s.name: s for s in data.config.shapes}
    util = {lr: _utilisation(sub, weights, plans, shapes) for lr, sub in df.groupby("load_run", sort=False)}
    pr = pr.assign(
        utilisation=pr["load_run"].map(util).astype(float),
        label=pr["scenario"].map(label_of),
        run_dir=str(run_dir),
    )
    plan_loads = {lr: _plan_load(run_dir, lr) for lr in pr["load_run"].unique()}
    for m in PLAN_METRICS:
        pr[m] = pr["load_run"].map(lambda lr, m=m: plan_loads[lr][m]).astype(float)
    tags = pr.set_index("load_run")[["label", "seed", "scenario", "run_dir"]]

    agg = pt.groupby(["load_run", "tenant"], dropna=False).agg(
        requests=("requests", "sum"), refused=("refused", "sum"), refused_capacity=("refused_capacity", "sum"),
        revoked=("revoked", "sum"), context_bursts=("context_bursts", "sum"), output_tokens=("output_tokens", "sum"),
        per_token_usd=("per_token_usd", "sum"), floor_violations=("floor_violations", "sum"),
        floor_judged=("floor_judged", "sum"),
        start_violations=("start_violations", "sum"), start_judged=("start_judged", "sum"),
        start_p95_s=("start_p95_s", "mean"),
        queued_p50_ms=("queued_p50_ms", "mean"), queued_p95_ms=("queued_p95_ms", "mean"),
        ttft_p50_ms=("ttft_p50_ms", "mean"), ttft_p95_ms=("ttft_p95_ms", "mean"),
        delivered_tps_mean=("delivered_tps_mean", "mean"),
    )
    ok = df[df["ok"]]
    on_demand = ok[~ok["reserved"]].groupby(["load_run", "tenant"])["shape_usd"].sum()
    fees = rb.groupby(["load_run", "tenant"])["reservation_usd"].sum() if len(rb) else pd.Series(dtype=float)
    agg["tenant_usd"] = on_demand.reindex(agg.index).fillna(0.0) + fees.reindex(agg.index).fillna(0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        agg["floor_violation_share"] = np.where(agg["floor_judged"] > 0, agg["floor_violations"] / agg["floor_judged"], np.nan)
        agg["start_violation_share"] = np.where(agg["start_judged"] > 0, agg["start_violations"] / agg["start_judged"], np.nan)
    agg = agg.reset_index()
    for col in ("label", "seed", "scenario", "run_dir"):
        agg[col] = agg["load_run"].map(tags[col])

    req = ok[["load_run", "tenant", "shape_usd", "bound_usd"]].copy()
    for col in ("label", "seed", "run_dir"):
        req[col] = req["load_run"].map(tags[col])
    return pr, agg, req, basis


def _derive(runs: pd.DataFrame) -> pd.DataFrame:
    """The ratio metrics of RUN_METRICS that per_run does not carry."""
    runs = runs.copy()
    with np.errstate(divide="ignore", invalid="ignore"):
        runs["shape_rev_over_cost"] = runs["tenant_usd"] / runs["gpu_cost_usd"]
        runs["token_rev_over_cost"] = runs["per_token_usd"] / runs["gpu_cost_usd"]
        runs["gross_margin_shape"] = 1.0 - runs["gpu_cost_usd"] / runs["tenant_usd"]
        runs["gross_margin_token"] = 1.0 - runs["gpu_cost_usd"] / runs["per_token_usd"]
        runs["floor_violation_rate"] = runs["floor_violations"] / runs["ok"]
        runs["start_violation_rate"] = runs["start_violations"] / runs["start_judged"]
        runs["refusal_rate"] = runs["refused"] / runs["requests"]
    cols = ["shape_rev_over_cost", "token_rev_over_cost", "gross_margin_shape", "gross_margin_token",
            "floor_violation_rate", "start_violation_rate", "refusal_rate"]
    runs[cols] = runs[cols].replace([np.inf, -np.inf], np.nan)
    return runs


def load_campaign(run_dirs: list[str | Path], manifest: Manifest, config_dir: str | Path | None = None) -> Campaign:
    """Load every run directory, keep the runs whose label the manifest knows, and for a repeated
    (label, seed) keep the load run that started last."""
    cfg_dir = Path(config_dir) if config_dir else None
    runs, tenants, requests, basis = [], [], [], "none"
    for d in run_dirs:
        pr, agg, req, b = _load_one(Path(d), cfg_dir)
        if pr.empty:
            continue
        runs.append(pr)
        tenants.append(agg)
        requests.append(req)
        basis = b
    if not runs:
        empty = pd.DataFrame()
        return Campaign(manifest, empty, empty, empty, empty, empty, basis)
    all_runs = _derive(pd.concat(runs, ignore_index=True))
    all_runs["key"] = all_runs["run_dir"] + "::" + all_runs["load_run"]
    known = all_runs["label"].isin(list(manifest.settings))
    unassigned = all_runs.loc[~known, ["scenario", "label", "seed", "load_run", "run_dir"]].reset_index(drop=True)
    assigned = all_runs[known].copy()
    assigned["_started"] = pd.to_datetime(assigned["started_at"], utc=True, errors="coerce", format="ISO8601")
    assigned = assigned.sort_values(["label", "seed", "_started"], na_position="first", kind="stable")
    dup_group = assigned.duplicated(["label", "seed"], keep=False)
    assigned["kept"] = ~assigned.duplicated(["label", "seed"], keep="last")
    duplicates = assigned.loc[dup_group, ["label", "seed", "load_run", "started_at", "run_dir", "kept"]].reset_index(drop=True)
    kept = assigned[assigned["kept"]].drop(columns=["_started", "kept"]).reset_index(drop=True)
    keys = set(kept["key"])
    ten = pd.concat(tenants, ignore_index=True)
    ten = ten[(ten["run_dir"] + "::" + ten["load_run"]).isin(keys)].reset_index(drop=True)
    req = pd.concat(requests, ignore_index=True)
    req = req[(req["run_dir"] + "::" + req["load_run"]).isin(keys)].reset_index(drop=True)
    return Campaign(manifest, kept, ten, req, unassigned, duplicates, basis)


# ------------------------------------------------------------------------------------- statistics


def _ci_rows(frame: pd.DataFrame, keys: list[str], metrics: list[str]) -> pd.DataFrame:
    """Long table: per group (`keys`) and metric, n, mean, sd and the 95% t-interval."""
    rows = []
    if frame.empty:
        return pd.DataFrame(columns=[*keys, "metric", "n", "mean", "sd", "ci_lo", "ci_hi"])
    for key, sub in frame.groupby(keys, sort=False):
        key = key if isinstance(key, tuple) else (key,)
        for m in metrics:
            mean, sd, n, lo, hi = mean_ci(sub[m]) if m in sub else (np.nan, np.nan, 0, np.nan, np.nan)
            rows.append({**dict(zip(keys, key, strict=True)), "metric": m, "n": n, "mean": mean, "sd": sd, "ci_lo": lo, "ci_hi": hi})
    return pd.DataFrame(rows)


def settings_table(c: Campaign) -> pd.DataFrame:
    """settings.csv: per label and run-level metric, n, mean, sd, ci_lo, ci_hi (95% Student t)."""
    return _ci_rows(c.runs, ["label"], [*RUN_METRICS, *PLAN_METRICS])


def tenants_table(c: Campaign) -> pd.DataFrame:
    """tenants.csv: per label, tenant and tenant-level metric, n, mean, sd, ci_lo, ci_hi."""
    return _ci_rows(c.tenants, ["label", "tenant"], list(TENANT_METRICS))


PAIRED_COLUMNS = ["setting", "baseline", "scope", "tenant", "metric", "n", "mean_baseline", "mean_setting", "diff",
                  "diff_ci_lo", "diff_ci_hi", "rel_diff", "p_t", "p_wilcoxon", "hedges_g", "headline", "p_holm"]


def paired_table(c: Campaign) -> pd.DataFrame:
    """paired.csv: for each manifest pair, each run-level metric (scope "run") and each tenant-level
    metric per tenant (scope "tenant"), the seed-paired comparison (report.stats.paired: difference
    = setting − baseline on the common seeds, its 95% CI, paired t and Wilcoxon p, Hedges' g,
    relative difference). `p_holm` is filled for the headline pairs' headline metrics only, Holm
    applied across all those rows together; pairs without a common seed produce no rows."""
    rows = []
    if c.runs.empty:
        return pd.DataFrame(columns=PAIRED_COLUMNS)
    for setting, baseline in c.manifest.pairs:
        rs = c.runs[c.runs["label"] == setting].set_index("seed")
        rb = c.runs[c.runs["label"] == baseline].set_index("seed")
        seeds = sorted(set(rs.index) & set(rb.index))
        if not seeds:
            continue
        head = (setting, baseline) in c.manifest.headline
        for m in RUN_METRICS:
            res = paired(rb.loc[seeds, m], rs.loc[seeds, m])
            rows.append(_paired_row(setting, baseline, "run", "", m, res, head and m in HEADLINE_METRICS))
        ts_all = c.tenants[c.tenants["label"] == setting]
        tb_all = c.tenants[c.tenants["label"] == baseline]
        for tenant in sorted(set(ts_all["tenant"]) & set(tb_all["tenant"])):
            ts = ts_all[ts_all["tenant"] == tenant].set_index("seed")
            tb = tb_all[tb_all["tenant"] == tenant].set_index("seed")
            common = sorted(set(ts.index) & set(tb.index))
            if not common:
                continue
            for m in TENANT_METRICS:
                res = paired(tb.loc[common, m], ts.loc[common, m])
                rows.append(_paired_row(setting, baseline, "tenant", tenant, m, res, False))
    out = pd.DataFrame(rows, columns=PAIRED_COLUMNS)
    if out.empty:
        return out
    mask = out["headline"].astype(bool)
    out.loc[mask, "p_holm"] = holm(out.loc[mask, "p_t"])
    return out


def _paired_row(setting: str, baseline: str, scope: str, tenant: str, metric: str, res: dict, headline: bool) -> dict:
    return {
        "setting": setting, "baseline": baseline, "scope": scope, "tenant": tenant, "metric": metric, "n": res["n"],
        "mean_baseline": res["mean_a"], "mean_setting": res["mean_b"], "diff": res["diff"],
        "diff_ci_lo": res["diff_lo"], "diff_ci_hi": res["diff_hi"], "rel_diff": res["rel_diff"],
        "p_t": res["p_t"], "p_wilcoxon": res["p_wilcoxon"], "hedges_g": res["hedges_g"],
        "headline": headline, "p_holm": np.nan,
    }


def predictability_table(c: Campaign) -> pd.DataFrame:
    """predictability.csv: per label, tenant and billing scheme, the tenant's bill across seeds: n,
    mean, sd and cv = sd ÷ mean (lower = a more predictable monthly bill). The quote columns
    describe shape billing and repeat on both scheme rows of a (label, tenant): over the label's
    completed requests with a positive quote, quote_ratio = shape_usd ÷ bound_usd (billed ÷ the
    quoted cap, always ≤ 1; the shortfall is the burst discount) at its p50 and p95, and
    share_at_quote = the share of requests billed the full quote, i.e. within 0.1% of it: a paced
    request lands on its quote to within the timestamps' resolution, not exactly."""
    cols = ["label", "tenant", "scheme", "n", "mean_usd", "sd_usd", "cv", "requests_quoted", "quote_ratio_p50",
            "quote_ratio_p95", "share_at_quote"]
    rows = []
    if c.tenants.empty:
        return pd.DataFrame(columns=cols)
    for (label, tenant), sub in c.tenants.groupby(["label", "tenant"], sort=False):
        q = c.requests[(c.requests["label"] == label) & (c.requests["tenant"] == tenant) & (c.requests["bound_usd"] > 0)]
        ratio = (q["shape_usd"] / q["bound_usd"]).to_numpy(dtype=float)
        quote = {
            "requests_quoted": len(ratio),
            "quote_ratio_p50": float(np.percentile(ratio, 50)) if len(ratio) else np.nan,
            "quote_ratio_p95": float(np.percentile(ratio, 95)) if len(ratio) else np.nan,
            "share_at_quote": float(np.mean(ratio >= 1.0 - AT_QUOTE_TOL)) if len(ratio) else np.nan,
        }
        for scheme, col in SCHEMES:
            mean, sd, n, _lo, _hi = mean_ci(sub[col])
            cv = sd / mean if n >= 2 and mean else np.nan
            rows.append({"label": label, "tenant": tenant, "scheme": scheme, "n": n, "mean_usd": mean, "sd_usd": sd, "cv": cv, **quote})
    return pd.DataFrame(rows, columns=cols)


def sweep_table(c: Campaign) -> pd.DataFrame:
    """sweep.csv: for every family, series and x: the label, n runs, and mean / ci_lo / ci_hi of the
    SWEEP_METRICS plus the chat tenant's queue p95 (ms). A point whose label has no runs yet keeps
    its row with n = 0, so the table shows which sweep points are still missing."""
    rows = []
    for fam in c.manifest.families.values():
        for series, pts in fam.series.items():
            for x, label in pts:
                sub = c.runs[c.runs["label"] == label] if not c.runs.empty else pd.DataFrame()
                row = {"family": fam.name, "x_name": fam.x_name, "series": series, "x": x, "label": label, "n": len(sub)}
                for m in SWEEP_METRICS:
                    mean, _sd, _n, lo, hi = mean_ci(sub[m]) if len(sub) else (np.nan, np.nan, 0, np.nan, np.nan)
                    row.update({f"{m}_mean": mean, f"{m}_ci_lo": lo, f"{m}_ci_hi": hi})
                t = c.tenants[(c.tenants["label"] == label) & (c.tenants["tenant"] == SWEEP_TENANT)] if not c.tenants.empty else pd.DataFrame()
                mean, _sd, _n, lo, hi = mean_ci(t["queued_p95_ms"]) if len(t) else (np.nan, np.nan, 0, np.nan, np.nan)
                row.update({f"{SWEEP_TENANT_METRIC}_mean": mean, f"{SWEEP_TENANT_METRIC}_ci_lo": lo, f"{SWEEP_TENANT_METRIC}_ci_hi": hi})
                rows.append(row)
    return pd.DataFrame(rows)


MARGIN_COLUMNS = ["kind", "scheme", "label", "seed", "utilisation", "shape_rev_over_cost", "token_rev_over_cost", "n", "slope",
                  "intercept", "r2", "breakeven_utilisation"]


def margin_fits(c: Campaign) -> dict[str, LinearFit]:
    """Per scheme, the least-squares line of revenue ÷ GPU cost against utilisation over every
    kept run of every label."""
    if c.runs.empty:
        return {s: LinearFit(np.nan, np.nan, np.nan, 0) for s, _ in SCHEMES}
    return {
        "shape": linear_fit(c.runs["utilisation"], c.runs["shape_rev_over_cost"]),
        "per_token": linear_fit(c.runs["utilisation"], c.runs["token_rev_over_cost"]),
    }


def margin_table(c: Campaign, fits: dict[str, LinearFit]) -> pd.DataFrame:
    """margin_curve.csv: "fit" rows (scheme, n, slope, intercept, r2 and breakeven_utilisation,
    the utilisation where the fitted revenue ÷ cost line crosses 1.0) and one "point" row per
    kept run (label, seed, utilisation, shape_rev_over_cost, token_rev_over_cost)."""
    rows = [
        {"kind": "fit", "scheme": scheme, "n": fit.n, "slope": fit.slope, "intercept": fit.intercept, "r2": fit.r2,
         "breakeven_utilisation": fit.x_at(1.0)}
        for scheme, fit in fits.items()
    ]
    if not c.runs.empty:
        for r in c.runs.itertuples(index=False):
            rows.append({"kind": "point", "label": r.label, "seed": int(r.seed), "utilisation": r.utilisation,
                         "shape_rev_over_cost": r.shape_rev_over_cost, "token_rev_over_cost": r.token_rev_over_cost})
    return pd.DataFrame(rows, columns=MARGIN_COLUMNS)


# ------------------------------------------------------------------------------------------- html


def _nan(v) -> bool:
    return v is None or (isinstance(v, float) and np.isnan(v))


def _pm(fmt: Callable, mean, lo, hi) -> str:
    """'mean ± half-width' of a 95% interval; the mean alone when there is no interval (n < 2)."""
    if _nan(mean):
        return "–"
    if _nan(hi):
        return fmt(mean)
    return f"{fmt(mean)} ± {fmt(hi - mean)}"


def _fmt_pct0(v: float) -> str:
    return f"{v * 100:.0f}%"


_fmt_p = html.fmt_f(4)


def _metric_fmt(scope: str, metric: str) -> Callable:
    reg = {**RUN_METRICS, **PLAN_METRICS} if scope == "run" else TENANT_METRICS
    return reg[metric][1] if metric in reg else html.fmt_f(3)


def _pm_frame(ci: pd.DataFrame, keys: list[str], metrics: list[str], reg: dict) -> pd.DataFrame:
    """Pivot a long CI table into one text cell 'mean ± h' per metric, plus n."""
    if ci.empty:
        return pd.DataFrame()
    rows = []
    for key, sub in ci.groupby(keys, sort=False):
        key = key if isinstance(key, tuple) else (key,)
        by = sub.set_index("metric")
        row = dict(zip(keys, key, strict=True))
        row["n"] = int(by["n"].max())
        for m in metrics:
            r = by.loc[m] if m in by.index else None
            row[m] = "–" if r is None else _pm(reg[m][1], r["mean"], r["ci_lo"], r["ci_hi"])
        rows.append(row)
    return pd.DataFrame(rows)


def _overview_section(c: Campaign) -> str:
    rows = []
    for label, desc in c.manifest.settings.items():
        sub = c.runs[c.runs["label"] == label] if not c.runs.empty else pd.DataFrame()
        seeds = ", ".join(str(int(s)) for s in sorted(sub["seed"])) if len(sub) else "–"
        rows.append({"label": label, "desc": desc, "runs": len(sub), "seeds": seeds})
    tbl = html.table(
        pd.DataFrame(rows),
        [("label", "label", html.fmt_text), ("desc", "setting", html.fmt_text), ("runs", "runs", html.fmt_int),
         ("seeds", "seeds", html.fmt_text)],
        left=2,
    )
    n_runs = len(c.runs)
    n_labels = int(c.runs["label"].nunique()) if n_runs else 0
    parts = [
        (f"<p>{n_runs} load run(s) kept over {n_labels} setting(s); seat weights for utilisation from "
         f"{escape(c.seat_basis)}. Every interval is a two-sided 95% Student-t interval over the runs (one "
         f"per seed); n &lt; 2 gives a mean without one.</p>"),
        f'<div class="card">{tbl}</div>',
    ]
    if len(c.unassigned):
        parts.append("<h3>Unassigned runs (label not in the manifest; ignored)</h3>")
        parts.append('<div class="card">' + html.table(
            c.unassigned,
            [("scenario", "scenario", html.fmt_text), ("label", "label", html.fmt_text), ("seed", "seed", html.fmt_int),
             ("load_run", "load run", html.fmt_text), ("run_dir", "run dir", html.fmt_text)],
            left=2,
        ) + "</div>")
    if len(c.duplicates):
        parts.append("<h3>Duplicate (label, seed) runs: the latest start is kept</h3>")
        parts.append('<div class="card">' + html.table(
            c.duplicates.assign(kept=c.duplicates["kept"].map({True: "kept", False: "dropped"})),
            [("label", "label", html.fmt_text), ("seed", "seed", html.fmt_int), ("load_run", "load run", html.fmt_text),
             ("started_at", "started", html.fmt_text), ("run_dir", "run dir", html.fmt_text), ("kept", "", html.fmt_text)],
        ) + "</div>")
    return "".join(parts)


def _settings_section(c: Campaign, st: pd.DataFrame) -> str:
    if c.runs.empty:
        return '<p class="muted">no runs</p>'
    shown = [*HEADLINE_METRICS, "plan_open_steps"]  # the offered load last, as the covariate it is
    frame = _pm_frame(st, ["label"], shown, {**RUN_METRICS, **PLAN_METRICS})
    cols = [("label", "label", html.fmt_text), ("n", "n", html.fmt_int)]
    cols += [(m, m.replace("_", " "), html.fmt_text) for m in shown]
    tbl = html.table(frame, cols)
    means = st.pivot(index="label", columns="metric", values="mean").reindex(frame["label"])
    labels = frame["label"].tolist()
    c1 = charts.bar_chart(
        "Revenue ÷ GPU cost per setting (mean over seeds)", labels,
        {"shape": means["shape_rev_over_cost"].tolist(), "per-token": means["token_rev_over_cost"].tolist()},
        subtitle="1.0 is break-even; intervals in the table", y_label="revenue ÷ cost",
    )
    c2 = charts.bar_chart(
        "Output tokens per GPU-hour and floor violation rate", labels,
        {"tokens / GPU-h (K)": (means["output_tokens_per_gpu_hour"] / 1e3).tolist(),
         "floor violations, %": (means["floor_violation_rate"] * 100).tolist()},
        subtitle="means over seeds; two units on one axis, read each series against its legend",
    )
    return f'<div class="card"><div class="grid">{c1}{c2}</div>{tbl}</div>'


def _paired_html_table(rows: pd.DataFrame) -> str:
    head = ["metric", "baseline", "setting", "diff", "95% CI of diff", "rel.", "p (paired t)", "p (Wilcoxon)", "p (Holm)",
            "Hedges' g"]
    th = "".join(f'<th class="{"l" if i == 0 else ""}">{escape(h)}</th>' for i, h in enumerate(head))
    body = []
    for r in rows.itertuples(index=False):
        fmt = _metric_fmt(r.scope, r.metric)
        name = r.metric if r.scope == "run" else f"{r.tenant} · {r.metric}"
        bold = ' style="font-weight:600"' if not _nan(r.p_t) and r.p_t < SIGNIFICANCE else ""
        cells = [
            f'<td class="l">{escape(name)}</td>', f"<td>{fmt(r.mean_baseline)}</td>", f"<td>{fmt(r.mean_setting)}</td>",
            f"<td>{fmt(r.diff)}</td>", f"<td>{fmt(r.diff_ci_lo)} … {fmt(r.diff_ci_hi)}</td>",
            f"<td>{html.fmt_pct(r.rel_diff)}</td>", f"<td>{_fmt_p(r.p_t)}</td>", f"<td>{_fmt_p(r.p_wilcoxon)}</td>",
            f"<td>{_fmt_p(r.p_holm)}</td>", f"<td>{html.fmt_f(2)(r.hedges_g)}</td>",
        ]
        body.append(f"<tr{bold}>{''.join(cells)}</tr>")
    return f'<div class="tblwrap"><table><thead><tr>{th}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'


def _paired_section(c: Campaign, pt: pd.DataFrame) -> str:
    intro = (
        "<p>Each table pairs a setting with its baseline on the seeds both have (one load run each); diff = setting − "
        "baseline with its 95% t-interval, two-sided paired t-test and Wilcoxon signed-rank p-values, and Hedges' g on "
        f"the differences. Rows with p (paired t) &lt; {SIGNIFICANCE} are bold. p (Holm) is the Holm–Bonferroni-adjusted "
        "p across all headline rows (the headline pairs × the headline metrics) and is empty elsewhere. With two or "
        "three seeds the Wilcoxon test cannot go below 0.25 / 0.5; it becomes informative from about six seeds.</p>"
    )
    parts = [intro]
    for setting, baseline in c.manifest.pairs:
        sub = pt[(pt["setting"] == setting) & (pt["baseline"] == baseline)] if not pt.empty else pd.DataFrame()
        if sub.empty:
            parts.append(f'<p class="muted">{escape(setting)} vs {escape(baseline)}: no common seeds yet.</p>')
            continue
        n = int(sub.loc[sub["scope"] == "run", "n"].max())
        parts.append(
            f"<h3>{escape(setting)} vs {escape(baseline)}</h3>"
            f'<p class="muted">{escape(c.manifest.settings.get(setting, ""))} — against — '
            f'{escape(c.manifest.settings.get(baseline, ""))}; {n} common seed(s).</p>'
            f'<div class="card">{_paired_html_table(sub)}</div>'
        )
    return "".join(parts)


def _predictability_section(c: Campaign, pred: pd.DataFrame) -> str:
    if pred.empty:
        return '<p class="muted">no runs</p>'
    wide = pred.pivot_table(index=["label", "tenant"], columns="scheme", values=["mean_usd", "cv", "n"], aggfunc="first")
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    wide = wide.reset_index()
    quote = pred.drop_duplicates(["label", "tenant"])[["label", "tenant", "requests_quoted", "quote_ratio_p50", "quote_ratio_p95", "share_at_quote"]]
    chart_parts = []
    for tenant, sub in wide.groupby("tenant", sort=False):
        sub = sub[sub["n_shape"] >= 2]
        if sub.empty:
            continue
        chart_parts.append(charts.bar_chart(
            f"CV of the bill across seeds – {tenant}", sub["label"].tolist(),
            {"per-token": (sub["cv_per_token"] * 100).tolist(), "shape": (sub["cv_shape"] * 100).tolist()},
            subtitle="sd ÷ mean of the tenant's bill over the seeds; lower = more predictable", y_label="CV, %",
            fmt=lambda v: f"{v:.1f}%",
        ))
    t1 = html.table(
        wide,
        [("label", "label", html.fmt_text), ("tenant", "tenant", html.fmt_text), ("n_shape", "n", html.fmt_int),
         ("mean_usd_per_token", "per-token mean", html.fmt_usd), ("cv_per_token", "per-token CV", html.fmt_pct),
         ("mean_usd_shape", "shape mean", html.fmt_usd), ("cv_shape", "shape CV", html.fmt_pct)],
        left=2,
    )
    t2 = html.table(
        quote,
        [("label", "label", html.fmt_text), ("tenant", "tenant", html.fmt_text), ("requests_quoted", "requests quoted", html.fmt_int),
         ("quote_ratio_p50", "billed ÷ quote p50", html.fmt_pct), ("quote_ratio_p95", "billed ÷ quote p95", html.fmt_pct),
         ("share_at_quote", "billed the full quote (within 0.1%)", html.fmt_pct)],
        left=2,
    )
    grid = f'<div class="grid">{"".join(chart_parts)}</div>' if chart_parts else ""
    return (
        f'<div class="card">{grid}<h3>Bill across seeds per scheme</h3>{t1}'
        f"<h3>Per-request bill against the quote (shape billing)</h3>"
        f'<p class="muted">quote = the shape bound (TTFT + tokens ÷ tps at the shape price, plus prefill); billed ÷ quote '
        f"&lt; 1 is the burst discount the tenant received.</p>{t2}</div>"
    )


def _sweep_chart(c: Campaign, fam: Family, sw: pd.DataFrame, title: str, specs: list[tuple[str, str]], *,
                 scale: float = 1.0, fmt: Callable = charts.fmt_value, y_ref: float | None = None) -> str:
    """One sweep chart: per series and metric spec (metric, legend suffix), the per-seed points and
    a line through the means."""
    points: dict[str, tuple[list[float], list[float]]] = {}
    lines: dict[str, tuple[list[float], list[float]]] = {}
    for series, pts in fam.series.items():
        for metric, suffix in specs:
            name = f"{series} · {suffix}" if suffix else series
            xs, ys = [], []
            for x, label in pts:
                sub = c.runs[c.runs["label"] == label] if not c.runs.empty else pd.DataFrame()
                vals = sub[metric].dropna().tolist() if len(sub) else []
                xs += [x] * len(vals)
                ys += [v * scale for v in vals]
            means = sw[(sw["family"] == fam.name) & (sw["series"] == series) & (sw["n"] > 0)].sort_values("x")
            points[name] = (xs, ys)
            lines[name] = (means["x"].tolist(), (means[f"{metric}_mean"] * scale).tolist())
    return charts.scatter_chart(
        f"{title} – {fam.name}", points, lines=lines, x_label=fam.x_name, fmt=fmt,
        subtitle="dots: one per seed; line: through the means", y_ref=y_ref,
        y_ref_label="break-even" if y_ref is not None else "",
    )


def _sweeps_section(c: Campaign, sw: pd.DataFrame) -> str:
    if sw.empty:
        return '<p class="muted">no families in the manifest</p>'
    parts = []
    for fam in c.manifest.families.values():
        rows = sw[sw["family"] == fam.name]
        if rows["n"].sum() == 0:
            parts.append(f'<h3>{escape(fam.name)}</h3><p class="muted">no runs yet for this sweep.</p>')
            continue
        cs = [
            _sweep_chart(c, fam, sw, "Revenue ÷ GPU cost", [("shape_rev_over_cost", "shape"), ("token_rev_over_cost", "per-token")], y_ref=1.0),
            _sweep_chart(c, fam, sw, "Output tokens per GPU-hour", [("output_tokens_per_gpu_hour", "")]),
            _sweep_chart(c, fam, sw, "Floor violation rate", [("floor_violation_rate", "")], scale=100.0, fmt=lambda v: f"{v:.1f}%"),
            _sweep_chart(c, fam, sw, "Refusal rate", [("refusal_rate", "")], scale=100.0, fmt=lambda v: f"{v:.1f}%"),
            _sweep_chart(c, fam, sw, "Utilisation", [("utilisation", "")], scale=100.0, fmt=_fmt_pct0),
        ]
        text = rows.assign(**{
            m: [_pm(RUN_METRICS[m][1], a, lo, hi) for a, lo, hi in zip(rows[f"{m}_mean"], rows[f"{m}_ci_lo"], rows[f"{m}_ci_hi"], strict=True)]
            for m in SWEEP_METRICS
        })
        text[SWEEP_TENANT_METRIC] = [
            _pm(html.fmt_int, a, lo, hi)
            for a, lo, hi in zip(rows[f"{SWEEP_TENANT_METRIC}_mean"], rows[f"{SWEEP_TENANT_METRIC}_ci_lo"], rows[f"{SWEEP_TENANT_METRIC}_ci_hi"], strict=True)
        ]
        cols = [("series", "series", html.fmt_text), ("x", fam.x_name, charts.fmt_value), ("label", "label", html.fmt_text), ("n", "n", html.fmt_int)]
        cols += [(m, m.replace("_", " "), html.fmt_text) for m in SWEEP_METRICS]
        cols += [(SWEEP_TENANT_METRIC, f"{SWEEP_TENANT} queue p95 ms", html.fmt_text)]
        tbl = html.table(text, cols, left=1)
        parts.append(f"<h3>{escape(fam.name)}: {escape(fam.x_name)}</h3><div class=\"card\"><div class=\"grid\">{''.join(cs)}</div>{tbl}</div>")
    return "".join(parts)


def _margin_section(c: Campaign, fits: dict[str, LinearFit], mc: pd.DataFrame) -> str:
    if c.runs.empty:
        return '<p class="muted">no runs</p>'
    xmax = float(np.nanmax(c.runs["utilisation"])) if c.runs["utilisation"].notna().any() else 1.0
    points = {
        "shape": (c.runs["utilisation"].tolist(), c.runs["shape_rev_over_cost"].tolist()),
        "per-token": (c.runs["utilisation"].tolist(), c.runs["token_rev_over_cost"].tolist()),
    }
    lines = {}
    for name, key in (("shape", "shape"), ("per-token", "per_token")):
        fit = fits[key]
        if fit.n >= 2 and not np.isnan(fit.slope):
            lines[name] = ([0.0, xmax], [fit.y_at(0.0), fit.y_at(xmax)])
    chart = charts.scatter_chart(
        "Revenue ÷ GPU cost against utilisation, every run of every setting", points, lines=lines,
        x_label="utilisation (seat-seconds sold ÷ offered)", y_label="revenue ÷ GPU cost",
        subtitle="one dot per load run; lines: least-squares fits; dashed: break-even", x_fmt=_fmt_pct0, y_ref=1.0,
        y_ref_label="break-even",
    )
    fit_rows = mc[mc["kind"] == "fit"].assign(scheme=lambda d: d["scheme"].map(SCHEME_LABELS).fillna(d["scheme"]))
    tbl = html.table(
        fit_rows,
        [("scheme", "scheme", html.fmt_text), ("n", "runs", html.fmt_int), ("slope", "slope (per unit utilisation)", html.fmt_f(3)),
         ("intercept", "intercept", html.fmt_f(3)), ("r2", "r²", html.fmt_f(3)),
         ("breakeven_utilisation", "break-even utilisation", html.fmt_pct)],
    )
    return (
        f'<div class="card">{chart}<p class="muted">The fit pools every setting, so it is a campaign-wide tendency, not a '
        f"controlled curve; the load sweep above holds the setting fixed. Break-even is where the fitted line reaches 1.0.</p>{tbl}</div>"
    )


def _tenants_section(c: Campaign, tt: pd.DataFrame) -> str:
    if tt.empty:
        return '<p class="muted">no runs</p>'
    parts = []
    metrics = list(TENANT_METRICS)
    for label in c.manifest.settings:
        sub = tt[tt["label"] == label]
        if sub.empty:
            continue
        frame = _pm_frame(sub, ["label", "tenant"], metrics, TENANT_METRICS)
        cols = [("tenant", "tenant", html.fmt_text), ("n", "n", html.fmt_int)]
        cols += [(m, m.replace("_", " "), html.fmt_text) for m in metrics]
        parts.append(f"<h3>{escape(label)}</h3><div class=\"card\">{html.table(frame, cols)}</div>")
    return "".join(parts)


def _definitions_section() -> str:
    items = [f"<dt>{escape(m)}</dt><dd>{escape(d)}</dd>" for m, (d, _) in RUN_METRICS.items()]
    items += [f"<dt>{escape(m)} (tenant)</dt><dd>{escape(d)}</dd>" for m, (d, _) in TENANT_METRICS.items()
              if m not in RUN_METRICS or m == "tenant_usd"]
    items += [f"<dt>{escape(m)}</dt><dd>{escape(d)}</dd>" for m, (d, _) in PLAN_METRICS.items()]
    methods = [
        ("95% CI", "two-sided Student-t interval for the mean over the runs of a setting (one per seed); none when n < 2."),
        ("paired comparison", ("setting − baseline on the seeds both have; the t-interval, paired t-test and Wilcoxon "
                               "signed-rank test are on those differences.")),
        ("Hedges' g", "mean difference ÷ sd of the differences × (1 − 3 ÷ (4n − 5)): the standardised paired effect size."),
        ("p (Holm)", "Holm–Bonferroni step-down adjustment across the headline rows (headline pairs × headline metrics)."),
        ("CV", "sd ÷ mean of a tenant's bill across seeds under one billing scheme."),
        ("billed ÷ quote", "a request's shape bill ÷ its shape bound (the quote the tenant could have been shown)."),
        ("break-even utilisation", "where the least-squares line of revenue ÷ GPU cost against utilisation crosses 1.0."),
    ]
    items += [f"<dt>{escape(k)}</dt><dd>{escape(v)}</dd>" for k, v in methods]
    return f"<dl>{''.join(items)}</dl>"


# ------------------------------------------------------------------------------------------ build


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, lineterminator="\n")


def build_campaign(run_dirs: list[str | Path], manifest_path: str | Path, out_dir: str | Path,
                   config_dir: str | Path | None = None) -> Path:
    """Load the runs, compute every table, write the CSVs and campaign.html into `out_dir`."""
    manifest = load_manifest(manifest_path)
    c = load_campaign(run_dirs, manifest, config_dir)
    st = settings_table(c)
    tt = tenants_table(c)
    pt = paired_table(c)
    pred = predictability_table(c)
    sw = sweep_table(c)
    fits = margin_fits(c)
    mc = margin_table(c, fits)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    _write_csv(st, out / "settings.csv")
    _write_csv(tt, out / "tenants.csv")
    _write_csv(pt, out / "paired.csv")
    _write_csv(pred, out / "predictability.csv")
    _write_csv(sw, out / "sweep.csv")
    _write_csv(mc, out / "margin_curve.csv")
    runs_cols = ["label", "seed", "load_run", "run_dir", "started_at", "duration_s", *RUN_METRICS, *PLAN_METRICS]
    if not c.runs.empty:
        _write_csv(c.runs[runs_cols], out / "runs.csv")

    n_pairs = int(pt[["setting", "baseline"]].drop_duplicates().shape[0]) if not pt.empty else 0
    intro = (
        f'<p class="lead">{len(c.runs)} load run(s), {int(c.runs["label"].nunique()) if len(c.runs) else 0} of '
        f"{len(manifest.settings)} settings, {n_pairs} of {len(manifest.pairs)} paired comparisons with data. "
        f"Manifest: {escape(str(manifest_path))}.</p>"
    )
    sections = [
        ("overview", "Overview", _overview_section(c)),
        ("settings", "Settings", _settings_section(c, st)),
        ("paired", "Paired comparisons", _paired_section(c, pt)),
        ("predictability", "Predictability", _predictability_section(c, pred)),
        ("sweeps", "Sweeps", _sweeps_section(c, sw)),
        ("margin", "Margin curve", _margin_section(c, fits, mc)),
        ("tenants", "Tenants", _tenants_section(c, tt)),
        ("definitions", "Definitions", _definitions_section()),
    ]
    (out / "campaign.html").write_text(html.page("slotgw campaign", intro, sections), encoding="utf-8", newline="\n")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="statistics over many slotgw load runs: settings, paired comparisons, sweeps")
    p.add_argument("run_dirs", nargs="+", help="runs/<run_id> directories (each with meter.sqlite)")
    p.add_argument("--manifest", default="config/phase3/campaign.yaml", help="campaign manifest (labels, pairs, families)")
    p.add_argument("--out", required=True, help="output directory for the CSVs and campaign.html")
    p.add_argument("--config-dir", default=os.environ.get("SLOTGW_CONFIG_DIR", "config"),
                   help="fallback config when a run has no config snapshot")
    args = p.parse_args()
    out = build_campaign(args.run_dirs, args.manifest, args.out, args.config_dir)
    print(f"campaign report written to {out / 'campaign.html'}")


if __name__ == "__main__":
    main()
