"""The scorecard and the re-pricing frontier (docs/PHASE3.md section 9, item A).

Pricing does not change scheduling, so every price table's revenue is already in the campaign's
meters: a reservation fee scales with the reservation price factor, and an SLA credit refunds the
runtime price of a completed request whose decode floor was missed (in whole or in part). The pass
criteria below were fixed in PHASE3 section 9 before any price was tried; this module reports the
whole frontier (every setting under every table of the grid) and marks which rows pass, instead of
choosing a table. Offline re-pricing holds behaviour fixed: real tenants would change what they buy
and what they send if prices changed, which the meters cannot show.

Scorecard (per setting; "baseline" = the on-demand bursting setting of the same workload, paired
by seed; "per-token" = the same runs billed per token):
  reserved tenant   bill CV across seeds <= 2 % and below the tenant's per-token CV on the same runs;
                    start-bound and decode-floor violation shares not above the baseline's for the tenant
  on-demand tenant  start p95 <= 1.5 x the baseline's; refusal rate within one point of the baseline's
  provider          gross margin at the table >= the baseline's at the campaign table, and >= the
                    per-token margin of the same runs; output tokens per GPU-hour >= 95 % of the baseline's
  everyone          decode-floor violation share per tenant not above the baseline's
The reserved tenant's bill against its per-token bill is reported, not passed or failed.
"""

from __future__ import annotations

from dataclasses import dataclass
from html import escape

import numpy as np
import pandas as pd

from . import charts, html
from .stats import mean_ci

FACTORS: tuple[float, ...] = (1.0, 2.3, 3.2, 4.0, 5.0, 6.0)  # reservation price factors tried (x the shape's price per second)
CREDITS: tuple[float, ...] = (0.0, 0.5, 1.0)  # SLA credit: share of the runtime price refunded on a floor-missed request
CRITERIA = {
    "reserved_cv_max": 0.02,  # a reserved tenant's bill CV across seeds
    "on_demand_start_p95_ratio_max": 1.5,  # an on-demand tenant's start p95 against the baseline's
    "refusal_rate_delta_max": 0.01,  # an on-demand tenant's refusal rate against the baseline's
    "output_share_min": 0.95,  # output tokens per GPU-hour against the baseline's
}
BASELINE_LABEL = "burst"  # on-demand slot billing with bursting: the status quo every setting is held against

SCORECARD_COLUMNS = ["label", "baseline", "scope", "tenant", "metric", "n", "value", "baseline_value", "criterion", "pass"]
FRONTIER_COLUMNS = ["label", "baseline", "n", "factor", "credit", "campaign_table", "revenue_usd", "revenue_ci_lo", "revenue_ci_hi",
                    "gross_margin", "baseline_gross_margin", "per_token_gross_margin", "pass_margin",
                    "reserved_tenants", "reserved_cv_max", "reserved_cv_per_token_max", "reserved_bill_ratio_min",
                    "reserved_bill_ratio_max", "pass_reserved_cv", "pass_reserved_service", "pass_on_demand", "pass_output",
                    "pass_floor", "pass_all"]


@dataclass
class Workloads:
    """Which labels share a workload (the loadgen scenario without the label suffix) and which
    on-demand bursting label each one is held against."""

    workload: dict[str, str]
    baseline: dict[str, str | None]


def _workload(scenario) -> str:
    return scenario.rsplit("+", 1)[0].strip() if isinstance(scenario, str) and "+" in scenario else str(scenario)


def baseline_label(c) -> str:
    return str(getattr(c.manifest, "scorecard", {}).get("baseline") or BASELINE_LABEL)


def workloads(c) -> Workloads:
    """A label's workload is its loadgen scenario without the label suffix, unless the manifest's
    scorecard.workload_of says otherwise (resv-p95 runs mixed-4090 with other reservation sizes). Its
    baseline: the scorecard's baseline label when that ran the same workload; else its manifest pair
    baseline when that ran the same workload (load1.5-burst -> load1.5-burstoff); else, for a label
    that is itself a pair's baseline, that pair's setting of the same workload (load1.5-burstoff ->
    load1.5-burst, the workload's bursting setting); else none."""
    if c.runs.empty:
        return Workloads({}, {})
    overrides = getattr(c.manifest, "scorecard", {}).get("workload_of") or {}
    wl = {label: str(overrides.get(label) or _workload(sub["scenario"].iloc[0])) for label, sub in c.runs.groupby("label", sort=False)}
    bl = baseline_label(c)
    pairs = dict(c.manifest.pairs)
    settings_of = {}  # a pair's baseline -> the pair's setting (the first in manifest order)
    for setting, baseline in c.manifest.pairs:
        settings_of.setdefault(baseline, setting)
    base: dict[str, str | None] = {}
    for label, w in wl.items():
        if bl in wl and wl[bl] == w:
            base[label] = bl
        elif label in pairs and pairs[label] in wl and wl[pairs[label]] == w:
            base[label] = pairs[label]
        elif label in settings_of and settings_of[label] in wl and wl[settings_of[label]] == w:
            base[label] = settings_of[label]
        else:
            base[label] = None
    return Workloads(wl, base)


def reserved_tenants(c) -> dict[str, list[str]]:
    """label -> the tenants that hold a reservation (a fee) in every kept run of the label."""
    if c.tenants.empty or "reservation_usd" not in c.tenants:
        return {}
    fee = c.tenants.groupby(["label", "tenant"])["reservation_usd"].min()
    out: dict[str, list[str]] = {}
    for (label, tenant), v in fee.items():
        if v > 0:
            out.setdefault(label, []).append(tenant)
    return {k: sorted(v) for k, v in out.items()}


def credit_base(c) -> pd.Series:
    """Per (run key, tenant): the runtime price of the tenant's completed requests whose decode floor
    was missed, i.e. what a 100 % SLA credit refunds."""
    req = c.requests
    if req.empty or "floor_violation" not in req:
        return pd.Series(dtype=float)
    viol = req[req["floor_violation"].astype(bool)]
    runtime = (viol["shape_usd"] - viol.get("prompt_usd", 0.0)).clip(lower=0.0)
    key = viol["run_dir"] + "::" + viol["load_run"]
    return runtime.groupby([key, viol["tenant"]]).sum()


def tenant_bills(c, factor: float, credit: float, base: pd.Series | None = None) -> pd.DataFrame:
    """One row per kept run and tenant: the tenant's bill under the price table (reservation factor,
    SLA credit): on-demand shape bills + fee x factor / the run's factor - credit x the floor-missed
    runtime price, floored at zero; with the per-token bill of the same requests."""
    cols = ["label", "seed", "load_run", "run_dir", "tenant", "on_demand_usd", "fee_usd", "credit_usd", "bill_usd", "per_token_usd"]
    if c.tenants.empty:
        return pd.DataFrame(columns=cols)
    t = c.tenants[["label", "seed", "load_run", "run_dir", "tenant", "tenant_usd", "reservation_usd", "per_token_usd"]].copy()
    key = t["run_dir"] + "::" + t["load_run"]
    f0 = c.runs.assign(key=c.runs["run_dir"] + "::" + c.runs["load_run"]).set_index("key")["reservation_factor"]
    t["on_demand_usd"] = t["tenant_usd"] - t["reservation_usd"]
    t["fee_usd"] = t["reservation_usd"] * factor / key.map(f0).astype(float)
    base = credit_base(c) if base is None else base
    idx = pd.MultiIndex.from_arrays([key, t["tenant"]])
    t["credit_usd"] = credit * base.reindex(idx).fillna(0.0).to_numpy() if len(base) else 0.0
    t["bill_usd"] = (t["on_demand_usd"] + t["fee_usd"] - t["credit_usd"]).clip(lower=0.0)
    return t[cols].reset_index(drop=True)


def _common_seeds(c, label: str, baseline: str | None) -> list[int]:
    if baseline is None:
        return sorted(c.runs.loc[c.runs["label"] == label, "seed"].astype(int))
    a = set(c.runs.loc[c.runs["label"] == label, "seed"].astype(int))
    b = set(c.runs.loc[c.runs["label"] == baseline, "seed"].astype(int))
    return sorted(a & b)


def _tenant_mean(c, label: str, tenant: str, metric: str, seeds: list[int]) -> float:
    t = c.tenants
    sub = t[(t["label"] == label) & (t["tenant"] == tenant) & (t["seed"].astype(int).isin(seeds))]
    return float(sub[metric].mean()) if len(sub) else np.nan


def _run_mean(c, label: str, metric: str, seeds: list[int]) -> float:
    r = c.runs
    sub = r[(r["label"] == label) & (r["seed"].astype(int).isin(seeds))]
    return float(sub[metric].mean()) if len(sub) else np.nan


def _pass(cond) -> float:
    """1.0 pass, 0.0 fail, NaN when a side is missing (not assessable)."""
    if cond is None or (isinstance(cond, float) and np.isnan(cond)):
        return np.nan
    return 1.0 if bool(cond) else 0.0


def _le(a: float, b: float, tol: float = 1e-12) -> float:
    if np.isnan(a) or np.isnan(b):
        return np.nan
    return _pass(a <= b + tol)


def scorecard_table(c) -> pd.DataFrame:
    """scorecard.csv: the price-independent rows of the scorecard, one per (label, tenant, metric), each
    against the label's baseline on their common seeds: a reserved tenant's start-bound and decode-floor
    violation shares, an on-demand tenant's start p95 and refusal rate, every tenant's floor share, and
    the run's output tokens per GPU-hour. `pass` is 1, 0 or NaN (no baseline, or no common seed)."""
    if c.runs.empty:
        return pd.DataFrame(columns=SCORECARD_COLUMNS)
    wl = workloads(c)
    reserved = reserved_tenants(c)
    crit = CRITERIA
    rows = []
    for label in c.manifest.settings:
        if label not in wl.baseline:
            continue
        base = wl.baseline[label]
        seeds = _common_seeds(c, label, base)
        n = len(seeds)
        tenants = sorted(c.tenants.loc[c.tenants["label"] == label, "tenant"].unique())
        for tenant in tenants:
            role = "reserved" if tenant in reserved.get(label, []) else "on-demand"
            for metric in ("floor_violation_share", "start_violation_share", "start_p95_s", "refusal_rate"):
                if metric == "refusal_rate":
                    val = _tenant_refusal_rate(c, label, tenant, seeds)
                    bval = _tenant_refusal_rate(c, base, tenant, seeds) if base else np.nan
                else:
                    val = _tenant_mean(c, label, tenant, metric, seeds)
                    bval = _tenant_mean(c, base, tenant, metric, seeds) if base else np.nan
                if metric == "floor_violation_share":
                    criterion, ok = "not above the baseline's", _le(val, bval)
                elif metric == "start_violation_share":
                    if role != "reserved":
                        continue
                    criterion, ok = "not above the baseline's (reserved tenant)", _le(val, bval)
                elif metric == "start_p95_s":
                    if role == "reserved":
                        continue
                    criterion, ok = f"<= {crit['on_demand_start_p95_ratio_max']:g} x the baseline's", _le(val, crit["on_demand_start_p95_ratio_max"] * bval)
                else:
                    if role == "reserved":
                        continue
                    criterion, ok = f"within {crit['refusal_rate_delta_max'] * 100:g} point(s) of the baseline's", _le(val, bval + crit["refusal_rate_delta_max"])
                rows.append({"label": label, "baseline": base, "scope": role, "tenant": tenant, "metric": metric, "n": n,
                             "value": val, "baseline_value": bval, "criterion": criterion, "pass": ok})
        val = _run_mean(c, label, "output_tokens_per_gpu_hour", seeds)
        bval = _run_mean(c, base, "output_tokens_per_gpu_hour", seeds) if base else np.nan
        rows.append({"label": label, "baseline": base, "scope": "run", "tenant": "", "metric": "output_tokens_per_gpu_hour", "n": n,
                     "value": val, "baseline_value": bval, "criterion": f">= {crit['output_share_min'] * 100:g} % of the baseline's",
                     "pass": _pass(val >= crit["output_share_min"] * bval - 1e-9) if not (np.isnan(val) or np.isnan(bval)) else np.nan})
    return pd.DataFrame(rows, columns=SCORECARD_COLUMNS)


def _tenant_refusal_rate(c, label: str | None, tenant: str, seeds: list[int]) -> float:
    if label is None:
        return np.nan
    t = c.tenants
    sub = t[(t["label"] == label) & (t["tenant"] == tenant) & (t["seed"].astype(int).isin(seeds))]
    req = float(sub["requests"].sum())
    return float(sub["refused"].sum()) / req if req else np.nan


def _service_passes(sc: pd.DataFrame, label: str) -> dict[str, float]:
    """The label's price-independent passes folded per party: all reserved tenants' service rows, all
    on-demand tenants' rows, the output row, every tenant's floor row. NaN when nothing was assessable."""
    sub = sc[sc["label"] == label]

    def fold(mask) -> float:
        vals = sub.loc[mask, "pass"].dropna()
        if vals.empty:
            return np.nan
        return 1.0 if bool((vals > 0.5).all()) else 0.0

    res = (sub["scope"] == "reserved") & sub["metric"].isin(["start_violation_share", "floor_violation_share"])
    od = (sub["scope"] == "on-demand") & sub["metric"].isin(["start_p95_s", "refusal_rate"])
    return {
        "pass_reserved_service": fold(res),
        "pass_on_demand": fold(od),
        "pass_output": fold(sub["metric"] == "output_tokens_per_gpu_hour"),
        "pass_floor": fold(sub["metric"] == "floor_violation_share"),
    }


def _cv(values: pd.Series) -> float:
    mean, sd, n, _lo, _hi = mean_ci(values)
    return sd / mean if n >= 2 and mean else np.nan


def frontier_table(c, sc: pd.DataFrame | None = None, factors: tuple[float, ...] = FACTORS, credits: tuple[float, ...] = CREDITS) -> pd.DataFrame:
    """frontier.csv: one row per setting and price table (reservation factor x SLA credit; a setting
    without a reservation only varies the credit, its factor column is the campaign's). Revenue, the
    provider's gross margin against the baseline's at the campaign table and against the per-token
    margin of the same runs, the reserved tenants' bill CV (max over them) and their bill against
    per-token (min and max over them), and the pass columns; pass_all folds every criterion."""
    if c.runs.empty:
        return pd.DataFrame(columns=FRONTIER_COLUMNS)
    sc = scorecard_table(c) if sc is None else sc
    wl = workloads(c)
    reserved = reserved_tenants(c)
    base_credit = credit_base(c)
    runs = c.runs.assign(key=c.runs["run_dir"] + "::" + c.runs["load_run"])
    f0_by_label = runs.groupby("label")["reservation_factor"].first()

    def margins(bills: pd.DataFrame, label: str, seeds: list[int]) -> tuple[pd.Series, pd.Series]:
        """Per seed: revenue and gross margin of the label's kept runs under the bills frame."""
        sub = bills[(bills["label"] == label) & (bills["seed"].astype(int).isin(seeds))]
        rev = sub.groupby(["run_dir", "load_run"])["bill_usd"].sum()
        cost = runs.set_index(["run_dir", "load_run"])["gpu_cost_usd"].reindex(rev.index)
        with np.errstate(divide="ignore", invalid="ignore"):
            margin = 1.0 - cost / rev
        return rev, margin.replace([np.inf, -np.inf], np.nan)

    campaign_bills: dict[str, pd.DataFrame] = {}
    rows = []
    for label in c.manifest.settings:
        if label not in wl.baseline:
            continue
        base = wl.baseline[label]
        seeds = _common_seeds(c, label, base)
        f0 = float(f0_by_label.get(label, np.nan))
        res_tenants = reserved.get(label, [])
        tables = [(f, cr) for f in factors for cr in credits] if res_tenants else [(f0, cr) for cr in credits]
        if (f0, 0.0) not in tables:
            tables.insert(0, (f0, 0.0))
        service = _service_passes(sc, label)
        b_margin = np.nan
        if base is not None:
            if base not in campaign_bills:
                campaign_bills[base] = tenant_bills(c, float(f0_by_label.get(base, f0)), 0.0, base_credit)
            _rev, bm = margins(campaign_bills[base], base, seeds)
            b_margin = float(bm.mean()) if len(bm) else np.nan
        pt_margin = _run_mean(c, label, "gross_margin_token", seeds)
        for f, cr in tables:
            bills = tenant_bills(c, f, cr, base_credit)
            rev, margin = margins(bills, label, seeds)
            rev_mean, _sd, n, lo, hi = mean_ci(rev)
            gm = float(margin.mean()) if len(margin) else np.nan
            cvs, cvs_pt, ratios = [], [], []
            for tenant in res_tenants:
                tb = bills[(bills["label"] == label) & (bills["tenant"] == tenant) & (bills["seed"].astype(int).isin(seeds))]
                cvs.append(_cv(tb["bill_usd"]))
                cvs_pt.append(_cv(tb["per_token_usd"]))
                pt = float(tb["per_token_usd"].mean()) if len(tb) else np.nan
                ratios.append(float(tb["bill_usd"].mean()) / pt if pt else np.nan)
            cv_max = float(np.nanmax(cvs)) if cvs and not all(np.isnan(cvs)) else np.nan
            cv_pt_max = float(np.nanmax(cvs_pt)) if cvs_pt and not all(np.isnan(cvs_pt)) else np.nan
            if res_tenants:
                pass_cv = np.nan if np.isnan(cv_max) else _pass(
                    cv_max <= CRITERIA["reserved_cv_max"] and all(
                        (not np.isnan(a) and not np.isnan(b) and a < b) for a, b in zip(cvs, cvs_pt, strict=True)))
            else:
                pass_cv = np.nan
            pass_margin = np.nan if np.isnan(gm) or (np.isnan(b_margin) and np.isnan(pt_margin)) else _pass(
                (np.isnan(b_margin) or gm >= b_margin - 1e-9) and (np.isnan(pt_margin) or gm >= pt_margin - 1e-9))
            parts = [pass_margin, pass_cv, *service.values()]
            assessed = [p for p in parts if not np.isnan(p)]
            pass_all = np.nan if not assessed else (1.0 if all(p > 0.5 for p in assessed) else 0.0)
            rows.append({
                "label": label, "baseline": base, "n": n, "factor": f, "credit": cr,
                "campaign_table": bool(abs(f - f0) < 1e-9 and cr == 0.0),
                "revenue_usd": rev_mean, "revenue_ci_lo": lo, "revenue_ci_hi": hi,
                "gross_margin": gm, "baseline_gross_margin": b_margin, "per_token_gross_margin": pt_margin, "pass_margin": pass_margin,
                "reserved_tenants": ",".join(res_tenants), "reserved_cv_max": cv_max, "reserved_cv_per_token_max": cv_pt_max,
                "reserved_bill_ratio_min": float(np.nanmin(ratios)) if ratios and not all(np.isnan(ratios)) else np.nan,
                "reserved_bill_ratio_max": float(np.nanmax(ratios)) if ratios and not all(np.isnan(ratios)) else np.nan,
                "pass_reserved_cv": pass_cv, **service, "pass_all": pass_all,
            })
    return pd.DataFrame(rows, columns=FRONTIER_COLUMNS)


# ------------------------------------------------------------------------------------------- html


def _fmt_pass(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "–"
    return "pass" if float(v) > 0.5 else "FAIL"


def _fmt_ratio(v) -> str:
    return "–" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{float(v):.2f}×"


def _fmt_factor(v) -> str:
    return "–" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{float(v):g}×"


def _fmt_credit(v) -> str:
    return "–" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{float(v) * 100:.0f}%"


def frontier_section(c, sc: pd.DataFrame, fr: pd.DataFrame) -> str:
    """The HTML section: the criteria as fixed, the service scorecard per setting, the frontier (the
    campaign table's row and every passing row per setting; the whole grid is frontier.csv) and a chart
    of the provider's margin against the reserved tenants' bill ratio over the factor grid."""
    if fr.empty:
        return '<p class="muted">no runs</p>'
    crit = CRITERIA
    criteria = (
        "<h3>The scorecard, fixed before any price was tried (PHASE3.md section 9)</h3>"
        '<div class="tblwrap"><table><thead><tr><th class="l">party</th><th class="l">metric</th><th class="l">pass</th></tr></thead><tbody>'
        f'<tr><td class="l">reserved tenant</td><td class="l">bill CV across seeds</td><td class="l">at most {crit["reserved_cv_max"] * 100:g} %, and below the tenant\'s per-token CV on the same runs</td></tr>'
        '<tr><td class="l">reserved tenant</td><td class="l">start-bound and decode-floor violation shares</td><td class="l">not above the baseline\'s for the same tenant</td></tr>'
        f'<tr><td class="l">on-demand tenants</td><td class="l">start p95; refusal rate</td><td class="l">at most {crit["on_demand_start_p95_ratio_max"]:g} × the baseline\'s; within {crit["refusal_rate_delta_max"] * 100:g} point of the baseline\'s</td></tr>'
        '<tr><td class="l">provider</td><td class="l">gross margin at the price table</td><td class="l">at least the baseline\'s at the campaign table, and at least the per-token margin of the same runs</td></tr>'
        f'<tr><td class="l">provider</td><td class="l">output tokens per GPU-hour</td><td class="l">at least {crit["output_share_min"] * 100:g} % of the baseline\'s</td></tr>'
        '<tr><td class="l">everyone</td><td class="l">decode-floor violation share per tenant</td><td class="l">not above the baseline\'s</td></tr>'
        "</tbody></table></div>"
        f'<p class="muted">baseline = <code>{escape(baseline_label(c))}</code> (on-demand slot billing with bursting) for settings of the same '
        "workload, else the setting's manifest pair of the same workload; every comparison is paired by seed on the common seeds. "
        "A price table moves the provider's margin and the reserved tenants' bills; it cannot move anyone's waiting time or floor. "
        "The reserved tenant's bill against its per-token bill is reported, not passed or failed. Offline re-pricing holds "
        "behaviour fixed: real tenants would change what they buy and send if prices changed, which these meters cannot show.</p>"
    )
    # the service scorecard, one row per setting
    labels = list(dict.fromkeys(fr["label"]))
    svc_rows = []
    for label in labels:
        sub = sc[sc["label"] == label]
        row = {"label": label, "baseline": fr.loc[fr["label"] == label, "baseline"].iloc[0] or "–",
               "n": int(fr.loc[fr["label"] == label, "n"].iloc[0])}
        row.update(_service_passes(sc, label))
        od = sub[(sub["scope"] == "on-demand") & (sub["metric"] == "start_p95_s")]
        with np.errstate(divide="ignore", invalid="ignore"):
            row["start_ratio_max"] = float((od["value"] / od["baseline_value"]).max()) if len(od) else np.nan
        out = sub[sub["metric"] == "output_tokens_per_gpu_hour"]
        with np.errstate(divide="ignore", invalid="ignore"):
            row["output_share"] = float((out["value"] / out["baseline_value"]).iloc[0]) if len(out) else np.nan
        fl = sub[sub["metric"] == "floor_violation_share"]
        row["floor_worst_delta"] = float((fl["value"] - fl["baseline_value"]).max()) if len(fl) else np.nan
        svc_rows.append(row)
    svc = pd.DataFrame(svc_rows)
    t_service = html.table(
        svc,
        [("label", "setting", html.fmt_text), ("baseline", "baseline", html.fmt_text), ("n", "n", html.fmt_int),
         ("pass_reserved_service", "reserved: start + floor", _fmt_pass), ("start_ratio_max", "on-demand start p95 ÷ baseline (worst)", _fmt_ratio),
         ("pass_on_demand", "on-demand: start + refusals", _fmt_pass), ("output_share", "output/GPU-h ÷ baseline", html.fmt_pct),
         ("pass_output", "output", _fmt_pass), ("floor_worst_delta", "floor share − baseline (worst tenant)", html.fmt_pct),
         ("pass_floor", "floor", _fmt_pass)],
        left=2,
    )
    # the frontier: the campaign table's row plus every passing row per setting
    show = fr[fr["campaign_table"] | (fr["pass_all"] > 0.5)].copy()
    t_frontier = html.table(
        show,
        [("label", "setting", html.fmt_text), ("factor", "reservation factor", _fmt_factor), ("credit", "SLA credit", _fmt_credit),
         ("n", "n", html.fmt_int), ("revenue_usd", "revenue per run", html.fmt_usd), ("gross_margin", "gross margin", html.fmt_pct),
         ("baseline_gross_margin", "baseline margin", html.fmt_pct), ("per_token_gross_margin", "per-token margin", html.fmt_pct),
         ("pass_margin", "margin", _fmt_pass), ("reserved_cv_max", "reserved CV (worst)", html.fmt_pct),
         ("reserved_bill_ratio_min", "reserved bill ÷ per-token (min)", _fmt_ratio), ("reserved_bill_ratio_max", "(max)", _fmt_ratio),
         ("pass_reserved_cv", "reserved CV", _fmt_pass), ("pass_all", "all criteria", _fmt_pass)],
        left=1,
    )
    n_pass = int((fr["pass_all"] > 0.5).sum())
    n_settings_pass = int(fr.loc[fr["pass_all"] > 0.5, "label"].nunique())
    # chart: provider margin against the reserved tenants' bill ratio (max over them), one series per reserved setting, credit 0
    points: dict[str, tuple[list[float], list[float]]] = {}
    for label in labels:
        sub = fr[(fr["label"] == label) & (fr["credit"] == 0.0) & (fr["reserved_tenants"] != "")]
        if len(sub) >= 2:
            points[label] = (sub["reserved_bill_ratio_max"].tolist(), (sub["gross_margin"] * 100).tolist())
    chart = ""
    if points:
        b_margin = fr.loc[(fr["label"] == baseline_label(c)) & fr["campaign_table"], "gross_margin"]
        y_ref = float(b_margin.iloc[0] * 100) if len(b_margin) and not np.isnan(b_margin.iloc[0]) else None
        chart = charts.scatter_chart(
            "Provider margin against the reserved tenant's bill, over the reservation price factor",
            points, lines=points,
            subtitle=f"each point a reservation factor ({', '.join(f'{f:g}' for f in FACTORS)}), no SLA credit; x = the worst reserved tenant's bill ÷ its per-token bill",
            x_label="reserved tenant's bill ÷ per-token bill", y_label="gross margin, %", fmt=lambda v: f"{v:.0f}%", x_fmt=lambda v: f"{v:.1f}×",
            y_ref=y_ref, y_ref_label=f"{baseline_label(c)} margin at the campaign table",
        )
    return (
        f'<div class="card">{criteria}</div>'
        f'<div class="card"><h3>Service, price-independent: each setting against its baseline</h3>{t_service}</div>'
        f'<div class="card"><h3>The frontier: the campaign table and every passing table per setting</h3>'
        f'<p class="muted">{n_pass} of {len(fr)} (setting, price table) rows pass every criterion, in {n_settings_pass} setting(s); '
        f"the whole grid is <code>frontier.csv</code>, the rows behind the service passes <code>scorecard.csv</code>. "
        f"campaign table = the factor the runs were billed at and no credit.</p>{t_frontier}"
        f'{"<div class=grid>" + chart + "</div>" if chart else ""}</div>'
    )
