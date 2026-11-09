from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats as sps

from report import stats
from report.campaign import build_campaign, label_of, load_manifest
from slotgw.meter import Meter, RequestRecord

REPO = Path(__file__).resolve().parents[1]
MANIFEST = REPO / "config" / "phase3" / "campaign.yaml"


# ----------------------------------------------------------------------------------------- stats


def test_mean_ci_matches_scipy_and_handles_small_n():
    vals = [10.0, 12.0, 11.0, 14.0]
    mean, sd, n, lo, hi = stats.mean_ci(vals)
    ref = sps.t.interval(0.95, len(vals) - 1, loc=np.mean(vals), scale=sps.sem(vals))
    assert (mean, sd, n) == (pytest.approx(11.75), pytest.approx(np.std(vals, ddof=1)), 4)
    assert (lo, hi) == (pytest.approx(ref[0]), pytest.approx(ref[1]))
    mean1, sd1, n1, lo1, hi1 = stats.mean_ci([3.0, float("nan")])
    assert (mean1, n1) == (3.0, 1) and math.isnan(sd1) and math.isnan(lo1) and math.isnan(hi1)
    assert stats.mean_ci([])[2] == 0 and math.isnan(stats.mean_ci([])[0])


def test_paired_matches_scipy_on_a_tiny_data_set():
    a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]  # baseline
    b = [1.5, 2.1, 3.8, 4.4, 5.9, 6.3]  # setting
    r = stats.paired(a, b)
    d = np.array(b) - np.array(a)
    assert r["n"] == 6 and r["mean_a"] == pytest.approx(3.5) and r["mean_b"] == pytest.approx(np.mean(b))
    assert r["diff"] == pytest.approx(d.mean()) and r["rel_diff"] == pytest.approx(d.mean() / 3.5)
    assert r["p_t"] == pytest.approx(sps.ttest_rel(b, a).pvalue)
    assert r["p_wilcoxon"] == pytest.approx(sps.wilcoxon(d).pvalue)
    half = sps.t.ppf(0.975, 5) * d.std(ddof=1) / math.sqrt(6)
    assert (r["diff_lo"], r["diff_hi"]) == (pytest.approx(d.mean() - half), pytest.approx(d.mean() + half))
    j = 1 - 3 / (4 * 6 - 5)
    assert r["hedges_g"] == pytest.approx(d.mean() / d.std(ddof=1) * j)
    # degenerate inputs: one pair, all-zero differences, NaN pairs dropped
    one = stats.paired([1.0], [2.0])
    assert one["n"] == 1 and one["diff"] == 1.0 and math.isnan(one["p_t"]) and math.isnan(one["diff_lo"])
    zero = stats.paired([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert zero["diff"] == 0 and math.isnan(zero["p_t"]) and math.isnan(zero["p_wilcoxon"]) and math.isnan(zero["hedges_g"])
    dropped = stats.paired([1.0, float("nan"), 3.0], [2.0, 5.0, 4.0])
    assert dropped["n"] == 2 and dropped["diff"] == pytest.approx(1.0)
    with pytest.raises(ValueError):
        stats.paired([1.0, 2.0], [1.0])


def test_holm_against_hand_computed_values():
    # m = 3 non-NaN: 0.01×3 = 0.03; 0.03×2 = 0.06; 0.04×1 = 0.04 -> running max 0.06
    assert stats.holm([0.01, 0.04, 0.03, float("nan")]) == pytest.approx([0.03, 0.06, 0.06, float("nan")], nan_ok=True)
    assert stats.holm([0.5, 0.9]) == pytest.approx([1.0, 1.0])  # 0.5×2 = 1.0 caps and carries to the next rank
    assert stats.holm([]) == []
    assert all(math.isnan(v) for v in stats.holm([float("nan"), float("nan")]))


def test_linear_fit_and_x_at():
    fit = stats.linear_fit([0.1, 0.2, 0.3, 0.4], [0.5, 1.0, 1.5, 2.0])  # y = 5x
    assert fit.slope == pytest.approx(5.0) and fit.intercept == pytest.approx(0.0, abs=1e-12)
    assert fit.r2 == pytest.approx(1.0) and fit.n == 4
    assert fit.x_at(1.0) == pytest.approx(0.2) and fit.y_at(0.2) == pytest.approx(1.0)
    flat = stats.linear_fit([1.0, 2.0, 3.0], [2.0, 2.0, 2.0])
    assert flat.slope == pytest.approx(0.0, abs=1e-12) and math.isnan(flat.x_at(1.0)) and math.isnan(flat.r2)
    assert stats.linear_fit([1.0], [2.0]).n == 1 and math.isnan(stats.linear_fit([1.0], [2.0]).slope)


# -------------------------------------------------------------------------------------- campaign


def test_label_of_and_manifest():
    assert label_of("mixed-4090+burstoff") == "burstoff"
    assert label_of("mixed-4090-load0.5+load0.5-burst") == "load0.5-burst"
    assert label_of("synthetic") is None and label_of(None) is None
    m = load_manifest(MANIFEST)
    assert "burst" in m.settings and ("burst", "burstoff") in m.pairs
    assert set(m.families) == {"load", "overcommit", "chunk"}
    assert m.families["load"].series["burst"][1] == (1.0, "burst")
    assert set(m.headline) <= set(m.pairs)


def _row(i: int, load_run: str, scenario: str, seed: int, tenant: str, shape: str, delivery: str, t: float,
         prompt: int, out: int, ttft: float, engine_rate: float, deliver_rate: float, status: str = "ok",
         lazy: bool = False) -> RequestRecord:
    rec = RequestRecord(
        request_id=f"chatcmpl-{load_run}-{i}", tenant=tenant, shape=shape, lazy=lazy, delivery=delivery, t_arrival=t,
        replica="local-2080ti", load_run=load_run, scenario=scenario, seed=seed, prompt_tokens=prompt,
    )
    rec.t_admitted = t + 0.001
    if status != "ok":
        rec.status, rec.http_status, rec.abort_reason = status, 503, "capacity_tps"
        return rec
    rec.t_first_token = rec.t_admitted + ttft
    rec.t_engine_done = rec.t_first_token + out / engine_rate
    rec.t_last_token = rec.t_first_token + out / deliver_rate
    rec.output_tokens = out
    rec.output_tokens_guaranteed = out
    rec.throttled_ms = max(0.0, (rec.t_last_token - rec.t_engine_done)) * 1000.0
    return rec


def _write_run(run_dir: Path, label: str, seed: int, started_at: str = "2026-11-08T00:00:00+00:00") -> None:
    """One synthetic gateway run with one load run: scenario 'mixed-4090+<label>'. The burst label
    delivers chat faster than the shape (a burst discount) and has fewer floor violations."""
    scenario = f"mixed-4090+{label}"
    load_run = f"mixed-4090-{label}-s{seed}"
    meter = Meter(run_dir / "meter.sqlite", run_id=f"{label}-s{seed}")
    t0 = 1_700_000_000.0 + seed * 10_000
    burst = label == "burst"
    i = 0
    for k in range(12):
        t = t0 + k * 7.0
        # chat: paced at 30 tps when bursting is off, 40 tps delivered when it is on (burst discount)
        meter.record(_row(i, load_run, scenario, seed, "chat", "small.8k.30tps", "stream", t, 600, 150 + 10 * seed, 0.3,
                          120, 40 if burst else 30)); i += 1
        # agent: 60 tps shape; in burst runs the engine keeps up on the even requests after the first 2·seed − 1
        slow = (not burst) or (k % 2 == 1) or (k < 2 * seed - 1)
        meter.record(_row(i, load_run, scenario, seed, "agent", "small.8k.60tps", "stream", t + 1, 1500, 200, 0.4,
                          40 if slow else 70, 40 if slow else 60)); i += 1
        meter.record(_row(i, load_run, scenario, seed, "batch", "small.8k.15tps.lazy", "complete", t + 2, 1000, 300, 0.5,
                          100, 18, lazy=True)); i += 1
    meter.record(_row(i, load_run, scenario, seed, "agent", "small.8k.60tps", "stream", t0 + 50.0, 8000, 300, 0, 1, 1,
                      status="refused")); i += 1
    meter.close()
    d = run_dir / "loads" / load_run
    d.mkdir(parents=True)
    (d / "loadrun.json").write_text(json.dumps({
        "load_run": load_run, "scenario": scenario, "label": label, "seed": seed, "started_at": started_at,
        "duration_s": 90, "gateway_pacing": {"enabled": True, "bucket_depth_s": 1.0},
    }), encoding="utf-8")
    with open(d / "samples.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["t", "replica", "kv_usage", "running", "waiting", "preemptions", "client_inflight"])
        for s in range(0, 90, 2):
            w.writerow([t0 + s, "local-2080ti", 0.1 + 0.005 * s, 3, 0, 0, 3])


def test_campaign_end_to_end(tmp_path):
    dirs = []
    for label in ("burstoff", "burst"):
        for seed in (1, 2, 3):
            d = tmp_path / f"{label}-s{seed}"
            _write_run(d, label, seed)
            dirs.append(d)
    # a duplicate (burst, seed 1) that started later replaces the first; an unknown label is listed and ignored
    dup = tmp_path / "burst-s1-again"
    _write_run(dup, "burst", 1, started_at="2026-11-08T06:00:00+00:00")
    odd = tmp_path / "odd-s1"
    _write_run(odd, "weird", 1)
    out = build_campaign([*dirs, dup, odd], MANIFEST, tmp_path / "campaign", config_dir=REPO / "config")

    for name in ("settings.csv", "tenants.csv", "paired.csv", "predictability.csv", "sweep.csv", "margin_curve.csv",
                 "runs.csv", "campaign.html"):
        assert (out / name).is_file(), name

    st = pd.read_csv(out / "settings.csv")
    assert set(st["label"]) == {"burstoff", "burst"}
    assert (st["n"] == 3).all()
    util = st[st["metric"] == "utilisation"].set_index("label")["mean"]
    assert (util > 0).all() and (util < 1).all()
    gm = st[(st["metric"] == "gross_margin_shape") & (st["label"] == "burst")].iloc[0]
    assert gm["ci_lo"] <= gm["mean"] <= gm["ci_hi"]
    runs = pd.read_csv(out / "runs.csv")
    assert len(runs) == 6 and runs[(runs["label"] == "burst") & (runs["seed"] == 1)]["run_dir"].iloc[0] == str(dup)

    pt = pd.read_csv(out / "paired.csv")
    row = pt[(pt["setting"] == "burst") & (pt["baseline"] == "burstoff") & (pt["scope"] == "run") & (pt["metric"] == "tenant_usd")]
    assert len(row) == 1 and row.iloc[0]["n"] == 3
    r = row.iloc[0]
    assert r["diff"] < 0 and r["diff_ci_lo"] <= r["diff"] <= r["diff_ci_hi"]  # bursting gives runtime away
    assert r["headline"] and not np.isnan(r["p_holm"]) and r["p_holm"] >= r["p_t"]
    fv = pt[(pt["setting"] == "burst") & (pt["scope"] == "run") & (pt["metric"] == "floor_violation_rate")].iloc[0]
    assert fv["diff"] < 0 and fv["p_t"] < 0.05  # 5, 4, 3 fewer violating agent requests in seeds 1-3
    assert np.isnan(pt[pt["scope"] == "tenant"]["p_holm"]).all()
    chat = pt[(pt["scope"] == "tenant") & (pt["tenant"] == "chat") & (pt["metric"] == "tenant_usd")].iloc[0]
    assert chat["n"] == 3 and chat["diff"] < 0
    assert set(pt["setting"]) == {"burst"}  # only pairs with common seeds appear

    pred = pd.read_csv(out / "predictability.csv")
    assert set(pred["scheme"]) == {"per_token", "shape"} and (pred["n"] == 3).all()
    chat_q = pred[(pred["label"] == "burst") & (pred["tenant"] == "chat")].iloc[0]
    assert chat_q["quote_ratio_p50"] < 1 and chat_q["share_at_quote"] == 0  # every burst chat request got a discount
    off_q = pred[(pred["label"] == "burstoff") & (pred["tenant"] == "chat")].iloc[0]
    assert off_q["share_at_quote"] == pytest.approx(1.0)

    sw = pd.read_csv(out / "sweep.csv")
    assert set(sw["family"]) == {"load", "overcommit", "chunk"}
    base = sw[(sw["family"] == "load") & (sw["series"] == "burst") & (sw["x"] == 1.0)].iloc[0]
    assert base["label"] == "burst" and base["n"] == 3 and base["tenant_usd_mean"] > 0
    assert (sw[sw["label"] == "load2-burst"]["n"] == 0).all()

    mc = pd.read_csv(out / "margin_curve.csv")
    assert (mc["kind"] == "point").sum() == 6 and set(mc.loc[mc["kind"] == "fit", "scheme"]) == {"shape", "per_token"}

    page = (out / "campaign.html").read_text(encoding="utf-8")
    assert "burst vs burstoff" in page
    for word in ("Overview", "Settings", "Paired comparisons", "Predictability", "Sweeps", "Margin curve", "Tenants"):
        assert word in page
    assert "weird" in page and "Unassigned" in page and "Duplicate" in page
    assert page.count("<svg") >= 6 and "<script" not in page
