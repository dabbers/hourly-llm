"""The scorecard and the re-pricing frontier (report/frontier.py, PHASE3.md section 9 A): bills under a
price table are the meters' bills re-priced, the criteria are the fixed ones, the frontier covers the grid."""

from __future__ import annotations

import numpy as np
import pytest

from report import frontier
from report.campaign import build_campaign, load_campaign, load_manifest

from .conftest import REPO
from .test_campaign import _write_run

MANIFEST = """
settings:
  burstoff: {desc: "on demand, bursting off"}
  burst: {desc: "on demand, bursting on"}
  resv-shared: {desc: "chat reserved"}
pairs:
  - [burst, burstoff]
  - [resv-shared, burst]
families: {}
headline:
  - [burst, burstoff]
"""


@pytest.fixture(scope="module")
def campaign(tmp_path_factory):
    root = tmp_path_factory.mktemp("frontier")
    dirs = []
    for label in ("burstoff", "burst", "resv-shared"):
        for seed in (1, 2, 3):
            d = root / f"{label}-s{seed}"
            _write_run(d, label, seed)
            dirs.append(d)
    manifest_path = root / "campaign.yaml"
    manifest_path.write_text(MANIFEST, encoding="utf-8")
    c = load_campaign(dirs, load_manifest(manifest_path), REPO / "config")
    return c, dirs, manifest_path, root


def _run_revenue(bills):
    return bills.groupby(["run_dir", "load_run"])["bill_usd"].sum()


def test_workloads_follow_the_scenario_the_manifest_overrides_and_the_pairs():
    import pandas as pd

    from report.campaign import Manifest

    runs = pd.DataFrame({
        "label": ["burst", "burstoff", "resv-p95", "load2-burst", "load2-burstoff", "flash-burst"],
        "scenario": ["mixed-4090+burst", "mixed-4090+burstoff", "mixed-4090-resv-p95+resv-p95", "mixed-4090-load2+load2-burst",
                     "mixed-4090-load2+load2-burstoff", "mixed-4090-flash+flash-burst"],
    })
    settings = dict.fromkeys(runs["label"], "")
    pairs = [("burst", "burstoff"), ("resv-p95", "burst"), ("load2-burst", "load2-burstoff")]

    class C:
        pass

    c = C()
    c.runs = runs
    # without an override the p95 scenario looks like another workload and has no baseline
    c.manifest = Manifest(settings, pairs, {}, [])
    wl = frontier.workloads(c)
    assert wl.baseline["burst"] == "burst" and wl.baseline["burstoff"] == "burst"
    assert wl.baseline["resv-p95"] is None
    assert wl.baseline["load2-burst"] == "load2-burstoff"  # its pair baseline of the same workload
    assert wl.baseline["load2-burstoff"] == "load2-burst"  # a pair's baseline is held against the pair's bursting setting
    assert wl.baseline["flash-burst"] is None  # another workload, no pair
    # the manifest says resv-p95 is the mixed-4090 workload with other reservation sizes
    c.manifest = Manifest(settings, pairs, {}, [], {"baseline": "burst", "workload_of": {"resv-p95": "mixed-4090"}})
    wl = frontier.workloads(c)
    assert wl.workload["resv-p95"] == "mixed-4090" and wl.baseline["resv-p95"] == "burst"


def test_the_campaign_table_reproduces_the_meters_bills(campaign):
    c, *_ = campaign
    f0 = float(c.runs["reservation_factor"].iloc[0])
    assert (c.runs["reservation_factor"] == f0).all()
    bills = frontier.tenant_bills(c, f0, 0.0)
    rev = _run_revenue(bills)
    want = c.runs.set_index(["run_dir", "load_run"])["tenant_usd"].reindex(rev.index)
    assert np.allclose(rev.to_numpy(), want.to_numpy(), atol=1e-9)
    assert (bills["credit_usd"] == 0).all()


def test_the_fee_scales_with_the_factor_and_only_the_fee(campaign):
    c, *_ = campaign
    f0 = float(c.runs["reservation_factor"].iloc[0])
    one = frontier.tenant_bills(c, f0, 0.0)
    two = frontier.tenant_bills(c, 2 * f0, 0.0)
    chat1 = one[(one["label"] == "resv-shared") & (one["tenant"] == "chat")].set_index("seed")
    chat2 = two[(two["label"] == "resv-shared") & (two["tenant"] == "chat")].set_index("seed")
    assert (chat1["on_demand_usd"] == 0).all()  # every chat request ran in the reservation: the bill is the fee alone
    assert np.allclose(chat2["fee_usd"], 2 * chat1["fee_usd"]) and np.allclose(chat2["bill_usd"], 2 * chat1["bill_usd"])
    others = (one["tenant"] != "chat") | (one["label"] != "resv-shared")
    assert np.allclose(one.loc[others, "bill_usd"], two.loc[others, "bill_usd"])  # on-demand bills do not move with the factor


def test_the_sla_credit_refunds_floor_missed_runtime_and_never_below_zero(campaign):
    c, *_ = campaign
    f0 = float(c.runs["reservation_factor"].iloc[0])
    base = frontier.credit_base(c)
    assert len(base) and (base > 0).all()
    none = frontier.tenant_bills(c, f0, 0.0).set_index(["label", "seed", "tenant"])
    full = frontier.tenant_bills(c, f0, 1.0).set_index(["label", "seed", "tenant"])
    half = frontier.tenant_bills(c, f0, 0.5).set_index(["label", "seed", "tenant"])
    # the agent tenant is delivered at 40 tps on a 60 tps shape in burstoff: every request misses the floor
    agent = none.index.get_level_values("tenant") == "agent"
    assert (full.loc[agent, "credit_usd"] > 0).all()
    assert np.allclose(half["credit_usd"], full["credit_usd"] / 2)
    assert (full["bill_usd"] <= none["bill_usd"] + 1e-12).all() and (full["bill_usd"] >= 0).all()
    runtime = none["on_demand_usd"] + none["fee_usd"]
    assert (full["credit_usd"] <= runtime + 1e-9).all()  # a credit is never more than what the tenant was billed


def test_scorecard_holds_every_setting_against_the_bursting_baseline(campaign):
    c, *_ = campaign
    sc = frontier.scorecard_table(c)
    assert set(sc["label"]) == {"burstoff", "burst", "resv-shared"}
    assert (sc["baseline"] == "burst").all()  # one workload: everything is held against burst, burst against itself
    burst = sc[sc["label"] == "burst"]
    assert (burst["pass"].dropna() == 1.0).all() and (burst["n"] == 3).all()
    resv = sc[(sc["label"] == "resv-shared") & (sc["tenant"] == "chat")]
    assert set(resv["scope"]) == {"reserved"}
    assert set(resv["metric"]) == {"floor_violation_share", "start_violation_share"}  # a reserved tenant is judged on service, not refusals
    agent = sc[(sc["label"] == "resv-shared") & (sc["tenant"] == "agent")]
    assert set(agent["metric"]) == {"floor_violation_share", "start_p95_s", "refusal_rate"}
    out = sc[sc["metric"] == "output_tokens_per_gpu_hour"]
    assert len(out) == 3 and (out["scope"] == "run").all()


def test_frontier_covers_the_grid_and_marks_the_campaign_table(campaign):
    c, *_ = campaign
    fr = frontier.frontier_table(c)
    f0 = float(c.runs["reservation_factor"].iloc[0])
    resv = fr[fr["label"] == "resv-shared"]
    grid = {(f, cr) for f in frontier.FACTORS for cr in frontier.CREDITS}
    assert set(zip(resv["factor"], resv["credit"], strict=True)) >= grid
    for label in ("burst", "burstoff"):
        sub = fr[fr["label"] == label]
        assert set(sub["credit"]) == set(frontier.CREDITS) and (sub["factor"] == f0).all()  # no reservation: only the credit varies
    assert (fr.groupby("label")["campaign_table"].sum() == 1).all()
    # the reserved chat tenant's bill is the fee alone: CV 0 at every factor, below its per-token CV
    assert (resv["reserved_tenants"] == "chat").all()
    assert np.allclose(resv["reserved_cv_max"], 0.0) and (resv["reserved_cv_per_token_max"] > 0).all()
    assert (resv["pass_reserved_cv"] == 1.0).all()
    # more fee, more revenue and margin, a dearer reservation relative to per-token
    no_credit = resv[resv["credit"] == 0.0].sort_values("factor")
    assert no_credit["revenue_usd"].is_monotonic_increasing and no_credit["gross_margin"].is_monotonic_increasing
    assert no_credit["reserved_bill_ratio_max"].is_monotonic_increasing
    # a credit lowers revenue at a fixed factor
    at_f0 = resv[np.isclose(resv["factor"], f0)].sort_values("credit")
    assert at_f0["revenue_usd"].is_monotonic_decreasing
    # the baseline's own row at the campaign table passes the margin criterion against itself
    b = fr[(fr["label"] == "burst") & fr["campaign_table"]].iloc[0]
    assert b["pass_margin"] == 1.0 and np.isclose(b["gross_margin"], b["baseline_gross_margin"])
    assert fr["pass_all"].isin([0.0, 1.0]).all()


def test_build_writes_the_scorecard_the_frontier_and_the_section(campaign):
    _c, dirs, manifest_path, root = campaign
    out = build_campaign(dirs, manifest_path, root / "out", REPO / "config")
    assert (out / "scorecard.csv").exists() and (out / "frontier.csv").exists()
    page = (out / "campaign.html").read_text(encoding="utf-8")
    assert "Scorecard and re-pricing frontier" in page and "frontier.csv" in page
    assert "fixed before any price was tried" in page
