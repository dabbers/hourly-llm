from __future__ import annotations

import csv
import json
from pathlib import Path

import pandas as pd
import pytest

from report.build import build_report
from slotgw.config import load_config
from slotgw.meter import Meter, RequestRecord

REPO = Path(__file__).resolve().parents[1]


def _row(i: int, load_run: str, seed: int, tenant: str, shape: str, delivery: str, t: float, prompt: int, out: int,
         ttft: float, engine_rate: float, deliver_rate: float, status: str = "ok", lazy: bool = False) -> RequestRecord:
    rec = RequestRecord(
        request_id=f"chatcmpl-{load_run}-{i}", tenant=tenant, shape=shape, lazy=lazy, delivery=delivery, t_arrival=t,
        replica="local-2080ti", load_run=load_run, scenario="synthetic", seed=seed, prompt_tokens=prompt,
    )
    rec.t_admitted = t + 0.001
    if status != "ok":
        rec.status, rec.http_status, rec.abort_reason = status, 429, "context_exceeds_shape"
        return rec
    rec.t_first_token = rec.t_admitted + ttft
    rec.t_engine_done = rec.t_first_token + out / engine_rate
    rec.t_last_token = rec.t_first_token + out / deliver_rate
    rec.output_tokens = out
    rec.output_tokens_guaranteed = out
    rec.throttled_ms = max(0.0, (rec.t_last_token - rec.t_engine_done)) * 1000.0
    return rec


def _write_run(run_dir: Path) -> None:
    meter = Meter(run_dir / "meter.sqlite", run_id="synthetic-run")
    base = 1_700_000_000.0
    i = 0
    for seed, load_run in ((1, "synthetic-s1"), (2, "synthetic-s2")):
        t0 = base + seed * 10_000
        for k in range(12):
            t = t0 + k * 7.0
            # chat: streamed, engine faster than the shape -> paced at 30 tps exactly after the head start
            meter.record(_row(i, load_run, seed, "chat", "small.8k.30tps", "stream", t, 600, 150 + 10 * seed, 0.3, 120, 36)); i += 1
            # agent: streamed, engine too slow for the 60 tps shape -> floor violations
            meter.record(_row(i, load_run, seed, "agent", "small.8k.60tps", "stream", t + 1, 1500, 200, 0.4, 40, 40)); i += 1
            # batch: lazy complete
            meter.record(_row(i, load_run, seed, "batch", "small.8k.15tps.lazy", "complete", t + 2, 1000, 300, 0.5, 100, 18, lazy=True)); i += 1
        # a long-prefill note-taker summary overlapping the chat streams, plus one refusal
        meter.record(_row(i, load_run, seed, "notetaker", "small.16k.30tps", "complete", t0 + 3.0, 6000, 200, 4.0, 100, 36)); i += 1
        meter.record(_row(i, load_run, seed, "agent", "small.8k.60tps", "stream", t0 + 50.0, 8000, 300, 0, 1, 1, status="refused")); i += 1
        d = run_dir / "loads" / load_run
        d.mkdir(parents=True)
        (d / "loadrun.json").write_text(json.dumps({
            "load_run": load_run, "scenario": "synthetic", "seed": seed, "started_at": "2026-11-06T00:00:00+00:00",
            "duration_s": 90, "gateway_pacing": {"enabled": True, "bucket_depth_s": 1.0},
        }), encoding="utf-8")
        with open(d / "samples.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["t", "replica", "kv_usage", "running", "waiting", "preemptions", "client_inflight"])
            for s in range(0, 90, 2):
                w.writerow([t0 + s, "local-2080ti", 0.1 + 0.005 * s, 3, 0, 0, 3])
    meter.close()


def test_report_builds_csvs_and_html(tmp_path):
    run_dir = tmp_path / "synthetic-run"
    _write_run(run_dir)
    out = build_report([run_dir], config_dir=REPO / "config")
    assert out == run_dir / "report"
    for name in ("per_request.csv", "per_tenant.csv", "summary.csv", "across_seeds.csv", "interference.csv",
                 "per_replica.csv", "refusals.csv", "market.csv", "list_prices.csv", "report.html"):
        assert (out / name).is_file(), name

    pr = pd.read_csv(out / "per_request.csv")
    ok = pr[pr["ok"]]
    assert (ok["billed_s"] <= ok["bound_s"] + 1e-9).all()
    chat = ok[ok["tenant"] == "chat"]
    assert chat["floor_violation"].sum() == 0 and chat["delivered_tps"].notna().all()
    assert chat["delivered_tps"].between(35.9, 36.1).all()  # delivered at 36 tok/s on a 30 tps shape
    agent = ok[ok["tenant"] == "agent"]
    assert agent["floor_violation"].all()  # 40 tok/s delivered on a 60 tps shape
    assert (agent["billed_s"] < agent["wall_s"]).all()  # capped: provider's slowness is not billed
    assert (pr[pr["tenant"] == "batch"]["per_token_usd"] > 0).all()

    pt = pd.read_csv(out / "per_tenant.csv")
    assert set(pt["tenant"]) == {"chat", "agent", "batch", "notetaker"}
    agent_pt = pt[(pt["tenant"] == "agent") & (pt["load_run"] == "synthetic-s1")].iloc[0]
    assert agent_pt["refused"] == 1 and agent_pt["ok"] == 12
    assert agent_pt["floor_violation_share"] == pytest.approx(1.0)

    summary = pd.read_csv(out / "summary.csv")
    assert len(summary) == 2 and set(summary["seed"]) == {1, 2}
    assert (summary["kv_usage_mean"] > 0).all() and (summary["concurrency_mean"] > 0).all()
    assert (summary["gpu_cost_usd"] > 0).all()

    xs = pd.read_csv(out / "across_seeds.csv")
    chat_xs = xs[xs["tenant"] == "chat"].iloc[0]
    assert chat_xs["seeds"] == 2 and chat_xs["shape_usd_cv"] > 0 and chat_xs["per_token_usd_cv"] > 0

    inter = pd.read_csv(out / "interference.csv")
    assert inter["during_long_prefill"].any()

    prp = pd.read_csv(out / "per_replica.csv")
    assert len(prp) == 2 and set(prp["replica"]) == {"local-2080ti"}
    assert (prp["fill_rate"] > 0).all() and (prp["breakeven_fill_rate"] > 0).all()
    assert prp["slot_plan"].iloc[0] == "8xsmall.8k.30tps"
    ref = pd.read_csv(out / "refusals.csv")
    assert set(ref["reason"]) == {"context_exceeds_shape"} and ref["n"].sum() == 2

    mk = pd.read_csv(out / "market.csv")
    refs = {r.name for r in load_config(REPO / "config").pricing.references}
    assert set(mk["reference"]) == refs and len(mk) == 2 * len(refs)
    assert mk["active"].sum() == 2  # the active reference once per load run
    assert (mk["revenue_usd"] > 0).all() and (mk["breakeven_gpu_usd_per_hour"] > 0).all()
    active = mk[mk["active"]]
    # the active reference prices the per-token bill, so its revenue is the run's per-token revenue
    assert active.set_index("load_run")["revenue_usd"].to_dict() == pytest.approx(summary.set_index("load_run")["per_token_usd"].to_dict())
    assert (mk["gross_margin_at_target"] >= mk["gross_margin"]).all()  # 70% target above the measured utilisation here
    lp = pd.read_csv(out / "list_prices.csv")
    margins = load_config(REPO / "config").pricing.provider.margin_targets
    assert len(lp) == 2 * len(margins) and set(lp["margin_target"]) == set(margins)
    assert (lp["output_usd_per_million"] > 0).all() and (lp["input_usd_per_million"] < lp["output_usd_per_million"]).all()
    assert (lp.sort_values("margin_target").groupby("load_run")["output_usd_per_million"].diff().dropna() > 0).all()  # more margin, higher price
    assert (lp["output_usd_per_million_at_target"] < lp["output_usd_per_million"]).all()
    # a 0% margin list price earns back exactly the GPU cost on the run's traffic
    zero = lp[lp["margin_target"] == 0].set_index("load_run")
    tok = load_config(REPO / "config").pricing.token
    for load_run, row in zero.iterrows():
        sub = ok[ok["load_run"] == load_run]
        disc = sub["lazy"].map({True: tok.batch_discount, False: 1.0})
        earned = ((sub["prompt_tokens"] * row["input_usd_per_million"] + sub["output_tokens"] * row["output_usd_per_million"]) * disc).sum() / 1e6
        assert earned == pytest.approx(summary.set_index("load_run").loc[load_run, "gpu_cost_usd"])

    page = (out / "report.html").read_text(encoding="utf-8")
    assert page.count("<svg") >= 7
    assert "Market reference prices" in page and "basket-3b" in page
    for word in ("chat", "agent", "batch", "notetaker", "Bills", "Predictability", "Timeline", "Definitions"):
        assert word in page
    assert "<script" not in page  # self-contained, no JavaScript


def test_report_from_two_run_dirs_and_custom_out(tmp_path):
    a, b = tmp_path / "run-a", tmp_path / "run-b"
    _write_run(a)
    _write_run(b)
    out = build_report([a, b], out_dir=tmp_path / "combined", config_dir=REPO / "config")
    summary = pd.read_csv(out / "summary.csv")
    assert len(summary) == 2  # same load-run ids in both dirs merge by id; rows overlap by request_id
    assert (out / "report.html").is_file()
