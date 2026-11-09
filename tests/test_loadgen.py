from __future__ import annotations

import csv
import json

import numpy as np
import pytest

from loadgen.arrivals import diurnal_times, hourmark_times, poisson_times
from loadgen.content import count_tokens_estimate, words
from loadgen.plan import build_plan
from loadgen.runner import ClientRecord, Counters, LoadRunner
from loadgen.scenario import HourmarkArrivals, Phase, Scenario


def _read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def tiny_scenario(tenants: list[str] | None = None, duration_s: float = 8.0) -> Scenario:
    all_tenants = [
        {
            "name": "chat",
            "api_key": "sk-local-chat",
            "shape": "small.8k.30tps",
            "profile": {"kind": "chat", "turns": [2, 2], "first_user_tokens": [20, 40], "user_tokens_per_turn": [20, 40],
                        "output_tokens": [8, 12], "think_s": [0.2, 0.5]},
            "arrivals": {"kind": "diurnal", "rate_per_min": 30},
            "reservation": {"slots": 1},
        },
        {
            "name": "agent",
            "api_key": "sk-local-agent",
            "shape": "small.8k.60tps",
            "profile": {"kind": "agent", "steps": [2, 2], "first_user_tokens": [30, 50], "growth_tokens": [20, 30],
                        "output_tokens": [8, 12], "pause_s": [0.2, 0.4]},
            "arrivals": {"kind": "bursty", "rate_on_per_min": 30, "rate_off_per_min": 0, "mean_on_s": 100, "mean_off_s": 1},
        },
        {
            "name": "notetaker",
            "api_key": "sk-local-notetaker",
            "shape": "small.16k.30tps",
            "profile": {"kind": "notetaker", "prompt_tokens": [100, 200], "output_tokens": [10, 15]},
            "arrivals": {"kind": "hourmark", "per_mark": 2, "sigma_s": 0.5, "background_per_min": 6},
        },
        {
            "name": "batch",
            "api_key": "sk-local-batch",
            "shape": "small.8k.15tps.lazy",
            "profile": {"kind": "batch", "prompt_tokens": [300, 400], "output_tokens": [10, 15], "concurrency": 1},
        },
    ]
    chosen = [t for t in all_tenants if tenants is None or t["name"] in tenants]
    return Scenario(
        name="tiny",
        hour_length_s=6,
        drain_s=10,
        sample_interval_s=1,
        phases=[Phase(name="p", duration_s=duration_s, rate_scale=1.0)],
        tenants=chosen,
    )


def test_plan_is_deterministic_per_seed():
    sc = tiny_scenario()
    a, b = build_plan(sc, 7), build_plan(sc, 7)
    assert a.to_json() == b.to_json()
    assert a.digest == b.digest
    c = build_plan(sc, 8)
    assert c.to_json() != a.to_json()
    assert a.n_requests > 0


def test_adding_a_tenant_does_not_change_the_others():
    only_chat = build_plan(tiny_scenario(["chat"]), 3)
    with_agent = build_plan(tiny_scenario(["chat", "agent"]), 3)
    chat_a = next(t for t in only_chat.tenants if t.name == "chat")
    chat_b = next(t for t in with_agent.tenants if t.name == "chat")
    assert json.dumps(chat_a.__dict__, default=lambda o: o.__dict__, sort_keys=True) == json.dumps(
        chat_b.__dict__, default=lambda o: o.__dict__, sort_keys=True
    )


def test_poisson_rate_and_diurnal_schedule():
    rng = np.random.default_rng(1)
    times = poisson_times(rng, 1.0, 0.0, 600.0)
    assert 520 <= len(times) <= 680  # mean 600, sd ~24
    assert times == sorted(times) and all(0 <= t < 600 for t in times)
    sc = Scenario(
        name="d", phases=[Phase(name="off", duration_s=300, rate_scale=0.0), Phase(name="on", duration_s=300, rate_scale=1.0)],
        tenants=tiny_scenario(["chat"]).tenants,
    )
    times = diurnal_times(np.random.default_rng(2), sc, 1.0)
    assert times and min(times) >= 300.0
    assert 240 <= len(times) <= 360


def test_hourmark_clusters_around_marks():
    spec = HourmarkArrivals(kind="hourmark", per_mark=5, sigma_s=1.0, background_per_min=0)
    times = hourmark_times(np.random.default_rng(3), spec, hour_length_s=100.0, duration=350.0)
    assert times == sorted(times)
    assert all(min(abs(t - m) for m in (100.0, 200.0, 300.0)) < 5.0 for t in times)
    assert 5 <= len(times) <= 30


def test_content_is_deterministic_and_sized():
    a, b = words(11, 100), words(11, 100)
    assert a == b
    assert count_tokens_estimate(a) == 100
    assert words(12, 100) != a
    assert count_tokens_estimate(words(5, 1)) == 1
    assert words(5, 0) == ""


async def test_loadgen_runs_against_gateway_and_tags_the_meter(gateway, meter_rows, tmp_path):
    sc = tiny_scenario()
    plan = build_plan(sc, 7)
    runner = LoadRunner(sc, plan, gateway.url, runs_dir=tmp_path, load_run_id="tiny-s7-test", progress_every_s=2)
    meta = await runner.run()
    out = runner.out_dir
    assert out is not None and out.name == "tiny-s7-test" and out.parent.name == "loads"
    assert (out / "plan.json").is_file() and (out / "loadrun.json").is_file()
    assert meta["plan_digest"] == plan.digest
    assert meta["counters"]["errors"] == 0
    assert meta["counters"]["ok"] >= 8
    assert meta["tokenizer"]["tokens_per_planned_token"] == pytest.approx(1.0, abs=0.02)
    # request timeouts follow each shape's queue contract from /v1/models: max_queue_s + the generation allowance
    assert runner.request_timeout("small.8k.15tps.lazy") == 600.0 + 3600.0
    assert runner.request_timeout("small.8k.30tps") == 600.0 + 30.0
    assert runner.request_timeout(None) == 600.0
    assert meta["request_timeouts_s"] == {
        "small.8k.30tps": 630.0, "small.8k.60tps": 630.0, "small.16k.30tps": 630.0, "small.8k.15tps.lazy": 4200.0,
    }

    client_rows = _read_csv(out / "client.csv")
    assert len(client_rows) == meta["counters"]["sent"]
    ok_rows = [r for r in client_rows if r["status"] == "ok"]
    assert all(r["request_id"].startswith("chatcmpl-") for r in ok_rows)
    assert all(float(r["billed_s"]) >= 0 for r in ok_rows)
    tenants = {r["tenant"] for r in ok_rows}
    assert tenants == {"chat", "agent", "notetaker", "batch"}
    streamed = [r for r in ok_rows if r["stream"] == "True"]
    assert streamed and all(int(r["chunks"]) >= 1 and int(r["output_tokens"]) >= 8 for r in streamed)

    rows = meter_rows(load_run="tiny-s7-test")
    assert len(rows) == len(ok_rows) + len([r for r in client_rows if r["status"] == "refused"])
    assert {r["scenario"] for r in rows} == {"tiny"} and {r["seed"] for r in rows} == {7}
    assert {r["request_id"] for r in rows} >= {r["request_id"] for r in ok_rows}

    samples = _read_csv(out / "samples.csv")
    assert len(samples) >= 5 and samples[0]["replica"] == "fake-0"
    assert {s["reserved_slots"] for s in samples} >= {"1"} and all(s["gw_revocations"] == "0" for s in samples)

    # the chat tenant's reservation was created at the start, served its requests and was released at the end
    (res,) = meta["reservations"]
    assert res["status"] == 201 and res["id"] and res["released"] is True and res["served"] >= 1
    assert res["shape"] == "small.8k.30tps" and res["slots"] == 1 and res["load_run"] == "tiny-s7-test"
    chat_rows = [r for r in rows if r["tenant"] == "chat" and r["status"] == "ok"]
    assert chat_rows and all(r["reservation_id"] == res["id"] for r in chat_rows)
    assert all(r["reservation_id"] is None for r in rows if r["tenant"] != "chat")


def test_reservation_windows_follow_phases():
    sc = tiny_scenario(["chat"])
    assert sc.reservation_windows(sc.tenants[0].reservation) == [(0.0, 8.0)]
    sc2 = Scenario(
        name="w", phases=[Phase(name="a", duration_s=10), Phase(name="b", duration_s=20), Phase(name="c", duration_s=5)],
        tenants=[{**sc.tenants[0].model_dump(), "reservation": {"slots": 2, "phases": ["b", "c"]}}],
    )
    assert sc2.reservation_windows(sc2.tenants[0].reservation) == [(10.0, 30.0), (30.0, 35.0)]
    with pytest.raises(ValueError, match="not a phase"):
        Scenario(name="w", phases=[Phase(name="a", duration_s=10)],
                 tenants=[{**sc.tenants[0].model_dump(), "reservation": {"slots": 1, "phases": ["zz"]}}])


def test_stream_error_events_classify_revocations_as_results():
    """The gateway revokes a context burst inside a 200 stream with an error event (code burst_revoked).
    That is a designed outcome under the grace policy, so loadgen records it as `revoked`; the run's
    exit code (errors == 0) must not trip on it. Any other in-stream error stays an error."""
    def rec():
        return ClientRecord(load_run="r", scenario="s", seed=1, tenant="agent", session="a-1", step=0,
                            shape="small.8k.60tps", stream=True, t_send=0.0)
    r = rec()
    LoadRunner._stream_error(r, {"code": "burst_revoked", "message": "context burst revoked: a guaranteed request needed the capacity"})
    assert r.status == "revoked"
    assert r.error.startswith("context burst revoked")
    r = rec()
    LoadRunner._stream_error(r, {"code": "upstream_error", "message": "boom"})
    assert r.status == "error"
    r = rec()
    LoadRunner._stream_error(r, {"message": "no code at all"})
    assert r.status == "error"
    assert "revoked" in Counters().as_dict()


def test_request_timeout_without_contracts_is_the_generation_allowance(tmp_path):
    """An older gateway (no max_queue_s in /v1/models) or an unreachable listing: every request keeps
    the blanket allowance; a shape whose contract is known adds its queue time."""
    sc = tiny_scenario(["batch", "chat"])
    runner = LoadRunner(sc, build_plan(sc, 1), "http://127.0.0.1:9", runs_dir=tmp_path)
    assert runner.request_timeout("small.8k.15tps.lazy") == 600.0
    runner.shape_contracts = {"small.8k.15tps.lazy": {"lazy": True, "max_queue_s": 3600}, "small.8k.30tps": {"lazy": False}}
    assert runner.request_timeout("small.8k.15tps.lazy") == 4200.0
    assert runner.request_timeout("small.8k.30tps") == 600.0  # contract listed without a queue time
