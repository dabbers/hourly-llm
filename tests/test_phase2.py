"""Phase 2, G0 (docs/PHASE2.md section 4): context bursts at the slowest tps class with a short
queue, prefix-cache hits in usage and bills, text capture for the output samples, the showcase lane
and the rate multiplier in the load generator, and the report's samples and cache-aware market."""

from __future__ import annotations

import asyncio
import json
import time

import httpx
import pandas as pd
import pytest
import yaml

from loadgen.content import words
from loadgen.plan import build_plan
from loadgen.runner import LoadRunner
from loadgen.scenario import Scenario
from report.build import build_report
from slotgw.billing import per_token_usd
from slotgw.config import (
    AdmissionSettings,
    BurstSettings,
    PriceReference,
    Pricing,
    ProviderPricing,
    TokenPricing,
)
from slotgw.meter import Meter
from slotgw.prices import build_references

from .test_loadgen import _read_csv, tiny_scenario
from .test_multi_replica import REPO, TWO_SEATS, Cluster, _admin, _post
from .test_report import _row

BIG = [{"role": "user", "content": words(1, 9_000)}]  # ~9k tokens: over the 8k shapes
LONG = [{"role": "user", "content": words(2, 400)}]


# ---- context bursts at the slowest tps class ---------------------------------------------------


async def _stream_final(url: str, body: dict, key: str) -> dict:
    async with httpx.AsyncClient(base_url=url, timeout=60) as c, c.stream(
        "POST", "/v1/chat/completions", json=body, headers={"Authorization": f"Bearer {key}"}
    ) as r:
        final = {"status": r.status_code}
        if r.status_code != 200:
            final["error"] = json.loads((await r.aread()).decode()).get("error")
            return final
        async for line in r.aiter_lines():
            if line.startswith("data:") and '"billing"' in line:
                final.update(json.loads(line[5:]))
    return final


async def test_context_burst_takes_the_slowest_seat_and_is_billed_as_that_shape(tmp_path):
    # TWO_SEATS: 3 seats at 15 tps, 2 at 30, 1 at 60. With one 30 tps request running, a 60 tps burst
    # does not fit (M4 behaviour); as a 15 tps burst it does, and its floor, pacing and price are the 15 tps shape's.
    cluster = Cluster(
        [TWO_SEATS], AdmissionSettings(), queue_s=5.0, meter_path=tmp_path / "m.sqlite",
        burst=BurstSettings(tps=False, max_context_tokens=16384), kv_tokens=60_000,
    )
    try:
        gw = cluster.gateway.url
        chat = asyncio.create_task(_post(gw, {"model": "small.8k.30tps", "messages": LONG, "max_tokens": 150, "ignore_eos": True}))
        await asyncio.sleep(0.5)
        final = await _stream_final(gw, {"model": "small.8k.60tps", "messages": BIG, "max_tokens": 40, "ignore_eos": True, "stream": True}, "sk-local-agent")
        assert final["status"] == 200, final
        slot = final["slotgw"]
        assert slot["class_ctx"] == "burst" and slot["shape"] == "small.8k.60tps" and slot["billed_shape"] == "small.8k.15tps"
        assert slot["queued_ms"] < 500  # placed at once: a 15 tps hold fits next to the 30 tps one
        bill = slot["billing"]
        assert bill["bound_s"] - slot["ttft_ms"] / 1000 == pytest.approx(40 / 15, abs=0.05)  # the 15 tps promise, not 60
        # paced at 15 tps: after the 15-token head start the engine's 40 tok/s is held back; at 60 tps nothing would be
        assert slot["throttled_ms"] > 300
        await chat
        rows = {r["request_id"]: r for r in cluster.rows()}
        brow = rows[slot["request_id"]]
        assert brow["billed_shape"] == "small.8k.15tps" and brow["shape"] == "small.8k.60tps" and brow["class_ctx"] == "burst"
        assert (await _admin(gw))["scheduler"]["bursts_admitted"] == 1
    finally:
        cluster.stop()


async def test_context_burst_of_its_own_class_fails_fast_on_the_short_burst_queue(tmp_path):
    cluster = Cluster(
        [TWO_SEATS], AdmissionSettings(), queue_s=5.0, meter_path=tmp_path / "m.sqlite",
        burst=BurstSettings(tps=False, max_context_tokens=16384, context_tps_class="own", context_max_queue_s=0.8),
        kv_tokens=60_000,
    )
    try:
        gw = cluster.gateway.url
        chat = asyncio.create_task(_post(gw, {"model": "small.8k.30tps", "messages": LONG, "max_tokens": 150, "ignore_eos": True}))
        await asyncio.sleep(0.5)
        t0 = time.perf_counter()
        r = await _post(gw, {"model": "small.8k.60tps", "messages": BIG, "max_tokens": 20, "ignore_eos": True}, key="sk-local-agent")
        waited = time.perf_counter() - t0
        assert r.status_code == 503 and r.json()["error"]["code"] == "capacity"
        assert 0.7 < waited < 2.5, waited  # the burst queue's 0.8 s, not the shape's 5 s
        await chat
        row = next(x for x in cluster.rows() if x["class_ctx"] == "burst")
        assert row["status"] == "refused" and row["abort_reason"] == "capacity:seat" and row["billed_shape"] == "small.8k.60tps"
    finally:
        cluster.stop()


# ---- cached prompt tokens -------------------------------------------------------------------------


async def test_prefix_cache_hits_are_metered_and_priced_at_the_cache_read_rate(tmp_path):
    cluster = Cluster([TWO_SEATS], AdmissionSettings(), queue_s=2.0, meter_path=tmp_path / "m.sqlite")
    try:
        gw = cluster.gateway.url
        body = {"model": "small.8k.30tps", "messages": LONG, "max_tokens": 10, "ignore_eos": True}
        first = (await _post(gw, body)).json()
        second = (await _post(gw, body)).json()  # the fake replica's prefix cache has seen this prompt
        assert first["usage"]["prompt_tokens_details"]["cached_tokens"] == 0
        cached = second["usage"]["prompt_tokens_details"]["cached_tokens"]
        assert 300 < cached <= second["usage"]["prompt_tokens"]
        assert second["slotgw"]["cached_tokens"] == cached
        b1, b2 = first["slotgw"]["billing"], second["slotgw"]["billing"]
        assert b2["per_token_usd"] < b1["per_token_usd"] and b2["prompt_usd"] < b1["prompt_usd"]
        rows = {r["request_id"]: r for r in cluster.rows()}
        assert rows[first["id"]]["cached_tokens"] == 0 and rows[second["id"]]["cached_tokens"] == cached
    finally:
        cluster.stop()


def test_cache_read_price_comes_from_the_reference_or_the_ratio():
    base = {"provider": ProviderPricing(gpu_usd_per_hour=1.0)}
    listed = PriceReference(name="r", input_usd_per_million=1.0, output_usd_per_million=2.0, cache_read_usd_per_million=0.1)
    p = Pricing(reference="r", references=[listed], **base)
    assert p.token.cache_read_price == pytest.approx(0.1) and p.cache_read_ratio == pytest.approx(0.1)
    unlisted = PriceReference(name="u", input_usd_per_million=1.0, output_usd_per_million=2.0)
    p = Pricing(reference="u", references=[unlisted], **base)
    assert p.token.cache_read_price == pytest.approx(0.25) and p.cache_read_ratio == pytest.approx(0.25)
    assert p.prompt_charge_usd(1000, 400) == pytest.approx((600 + 400 * 0.25) * 1.0 / 1e6)
    tok = TokenPricing(input_usd_per_million=1.0, output_usd_per_million=2.0)
    assert per_token_usd(1000, 100, False, tok, cached_tokens=400) == pytest.approx((600 + 100 + 200) / 1e6)
    assert per_token_usd(1000, 100, False, tok, cached_tokens=5000) == pytest.approx((250 + 200) / 1e6)  # clamped to the prompt


def test_fetched_references_carry_the_providers_cache_read_price():
    models = [
        {"id": "a/one", "context_length": 1000, "pricing": {"prompt": "0.0000001", "completion": "0.0000003", "input_cache_read": "0.00000002"}},
        {"id": "b/two", "context_length": 2000, "pricing": {"prompt": "0.0000002", "completion": "0.0000004"}},
        {"id": "c/three", "context_length": 3000, "pricing": {"prompt": "0.0000003", "completion": "0.0000005", "input_cache_read": "0.00000006"}},
    ]
    doc = build_references(models, {"one": "a/one", "two": "b/two", "three": "c/three"}, {"b": ["one", "two", "three"]}, source="t", fetched="2026-11-07")
    by = {r["name"]: r for r in doc["references"]}
    assert by["one"]["cache_read_usd_per_million"] == pytest.approx(0.02) and "cache_read_usd_per_million" not in by["two"]
    assert by["b"]["cache_read_usd_per_million"] == pytest.approx(0.04)  # median of the members that list one


# ---- text capture ---------------------------------------------------------------------------------


async def test_store_text_keeps_completions_and_showcase_prompts(tmp_path):
    cluster = Cluster([TWO_SEATS], AdmissionSettings(), queue_s=2.0, meter_path=tmp_path / "m.sqlite", store_text=True)
    try:
        gw = cluster.gateway.url
        body = {"model": "small.8k.30tps", "messages": [{"role": "user", "content": "Describe a ledger."}], "max_tokens": 12, "stream": True}
        async with httpx.AsyncClient(base_url=gw, timeout=30) as c:
            async with c.stream("POST", "/v1/chat/completions", json=body, headers={"Authorization": "Bearer sk-local-chat", "X-Slotgw-Showcase": "chat-ledger#0"}) as r:
                assert r.status_code == 200
                rid = r.headers["x-request-id"]
                async for _ in r.aiter_lines():
                    pass
            plain = (await c.post("/v1/chat/completions", json={**body, "stream": False}, headers={"Authorization": "Bearer sk-local-chat"})).json()
        texts = {t["request_id"]: t for t in cluster.rows("texts")}
        assert set(texts) == {rid, plain["id"]}
        assert json.loads(texts[rid]["prompt_json"]) == body["messages"] and len(texts[rid]["output_text"].split()) >= 10
        assert texts[plain["id"]]["prompt_json"] is None and texts[plain["id"]]["output_text"] == plain["choices"][0]["message"]["content"]
        reqs = {r["request_id"]: r for r in cluster.rows()}
        assert reqs[rid]["showcase"] == "chat-ledger#0" and reqs[plain["id"]]["showcase"] is None
    finally:
        cluster.stop()


async def test_store_text_off_keeps_no_text(tmp_path):
    cluster = Cluster([TWO_SEATS], AdmissionSettings(), queue_s=2.0, meter_path=tmp_path / "m.sqlite")
    try:
        r = await _post(cluster.gateway.url, {"model": "small.8k.30tps", "messages": LONG, "max_tokens": 5, "ignore_eos": True})
        assert r.status_code == 200 and cluster.rows("texts") == []
    finally:
        cluster.stop()


# ---- the showcase lane and the rate multiplier in the load generator -----------------------------


def _prompt_file(tmp_path, profile="chat"):
    p = tmp_path / f"{profile}.yaml"
    p.write_text(yaml.safe_dump({
        "profile": profile,
        "system": "Be brief.",
        "prompts": [
            {"id": f"{profile}-p1", "max_tokens": 12, "turns": ["Hello there, what is a ledger in this gateway?", "And what is a seat?"]},
            {"id": f"{profile}-p2", "max_tokens": 8, "turns": ["Name three colours and nothing else."]},
        ],
    }), encoding="utf-8")
    return p


def _showcase_scenario(tmp_path, duration_s=8.0, rate_multiplier=1.0) -> Scenario:
    sc = tiny_scenario(["chat", "batch"], duration_s=duration_s)
    chat, batch = (t.model_dump() for t in sc.tenants)
    chat["showcase"] = {"prompts": str(_prompt_file(tmp_path, "chat")), "every_s": 3.0, "offset_s": 0.5, "think_s": 0.2}
    batch["showcase"] = {"prompts": str(_prompt_file(tmp_path, "batch")), "every_s": 4.0, "offset_s": 1.0}
    return Scenario(**{**sc.model_dump(), "tenants": [chat, batch], "rate_multiplier": rate_multiplier})


def test_showcase_sessions_cycle_the_prompt_file_at_fixed_offsets(tmp_path):
    plan = build_plan(_showcase_scenario(tmp_path), 7)
    chat = next(t for t in plan.tenants if t.name == "chat")
    shows = sorted((s for s in chat.sessions if s.id.startswith("chat-show-")), key=lambda s: s.t_start)
    assert [s.t_start for s in shows] == [0.5, 3.5, 6.5]
    assert [[st.showcase for st in s.steps] for s in shows] == [["chat-p1"] * 2, ["chat-p2"], ["chat-p1"] * 2]
    assert shows[0].steps[1].pause_before_s == 0.2 and shows[0].steps[0].max_tokens == 12 and shows[0].steps[0].user_tokens > 5
    batch = next(t for t in plan.tenants if t.name == "batch")
    b_shows = [s for s in batch.sessions if s.t_start is not None]
    assert [s.t_start for s in b_shows] == [1.0, 5.0] and all(s.worker is None for s in b_shows)
    assert all(s.t_start is None for s in batch.sessions if not s.id.startswith("batch-show-"))  # the workers stay closed-loop
    # the synthetic plan is untouched by the showcase lane
    plain = build_plan(tiny_scenario(["chat", "batch"]), 7)
    plain_chat = next(t for t in plain.tenants if t.name == "chat")
    assert [s.id for s in plain_chat.sessions] == [s.id for s in chat.sessions if not s.id.startswith("chat-show-")]


def test_rate_multiplier_scales_open_loop_arrivals():
    one = build_plan(tiny_scenario(["chat", "notetaker"], duration_s=120.0), 5)
    three = build_plan(Scenario(**{**tiny_scenario(["chat", "notetaker"], duration_s=120.0).model_dump(), "rate_multiplier": 3.0}), 5)
    zero = build_plan(Scenario(**{**tiny_scenario(["chat", "notetaker"], duration_s=120.0).model_dump(), "rate_multiplier": 0.0}), 5)
    n = lambda p: sum(len(t.sessions) for t in p.tenants)
    assert n(zero) == 0 and n(three) > 2 * n(one) > 0


async def test_showcase_requests_carry_real_text_and_the_header(gateway, meter_rows, tmp_path):
    sc = _showcase_scenario(tmp_path)
    plan = build_plan(sc, 7)
    runner = LoadRunner(sc, plan, gateway.url, runs_dir=tmp_path / "runs", load_run_id="show-s7", progress_every_s=5, reservations=False)
    meta = await runner.run()
    assert meta["counters"]["errors"] == 0
    rows = _read_csv(runner.out_dir / "client.csv")
    shows = [r for r in rows if r["showcase"]]
    assert {r["showcase"] for r in shows} >= {"chat-p1#0", "chat-p1#1", "chat-p2#0", "batch-p1#0"}
    assert all(r["status"] == "ok" for r in shows), [r["error"] for r in shows if r["status"] != "ok"]
    first = next(r for r in shows if r["showcase"] == "chat-p1#0")
    assert int(first["output_tokens"]) <= 12  # a natural stop within the prompt's budget (no ignore_eos)
    metered = {r["request_id"]: r for r in meter_rows(load_run="show-s7")}
    assert all(metered[r["request_id"]]["showcase"] == r["showcase"] for r in shows)
    assert all(metered[r["request_id"]]["showcase"] is None for r in rows if not r["showcase"] and r["status"] == "ok")


# ---- the report -----------------------------------------------------------------------------------


def test_report_shows_samples_billed_shapes_and_cache_aware_market(tmp_path):
    run_dir = tmp_path / "p2-run"
    meter = Meter(run_dir / "meter.sqlite", run_id="p2-run")
    t0 = 1_700_000_000.0
    # a showcase chat request with stored text, a context burst billed as the 15 tps class, a cached note-taker summary
    show = _row(0, "p2-s1", 1, "chat", "small.8k.30tps", "stream", t0, 60, 40, 0.3, 120, 36)
    show.showcase, show.billed_shape = "chat-p1#0", "small.8k.30tps"
    meter.record(show)
    meter.record_text(show.request_id, "Ledgers count seats; seats count streams.", json.dumps([{"role": "user", "content": "What is a ledger?"}]))
    burst = _row(1, "p2-s1", 1, "agent", "small.8k.60tps", "stream", t0 + 5, 9000, 60, 0.4, 40, 16)
    burst.class_ctx, burst.billed_shape = "burst", "small.8k.15tps"
    meter.record(burst)
    cached = _row(2, "p2-s1", 1, "notetaker", "small.16k.30tps", "complete", t0 + 10, 6000, 200, 2.0, 100, 36)
    cached.cached_tokens, cached.billed_shape = 4000, "small.16k.30tps"
    meter.record(cached)
    meter.close()
    d = run_dir / "loads" / "p2-s1"
    d.mkdir(parents=True)
    (d / "loadrun.json").write_text(json.dumps({"load_run": "p2-s1", "scenario": "p2", "seed": 1, "duration_s": 30}), encoding="utf-8")

    out = build_report([run_dir], config_dir=REPO / "config")
    pr = pd.read_csv(out / "per_request.csv").set_index("request_id")
    b = pr.loc[burst.request_id]
    assert b["billed_shape"] == "small.8k.15tps" and b["tps"] == 15 and not b["floor_violation"]  # 16 tok/s meets the 15 tps class
    assert b["bound_s"] == pytest.approx(0.4 + 60 / 15, abs=1e-3)
    c = pr.loc[cached.request_id]
    assert c["cached_tokens"] == 4000
    full = 6000 * 0.0755 / 1e6 + 200 * 0.15 / 1e6  # basket-3b at full prompt price
    assert c["per_token_usd"] < full and c["per_token_usd"] == pytest.approx(full - 4000 * 0.0755 * 0.75 / 1e6, rel=1e-3)
    pt = pd.read_csv(out / "per_tenant.csv").set_index("tenant")
    assert pt.loc["notetaker", "cached_share"] == pytest.approx(4000 / 6000)
    mk = pd.read_csv(out / "market.csv")
    assert (mk["revenue_usd"] < mk["revenue_usd_uncached"]).all() and (mk["cache_read_usd_per_million"] < mk["input_usd_per_million"]).all()
    sm = pd.read_csv(out / "samples.csv")
    assert len(sm) == 1 and sm.iloc[0]["prompt_id"] == "chat-p1" and sm.iloc[0]["turn"] == 0
    assert sm.iloc[0]["output_text"].startswith("Ledgers") and "What is a ledger?" in sm.iloc[0]["prompt_json"]
    md = (out / "samples.md").read_text(encoding="utf-8")
    assert "chat · chat-p1 · turn 1" in md and "Ledgers count seats" in md
    page = (out / "report.html").read_text(encoding="utf-8")
    assert "Output samples" in page and "Ledgers count seats" in page and "billed as" not in page.split("Output samples")[0]


# ---- placement rules: pinned reservations and excluded shapes (Q6) -------------------------------


def test_scheduler_never_places_an_excluded_shape_and_pins_reservations():
    from dataclasses import dataclass

    from slotgw.admission import ReplicaLedger, ReservationCapacity, Scheduler
    from slotgw.calibration import ThroughputProfile
    from slotgw.config import Shape

    @dataclass
    class Dummy:
        id: str
        model_alias: str = "small"

    s30 = Shape(name="small.8k.30tps", model="small", max_context=8192, tps=30, max_queue_s=1)
    s60 = Shape(name="small.8k.60tps", model="small", max_context=8192, tps=60, max_queue_s=1)
    prof = ThroughputProfile.from_curve_string(TWO_SEATS)
    a = ReplicaLedger(Dummy("a"), prof, 181_936, 16, AdmissionSettings(), exclude_shapes=["small.8k.60tps"])
    b = ReplicaLedger(Dummy("b"), prof, 181_936, 16, AdmissionSettings())
    sched = Scheduler([a, b], AdmissionSettings())
    assert [x.id for x in sched.candidates(s60)] == ["b"] and [x.id for x in sched.candidates(s30)] == ["a", "b"]
    assert sched.place(s60, 100) is b
    res = sched.reserve("chat", s30, 2, replica_id="a")  # both slots on a, although b is emptier after the first
    assert {h.ledger.id for h in res.slots} == {"a"} and a.reserved_slots == 2 and b.reserved_slots == 0
    with pytest.raises(ReservationCapacity):  # a has 2 seats at 30 tps, both reserved: a third pinned slot does not fit
        sched.reserve("chat", s30, 1, replica_id="a")
    with pytest.raises(ValueError, match="unknown replica"):
        sched.reserve("chat", s30, 1, replica_id="zz")
    assert sched.place(s60, 100) is b and a.snapshot()["exclude_shapes"] == ["small.8k.60tps"]


async def test_reservation_can_be_pinned_to_a_replica_over_http(tmp_path):
    cluster = Cluster([TWO_SEATS, TWO_SEATS], AdmissionSettings(), queue_s=2.0, meter_path=tmp_path / "m.sqlite")
    try:
        gw = cluster.gateway.url
        async with httpx.AsyncClient(base_url=gw, timeout=10) as c:
            r = await c.post("/admin/reservations", json={"tenant": "chat", "shape": "small.8k.30tps", "slots": 2, "replica": "fake-1"})
            assert r.status_code == 201 and r.json()["replicas"] == ["fake-1"]
            r = await c.post("/admin/reservations", json={"tenant": "chat", "shape": "small.8k.30tps", "slots": 1, "replica": "fake-1"})
            assert r.status_code == 409  # fake-1's two 30 tps seats are taken
            r = await c.post("/admin/reservations", json={"tenant": "chat", "shape": "small.8k.30tps", "slots": 1, "replica": "nope"})
            assert r.status_code == 400
            r = await c.post("/admin/reservations", json={"tenant": "chat", "shape": "small.8k.30tps", "slots": 1})
            assert r.status_code == 201 and r.json()["replicas"] == ["fake-0"]  # unpinned: the emptier replica
        snap = await _admin(gw)
        assert {rep["id"]: rep["ledger"]["reserved_slots"] for rep in snap["replicas"]} == {"fake-0": 1, "fake-1": 2}
    finally:
        cluster.stop()


def test_loadgen_passes_the_reservation_pin_and_the_override():
    from loadgen.plan import build_plan as _bp

    sc = tiny_scenario(["chat"])
    sc.tenants[0].reservation.replica = "fake-0"
    plan = _bp(sc, 1)
    r = LoadRunner(sc, plan, "http://x", runs_dir=".", reservations=True)
    assert r.reservation_replica is None  # the scenario's own pin applies
    r2 = LoadRunner(sc, plan, "http://x", runs_dir=".", reservations=True, reservation_replica="none")
    assert r2.reservation_replica == "none"
    r3 = LoadRunner(sc, plan, "http://x", runs_dir=".", reservations=True, reservation_replica="fake-1")
    assert r3.reservation_replica == "fake-1"


def test_replica_ids_env_selects_a_subset_of_the_configured_replicas(monkeypatch):
    from slotgw.config import load_config

    monkeypatch.setenv("REPLICA_CURVE", "1:100,8:50")
    assert len(load_config(REPO / "config" / "phase2").replicas) == 3
    monkeypatch.setenv("SLOTGW_REPLICA_IDS", "runpod-4090-a")
    assert [r.id for r in load_config(REPO / "config" / "phase2").replicas] == ["runpod-4090-a"]
    monkeypatch.setenv("SLOTGW_REPLICA_IDS", "runpod-4090-a, runpod-4090-c")
    assert [r.id for r in load_config(REPO / "config" / "phase2").replicas] == ["runpod-4090-a", "runpod-4090-c"]
    monkeypatch.setenv("SLOTGW_REPLICA_IDS", "nope")
    with pytest.raises(ValueError, match="unknown replicas"):
        load_config(REPO / "config" / "phase2")
