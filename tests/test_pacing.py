from __future__ import annotations

import json
import time

import httpx
import pytest

from slotgw.clock import now
from slotgw.pacing import Pacer, TokenBucket

CHAT_KEY = {"Authorization": "Bearer sk-local-chat"}
MESSAGES = [{"role": "user", "content": "Write a long story about token buckets."}]


def test_bucket_starts_full_then_paces():
    b = TokenBucket(rate=10.0, depth=5.0, t0=100.0)
    assert b.schedule(5, 100.0) == (100.0, 0.0)  # depth delivered at once
    t, wait = b.schedule(1, 100.0)  # bucket empty: one token at 10/s
    assert t == pytest.approx(100.1) and wait == pytest.approx(0.1)
    t, wait = b.schedule(1, 100.0)  # queued behind the previous delivery
    assert t == pytest.approx(100.2) and wait == pytest.approx(0.1)
    t, wait = b.schedule(1, 103.0)  # a late token after a long gap is immediate (bucket refilled)
    assert t == 103.0 and wait == 0.0


def test_bucket_long_run_rate_is_exact():
    b = TokenBucket(rate=30.0, depth=30.0, t0=0.0)
    last = 0.0
    for _ in range(300):
        last, _ = b.schedule(1, 0.0)  # all tokens available at t=0
    # 30 immediate, then 270 at 30/s = 9 s
    assert last == pytest.approx(9.0)


def test_pacer_disabled_is_passthrough_and_counts_nothing_throttled():
    p = Pacer(tps=30, bucket_depth_s=1.0, enabled=False)
    assert p.schedule(100, 5.0) == 5.0
    assert p.throttled_ms == 0.0 and p.scheduled_tokens == 100


def test_pacer_throttled_time_is_wall_time_held_by_bucket():
    p = Pacer(tps=10, bucket_depth_s=0.0)
    t = 0.0
    for _ in range(10):
        t = p.schedule(1, 0.0)
    assert t == pytest.approx(1.0)
    assert p.throttled_ms == pytest.approx(1000.0)
    # tokens arriving slower than the rate are never throttled (one token of depth)
    p2 = Pacer(tps=10, bucket_depth_s=0.1)
    for i in range(5):
        p2.schedule(1, 10.0 + i * 0.5)
    assert p2.throttled_ms == 0.0


async def _stream_token_times(gateway, body, headers=CHAT_KEY):
    """(time, cumulative delivered tokens) per content chunk, plus the final chunk."""
    times: list[tuple[float, int]] = []
    final = None
    async with (
        httpx.AsyncClient(base_url=gateway.url, timeout=60) as c,
        c.stream("POST", "/v1/chat/completions", json=body, headers=headers) as r,
    ):
        assert r.status_code == 200
        async for line in r.aiter_lines():
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            obj = json.loads(data)
            if "error" in obj:
                raise AssertionError(obj)
            choice = obj["choices"][0]
            if choice["delta"].get("content"):
                times.append((now(), obj["slotgw"]["delivered_tokens"]))
            if choice["finish_reason"]:
                final = obj
    return times, final


async def test_stream_is_paced_at_shape_rate(gateway, meter_rows):
    """Fake engine runs at ~200 tok/s; the 30 tps shape must deliver at 30 tok/s (+-5%) once the
    one-second bucket is spent. Acceptance criterion of milestone 2."""
    n = 120
    body = {"model": "small.8k.30tps", "messages": MESSAGES, "max_tokens": n, "ignore_eos": True, "stream": True}
    times, final = await _stream_token_times(gateway, body)
    assert times[-1][1] == n
    depth = 30  # 1 s of 30 tps
    t0, c0 = next((t, c) for t, c in times if c >= depth)
    t1, c1 = times[-1]
    rate = (c1 - c0) / (t1 - t0)
    assert rate == pytest.approx(30.0, rel=0.05), f"steady-state delivered rate {rate:.1f} tok/s"
    assert t0 - times[0][0] < 1.0  # the bucket's head start went out fast
    assert final["slotgw"]["pacing"] is True
    assert final["slotgw"]["throttled_ms"] > 1500  # engine was ~3 s ahead of delivery
    bill = final["slotgw"]["billing"]
    assert bill["delivered_tokens"] == n
    assert bill["billed_s"] <= bill["bound_s"] + 1e-6
    assert bill["billed_s"] == pytest.approx(bill["wall_s"])  # cap not hit: delivery was faster than bound

    row = meter_rows(request_id=final["slotgw"]["request_id"])[0]
    assert row["throttled_ms"] > 1500
    assert row["t_engine_done"] < row["t_last_token"]
    assert row["output_tokens"] == n and row["wasted_tokens"] == 0


async def test_complete_response_is_released_at_virtual_pace(gateway, meter_rows):
    """60 tokens on a 30 tps shape: 30 immediately, 30 more over one second -> released ~1 s after
    the first token, even though the engine finished in ~0.3 s."""
    body = {"model": "small.8k.30tps", "messages": MESSAGES, "max_tokens": 60, "ignore_eos": True}
    t0 = time.perf_counter()
    async with httpx.AsyncClient(base_url=gateway.url, timeout=60) as c:
        r = await c.post("/v1/chat/completions", json=body, headers=CHAT_KEY)
    elapsed = time.perf_counter() - t0
    assert r.status_code == 200, r.text
    slot = r.json()["slotgw"]
    assert 0.95 <= elapsed <= 1.8, elapsed
    row = meter_rows(request_id=slot["request_id"])[0]
    held = row["t_last_token"] - row["t_engine_done"]
    assert held > 0.4, held  # response was held back by the virtual bucket
    assert row["throttled_ms"] > 400
    assert slot["billing"]["billed_s"] <= slot["billing"]["bound_s"] + 1e-6
    assert slot["billing"]["bound_s"] == pytest.approx(row["ttft_ms"] / 1000 + 60 / 30, abs=1e-3)


async def test_short_requests_within_bucket_depth_are_not_throttled(gateway):
    body = {"model": "small.8k.30tps", "messages": MESSAGES, "max_tokens": 10, "ignore_eos": True}
    async with httpx.AsyncClient(base_url=gateway.url, timeout=60) as c:
        r = await c.post("/v1/chat/completions", json=body, headers=CHAT_KEY)
    assert r.json()["slotgw"]["throttled_ms"] == 0.0
