from __future__ import annotations

import json

import httpx
import pytest

CHAT_KEY = {"Authorization": "Bearer sk-local-chat"}
MESSAGES = [{"role": "user", "content": "Summarise the meeting about token buckets in one line."}]


async def _post(gateway, body, headers=CHAT_KEY):
    async with httpx.AsyncClient(base_url=gateway.url, timeout=30) as c:
        return await c.post("/v1/chat/completions", json=body, headers=headers)


async def test_missing_key_is_401(gateway):
    r = await _post(gateway, {"messages": MESSAGES}, headers={})
    assert r.status_code == 401
    assert r.json()["error"]["code"] == "invalid_api_key"


async def test_wrong_key_is_401(gateway):
    r = await _post(gateway, {"messages": MESSAGES}, headers={"Authorization": "Bearer nope"})
    assert r.status_code == 401


async def test_models_lists_shapes(gateway):
    async with httpx.AsyncClient(base_url=gateway.url) as c:
        r = await c.get("/v1/models", headers=CHAT_KEY)
    assert r.status_code == 200
    ids = {m["id"] for m in r.json()["data"]}
    assert "small.8k.30tps" in ids and "small.8k.15tps.lazy" in ids
    lazy = next(m for m in r.json()["data"] if m["id"] == "small.8k.15tps.lazy")
    assert lazy["slotgw"]["lazy"] is True


async def test_complete_response_and_meter(gateway, meter_rows):
    r = await _post(
        gateway,
        {"model": "small.8k.30tps", "messages": MESSAGES, "max_tokens": 20, "ignore_eos": True},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["object"] == "chat.completion"
    assert body["model"] == "small.8k.30tps"
    assert body["choices"][0]["message"]["content"].strip()
    assert body["usage"]["completion_tokens"] == 20
    assert body["usage"]["prompt_tokens"] > 0
    rid = body["slotgw"]["request_id"]
    assert r.headers["x-request-id"] == rid

    rows = meter_rows(request_id=rid)
    assert len(rows) == 1
    row = rows[0]
    assert row["tenant"] == "chat"
    assert row["shape"] == "small.8k.30tps"
    assert row["delivery"] == "complete"
    assert row["status"] == "ok"
    assert row["output_tokens"] == 20
    assert row["output_tokens_guaranteed"] == 20
    assert row["prompt_tokens"] == body["usage"]["prompt_tokens"]
    assert row["t_arrival"] <= row["t_admitted"] <= row["t_first_token"] <= row["t_engine_done"] <= row["t_last_token"]
    assert row["ttft_ms"] is not None and row["ttft_ms"] >= 0
    assert row["e2e_ms"] > 0
    assert row["replica"] == "fake-0"


async def test_stream_response_and_meter(gateway, meter_rows):
    body = {"model": "small.8k.30tps", "messages": MESSAGES, "max_tokens": 15, "ignore_eos": True, "stream": True}
    content_chunks = 0
    final = None
    saw_done = False
    async with (
        httpx.AsyncClient(base_url=gateway.url, timeout=30) as c,
        c.stream("POST", "/v1/chat/completions", json=body, headers=CHAT_KEY) as r,
    ):
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        rid = r.headers["x-request-id"]
        async for line in r.aiter_lines():
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                saw_done = True
                break
            obj = json.loads(data)
            assert obj["model"] == "small.8k.30tps"
            choice = obj["choices"][0]
            if choice["delta"].get("content"):
                content_chunks += 1
            if choice["finish_reason"]:
                final = obj
    assert saw_done
    assert content_chunks == 15
    assert final is not None
    assert final["usage"]["completion_tokens"] == 15
    assert final["slotgw"]["request_id"] == rid

    row = meter_rows(request_id=rid)[0]
    assert row["delivery"] == "stream"
    assert row["status"] == "ok"
    assert row["output_tokens"] == 15
    assert row["t_first_token"] < row["t_last_token"]


async def test_default_shape_when_model_is_alias_or_missing(gateway):
    r = await _post(gateway, {"model": "small", "messages": MESSAGES, "max_tokens": 3, "ignore_eos": True})
    assert r.status_code == 200
    assert r.json()["model"] == "small.8k.30tps"  # chat tenant default
    r = await _post(gateway, {"messages": MESSAGES, "max_tokens": 3, "ignore_eos": True},
                    headers={"Authorization": "Bearer sk-local-batch"})
    assert r.status_code == 200
    assert r.json()["model"] == "small.8k.15tps.lazy"


async def test_shape_field_overrides_model(gateway):
    r = await _post(gateway, {"model": "small.8k.30tps", "shape": "small.16k.30tps", "messages": MESSAGES,
                              "max_tokens": 3, "ignore_eos": True})
    assert r.status_code == 200
    assert r.json()["model"] == "small.16k.30tps"


async def test_unknown_shape_is_404(gateway):
    r = await _post(gateway, {"model": "small.99k.1tps", "messages": MESSAGES})
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "model_not_found"


async def test_context_exceeding_shape_is_refused_and_metered(gateway, meter_rows):
    r = await _post(gateway, {"model": "small.8k.30tps", "messages": MESSAGES, "max_tokens": 9000})
    assert r.status_code == 429
    assert r.json()["error"]["code"] == "context_exceeds_shape"
    rows = [row for row in meter_rows(status="refused") if row["abort_reason"] == "context_exceeds_shape"]
    assert rows and rows[-1]["http_status"] == 429 and rows[-1]["output_tokens"] == 0


async def test_invalid_messages_is_400(gateway):
    r = await _post(gateway, {"model": "small.8k.30tps", "messages": []})
    assert r.status_code == 400


async def test_admin_replicas_reports_cache_config(gateway):
    async with httpx.AsyncClient(base_url=gateway.url) as c:
        r = await c.get("/admin/replicas")
    assert r.status_code == 200
    rep = r.json()["replicas"][0]
    assert rep["health"]["ok"] is True
    assert rep["health"]["preemptions"] == 0
    assert rep["cache"]["block_size"] == 16
    assert rep["cache"]["kv_tokens_total"] == 160_000


@pytest.mark.parametrize("n", [4])
async def test_concurrent_streams_all_complete(gateway, meter_rows, n):
    import asyncio

    async def one(i):
        return await _post(
            gateway,
            {"model": "small.8k.30tps", "messages": MESSAGES + [{"role": "user", "content": str(i)}],
             "max_tokens": 10, "ignore_eos": True},
        )

    results = await asyncio.gather(*(one(i) for i in range(n)))
    assert all(r.status_code == 200 for r in results)
    assert len({r.json()["slotgw"]["request_id"] for r in results}) == n
