"""OpenAI-compatible API. The OpenAI `model` field names the shape to run against
(docs/DESIGN.md section 4.1). Milestone 2: auth, shape check, per-request pacing, billing, meter."""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from . import __version__
from .billing import Bill, bill_request
from .clock import now
from .config import Config, Shape, Tenant
from .meter import Meter, RequestRecord
from .pacing import Pacer
from .replicas.base import Replica, ReplicaError, TokenChunk, Usage

log = logging.getLogger("slotgw.api")

_PASSTHROUGH_EXCLUDE = {"model", "shape", "stream", "stream_options"}


class GatewayState:
    def __init__(self, config: Config, meter: Meter, replicas: list[Replica]):
        self.config = config
        self.meter = meter
        self.replicas = replicas

    def select_replica(self, shape: Shape) -> Replica | None:
        # Milestone 2: first replica serving the shape's model. Milestone 3 replaces this with
        # ledger-based admission and shape-affinity scoring.
        for r in self.replicas:
            if r.model_alias == shape.model:
                return r
        return None

    def pacer_for(self, shape: Shape) -> Pacer:
        p = self.config.gateway.pacing
        return Pacer(shape.tps, p.bucket_depth_s, p.enabled)


def openai_error(status: int, message: str, code: str | None = None, type_: str = "invalid_request_error") -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"error": {"message": message, "type": type_, "param": None, "code": code}},
    )


def authenticate(request: Request, config: Config) -> Tenant | None:
    auth = request.headers.get("authorization", "")
    if not auth.lower().startswith("bearer "):
        return None
    return config.tenant_by_key(auth[7:].strip())


def resolve_shape(body: dict[str, Any], tenant: Tenant, config: Config) -> Shape | None:
    """`shape` wins, then `model` if it names a shape, else the tenant's default shape.
    A `model` that is a model alias (e.g. "small") or absent falls back to the default."""
    requested = body.get("shape") or body.get("model")
    if requested and requested in config.models:
        requested = None
    if not requested:
        return config.shape(tenant.default_shape)
    return config.shape(str(requested))


def _sse(obj: dict[str, Any]) -> str:
    return f"data: {json.dumps(obj, separators=(',', ':'))}\n\n"


def _load_tags(request: Request, rec: RequestRecord) -> None:
    """The load generator tags its requests so one gateway run can hold several seeded runs."""
    rec.load_run = request.headers.get("x-slotgw-run") or None
    rec.scenario = request.headers.get("x-slotgw-scenario") or None
    seed = request.headers.get("x-slotgw-seed")
    rec.seed = int(seed) if seed and seed.lstrip("-").isdigit() else None


def build_router() -> APIRouter:
    router = APIRouter()

    @router.get("/healthz")
    async def healthz(request: Request) -> dict[str, Any]:
        state: GatewayState = request.app.state.gw
        return {
            "ok": True,
            "version": __version__,
            "run_id": state.meter.run_id,
            "pacing": state.config.gateway.pacing.model_dump(),
        }

    @router.get("/admin/replicas")
    async def admin_replicas(request: Request) -> dict[str, Any]:
        state: GatewayState = request.app.state.gw
        out = []
        for r in state.replicas:
            h = await r.health()
            cc = await r.cache_config()
            out.append(
                {
                    "id": r.id,
                    "model": r.model_alias,
                    "hf_model": r.hf_model,
                    "health": {
                        "ok": h.ok,
                        "kv_usage": h.kv_usage,
                        "running": h.running,
                        "waiting": h.waiting,
                        "preemptions": h.preemptions,
                        "error": h.error,
                    },
                    "cache": None
                    if cc is None
                    else {
                        "num_gpu_blocks": cc.num_gpu_blocks,
                        "block_size": cc.block_size,
                        "kv_tokens_total": cc.kv_tokens_total,
                    },
                }
            )
        return {"replicas": out}

    @router.post("/admin/tokenize")
    async def admin_tokenize(request: Request):
        """Token count of a message list through the replica's tokenizer (chat template included).
        The load generator uses it once per run to check its prompt sizing."""
        state: GatewayState = request.app.state.gw
        try:
            body = await request.json()
        except json.JSONDecodeError:
            return openai_error(400, "request body is not valid JSON")
        messages = body.get("messages") if isinstance(body, dict) else None
        if not isinstance(messages, list) or not messages:
            return openai_error(400, "'messages' must be a non-empty list", code="invalid_messages")
        alias = body.get("model") or next(iter(state.config.models))
        replica = next((r for r in state.replicas if r.model_alias == alias), None)
        if replica is None:
            return openai_error(503, f"no replica serves model {alias!r}", code="no_replica", type_="server_error")
        try:
            count = await replica.tokenize(messages)
        except ReplicaError as e:
            return openai_error(502, f"replica tokenize failed: {e.detail}", code="upstream_error", type_="server_error")
        return {"model": alias, "count": count}

    @router.get("/v1/models")
    async def list_models(request: Request):
        state: GatewayState = request.app.state.gw
        tenant = authenticate(request, state.config)
        if tenant is None:
            return openai_error(401, "invalid or missing API key", code="invalid_api_key", type_="authentication_error")
        data = [
            {
                "id": s.name,
                "object": "model",
                "created": 0,
                "owned_by": "slotgw",
                "slotgw": {
                    "model": s.model,
                    "hf_model": state.config.hf_model(s.model),
                    "max_context": s.max_context,
                    "tps": s.tps,
                    "lazy": s.lazy,
                    "price_per_second": s.price_per_second,
                },
            }
            for s in state.config.shapes
        ]
        return {"object": "list", "data": data}

    @router.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        state: GatewayState = request.app.state.gw
        t_arrival = now()
        tenant = authenticate(request, state.config)
        if tenant is None:
            return openai_error(401, "invalid or missing API key", code="invalid_api_key", type_="authentication_error")
        try:
            body = await request.json()
        except json.JSONDecodeError:
            return openai_error(400, "request body is not valid JSON")
        if not isinstance(body, dict):
            return openai_error(400, "request body must be a JSON object")
        shape = resolve_shape(body, tenant, state.config)
        if shape is None:
            return openai_error(404, f"unknown shape {body.get('shape') or body.get('model')!r}", code="model_not_found")
        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            return openai_error(400, "'messages' must be a non-empty list", code="invalid_messages")
        stream = bool(body.get("stream", False))
        replica = state.select_replica(shape)
        request_id = "chatcmpl-" + uuid.uuid4().hex[:24]
        rec = RequestRecord(
            request_id=request_id,
            tenant=tenant.name,
            shape=shape.name,
            lazy=shape.lazy,
            delivery="stream" if stream else "complete",
            t_arrival=t_arrival,
            replica=replica.id if replica else None,
        )
        _load_tags(request, rec)
        if replica is None:
            rec.status, rec.http_status = "refused", 503
            state.meter.record(rec)
            return openai_error(503, f"no replica serves model {shape.model!r}", code="no_replica", type_="server_error")

        max_tokens = body.get("max_tokens") or body.get("max_completion_tokens") or shape.default_max_tokens
        try:
            max_tokens = int(max_tokens)
        except (TypeError, ValueError):
            return openai_error(400, "'max_tokens' must be an integer")

        try:
            prompt_tokens = await replica.tokenize(messages)
        except ReplicaError as e:
            rec.status, rec.http_status, rec.abort_reason = "upstream_error", 502, e.detail
            state.meter.record(rec)
            return openai_error(502, f"replica tokenize failed: {e.detail}", code="upstream_error", type_="server_error")
        rec.prompt_tokens = prompt_tokens

        if prompt_tokens + max_tokens > shape.max_context:
            rec.status, rec.http_status, rec.abort_reason = "refused", 429, "context_exceeds_shape"
            state.meter.record(rec)
            return openai_error(
                429,
                f"prompt ({prompt_tokens}) + max_tokens ({max_tokens}) exceeds shape {shape.name} context ({shape.max_context})",
                code="context_exceeds_shape",
                type_="shape_error",
            )

        payload = {k: v for k, v in body.items() if k not in _PASSTHROUGH_EXCLUDE}
        payload["max_tokens"] = max_tokens
        rec.t_admitted = now()  # milestone 2: no admission control, admitted on arrival
        rec.queued_ms = (rec.t_admitted - rec.t_arrival) * 1000.0
        created = int(rec.t_admitted)
        pacer = state.pacer_for(shape)

        if stream:
            return StreamingResponse(
                _stream_response(state, replica, shape, payload, rec, created, pacer),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Request-Id": request_id},
            )
        return await _complete_response(state, replica, shape, payload, rec, created, pacer)

    return router


def _usage_dict(rec: RequestRecord, usage: Usage | None, counted: int) -> dict[str, Any]:
    completion = usage.completion_tokens if usage else counted
    prompt = usage.prompt_tokens if usage and usage.prompt_tokens else rec.prompt_tokens
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
    }


def _bill(state: GatewayState, rec: RequestRecord, shape: Shape) -> Bill:
    return bill_request(
        shape=shape,
        pricing=state.config.pricing.token,
        prompt_tokens=rec.prompt_tokens,
        output_tokens=rec.output_tokens,
        wasted_tokens=rec.wasted_tokens,
        t_admitted=rec.t_admitted,
        t_first_token=rec.t_first_token,
        t_last_token=rec.t_last_token,
    )


def _slotgw_dict(rec: RequestRecord, shape: Shape, pacer: Pacer, bill: Bill | None) -> dict[str, Any]:
    d: dict[str, Any] = {
        "request_id": rec.request_id,
        "shape": shape.name,
        "replica": rec.replica,
        "ttft_ms": rec.ttft_ms,
        "queued_ms": rec.queued_ms,
        "throttled_ms": rec.throttled_ms,
        "pacing": pacer.enabled,
    }
    if bill is not None:
        d["billing"] = bill.as_dict()
    return d


def _finalize(rec: RequestRecord, usage: Usage | None, counted: int, pacer: Pacer) -> None:
    rec.output_tokens = usage.completion_tokens if usage else counted
    if usage and usage.prompt_tokens:
        rec.prompt_tokens = usage.prompt_tokens
    rec.output_tokens_guaranteed = rec.output_tokens  # milestone 2: everything is guaranteed
    rec.output_tokens_burst = 0
    rec.throttled_ms = pacer.throttled_ms


async def _complete_response(
    state: GatewayState,
    replica: Replica,
    shape: Shape,
    payload: dict[str, Any],
    rec: RequestRecord,
    created: int,
    pacer: Pacer,
) -> JSONResponse:
    parts: list[str] = []
    counted = 0
    usage: Usage | None = None
    finish: str | None = None
    t_release: float | None = None
    try:
        async for chunk in replica.stream_chat(payload, rec.request_id):
            if chunk.text or chunk.n_tokens:
                t_now = now()
                if rec.t_first_token is None:
                    rec.t_first_token = t_now
                counted += chunk.n_tokens
                parts.append(chunk.text)
                # Virtual pacing: run the chunk through the same bucket a stream would use.
                t_deliver = pacer.schedule(chunk.n_tokens, t_now)
                t_release = t_deliver if t_release is None else max(t_release, t_deliver)
            if chunk.usage:
                usage = chunk.usage
            if chunk.finish_reason:
                finish = chunk.finish_reason
        rec.t_engine_done = now()
    except ReplicaError as e:
        rec.status, rec.http_status, rec.abort_reason = "upstream_error", 502, e.detail
        rec.t_engine_done = now()
        _finalize(rec, usage, counted, pacer)
        state.meter.record(rec)
        return openai_error(502, f"replica error: {e.detail}", code="upstream_error", type_="server_error")
    except asyncio.CancelledError:
        rec.status, rec.abort_reason = "client_disconnect", "client_disconnect"
        _finalize(rec, usage, counted, pacer)
        rec.wasted_tokens = rec.output_tokens
        state.meter.record(rec)
        raise
    _finalize(rec, usage, counted, pacer)
    # Release the response when the paced stream would have delivered its last token (DESIGN 6).
    if t_release is not None:
        delay = t_release - now()
        if delay > 0:
            await asyncio.sleep(delay)
    rec.t_last_token = now()
    bill = _bill(state, rec, shape)
    state.meter.record(rec)
    return JSONResponse(
        {
            "id": rec.request_id,
            "object": "chat.completion",
            "created": created,
            "model": shape.name,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "".join(parts)},
                    "finish_reason": finish or "stop",
                }
            ],
            "usage": _usage_dict(rec, usage, counted),
            "slotgw": _slotgw_dict(rec, shape, pacer, bill),
        },
        headers={"X-Request-Id": rec.request_id},
    )


async def _stream_response(
    state: GatewayState,
    replica: Replica,
    shape: Shape,
    payload: dict[str, Any],
    rec: RequestRecord,
    created: int,
    pacer: Pacer,
) -> AsyncIterator[str]:
    """Consume the engine as fast as it produces (producer task), deliver at bucket pace."""
    produced = 0
    delivered = 0
    usage: Usage | None = None
    finish: str | None = None
    recorded = False
    queue: asyncio.Queue[TokenChunk | ReplicaError | None] = asyncio.Queue()

    async def produce() -> None:
        nonlocal produced, usage, finish
        try:
            async for chunk in replica.stream_chat(payload, rec.request_id):
                produced += chunk.n_tokens
                if chunk.usage:
                    usage = chunk.usage
                if chunk.finish_reason:
                    finish = chunk.finish_reason
                await queue.put(chunk)
            rec.t_engine_done = now()
            await queue.put(None)
        except ReplicaError as e:
            rec.t_engine_done = now()
            await queue.put(e)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001 - anything else is an upstream failure to the client
            rec.t_engine_done = now()
            await queue.put(ReplicaError(502, f"{type(e).__name__}: {e}"))

    producer = asyncio.create_task(produce())

    def chunk_obj(delta: dict[str, Any], finish_reason: str | None) -> dict[str, Any]:
        return {
            "id": rec.request_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": shape.name,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }

    try:
        yield _sse(chunk_obj({"role": "assistant", "content": ""}, None))
        while True:
            item = await queue.get()
            if item is None:
                break
            if isinstance(item, ReplicaError):
                raise item
            if item.text or item.n_tokens:
                t_now = now()
                if rec.t_first_token is None:
                    rec.t_first_token = t_now
                t_deliver = pacer.schedule(item.n_tokens, t_now)
                if t_deliver > t_now:
                    await asyncio.sleep(t_deliver - t_now)
                yield _sse(chunk_obj({"content": item.text}, None))
                delivered += item.n_tokens
                rec.t_last_token = now()
        _finalize(rec, usage, produced, pacer)
        if rec.t_last_token is None:
            rec.t_last_token = now()
        bill = _bill(state, rec, shape)
        final = chunk_obj({}, finish or "stop")
        final["usage"] = _usage_dict(rec, usage, produced)
        final["slotgw"] = _slotgw_dict(rec, shape, pacer, bill)
        yield _sse(final)
        yield "data: [DONE]\n\n"
        rec.status = "ok"
        state.meter.record(rec)
        recorded = True
    except ReplicaError as e:
        rec.status, rec.http_status, rec.abort_reason = "upstream_error", 502, e.detail
        _finalize(rec, usage, produced, pacer)
        rec.wasted_tokens = max(0, produced - delivered)
        state.meter.record(rec)
        recorded = True
        yield _sse({"error": {"message": f"replica error: {e.detail}", "type": "server_error", "code": "upstream_error"}})
        yield "data: [DONE]\n\n"
    except (asyncio.CancelledError, GeneratorExit):
        rec.status, rec.abort_reason = "client_disconnect", "client_disconnect"
        _finalize(rec, usage, produced, pacer)
        rec.wasted_tokens = max(0, produced - delivered)  # generated but never delivered
        state.meter.record(rec)
        recorded = True
        raise
    finally:
        if not producer.done():
            producer.cancel()  # aborts the upstream request; the engine stops generating
        if not recorded:
            rec.status = "error" if rec.status == "ok" else rec.status
            state.meter.record(rec)
