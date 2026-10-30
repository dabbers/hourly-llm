"""OpenAI-compatible API. The OpenAI `model` field names the shape to run against
(docs/DESIGN.md section 4.1). Milestone 1: auth, shape check, passthrough, meter."""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .clock import now
from .config import Config, Shape, Tenant
from .meter import Meter, RequestRecord
from .replicas.base import Replica, ReplicaError, Usage

log = logging.getLogger("slotgw.api")

_PASSTHROUGH_EXCLUDE = {"model", "shape", "stream", "stream_options"}


class GatewayState:
    def __init__(self, config: Config, meter: Meter, replicas: list[Replica]):
        self.config = config
        self.meter = meter
        self.replicas = replicas

    def select_replica(self, shape: Shape) -> Replica | None:
        # Milestone 1: first replica serving the shape's model. Milestone 3 replaces this with
        # ledger-based admission and shape-affinity scoring.
        for r in self.replicas:
            if r.model_alias == shape.model:
                return r
        return None


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


def build_router() -> APIRouter:
    router = APIRouter()

    @router.get("/healthz")
    async def healthz() -> dict[str, Any]:
        return {"ok": True}

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
        rec.t_admitted = now()  # milestone 1: no admission control, admitted on arrival
        rec.queued_ms = (rec.t_admitted - rec.t_arrival) * 1000.0
        created = int(rec.t_admitted)

        if stream:
            return StreamingResponse(
                _stream_response(state, replica, shape, payload, rec, created),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Request-Id": request_id},
            )
        return await _complete_response(state, replica, shape, payload, rec, created)

    return router


def _usage_dict(rec: RequestRecord, usage: Usage | None, counted: int) -> dict[str, Any]:
    completion = usage.completion_tokens if usage else counted
    prompt = usage.prompt_tokens if usage and usage.prompt_tokens else rec.prompt_tokens
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
    }


def _slotgw_dict(rec: RequestRecord, shape: Shape) -> dict[str, Any]:
    return {
        "request_id": rec.request_id,
        "shape": shape.name,
        "replica": rec.replica,
        "ttft_ms": rec.ttft_ms,
        "queued_ms": rec.queued_ms,
    }


def _finalize(rec: RequestRecord, usage: Usage | None, counted: int) -> None:
    rec.output_tokens = usage.completion_tokens if usage else counted
    if usage and usage.prompt_tokens:
        rec.prompt_tokens = usage.prompt_tokens
    rec.output_tokens_guaranteed = rec.output_tokens  # milestone 1: everything is guaranteed
    rec.output_tokens_burst = 0


async def _complete_response(
    state: GatewayState, replica: Replica, shape: Shape, payload: dict[str, Any], rec: RequestRecord, created: int
) -> JSONResponse:
    parts: list[str] = []
    counted = 0
    usage: Usage | None = None
    finish: str | None = None
    try:
        async for chunk in replica.stream_chat(payload, rec.request_id):
            if chunk.text or chunk.n_tokens:
                if rec.t_first_token is None:
                    rec.t_first_token = now()
                counted += chunk.n_tokens
                parts.append(chunk.text)
            if chunk.usage:
                usage = chunk.usage
            if chunk.finish_reason:
                finish = chunk.finish_reason
        rec.t_engine_done = now()
    except ReplicaError as e:
        rec.status, rec.http_status, rec.abort_reason = "upstream_error", 502, e.detail
        rec.t_engine_done = now()
        state.meter.record(rec)
        return openai_error(502, f"replica error: {e.detail}", code="upstream_error", type_="server_error")
    except asyncio.CancelledError:
        rec.status, rec.abort_reason = "client_disconnect", "client_disconnect"
        rec.wasted_tokens = counted
        state.meter.record(rec)
        raise
    _finalize(rec, usage, counted)
    rec.t_last_token = now()
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
            "slotgw": _slotgw_dict(rec, shape),
        },
        headers={"X-Request-Id": rec.request_id},
    )


async def _stream_response(
    state: GatewayState, replica: Replica, shape: Shape, payload: dict[str, Any], rec: RequestRecord, created: int
) -> AsyncIterator[str]:
    counted = 0
    usage: Usage | None = None
    finish: str | None = None
    recorded = False

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
        async for chunk in replica.stream_chat(payload, rec.request_id):
            if chunk.text or chunk.n_tokens:
                if rec.t_first_token is None:
                    rec.t_first_token = now()
                counted += chunk.n_tokens
                yield _sse(chunk_obj({"content": chunk.text}, None))
                rec.t_last_token = now()
            if chunk.usage:
                usage = chunk.usage
            if chunk.finish_reason:
                finish = chunk.finish_reason
        rec.t_engine_done = now()
        _finalize(rec, usage, counted)
        final = chunk_obj({}, finish or "stop")
        final["usage"] = _usage_dict(rec, usage, counted)
        final["slotgw"] = _slotgw_dict(rec, shape)
        yield _sse(final)
        yield "data: [DONE]\n\n"
        rec.t_last_token = now()
        rec.status = "ok"
        state.meter.record(rec)
        recorded = True
    except ReplicaError as e:
        rec.status, rec.http_status, rec.abort_reason = "upstream_error", 502, e.detail
        rec.t_engine_done = now()
        _finalize(rec, usage, counted)
        state.meter.record(rec)
        recorded = True
        yield _sse({"error": {"message": f"replica error: {e.detail}", "type": "server_error", "code": "upstream_error"}})
        yield "data: [DONE]\n\n"
    except (asyncio.CancelledError, GeneratorExit):
        rec.status, rec.abort_reason = "client_disconnect", "client_disconnect"
        _finalize(rec, usage, counted)
        state.meter.record(rec)
        recorded = True
        raise
    finally:
        if not recorded:
            rec.status = "error" if rec.status == "ok" else rec.status
            state.meter.record(rec)
