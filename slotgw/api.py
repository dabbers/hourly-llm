"""OpenAI-compatible API. The OpenAI `model` field names the shape to run against
(docs/DESIGN.md section 4.1): auth, shape check (context bursts), admission (ledgers, reservations,
queues), pacing with burst, billing, meter; admin endpoints for replicas and reservations."""

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
from .admission import Hold, QueueTimeout, ReservationCapacity, Scheduler, Unsellable
from .billing import Bill, bill_request
from .clock import now
from .config import Config, Shape, Tenant
from .meter import Meter, RequestRecord, ReservationRecord
from .pacing import Pacer
from .replicas.base import Replica, ReplicaError, TokenChunk, Usage

log = logging.getLogger("slotgw.api")

_PASSTHROUGH_EXCLUDE = {"model", "shape", "stream", "stream_options", "priority"}
PRIORITY_INTERACTIVE = 0  # vLLM --scheduling-policy priority: lower runs first
PRIORITY_LAZY = 5
PRIORITY_BURST = 10
RETRY_AFTER_S = 5


class BurstRevoked(Exception):
    """The burst manager aborted this context burst for a guaranteed admission."""


class GatewayState:
    def __init__(self, config: Config, meter: Meter, replicas: list[Replica], scheduler: Scheduler | None = None):
        self.config = config
        self.meter = meter
        self.replicas = replicas
        self.scheduler = scheduler
        self._by_id = {r.id: r for r in replicas}

    def select_replica(self, shape: Shape) -> Replica | None:
        """Without admission (milestone-2 mode): first replica serving the shape's model."""
        for r in self.replicas:
            if r.model_alias == shape.model:
                return r
        return None

    def replica_by_id(self, replica_id: str) -> Replica:
        return self._by_id[replica_id]

    def pacer_for(self, shape: Shape) -> Pacer:
        p = self.config.gateway.pacing
        return Pacer(shape.tps, p.bucket_depth_s, p.enabled)

    def context_burst_cap(self, shape: Shape) -> int:
        cap = self.config.gateway.burst.max_context_tokens
        if cap > 0:
            return cap
        return max(s.max_context for s in self.config.shapes if s.model == shape.model)

    def burst_shape_for(self, shape: Shape) -> Shape:
        """The shape a context burst of `shape` runs as (DESIGN 12, post-M4): a request with no
        floor takes the cheapest seat, so by default the model's slowest non-lazy shape sets its
        tps (seat check, pacing) and price; `context_tps_class: own` keeps the request's shape."""
        if self.config.gateway.burst.context_tps_class == "own":
            return shape
        candidates = [s for s in self.config.shapes if s.model == shape.model and not s.lazy]
        return min(candidates, key=lambda s: (s.tps, -s.max_context), default=shape)


def openai_error(
    status: int, message: str, code: str | None = None, type_: str = "invalid_request_error", headers: dict | None = None
) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"error": {"message": message, "type": type_, "param": None, "code": code}},
        headers=headers,
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
    rec.showcase = request.headers.get("x-slotgw-showcase") or None


def _store_text(state: GatewayState, rec: RequestRecord, payload: dict[str, Any], text: str) -> None:
    """Keep the completion (and a showcase request's prompt) when gateway.store_text is on."""
    if not state.config.gateway.store_text:
        return
    prompt_json = json.dumps(payload.get("messages"), ensure_ascii=False) if rec.showcase else None
    try:
        state.meter.record_text(rec.request_id, text, prompt_json)
    except Exception:  # the sample is a convenience, never worth failing the request
        log.exception("could not store text for %s", rec.request_id)


def _release(state: GatewayState, hold: Hold | None) -> None:
    if hold is not None and state.scheduler is not None:
        state.scheduler.release(hold)


def _prefill_done(state: GatewayState, hold: Hold | None) -> None:
    if hold is not None and not hold.prefill_done and state.scheduler is not None:
        state.scheduler.mark_prefill_done(hold)


def _reservation_record(state: GatewayState, res, shape: Shape) -> ReservationRecord:
    factor = state.config.pricing.shape.reservation_price_factor
    return ReservationRecord(
        reservation_id=res.id,
        tenant=res.tenant,
        shape=shape.name,
        slots=len(res.slots),
        t_start=res.t_start,
        t_end=res.t_end,
        price_per_second=(shape.price_per_second or 0.0) * factor,
        replicas=";".join(sorted({s.ledger.id for s in res.slots})),
        served=res.served,
        timeouts=res.timeouts,
        load_run=res.tags.get("load_run"),
        scenario=res.tags.get("scenario"),
        seed=res.tags.get("seed"),
    )


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
            "admission": state.config.gateway.admission.model_dump(),
            "burst": state.config.gateway.burst.model_dump(),
            "price_reference": state.config.pricing.token.reference,
        }

    @router.get("/admin/replicas")
    async def admin_replicas(request: Request) -> dict[str, Any]:
        state: GatewayState = request.app.state.gw
        ledgers = {ledg.id: ledg for ledg in state.scheduler.ledgers} if state.scheduler else {}
        out = []
        for r in state.replicas:
            h = await r.health()
            cc = await r.cache_config()
            ledg = ledgers.get(r.id)
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
                    "ledger": ledg.snapshot() if ledg else None,
                }
            )
        sched = None
        if state.scheduler:
            snap = state.scheduler.snapshot()
            sched = {k: v for k, v in snap.items() if k != "ledgers"}
        return {"replicas": out, "scheduler": sched}

    @router.get("/admin/reservations")
    async def admin_reservations(request: Request) -> dict[str, Any]:
        state: GatewayState = request.app.state.gw
        if state.scheduler is None:
            return {"reservations": []}
        return {"reservations": [r.snapshot() for r in state.scheduler.reservations.values()]}

    @router.post("/admin/reservations")
    async def admin_reserve(request: Request):
        """Create a standing reservation: {tenant, shape, slots, replica?, load_run?, scenario?, seed?}."""
        state: GatewayState = request.app.state.gw
        if state.scheduler is None:
            return openai_error(503, "admission control is off; reservations need ledgers", code="no_admission", type_="server_error")
        try:
            body = await request.json()
        except json.JSONDecodeError:
            return openai_error(400, "request body is not valid JSON")
        tenant = next((t for t in state.config.tenants if t.name == body.get("tenant")), None)
        if tenant is None:
            return openai_error(404, f"unknown tenant {body.get('tenant')!r}", code="tenant_not_found")
        shape = state.config.shape(str(body.get("shape") or tenant.default_shape))
        if shape is None:
            return openai_error(404, f"unknown shape {body.get('shape')!r}", code="model_not_found")
        try:
            slots = int(body.get("slots", 1))
        except (TypeError, ValueError):
            return openai_error(400, "'slots' must be an integer")
        tags = {k: body.get(k) for k in ("load_run", "scenario", "seed") if body.get(k) is not None}
        replica_id = body.get("replica") or None  # pin every slot to one replica (phase 2 placement rule)
        try:
            res = state.scheduler.reserve(tenant.name, shape, slots, body.get("id"), tags, replica_id=replica_id)
        except Unsellable:
            return openai_error(503, f"no replica can honour shape {shape.name}", code="shape_unsellable", type_="server_error")
        except ReservationCapacity as e:
            return openai_error(
                409, f"{slots} x {shape.name} does not fit ({e.reason}; {e.placed} placed)", code="reservation_capacity", type_="server_error"
            )
        except ValueError as e:
            return openai_error(400, str(e))
        state.meter.record_reservation(_reservation_record(state, res, shape))
        return JSONResponse(res.snapshot(), status_code=201)

    @router.delete("/admin/reservations/{reservation_id}")
    async def admin_unreserve(request: Request, reservation_id: str):
        state: GatewayState = request.app.state.gw
        if state.scheduler is None:
            return openai_error(503, "admission control is off", code="no_admission", type_="server_error")
        res = state.scheduler.unreserve(reservation_id)
        if res is None:
            return openai_error(404, f"no active reservation {reservation_id!r}", code="reservation_not_found")
        state.meter.record_reservation(_reservation_record(state, res, res.shape))
        return res.snapshot()

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
                    "max_queue_s": s.max_queue_s,  # how long the gateway may hold a request before refusing it
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

        burst_ctx = False
        need = prompt_tokens + max_tokens
        if need > shape.max_context:
            bs = state.config.gateway.burst
            if state.scheduler is not None and bs.enabled and bs.context and not shape.lazy and need <= state.context_burst_cap(shape):
                burst_ctx = True  # context burst: free KV beyond the shape, revocable (DESIGN 7)
                rec.class_ctx = "burst"
            else:
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
        payload["priority"] = PRIORITY_BURST if burst_ctx else (PRIORITY_LAZY if shape.lazy else PRIORITY_INTERACTIVE)

        # A context burst runs as its burst class (the model's slowest shape by default): that shape's
        # tps takes the seat, paces the stream and prices the runtime; the tenant's shape stays on the record.
        billed_shape = state.burst_shape_for(shape) if burst_ctx else shape
        rec.billed_shape = billed_shape.name
        pacer = state.pacer_for(billed_shape)
        hold: Hold | None = None
        if state.scheduler is not None:
            admission = state.config.gateway.admission
            preemptable = shape.lazy and not stream and admission.lazy_preempt == "requeue"
            try:
                hold = await state.scheduler.admit(
                    request_id, billed_shape, prompt_tokens, tenant=tenant.name, preemptable=preemptable,
                    kv_tokens=need if burst_ctx else None, burst_ctx=burst_ctx,
                )
            except Unsellable:
                rec.status, rec.http_status, rec.abort_reason = "refused", 503, "shape_unsellable"
                state.meter.record(rec)
                return openai_error(
                    503, f"no replica can honour shape {shape.name}", code="shape_unsellable", type_="server_error"
                )
            except QueueTimeout as e:
                rec.status, rec.http_status, rec.abort_reason = "refused", 503, f"capacity:{e.reason}"
                rec.queued_ms = e.waited_s * 1000.0
                state.meter.record(rec)
                return openai_error(
                    503,
                    f"no capacity for shape {shape.name} within {shape.max_queue_s:.0f}s ({e.reason})",
                    code="capacity",
                    type_="server_error",
                    headers={"Retry-After": str(RETRY_AFTER_S)},
                )
            replica = state.replica_by_id(hold.ledger.id)
            rec.replica = replica.id
            rec.reservation_id = hold.reservation_id if hold.reserved else None
            hold.pacer = pacer
            state.scheduler.dispatch()  # lends this request its burst rate
        rec.t_admitted = now()
        rec.queued_ms = (rec.t_admitted - rec.t_arrival) * 1000.0
        created = int(rec.t_admitted)

        if stream:
            return StreamingResponse(
                _stream_response(state, replica, shape, payload, rec, created, pacer, hold),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Request-Id": request_id},
            )
        return await _complete_response(state, replica, shape, payload, rec, created, pacer, hold)

    return router


def _usage_dict(rec: RequestRecord, usage: Usage | None, counted: int) -> dict[str, Any]:
    completion = usage.completion_tokens if usage else counted
    prompt = usage.prompt_tokens if usage and usage.prompt_tokens else rec.prompt_tokens
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
        "prompt_tokens_details": {"cached_tokens": rec.cached_tokens},
    }


def _bill(state: GatewayState, rec: RequestRecord, shape: Shape) -> Bill:
    billed = state.config.shape(rec.billed_shape) if rec.billed_shape else None
    return bill_request(
        shape=billed or shape,
        pricing=state.config.pricing,
        prompt_tokens=rec.prompt_tokens,
        output_tokens=rec.output_tokens,
        delivered_tokens=rec.delivered_tokens,
        t_admitted=rec.t_admitted,
        t_first_token=rec.t_first_token,
        t_last_token=rec.t_last_token,
        cached_tokens=rec.cached_tokens,
    )


def _slotgw_dict(rec: RequestRecord, shape: Shape, pacer: Pacer, bill: Bill | None) -> dict[str, Any]:
    d: dict[str, Any] = {
        "request_id": rec.request_id,
        "shape": shape.name,
        "billed_shape": rec.billed_shape or shape.name,
        "replica": rec.replica,
        "class_ctx": rec.class_ctx,
        "reservation_id": rec.reservation_id,
        "cached_tokens": rec.cached_tokens,
        "ttft_ms": rec.ttft_ms,
        "queued_ms": rec.queued_ms,
        "throttled_ms": rec.throttled_ms,
        "preempted_runs": rec.preempted_runs,
        "burst_tokens": rec.output_tokens_burst,
        "burst_rate_max": round(pacer.burst_rate_max, 2),
        "pacing": pacer.enabled,
    }
    if bill is not None:
        d["billing"] = {**bill.as_dict(), "reserved": rec.reservation_id is not None}
    return d


def _finalize(rec: RequestRecord, usage: Usage | None, counted: int, pacer: Pacer, delivered: int) -> None:
    rec.output_tokens = usage.completion_tokens if usage else counted
    if usage and usage.prompt_tokens:
        rec.prompt_tokens = usage.prompt_tokens
    rec.cached_tokens = max(0, min(usage.cached_tokens, rec.prompt_tokens)) if usage else 0
    rec.output_tokens_burst = min(rec.output_tokens, pacer.burst_tokens)
    rec.output_tokens_guaranteed = rec.output_tokens - rec.output_tokens_burst
    rec.throttled_ms = pacer.throttled_ms
    rec.delivered_tokens = max(0, delivered)


def _revoked(rec: RequestRecord) -> None:
    rec.status, rec.http_status, rec.abort_reason, rec.revoked = "revoked", 503, "burst_revoked", True


async def _consume_engine(
    state: GatewayState,
    replica: Replica,
    payload: dict[str, Any],
    rec: RequestRecord,
    pacer: Pacer,
    hold: Hold | None,
    progress: dict[str, int],
) -> tuple[list[str], int, Usage | None, str | None, float | None]:
    """One engine run for a complete response: drain the engine, run every chunk through the
    bucket virtually, free the seat when the engine is done (or fails, or is cancelled)."""
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
                    _prefill_done(state, hold)
                counted += chunk.n_tokens
                progress["counted"] = counted
                parts.append(chunk.text)
                t_deliver = pacer.schedule(chunk.n_tokens, t_now)
                t_release = t_deliver if t_release is None else max(t_release, t_deliver)
            if chunk.usage:
                usage = chunk.usage
            if chunk.finish_reason:
                finish = chunk.finish_reason
        rec.t_engine_done = now()
    finally:
        _release(state, hold)
    return parts, counted, usage, finish, t_release


async def _complete_response(
    state: GatewayState,
    replica: Replica,
    shape: Shape,
    payload: dict[str, Any],
    rec: RequestRecord,
    created: int,
    pacer: Pacer,
    hold: Hold | None,
) -> JSONResponse:
    while True:
        progress = {"counted": 0}
        task = asyncio.create_task(_consume_engine(state, replica, payload, rec, pacer, hold, progress))
        if hold is not None:
            hold.task = task
        try:
            parts, counted, usage, finish, t_release = await task
            break
        except ReplicaError as e:
            rec.status, rec.http_status, rec.abort_reason = "upstream_error", 502, e.detail
            rec.t_engine_done = now()
            _finalize(rec, None, progress["counted"], pacer, 0)
            rec.wasted_tokens += progress["counted"]
            state.meter.record(rec)
            return openai_error(502, f"replica error: {e.detail}", code="upstream_error", type_="server_error")
        except asyncio.CancelledError:
            if hold is not None and hold.revoked:
                rec.t_engine_done = now()
                _finalize(rec, None, progress["counted"], pacer, 0)
                rec.wasted_tokens += progress["counted"]
                _revoked(rec)
                state.meter.record(rec)
                return openai_error(
                    503, "context burst revoked: a guaranteed request needed the capacity", code="burst_revoked",
                    type_="server_error", headers={"Retry-After": str(RETRY_AFTER_S)},
                )
            if hold is not None and hold.preempted and state.scheduler is not None:
                # Policy `requeue`: the run is discarded unbilled and the request re-enters at the
                # head of the lazy queue (docs/DESIGN.md 4.1 step 9).
                rec.preempted_runs += 1
                rec.wasted_tokens += progress["counted"]
                rec.t_first_token = None
                rec.t_engine_done = None
                pacer = state.pacer_for(shape)
                try:
                    hold = await state.scheduler.admit(
                        rec.request_id, shape, rec.prompt_tokens, tenant=rec.tenant, preemptable=True, front=True
                    )
                except QueueTimeout as e:
                    rec.status, rec.http_status, rec.abort_reason = "refused", 503, f"capacity:{e.reason}"
                    rec.queued_ms = (now() - rec.t_arrival) * 1000.0
                    state.meter.record(rec)
                    return openai_error(
                        503, f"no capacity after preemption ({e.reason})", code="capacity", type_="server_error",
                        headers={"Retry-After": str(RETRY_AFTER_S)},
                    )
                replica = state.replica_by_id(hold.ledger.id)
                rec.replica = replica.id
                hold.pacer = pacer
                rec.t_admitted = now()  # the billed run starts now; everything before is unbilled
                rec.queued_ms = (rec.t_admitted - rec.t_arrival) * 1000.0
                continue
            rec.status, rec.abort_reason = "client_disconnect", "client_disconnect"
            _finalize(rec, None, progress["counted"], pacer, 0)
            rec.wasted_tokens += progress["counted"]
            state.meter.record(rec)
            raise
    _finalize(rec, usage, counted, pacer, delivered=usage.completion_tokens if usage else counted)
    # Release the response when the paced stream would have delivered its last token (DESIGN 6).
    if t_release is not None:
        delay = t_release - now()
        if delay > 0:
            await asyncio.sleep(delay)
    rec.t_last_token = now()
    bill = _bill(state, rec, shape)
    state.meter.record(rec)
    _store_text(state, rec, payload, "".join(parts))
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
    hold: Hold | None,
) -> AsyncIterator[str]:
    """Consume the engine as fast as it produces (producer task), deliver at bucket pace. The seat
    is freed when the engine finishes, which can be well before delivery ends."""
    produced = 0
    delivered = 0
    usage: Usage | None = None
    finish: str | None = None
    recorded = False
    texts: list[str] = []  # delivered text, kept for the report's output samples when store_text is on
    queue: asyncio.Queue[TokenChunk | ReplicaError | BurstRevoked | None] = asyncio.Queue()

    async def produce() -> None:
        nonlocal produced, usage, finish
        try:
            async for chunk in replica.stream_chat(payload, rec.request_id):
                if (chunk.text or chunk.n_tokens) and produced == 0:
                    _prefill_done(state, hold)
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
            if hold is not None and hold.revoked:
                rec.t_engine_done = now()
                await queue.put(BurstRevoked())
                return
            raise
        except Exception as e:  # noqa: BLE001 - anything else is an upstream failure to the client
            rec.t_engine_done = now()
            await queue.put(ReplicaError(502, f"{type(e).__name__}: {e}"))
        finally:
            _release(state, hold)

    producer = asyncio.create_task(produce())
    if hold is not None:
        hold.task = producer

    def chunk_obj(delta: dict[str, Any], finish_reason: str | None, delivered_tokens: int | None = None) -> dict[str, Any]:
        obj: dict[str, Any] = {
            "id": rec.request_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": shape.name,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }
        if delivered_tokens is not None:  # a chunk may carry several tokens; clients can count them
            obj["slotgw"] = {"delivered_tokens": delivered_tokens}
        return obj

    try:
        yield _sse(chunk_obj({"role": "assistant", "content": ""}, None))
        while True:
            item = await queue.get()
            if item is None:
                break
            if isinstance(item, (ReplicaError, BurstRevoked)):
                raise item
            if item.text or item.n_tokens:
                t_now = now()
                if rec.t_first_token is None:
                    rec.t_first_token = t_now
                t_deliver = pacer.schedule(item.n_tokens, t_now)
                if t_deliver > t_now:
                    await asyncio.sleep(t_deliver - t_now)
                delivered += item.n_tokens
                if item.text:
                    texts.append(item.text)
                yield _sse(chunk_obj({"content": item.text}, None, delivered))
                rec.t_last_token = now()
        _finalize(rec, usage, produced, pacer, delivered)
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
    except BurstRevoked:
        _finalize(rec, usage, produced, pacer, delivered)
        rec.wasted_tokens = max(0, produced - delivered)
        _revoked(rec)
        state.meter.record(rec)
        recorded = True
        yield _sse({"error": {"message": "context burst revoked: a guaranteed request needed the capacity", "type": "server_error", "code": "burst_revoked"}})
        yield "data: [DONE]\n\n"
    except ReplicaError as e:
        rec.status, rec.http_status, rec.abort_reason = "upstream_error", 502, e.detail
        _finalize(rec, usage, produced, pacer, delivered)
        rec.wasted_tokens = max(0, produced - delivered)
        state.meter.record(rec)
        recorded = True
        yield _sse({"error": {"message": f"replica error: {e.detail}", "type": "server_error", "code": "upstream_error"}})
        yield "data: [DONE]\n\n"
    except (asyncio.CancelledError, GeneratorExit):
        rec.status, rec.abort_reason = "client_disconnect", "client_disconnect"
        _finalize(rec, usage, produced, pacer, delivered)
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
        if texts or rec.showcase:
            _store_text(state, rec, payload, "".join(texts))
