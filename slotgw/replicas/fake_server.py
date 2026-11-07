"""In-process fake vLLM replica: OpenAI chat completions (stream and complete) with a configurable
step-rate curve and KV budget, plus /tokenize, /metrics and /v1/models. Used by tests and by the
multi-replica scheduler tests; also runnable as a service (slotgw-fake-replica)."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from itertools import pairwise
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse

MIN_SLEEP_S = 0.016  # below the clock resolution of some event loops a sleep returns at once

WORDS = ["the", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog", "while", "the", "meter", "counts", "every", "token", "that", "the", "fake", "replica", "emits", "at", "a", "calibrated", "rate"]


def parse_curve(text: str) -> list[tuple[int, float]]:
    pts = []
    for part in text.split(","):
        n, rate = part.split(":")
        pts.append((int(n), float(rate)))
    pts.sort()
    return pts


def step_rate(curve: list[tuple[int, float]], n: int) -> float:
    """Per-stream decode tokens/s at n concurrent streams, linear between calibration points."""
    n = max(1, n)
    if n <= curve[0][0]:
        return curve[0][1]
    for (n0, r0), (n1, r1) in pairwise(curve):
        if n0 <= n <= n1:
            if n1 == n0:
                return r1
            return r0 + (r1 - r0) * (n - n0) / (n1 - n0)
    return curve[-1][1]


_TOKEN_RE = re.compile(r"\w+|[^\w\s]")


def estimate_tokens(messages: list[dict[str, Any]]) -> int:
    """Roughly one token per word or punctuation mark, plus chat-template overhead. Matches how
    the load generator sizes its prompts (loadgen/content.py) so fake runs see realistic counts."""
    total = 3
    for m in messages:
        content = m.get("content") or ""
        if isinstance(content, list):
            content = " ".join(str(p.get("text", "")) for p in content if isinstance(p, dict))
        total += len(_TOKEN_RE.findall(str(content))) + 4
    return total


@dataclass
class FakeState:
    model: str
    kv_tokens: int
    block_size: int
    curve: list[tuple[int, float]]
    prefill_tps: float
    running: dict[str, int] = field(default_factory=dict)  # request_id -> context tokens held
    completed: int = 0
    generated_tokens: int = 0
    prompt_tokens: int = 0
    prefix_cache: bool = True
    seen_prefixes: set[str] = field(default_factory=set)  # hashes of message-list prefixes already prefilled

    @property
    def kv_used(self) -> int:
        return sum(self.running.values())

    def cached_tokens(self, messages: list[dict[str, Any]], prompt_tokens: int) -> int:
        """Emulate vLLM's prefix cache: the longest message-list prefix prefilled before counts as
        cached (whole blocks), so a multi-turn session's history is cached from its second turn on."""
        if not self.prefix_cache:
            return 0
        keys = [hashlib.sha1(json.dumps(messages[:k], sort_keys=True).encode()).hexdigest() for k in range(1, len(messages) + 1)]
        cached = 0
        for k in range(len(messages), 0, -1):
            if keys[k - 1] in self.seen_prefixes:
                cached = estimate_tokens(messages[:k]) // self.block_size * self.block_size
                break
        if len(self.seen_prefixes) > 100_000:
            self.seen_prefixes.clear()
        self.seen_prefixes.update(keys)
        return max(0, min(cached, prompt_tokens))


def create_fake_app(
    model: str = "Qwen/Qwen2.5-3B-Instruct-AWQ",
    kv_tokens: int = 160_000,
    block_size: int = 16,
    curve: str = "1:60,8:40,16:30,32:20",
    prefill_tps: float = 2000.0,
    prefix_cache: bool = True,
) -> FastAPI:
    state = FakeState(
        model=model,
        kv_tokens=kv_tokens,
        block_size=block_size,
        curve=parse_curve(curve),
        prefill_tps=prefill_tps,
        prefix_cache=prefix_cache,
    )
    app = FastAPI(title="fake-vllm")
    app.state.fake = state

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {"ok": True}

    @app.get("/v1/models")
    async def models() -> dict[str, Any]:
        return {"object": "list", "data": [{"id": state.model, "object": "model", "owned_by": "fake"}]}

    @app.post("/tokenize")
    async def tokenize(request: Request) -> dict[str, Any]:
        body = await request.json()
        if "messages" in body:
            count = estimate_tokens(body["messages"])
        else:
            count = len(str(body.get("prompt", ""))) // 4 + 1
        return {"count": count, "max_model_len": 32768, "tokens": []}

    @app.get("/metrics")
    async def metrics() -> PlainTextResponse:
        lines = [
            f'vllm:cache_config_info{{block_size="{state.block_size}",num_gpu_blocks="{state.kv_tokens // state.block_size}",cache_dtype="auto"}} 1.0',
            f"vllm:num_requests_running {len(state.running)}",
            "vllm:num_requests_waiting 0",
            f"vllm:kv_cache_usage_perc {min(1.0, state.kv_used / state.kv_tokens):.6f}",
            "vllm:num_preemptions_total 0",
            f"vllm:generation_tokens_total {state.generated_tokens}",
            f"vllm:prompt_tokens_total {state.prompt_tokens}",
            f"vllm:request_success_total {state.completed}",
        ]
        return PlainTextResponse("\n".join(lines) + "\n")

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        body = await request.json()
        messages = body.get("messages") or []
        prompt_tokens = estimate_tokens(messages)
        cached_tokens = state.cached_tokens(messages, prompt_tokens)
        max_tokens = int(body.get("max_tokens") or 64)
        ignore_eos = bool(body.get("ignore_eos", False))
        if ignore_eos:
            n_out = max_tokens
        else:
            n_out = min(max_tokens, 1 + (sum(ord(c) for c in json.dumps(messages)) % 40))
        stream = bool(body.get("stream", False))
        include_usage = bool((body.get("stream_options") or {}).get("include_usage", False))
        continuous = bool((body.get("stream_options") or {}).get("continuous_usage_stats", False))
        request_id = "chatcmpl-" + uuid.uuid4().hex[:24]
        created = int(time.time())
        context = prompt_tokens + n_out

        async def generate():
            """Yields (tokens_done, k, text): k tokens due by now. Tokens are scheduled on absolute
            due times at the current step rate; sleeps never go below MIN_SLEEP_S (event-loop
            clocks are coarse on some platforms), and tokens that fell due meanwhile go out in one
            chunk, so the long-run rate matches the curve on every platform."""
            state.running[request_id] = context
            state.prompt_tokens += prompt_tokens
            try:
                await asyncio.sleep((prompt_tokens - cached_tokens) / state.prefill_tps)  # cache hits skip prefill
                due = time.perf_counter()
                i = 0
                while i < n_out:
                    rate = step_rate(state.curve, len(state.running))
                    due += 1.0 / rate
                    delay = due - time.perf_counter()
                    if delay > 0:
                        await asyncio.sleep(max(delay, MIN_SLEEP_S))
                    k = 1
                    while i + k < n_out and due + k / rate <= time.perf_counter():
                        k += 1
                    due += (k - 1) / rate
                    state.generated_tokens += k
                    text = "".join(WORDS[(i + j) % len(WORDS)] + " " for j in range(k))
                    i += k
                    yield i, k, text
            finally:
                state.running.pop(request_id, None)
                state.completed += 1

        finish_reason = "length" if n_out >= max_tokens else "stop"

        def usage_obj(completion_tokens: int) -> dict[str, Any]:
            return {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "prompt_tokens_details": {"cached_tokens": cached_tokens},
            }

        if not stream:
            parts: list[str] = []
            async for _, _, text in generate():
                parts.append(text)
            return JSONResponse(
                {
                    "id": request_id,
                    "object": "chat.completion",
                    "created": created,
                    "model": state.model,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "".join(parts)},
                            "finish_reason": finish_reason,
                        }
                    ],
                    "usage": usage_obj(n_out),
                }
            )

        def chunk(delta: dict[str, Any], finish: str | None, completion_tokens: int | None) -> str:
            obj: dict[str, Any] = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": state.model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
            }
            if completion_tokens is not None:
                obj["usage"] = usage_obj(completion_tokens)
            return f"data: {json.dumps(obj)}\n\n"

        async def sse():
            yield chunk({"role": "assistant", "content": ""}, None, 0 if continuous else None)
            done = 0
            async for done, _, text in generate():
                yield chunk({"content": text}, None, done if continuous else None)
            yield chunk({}, finish_reason, done if (continuous or include_usage) else None)
            if include_usage and not continuous:
                obj = {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": state.model,
                    "choices": [],
                    "usage": usage_obj(done),
                }
                yield f"data: {json.dumps(obj)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(sse(), media_type="text/event-stream")

    return app


def main() -> None:
    import uvicorn

    p = argparse.ArgumentParser(description="fake vLLM replica")
    p.add_argument("--host", default=os.environ.get("FAKE_HOST", "0.0.0.0"))
    p.add_argument("--port", type=int, default=int(os.environ.get("FAKE_PORT", "8001")))
    p.add_argument("--model", default=os.environ.get("FAKE_MODEL", "Qwen/Qwen2.5-3B-Instruct-AWQ"))
    p.add_argument("--kv-tokens", type=int, default=int(os.environ.get("FAKE_KV_TOKENS", "160000")))
    p.add_argument("--curve", default=os.environ.get("FAKE_RATE_CURVE", "1:60,8:40,16:30,32:20"))
    p.add_argument("--prefill-tps", type=float, default=float(os.environ.get("FAKE_PREFILL_TPS", "2000")))
    args = p.parse_args()
    app = create_fake_app(
        model=args.model, kv_tokens=args.kv_tokens, curve=args.curve, prefill_tps=args.prefill_tps
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
