"""HTTP client for a vLLM OpenAI-compatible server (also used for the fake replica)."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx

from ..config import ReplicaConfig
from ..prom import first_labels, first_value, parse_prometheus
from .base import CacheConfig, ReplicaError, ReplicaHealth, TokenChunk, Usage


class VllmReplica:
    def __init__(self, cfg: ReplicaConfig, hf_model: str, client: httpx.AsyncClient):
        self.id = cfg.id
        self.model_alias = cfg.model
        self.hf_model = hf_model
        self.url = cfg.url.rstrip("/")
        self._client = client

    async def tokenize(self, messages: list[dict[str, Any]]) -> int:
        body = {"model": self.hf_model, "messages": messages, "add_generation_prompt": True}
        try:
            r = await self._client.post(f"{self.url}/tokenize", json=body)
        except httpx.HTTPError as e:
            raise ReplicaError(502, f"tokenize failed: {e}") from e
        if r.status_code != 200:
            raise ReplicaError(r.status_code, r.text[:500])
        return int(r.json()["count"])

    async def stream_chat(self, payload: dict[str, Any], request_id: str) -> AsyncIterator[TokenChunk]:
        body = dict(payload)
        body["model"] = self.hf_model
        body["stream"] = True
        body["stream_options"] = {"include_usage": True, "continuous_usage_stats": True}
        headers = {"X-Request-Id": request_id}
        prev_completion = 0
        try:
            async with self._client.stream(
                "POST", f"{self.url}/v1/chat/completions", json=body, headers=headers
            ) as r:
                if r.status_code != 200:
                    text = (await r.aread()).decode("utf-8", "replace")
                    raise ReplicaError(r.status_code, text[:500])
                async for line in r.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        obj = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    text = ""
                    finish = None
                    choices = obj.get("choices") or []
                    if choices:
                        delta = choices[0].get("delta") or {}
                        text = delta.get("content") or ""
                        finish = choices[0].get("finish_reason")
                    usage_obj = obj.get("usage")
                    usage = None
                    n_tokens = 0
                    if usage_obj and usage_obj.get("completion_tokens") is not None:
                        cum = int(usage_obj["completion_tokens"])
                        n_tokens = max(0, cum - prev_completion)
                        prev_completion = cum
                        details = usage_obj.get("prompt_tokens_details") or {}
                        usage = Usage(
                            prompt_tokens=int(usage_obj.get("prompt_tokens") or 0),
                            completion_tokens=cum,
                            cached_tokens=int(details.get("cached_tokens") or 0),
                        )
                    elif text:
                        n_tokens = 1
                    yield TokenChunk(text=text, n_tokens=n_tokens, finish_reason=finish, usage=usage)
        except httpx.HTTPError as e:
            raise ReplicaError(502, f"upstream stream failed: {e}") from e

    async def _metrics(self) -> list:
        r = await self._client.get(f"{self.url}/metrics")
        if r.status_code != 200:
            raise ReplicaError(r.status_code, r.text[:200])
        return parse_prometheus(r.text)

    async def health(self) -> ReplicaHealth:
        try:
            samples = await self._metrics()
        except (httpx.HTTPError, ReplicaError) as e:
            return ReplicaHealth(ok=False, error=str(e))
        kv = first_value(samples, "vllm:kv_cache_usage_perc")
        if kv is None:
            kv = first_value(samples, "vllm:gpu_cache_usage_perc")
        running = first_value(samples, "vllm:num_requests_running")
        waiting = first_value(samples, "vllm:num_requests_waiting")
        preempt = first_value(samples, "vllm:num_preemptions_total")
        raw = {s.name: s.value for s in samples if s.name.startswith("vllm:") and not s.labels}
        return ReplicaHealth(
            ok=True,
            kv_usage=kv,
            running=int(running) if running is not None else None,
            waiting=int(waiting) if waiting is not None else None,
            preemptions=int(preempt) if preempt is not None else None,
            raw=raw,
        )

    async def cache_config(self) -> CacheConfig | None:
        try:
            samples = await self._metrics()
        except (httpx.HTTPError, ReplicaError):
            return None
        labels = first_labels(samples, "vllm:cache_config_info")
        if labels is None:
            return None

        def _int(key: str) -> int | None:
            v = labels.get(key)
            try:
                return int(v) if v not in (None, "", "None") else None
            except ValueError:
                return None

        return CacheConfig(
            num_gpu_blocks=_int("num_gpu_blocks"), block_size=_int("block_size"), raw=labels
        )
