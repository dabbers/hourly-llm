"""Replica interface (docs/DESIGN.md section 4.2). Real vLLM servers and in-process fakes both
speak the OpenAI API plus vLLM's /tokenize and /metrics, so one client class serves both."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class Usage:
    prompt_tokens: int
    completion_tokens: int


@dataclass
class TokenChunk:
    text: str = ""
    n_tokens: int = 0
    finish_reason: str | None = None
    usage: Usage | None = None


@dataclass
class ReplicaHealth:
    ok: bool
    kv_usage: float | None = None  # 0..1
    running: int | None = None
    waiting: int | None = None
    preemptions: int | None = None
    error: str | None = None
    raw: dict[str, float] = field(default_factory=dict)


@dataclass
class CacheConfig:
    num_gpu_blocks: int | None
    block_size: int | None
    raw: dict[str, str]

    @property
    def kv_tokens_total(self) -> int | None:
        if self.num_gpu_blocks is None or self.block_size is None:
            return None
        return self.num_gpu_blocks * self.block_size


class ReplicaError(Exception):
    def __init__(self, status: int, detail: str):
        super().__init__(f"replica error {status}: {detail}")
        self.status = status
        self.detail = detail


class Replica(Protocol):
    id: str
    model_alias: str
    hf_model: str

    async def tokenize(self, messages: list[dict[str, Any]]) -> int: ...

    def stream_chat(self, payload: dict[str, Any], request_id: str) -> AsyncIterator[TokenChunk]: ...

    async def health(self) -> ReplicaHealth: ...

    async def cache_config(self) -> CacheConfig | None: ...
