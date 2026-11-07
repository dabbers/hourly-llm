"""Synthetic prompt text sized in tokens.

Prompts are built from a vocabulary of common English words that are single tokens in GPT-style
BPE tokenizers (Qwen2.5 included; see docs/ASSUMPTIONS.md for the measured ratio). Every word and
every sentence-ending period counts as one token, so `words(seed, n)` is n tokens of message
content; the chat template adds a fixed overhead the gateway's tokenizer counts on top. The text
is deterministic in (seed, n) so a plan can store sizes and seeds instead of megabytes of text.
"""

from __future__ import annotations

import numpy as np

VOCAB = [
    "the", "and", "that", "with", "from", "this", "have", "will", "were", "they", "their",
    "which", "about", "there", "would", "other", "could", "these", "first", "after", "should",
    "where", "before", "might", "those", "being", "under", "while", "during", "between",
    "without", "through", "against", "because", "however", "another", "around", "several",
    "within", "team", "meeting", "notes", "action", "items", "review", "plan", "budget",
    "launch", "product", "design", "customer", "support", "request", "update", "status",
    "report", "summary", "decision", "option", "risk", "issue", "change", "schedule", "week",
    "month", "quarter", "year", "today", "tomorrow", "morning", "afternoon", "call", "email",
    "message", "project", "task", "owner", "deadline", "priority", "result", "outcome", "goal",
    "metric", "number", "value", "cost", "price", "time", "data", "model", "system", "service",
    "server", "client", "network", "storage", "memory", "cache", "token", "rate", "limit",
    "queue", "batch", "stream", "shape", "slot", "billing", "invoice", "account", "usage",
    "traffic", "load", "test", "build", "release", "version", "feature", "bug", "fix", "error",
    "warning", "check", "verify", "confirm", "agree", "discuss", "propose", "suggest", "decide",
    "assign", "follow", "close", "open", "start", "finish", "deliver", "share", "send",
    "receive", "create", "remove", "add", "move", "copy", "save", "load", "read", "write",
    "run", "stop", "wait", "again", "still", "only", "also", "very", "much", "more", "most",
    "less", "least", "same", "different", "better", "worse", "small", "large", "long", "short",
    "high", "low", "fast", "slow", "early", "late", "new", "old", "next", "last", "main",
    "other", "people", "person", "group", "company", "office", "market", "sales", "growth",
    "revenue", "margin", "profit", "share", "question", "answer", "reason", "example", "detail",
    "context", "point", "case", "part", "side", "level", "step", "order",
]


def words(seed: int, n_tokens: int, sentence_len: tuple[int, int] = (8, 16)) -> str:
    """n_tokens of content: words plus periods, each one token; deterministic in (seed, n)."""
    if n_tokens <= 0:
        return ""
    rng = np.random.default_rng(seed)
    out: list[str] = []
    remaining = n_tokens
    while remaining > 0:
        length = int(rng.integers(sentence_len[0], sentence_len[1] + 1))
        # a sentence is `length` words plus one period, unless the budget ends first
        n_words = min(length, remaining - 1) if remaining > 1 else remaining
        idx = rng.integers(0, len(VOCAB), size=n_words)
        sentence = [VOCAB[i] for i in idx]
        sentence[0] = sentence[0].capitalize()
        remaining -= n_words
        if remaining > 0:
            out.append(" ".join(sentence) + ".")
            remaining -= 1
        else:
            out.append(" ".join(sentence))
    return " ".join(out)


def count_tokens_estimate(text: str) -> int:
    """What the vocabulary promises: words + periods."""
    return sum(1 + w.endswith(".") for w in text.split())
