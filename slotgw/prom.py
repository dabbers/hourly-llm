"""Minimal Prometheus text-format parser (enough for vLLM's /metrics)."""

from __future__ import annotations

import re
from dataclasses import dataclass

_LABEL_RE = re.compile(r'([A-Za-z_][A-Za-z0-9_]*)="((?:[^"\\]|\\.)*)"')
_LINE_RE = re.compile(r"^([A-Za-z_:][A-Za-z0-9_:]*)(\{(.*)\})?\s+(\S+)(\s+\S+)?$")


@dataclass
class Sample:
    name: str
    labels: dict[str, str]
    value: float


def parse_prometheus(text: str) -> list[Sample]:
    out: list[Sample] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = _LINE_RE.match(line)
        if not m:
            continue
        name, _, label_text, value_text, _ = m.groups()
        labels = {k: v.replace('\\"', '"') for k, v in _LABEL_RE.findall(label_text or "")}
        try:
            value = float(value_text)
        except ValueError:
            continue
        out.append(Sample(name=name, labels=labels, value=value))
    return out


def first_value(samples: list[Sample], name: str) -> float | None:
    for s in samples:
        if s.name == name:
            return s.value
    return None


def first_labels(samples: list[Sample], name: str) -> dict[str, str] | None:
    for s in samples:
        if s.name == name:
            return s.labels
    return None
