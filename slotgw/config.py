"""Configuration models and loaders (config/*.yaml)."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator

_ENV_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def expand_env(text: str) -> str:
    """Expand ${VAR} and ${VAR:-default} in YAML text before parsing."""
    return _ENV_RE.sub(lambda m: os.environ.get(m.group(1), m.group(2) or ""), text)


class Shape(BaseModel):
    """A per-request execution contract (see docs/DESIGN.md section 3)."""

    name: str
    model: str
    max_context: int = Field(gt=0)
    tps: float = Field(gt=0)
    price_per_second: float = Field(ge=0)
    lazy: bool = False
    max_queue_s: float = 30.0
    default_max_tokens: int = 512


class Tenant(BaseModel):
    name: str
    api_key: str
    default_shape: str
    max_concurrent: int | None = None


class ReplicaConfig(BaseModel):
    id: str
    kind: Literal["vllm", "fake"] = "vllm"
    url: str
    model: str


class TokenPricing(BaseModel):
    input_usd_per_million: float
    output_usd_per_million: float
    batch_discount: float = 0.5


class ProviderPricing(BaseModel):
    gpu_usd_per_hour: float


class Pricing(BaseModel):
    token: TokenPricing
    provider: ProviderPricing


class PacingSettings(BaseModel):
    """Per-request delivery pacing (docs/DESIGN.md section 6)."""

    enabled: bool = True
    bucket_depth_s: float = Field(default=1.0, ge=0)


class GatewaySettings(BaseModel):
    pacing: PacingSettings = Field(default_factory=PacingSettings)


class Config(BaseModel):
    shapes: list[Shape]
    tenants: list[Tenant]
    models: dict[str, str]
    replicas: list[ReplicaConfig]
    pricing: Pricing
    gateway: GatewaySettings = Field(default_factory=GatewaySettings)

    @model_validator(mode="after")
    def _check_references(self) -> Config:
        names = [s.name for s in self.shapes]
        if len(set(names)) != len(names):
            raise ValueError("duplicate shape names")
        for s in self.shapes:
            if s.model not in self.models:
                raise ValueError(f"shape {s.name}: unknown model alias {s.model!r}")
        keys = [t.api_key for t in self.tenants]
        if len(set(keys)) != len(keys):
            raise ValueError("duplicate tenant api keys")
        for t in self.tenants:
            if t.default_shape not in names:
                raise ValueError(f"tenant {t.name}: unknown default shape {t.default_shape!r}")
        for r in self.replicas:
            if r.model not in self.models:
                raise ValueError(f"replica {r.id}: unknown model alias {r.model!r}")
        return self

    def shape(self, name: str) -> Shape | None:
        return next((s for s in self.shapes if s.name == name), None)

    def tenant_by_key(self, api_key: str) -> Tenant | None:
        return next((t for t in self.tenants if t.api_key == api_key), None)

    def hf_model(self, alias: str) -> str:
        return self.models[alias]


def _read_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(expand_env(path.read_text(encoding="utf-8"))) or {}


def load_config(config_dir: str | Path | None = None) -> Config:
    base = Path(config_dir or os.environ.get("SLOTGW_CONFIG_DIR", "config"))
    shapes = _read_yaml(base / "shapes.yaml")
    tenants = _read_yaml(base / "tenants.yaml")
    replicas = _read_yaml(base / "replicas.yaml")
    pricing = _read_yaml(base / "pricing.yaml")
    gateway = _read_yaml(base / "gateway.yaml") if (base / "gateway.yaml").exists() else {}
    return Config(
        shapes=shapes.get("shapes", []),
        tenants=tenants.get("tenants", []),
        models=replicas.get("models", {}),
        replicas=replicas.get("replicas", []),
        pricing=pricing,
        gateway=gateway,
    )
