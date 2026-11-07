"""Configuration models and loaders (config/*.yaml)."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

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
    price_per_second: float | None = Field(default=None, ge=0)  # None = derived from the active price reference
    price_premium: float = Field(default=1.0, gt=0)  # multiplier on the derived price (e.g. 1.25 for 16k context)
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
    calibration: str | None = None  # calibration/<replica>.yaml (docs/DESIGN.md 5.2)
    curve: str | None = None  # inline "n:tps,..." alternative, e.g. for fake replicas
    curve_context: int | None = None  # which calibration level to use; default: the longest
    kv_tokens: int | None = None  # override the KV budget read from the replica
    slot_plan: dict[str, int] = Field(default_factory=dict)  # sellable slots, e.g. {small.8k.30tps: 8}


class PriceReference(BaseModel):
    """A per-token market price point (config/price_references.yaml, written by slotgw-fetch-prices)."""

    name: str
    input_usd_per_million: float = Field(ge=0)
    output_usd_per_million: float = Field(ge=0)
    model: str | None = None  # provider model id the prices belong to
    basket: list[str] = Field(default_factory=list)  # member names when this is the median of several
    context_length: int | None = None


class TokenPricing(BaseModel):
    """The per-token reference used for bills: the active price reference, or custom numbers."""

    input_usd_per_million: float = Field(default=0.0, ge=0)
    output_usd_per_million: float = Field(default=0.0, ge=0)
    batch_discount: float = 0.5
    reference: str = "custom"  # which price reference the prices came from (set by Pricing)


class ProviderPricing(BaseModel):
    gpu_usd_per_hour: float
    target_utilization: float = Field(default=0.70, gt=0, le=1)  # second cost-per-token basis next to the measured one
    margin_targets: list[float] = Field(default_factory=lambda: [0.0, 0.25, 0.5])  # gross margins the report lists prices at

    @field_validator("margin_targets")
    @classmethod
    def _margins(cls, v: list[float]) -> list[float]:
        if any(not 0 <= m < 1 for m in v):
            raise ValueError("margin_targets must be in [0, 1)")
        return v


class ShapePricing(BaseModel):
    """Extra charges on top of a shape's price_per_second (docs/DESIGN.md section 8)."""

    prompt_usd_per_million: float | None = Field(default=None, ge=0)  # prices prefill; None/auto = reference input price
    reservation_price_factor: float = Field(default=1.0, gt=0)  # reserved slot price = price_per_second x this

    @field_validator("prompt_usd_per_million", mode="before")
    @classmethod
    def _auto(cls, v: Any) -> Any:
        return None if isinstance(v, str) and v.strip().lower() in ("", "auto") else v


class Pricing(BaseModel):
    token: TokenPricing = Field(default_factory=TokenPricing)
    provider: ProviderPricing
    shape: ShapePricing = Field(default_factory=ShapePricing)
    reference: str | None = None  # a name in `references`, or "custom" for the token block's own numbers
    references: list[PriceReference] = Field(default_factory=list)  # from price_references.yaml
    references_source: str | None = None
    references_fetched: str | None = None

    def find_reference(self, name: str) -> PriceReference | None:
        return next((r for r in self.references if r.name == name), None)

    @model_validator(mode="after")
    def _resolve(self) -> Pricing:
        if self.reference and self.reference != "custom":
            ref = self.find_reference(self.reference)
            if ref is None:
                raise ValueError(f"unknown price reference {self.reference!r}; known: {[r.name for r in self.references]}")
            self.token.input_usd_per_million = ref.input_usd_per_million
            self.token.output_usd_per_million = ref.output_usd_per_million
            self.token.reference = ref.name
        elif "output_usd_per_million" not in self.token.model_fields_set:
            raise ValueError("pricing: set `reference:` to a price reference or give token prices")
        if self.shape.prompt_usd_per_million is None:
            self.shape.prompt_usd_per_million = self.token.input_usd_per_million
        return self


class PacingSettings(BaseModel):
    """Per-request delivery pacing (docs/DESIGN.md section 6)."""

    enabled: bool = True
    bucket_depth_s: float = Field(default=1.0, ge=0)


class AdmissionSettings(BaseModel):
    """Ledgers, seat rule and queues (docs/DESIGN.md sections 5.2-5.4)."""

    enabled: bool = True
    kv_margin: float = Field(default=0.05, ge=0, lt=1)
    tps_margin: float = Field(default=0.10, ge=0, lt=1)
    tps_overcommit: float = Field(default=1.0, ge=1.0)
    lazy_max_share: float = Field(default=0.5, ge=0, le=1)
    lazy_preempt: Literal["never", "requeue"] = "never"
    max_prefill_tokens_in_flight: int = Field(default=0, ge=0)  # 0 = no cap
    dispatch_tick_s: float = Field(default=0.25, gt=0)
    health_interval_s: float = Field(default=5.0, gt=0)
    reservation_spill: bool = False  # a reserved tenant whose slots are busy may compete on demand


class BurstSettings(BaseModel):
    """Burst manager (docs/DESIGN.md section 7): lend idle step rate and free KV, revocably."""

    enabled: bool = True
    tps: bool = True  # lend the step rate above each running request's guarantee to its pacer
    context: bool = True  # admit over-context requests into free KV with low engine priority
    revoke_policy: Literal["finish", "grace", "abort"] = "grace"  # when a guaranteed admission needs the KV
    grace_s: float = Field(default=15.0, ge=0)
    max_context_tokens: int = Field(default=0, ge=0)  # cap for context bursts; 0 = the model's largest shape


class GatewaySettings(BaseModel):
    pacing: PacingSettings = Field(default_factory=PacingSettings)
    admission: AdmissionSettings = Field(default_factory=AdmissionSettings)
    burst: BurstSettings = Field(default_factory=BurstSettings)


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
        tok = self.pricing.token
        for s in self.shapes:
            if s.price_per_second is None:  # parity with the reference: price_per_second / tps = $/output token
                s.price_per_second = tok.output_usd_per_million / 1e6 * s.tps * s.price_premium * (tok.batch_discount if s.lazy else 1.0)
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
    refs_path = base / "price_references.yaml"
    if refs_path.exists():  # runs snapshotted before the file existed keep their token prices
        refs = _read_yaml(refs_path)
        pricing = {**pricing, "references": refs.get("references", []), "references_source": refs.get("source"),
                   "references_fetched": str(refs.get("fetched") or "")}
    gateway = _read_yaml(base / "gateway.yaml") if (base / "gateway.yaml").exists() else {}
    return Config(
        shapes=shapes.get("shapes", []),
        tenants=tenants.get("tenants", []),
        models=replicas.get("models", {}),
        replicas=replicas.get("replicas", []),
        pricing=pricing,
        gateway=gateway,
    )
