"""Scenario file models (docs/DESIGN.md section 9): tenants with a profile and an arrival process,
plus a phase timeline. Loaded from config/scenarios/*.yaml."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, Field, model_validator

IntRange = tuple[int, int]
FloatRange = tuple[float, float]


class Phase(BaseModel):
    name: str
    duration_s: float = Field(gt=0)
    rate_scale: float = Field(default=1.0, ge=0)


class ChatProfile(BaseModel):
    """Streamed multi-turn conversations; the prompt grows by each user turn plus the reply."""

    kind: Literal["chat"]
    turns: IntRange = (3, 10)
    first_user_tokens: IntRange = (200, 800)
    user_tokens_per_turn: IntRange = (200, 800)
    output_tokens: IntRange = (50, 300)
    think_s: FloatRange = (5.0, 30.0)
    stream: bool = True


class AgentProfile(BaseModel):
    """Streamed agentic loops: sequential steps, each adding a tool result to the context."""

    kind: Literal["agent"]
    steps: IntRange = (5, 15)
    first_user_tokens: IntRange = (1000, 2000)
    growth_tokens: IntRange = (500, 1500)
    output_tokens: IntRange = (100, 400)
    pause_s: FloatRange = (1.0, 5.0)
    stream: bool = True


class NotetakerProfile(BaseModel):
    """One complete (non-streamed) summary of a long transcript per session."""

    kind: Literal["notetaker"]
    prompt_tokens: IntRange = (3000, 8000)
    output_tokens: IntRange = (150, 400)
    stream: bool = False


class BatchProfile(BaseModel):
    """Closed loop: `concurrency` workers each send the next request as soon as one completes."""

    kind: Literal["batch"]
    prompt_tokens: IntRange = (500, 2000)
    output_tokens: IntRange = (200, 600)
    concurrency: int = Field(default=2, ge=1)
    stream: bool = False


Profile = Annotated[ChatProfile | AgentProfile | NotetakerProfile | BatchProfile, Field(discriminator="kind")]


class PoissonArrivals(BaseModel):
    kind: Literal["poisson"]
    rate_per_min: float = Field(ge=0)


class DiurnalArrivals(BaseModel):
    """Poisson whose rate is `rate_per_min x phase.rate_scale` (a rate schedule over the timeline)."""

    kind: Literal["diurnal"]
    rate_per_min: float = Field(ge=0)


class BurstyArrivals(BaseModel):
    """On/off Markov-modulated Poisson: exponential on and off periods with their own rates."""

    kind: Literal["bursty"]
    rate_on_per_min: float = Field(ge=0)
    rate_off_per_min: float = Field(default=0.0, ge=0)
    mean_on_s: float = Field(gt=0)
    mean_off_s: float = Field(gt=0)


class HourmarkArrivals(BaseModel):
    """Clusters around each hour mark (k x hour_length_s, k >= 1), Gaussian with `sigma_s`,
    Poisson(`per_mark`) requests per mark, plus a Poisson background between marks."""

    kind: Literal["hourmark"]
    per_mark: float = Field(ge=0)
    sigma_s: float = Field(ge=0)
    background_per_min: float = Field(default=0.0, ge=0)
    offset_s: float = 0.0


class ClosedLoop(BaseModel):
    kind: Literal["closed_loop"]


Arrivals = Annotated[
    PoissonArrivals | DiurnalArrivals | BurstyArrivals | HourmarkArrivals | ClosedLoop,
    Field(discriminator="kind"),
]


class ReservationSpec(BaseModel):
    """A standing reservation the load generator creates for the tenant (docs/DESIGN.md section 8):
    `slots` of `shape` (default: the tenant's shape) for the whole run, or one window per named phase."""

    slots: int = Field(ge=1)
    shape: str | None = None
    phases: list[str] = Field(default_factory=list)
    replica: str | None = None  # pin the slots to one replica (phase 2 placement rule); --reservation-replica overrides


class ShowcaseSpec(BaseModel):
    """The showcase lane (docs/PHASE2.md section 6): real, hand-written prompts from a prompt file,
    one session every `every_s` seconds from `offset_s`, cycling through the file. They go through
    the same shape, admission, pacing and billing as the synthetic traffic, stop at end-of-sequence
    like real requests, and carry `X-Slotgw-Showcase` so the gateway keeps their text."""

    prompts: str  # config/prompts/<profile>.yaml, relative to the working directory or the scenario file
    every_s: float = Field(default=180.0, gt=0)
    offset_s: float = Field(default=30.0, ge=0)
    max_tokens: int | None = None  # overrides the prompt file's per-prompt budget
    think_s: float = Field(default=3.0, ge=0)  # pause between the turns of one prompt


class ShowcasePrompt(BaseModel):
    id: str
    turns: list[str] = Field(min_length=1)
    max_tokens: int = Field(default=300, ge=1)


class PromptFile(BaseModel):
    profile: str
    system: str | None = None
    prompts: list[ShowcasePrompt] = Field(min_length=1)


def resolve_prompts_path(spec_path: str, scenario_path: str | Path | None = None) -> Path:
    """The prompt file: as given (relative to the working directory), else relative to the scenario
    file's directory, else to the config directory above it (config/scenarios -> config)."""
    p = Path(spec_path)
    if p.is_file() or scenario_path is None:
        return p
    base = Path(scenario_path).resolve().parent
    for candidate in (base / p, base.parent / p, base.parent.parent / p):
        if candidate.is_file():
            return candidate
    return p


def load_prompts(path: str | Path) -> PromptFile:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    pf = PromptFile.model_validate(data)
    ids = [p.id for p in pf.prompts]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path}: duplicate prompt ids")
    return pf


class TenantSpec(BaseModel):
    name: str
    api_key: str
    shape: str | None = None  # None: the tenant's default shape (gateway decides)
    profile: Profile
    arrivals: Arrivals | None = None
    reservation: ReservationSpec | None = None
    showcase: ShowcaseSpec | None = None

    @model_validator(mode="after")
    def _closed_loop_for_batch(self) -> TenantSpec:
        if self.profile.kind == "batch":
            if self.arrivals is not None and self.arrivals.kind != "closed_loop":
                raise ValueError(f"tenant {self.name}: batch profile runs closed loop; drop `arrivals`")
            self.arrivals = ClosedLoop(kind="closed_loop")
        elif self.arrivals is None or self.arrivals.kind == "closed_loop":
            raise ValueError(f"tenant {self.name}: profile {self.profile.kind} needs an open-loop arrival process")
        return self


class Scenario(BaseModel):
    name: str
    hour_length_s: float = Field(default=600.0, gt=0)
    drain_s: float = Field(default=60.0, ge=0)
    sample_interval_s: float = Field(default=2.0, gt=0)
    rate_multiplier: float = Field(default=1.0, ge=0)  # scales every open-loop arrival rate: the sizing knob for a bigger replica
    phases: list[Phase] = Field(min_length=1)
    tenants: list[TenantSpec] = Field(min_length=1)

    @property
    def duration_s(self) -> float:
        return sum(p.duration_s for p in self.phases)

    def rate_scale_at(self, t: float) -> float:
        acc = 0.0
        for p in self.phases:
            acc += p.duration_s
            if t < acc:
                return p.rate_scale
        return 0.0

    @property
    def max_rate_scale(self) -> float:
        return max(p.rate_scale for p in self.phases)

    def phase_window(self, name: str) -> tuple[float, float]:
        """(start, end) offsets of a phase; raises for an unknown phase name."""
        acc = 0.0
        for p in self.phases:
            if p.name == name:
                return acc, acc + p.duration_s
            acc += p.duration_s
        raise ValueError(f"unknown phase {name!r}")

    def reservation_windows(self, spec: ReservationSpec) -> list[tuple[float, float]]:
        if not spec.phases:
            return [(0.0, self.duration_s)]
        return [self.phase_window(n) for n in spec.phases]

    @model_validator(mode="after")
    def _check_reservation_phases(self) -> Scenario:
        names = {p.name for p in self.phases}
        for t in self.tenants:
            if t.reservation is not None:
                for n in t.reservation.phases:
                    if n not in names:
                        raise ValueError(f"tenant {t.name}: reservation phase {n!r} is not a phase of the scenario")
        return self


def load_scenario(path: str | Path) -> Scenario:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return Scenario.model_validate(data)
