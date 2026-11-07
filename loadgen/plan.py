"""Seeded plan builder: scenario + seed -> every session, step size and arrival offset, before any
request is sent. The plan is what makes runs reproducible (milestone 2 acceptance: two runs with the
same seed produce identical request sequences). Each tenant draws from a child generator keyed by
its name, so adding a tenant does not change the others' plans."""

from __future__ import annotations

import hashlib
import json
import math
import zlib
from dataclasses import asdict, dataclass, field

import numpy as np

from .arrivals import arrival_times
from .scenario import (
    AgentProfile,
    BatchProfile,
    ChatProfile,
    NotetakerProfile,
    Scenario,
    TenantSpec,
)


@dataclass
class Step:
    i: int
    user_tokens: int  # tokens of the new user message (the history carries the rest)
    max_tokens: int
    stream: bool
    pause_before_s: float  # think time or tool pause before this step (0 for the first)
    content_seed: int


@dataclass
class Session:
    id: str
    t_start: float | None  # None for closed-loop workers (they start at 0 and never wait)
    worker: int | None
    steps: list[Step]


@dataclass
class TenantPlan:
    name: str
    api_key: str
    shape: str | None
    profile_kind: str
    sessions: list[Session] = field(default_factory=list)

    @property
    def n_requests(self) -> int:
        return sum(len(s.steps) for s in self.sessions)


@dataclass
class Plan:
    scenario: str
    seed: int
    duration_s: float
    hour_length_s: float
    tenants: list[TenantPlan]

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=1, sort_keys=True)

    @property
    def digest(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:16]

    @property
    def n_requests(self) -> int:
        return sum(t.n_requests for t in self.tenants)

    @property
    def n_open_loop_requests(self) -> int:
        """Planned requests of open-loop tenants; closed-loop workers are over-provisioned by design."""
        return sum(len(s.steps) for t in self.tenants for s in t.sessions if s.t_start is not None)

    @property
    def n_workers(self) -> int:
        return len({(t.name, s.worker) for t in self.tenants for s in t.sessions if s.t_start is None})


def tenant_rng(seed: int, name: str) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(zlib.crc32(name.encode()),)))


def _int(rng: np.random.Generator, r: tuple[int, int]) -> int:
    return int(rng.integers(r[0], r[1] + 1))


def _float(rng: np.random.Generator, r: tuple[float, float]) -> float:
    return float(rng.uniform(r[0], r[1]))


def _seed(rng: np.random.Generator) -> int:
    return int(rng.integers(0, 2**31 - 1))


def _session_steps(rng: np.random.Generator, profile) -> list[Step]:
    if isinstance(profile, ChatProfile):
        n = _int(rng, profile.turns)
        steps = []
        for i in range(n):
            steps.append(
                Step(
                    i=i,
                    user_tokens=_int(rng, profile.first_user_tokens if i == 0 else profile.user_tokens_per_turn),
                    max_tokens=_int(rng, profile.output_tokens),
                    stream=profile.stream,
                    pause_before_s=0.0 if i == 0 else _float(rng, profile.think_s),
                    content_seed=_seed(rng),
                )
            )
        return steps
    if isinstance(profile, AgentProfile):
        n = _int(rng, profile.steps)
        return [
            Step(
                i=i,
                user_tokens=_int(rng, profile.first_user_tokens if i == 0 else profile.growth_tokens),
                max_tokens=_int(rng, profile.output_tokens),
                stream=profile.stream,
                pause_before_s=0.0 if i == 0 else _float(rng, profile.pause_s),
                content_seed=_seed(rng),
            )
            for i in range(n)
        ]
    if isinstance(profile, NotetakerProfile | BatchProfile):
        return [
            Step(
                i=0,
                user_tokens=_int(rng, profile.prompt_tokens),
                max_tokens=_int(rng, profile.output_tokens),
                stream=profile.stream,
                pause_before_s=0.0,
                content_seed=_seed(rng),
            )
        ]
    raise ValueError(f"unknown profile {profile!r}")


def _plan_tenant(scenario: Scenario, seed: int, spec: TenantSpec) -> TenantPlan:
    rng = tenant_rng(seed, spec.name)
    plan = TenantPlan(name=spec.name, api_key=spec.api_key, shape=spec.shape, profile_kind=spec.profile.kind)
    if isinstance(spec.profile, BatchProfile):
        # Closed loop: each worker gets more single-step sessions than it can possibly finish.
        per_worker = math.ceil(scenario.duration_s / 2.0) + 5
        for w in range(spec.profile.concurrency):
            for j in range(per_worker):
                plan.sessions.append(
                    Session(id=f"{spec.name}-w{w}-{j}", t_start=None, worker=w, steps=_session_steps(rng, spec.profile))
                )
        return plan
    assert spec.arrivals is not None
    times = arrival_times(rng, spec.arrivals, scenario)
    for j, t in enumerate(times):
        plan.sessions.append(
            Session(id=f"{spec.name}-{j}", t_start=round(float(t), 3), worker=None, steps=_session_steps(rng, spec.profile))
        )
    return plan


def build_plan(scenario: Scenario, seed: int) -> Plan:
    return Plan(
        scenario=scenario.name,
        seed=seed,
        duration_s=scenario.duration_s,
        hour_length_s=scenario.hour_length_s,
        tenants=[_plan_tenant(scenario, seed, t) for t in scenario.tenants],
    )
