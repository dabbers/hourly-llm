"""Executes a plan against the gateway: open-loop sessions start at their planned offsets, closed-loop
workers chain requests, every request is recorded client-side (client.csv) and the gateway's replica
view is sampled (samples.csv). Output lands next to the gateway's meter:
runs/<gateway_run_id>/loads/<load_run_id>/."""

from __future__ import annotations

import asyncio
import csv
import json
import logging
import shutil
import time
from dataclasses import asdict, dataclass, fields
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from slotgw.clock import now

from .content import words
from .plan import Plan, Session, Step, TenantPlan
from .scenario import PromptFile, Scenario, load_prompts, resolve_prompts_path

log = logging.getLogger("loadgen")


@dataclass
class ClientRecord:
    load_run: str
    scenario: str
    seed: int
    tenant: str
    session: str
    step: int
    shape: str | None
    stream: bool
    t_send: float
    t_first_byte: float | None = None
    t_first_token: float | None = None
    t_done: float | None = None
    status: str = "ok"  # ok | refused | error | cancelled
    http_status: int | None = None
    request_id: str | None = None
    user_tokens: int = 0
    max_tokens: int = 0
    prompt_tokens: int | None = None
    output_tokens: int | None = None
    chunks: int = 0
    billed_s: float | None = None
    shape_usd: float | None = None
    per_token_usd: float | None = None
    throttled_ms: float | None = None
    error: str | None = None
    showcase: str | None = None  # prompt id of a showcase request (real text, kept by the gateway)

    @property
    def ttft_ms(self) -> float | None:
        return None if self.t_first_token is None else (self.t_first_token - self.t_send) * 1000.0

    @property
    def e2e_ms(self) -> float | None:
        return None if self.t_done is None else (self.t_done - self.t_send) * 1000.0


CLIENT_COLUMNS = [f.name for f in fields(ClientRecord)] + ["ttft_ms", "e2e_ms"]
SAMPLE_COLUMNS = [
    "t", "replica", "kv_usage", "running", "waiting", "preemptions", "client_inflight",
    # gateway ledger view (milestone 3; empty when admission is off)
    "ledger_running", "ledger_lazy_running", "ledger_kv_held", "ledger_kv_usable", "seats_now",
    "prefill_tokens_in_flight", "queue_interactive", "queue_lazy", "gw_preemptions", "gw_timeouts",
    # milestone 4: reservations and burst
    "engine_active", "reserved_slots", "reserved_idle", "burst_running", "burst_rate_lent", "step_rate_now",
    "queue_burst", "gw_revocations", "gw_bursts_admitted",
]


class CsvSink:
    def __init__(self, path: Path, columns: list[str]):
        self._f = open(path, "w", newline="", encoding="utf-8")  # noqa: SIM115 - long-lived sink
        self._w = csv.DictWriter(self._f, fieldnames=columns, extrasaction="ignore")
        self._w.writeheader()
        self.rows = 0

    def write(self, row: dict[str, Any]) -> None:
        self._w.writerow(row)
        self._f.flush()
        self.rows += 1

    def close(self) -> None:
        self._f.close()


@dataclass
class Counters:
    sent: int = 0
    ok: int = 0
    refused: int = 0
    errors: int = 0
    cancelled: int = 0
    inflight: int = 0
    output_tokens: int = 0
    prompt_tokens: int = 0

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


class LoadRunner:
    def __init__(
        self,
        scenario: Scenario,
        plan: Plan,
        gateway_url: str,
        runs_dir: str | Path,
        load_run_id: str | None = None,
        config_dir: str | Path | None = None,
        scenario_path: str | Path | None = None,
        progress_every_s: float = 10.0,
        label: str | None = None,
        reservations: bool = True,
        reservation_replica: str | None = None,
    ):
        self.scenario = scenario
        self.plan = plan
        self.reservations_enabled = reservations
        self.reservation_replica = reservation_replica  # None: the scenario's `replica:`; "none": unpinned; else this replica
        self.reservation_log: list[dict[str, Any]] = []
        self.gateway_url = gateway_url.rstrip("/")
        self.runs_dir = Path(runs_dir)
        stem = f"{scenario.name}-{label}" if label else scenario.name
        self.load_run_id = load_run_id or f"{stem}-s{plan.seed}-{time.strftime('%Y%m%d-%H%M%S')}"
        self.config_dir = Path(config_dir) if config_dir else None
        self.scenario_path = Path(scenario_path) if scenario_path else None
        self.progress_every_s = progress_every_s
        self.label = label
        self.scenario_label = f"{scenario.name}+{label}" if label else scenario.name
        self.counters = Counters()
        self.out_dir: Path | None = None
        self.gateway_run_id: str | None = None
        self._t0 = 0.0  # perf_counter at timeline start
        self._t_end = 0.0
        self._client_sink: CsvSink | None = None
        self._sample_sink: CsvSink | None = None
        self._prompts: dict[str, PromptFile] = {
            spec.name: load_prompts(resolve_prompts_path(spec.showcase.prompts, scenario_path))
            for spec in scenario.tenants
            if spec.showcase is not None
        }

    # ---- timeline helpers -------------------------------------------------------------------

    def elapsed(self) -> float:
        return time.perf_counter() - self._t0

    def past_end(self) -> bool:
        return time.perf_counter() >= self._t_end

    async def sleep_until(self, offset_s: float) -> None:
        delay = self._t0 + offset_s - time.perf_counter()
        if delay > 0:
            await asyncio.sleep(delay)

    # ---- lifecycle --------------------------------------------------------------------------

    async def run(self) -> dict[str, Any]:
        timeout = httpx.Timeout(600.0, connect=10.0)
        limits = httpx.Limits(max_connections=512, max_keepalive_connections=64)
        async with httpx.AsyncClient(base_url=self.gateway_url, timeout=timeout, limits=limits) as client:
            health = await self._preflight(client)
            self.gateway_run_id = str(health.get("run_id") or "unknown")
            self.out_dir = self.runs_dir / self.gateway_run_id / "loads" / self.load_run_id
            self.out_dir.mkdir(parents=True, exist_ok=True)
            (self.out_dir / "plan.json").write_text(self.plan.to_json(), encoding="utf-8")
            self._snapshot_config()
            tokenizer = await self._tokenizer_check(client)
            self._client_sink = CsvSink(self.out_dir / "client.csv", CLIENT_COLUMNS)
            self._sample_sink = CsvSink(self.out_dir / "samples.csv", SAMPLE_COLUMNS)
            started = datetime.now(UTC)
            log.info(
                "load run %s: scenario %s seed %d, %d planned open-loop requests + %d closed-loop workers "
                "over %.0f s (+%.0f s drain) -> %s",
                self.load_run_id, self.scenario.name, self.plan.seed, self.plan.n_open_loop_requests,
                self.plan.n_workers, self.scenario.duration_s, self.scenario.drain_s, self.out_dir,
            )
            self._t0 = time.perf_counter()
            self._t_end = self._t0 + self.scenario.duration_s
            sampler = asyncio.create_task(self._sample_loop(client))
            progress = asyncio.create_task(self._progress_loop())
            reservers = [
                asyncio.create_task(self._reservation_window(client, spec, w0, w1))
                for spec in self.scenario.tenants
                if spec.reservation is not None and self.reservations_enabled
                for (w0, w1) in self.scenario.reservation_windows(spec.reservation)
            ]
            try:
                await asyncio.gather(*(self._drive_tenant(client, tp) for tp in self.plan.tenants))
            finally:
                progress.cancel()
                sampler.cancel()
                for t in (progress, sampler):
                    try:
                        await t
                    except (asyncio.CancelledError, httpx.HTTPError):
                        pass
                for t in reservers:
                    t.cancel()
                await asyncio.gather(*reservers, return_exceptions=True)
                await self._release_reservations(client)
                self._client_sink.close()
                self._sample_sink.close()
            ended = datetime.now(UTC)
            meta = {
                "load_run": self.load_run_id,
                "scenario": self.scenario_label,
                "label": self.label,
                "seed": self.plan.seed,
                "gateway_url": self.gateway_url,
                "gateway_run_id": self.gateway_run_id,
                "gateway_pacing": health.get("pacing"),
                "gateway_admission": health.get("admission"),
                "gateway_burst": health.get("burst"),
                "price_reference": health.get("price_reference"),
                "reservations": self.reservation_log,
                "replicas": health.get("replicas"),  # id, model, KV budget (cache) at the start of the run
                "plan_digest": self.plan.digest,
                "planned_open_loop_requests": self.plan.n_open_loop_requests,
                "closed_loop_workers": self.plan.n_workers,
                "duration_s": self.scenario.duration_s,
                "drain_s": self.scenario.drain_s,
                "started_at": started.isoformat(),
                "ended_at": ended.isoformat(),
                "elapsed_s": round((ended - started).total_seconds(), 3),
                "tokenizer": tokenizer,
                "counters": self.counters.as_dict(),
                "client_rows": self._client_sink.rows,
                "sample_rows": self._sample_sink.rows,
            }
            (self.out_dir / "loadrun.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
            log.info("done: %s", json.dumps(self.counters.as_dict()))
            return meta

    async def _preflight(self, client: httpx.AsyncClient) -> dict[str, Any]:
        try:
            r = await client.get("/healthz", timeout=10)
            r.raise_for_status()
            health = r.json()
            r = await client.get("/admin/replicas", timeout=30)
            health["replicas"] = r.json().get("replicas", []) if r.status_code == 200 else []
        except httpx.HTTPError as e:
            raise SystemExit(f"gateway {self.gateway_url} not reachable: {e}") from e
        return health

    def _snapshot_config(self) -> None:
        assert self.out_dir is not None
        snap = self.out_dir / "config"
        snap.mkdir(exist_ok=True)
        if self.config_dir and self.config_dir.is_dir():
            for p in self.config_dir.glob("*.yaml"):
                shutil.copy(p, snap / p.name)
        if self.scenario_path and self.scenario_path.is_file():
            shutil.copy(self.scenario_path, snap / "scenario.yaml")
        else:
            (snap / "scenario.yaml").write_text(
                json.dumps(self.scenario.model_dump(mode="json"), indent=1), encoding="utf-8"
            )

    async def _tokenizer_check(self, client: httpx.AsyncClient) -> dict[str, Any] | None:
        """Measure the chat-template overhead and tokens-per-word of the vocabulary through the
        replica's tokenizer, so prompt sizes in the plan can be read against real counts."""
        try:
            r = await client.post("/admin/tokenize", json={"messages": [{"role": "user", "content": "Hello"}]})
            r.raise_for_status()
            overhead = int(r.json()["count"]) - 1
            sample = words(12345, 400)
            r = await client.post("/admin/tokenize", json={"messages": [{"role": "user", "content": sample}]})
            r.raise_for_status()
            ratio = (int(r.json()["count"]) - overhead) / 400.0
        except (httpx.HTTPError, KeyError, ValueError) as e:
            log.warning("tokenizer check skipped: %s", e)
            return None
        log.info("tokenizer: chat-template overhead %d tokens, %.3f tokens per planned token", overhead, ratio)
        if abs(ratio - 1.0) > 0.05:
            log.warning("prompt sizing is off by %.0f%%: adjust loadgen/content.py VOCAB", (ratio - 1) * 100)
        return {"template_overhead_tokens": overhead, "tokens_per_planned_token": round(ratio, 4)}

    # ---- reservations -----------------------------------------------------------------------

    async def _reservation_window(self, client: httpx.AsyncClient, spec, w0: float, w1: float) -> None:
        """Hold the tenant's reservation from offset w0 to w1 of the timeline."""
        await self.sleep_until(w0)
        body = {
            "tenant": spec.name,
            "shape": spec.reservation.shape or spec.shape,
            "slots": spec.reservation.slots,
            "load_run": self.load_run_id,
            "scenario": self.scenario_label,
            "seed": self.plan.seed,
        }
        replica = spec.reservation.replica if self.reservation_replica is None else self.reservation_replica
        if replica and replica.lower() != "none":
            body["replica"] = replica
        entry: dict[str, Any] = {**body, "t_start_offset": w0, "t_end_offset": w1, "id": None, "status": None}
        self.reservation_log.append(entry)
        try:
            r = await client.post("/admin/reservations", json=body, timeout=30)
            entry["status"] = r.status_code
            if r.status_code != 201:
                entry["error"] = r.text[:200]
                log.warning("reservation for %s refused: %s %s", spec.name, r.status_code, r.text[:200])
                return
            entry["id"] = r.json().get("id")
            entry["replicas"] = r.json().get("replicas")
            log.info("reservation %s: %d x %s for %s until t=%.0fs", entry["id"], body["slots"], body["shape"], spec.name, w1)
        except httpx.HTTPError as e:
            entry["error"] = f"{type(e).__name__}: {e}"[:200]
            log.warning("reservation for %s failed: %s", spec.name, e)
            return
        try:
            await self.sleep_until(w1)
        finally:
            await self._delete_reservation(client, entry)

    async def _delete_reservation(self, client: httpx.AsyncClient, entry: dict[str, Any]) -> None:
        if not entry.get("id") or entry.get("released"):
            return
        try:
            r = await client.delete(f"/admin/reservations/{entry['id']}", timeout=30)
            entry["released"] = r.status_code == 200
            if r.status_code == 200:
                entry["served"] = r.json().get("served")
                entry["timeouts"] = r.json().get("timeouts")
        except httpx.HTTPError as e:
            log.warning("could not release reservation %s: %s", entry["id"], e)

    async def _release_reservations(self, client: httpx.AsyncClient) -> None:
        for entry in self.reservation_log:
            await self._delete_reservation(client, entry)

    # ---- drivers ----------------------------------------------------------------------------

    async def _drive_tenant(self, client: httpx.AsyncClient, tp: TenantPlan) -> None:
        """Closed-loop sessions (worker set) chain on their worker; open-loop ones (t_start set,
        including a batch tenant's showcase sessions) start at their offsets."""
        tasks: set[asyncio.Task] = set()
        workers: dict[int, list[Session]] = {}
        for s in tp.sessions:
            if s.t_start is None:
                workers.setdefault(s.worker or 0, []).append(s)
        for w, sessions in sorted(workers.items()):
            tasks.add(asyncio.create_task(self._worker_loop(client, tp, w, sessions)))
        open_loop = sorted((s for s in tp.sessions if s.t_start is not None), key=lambda s: s.t_start or 0.0)
        for s in open_loop:
            await self.sleep_until(s.t_start or 0.0)
            if self.past_end():
                break
            t = asyncio.create_task(self._run_session(client, tp, s))
            tasks.add(t)
            t.add_done_callback(tasks.discard)
        pending = [t for t in tasks if not t.done()]
        if pending:
            deadline = self._t_end + self.scenario.drain_s - time.perf_counter()
            _, still = await asyncio.wait(pending, timeout=max(0.0, deadline))
            for t in still:
                t.cancel()
            if still:
                await asyncio.gather(*still, return_exceptions=True)
                log.warning("tenant %s: %d sessions cancelled at drain deadline", tp.name, len(still))

    async def _worker_loop(self, client: httpx.AsyncClient, tp: TenantPlan, worker: int, sessions: list[Session]) -> None:
        await asyncio.sleep(0.5 * worker)  # stagger workers a little
        for s in sessions:
            if self.past_end():
                return
            await self._run_session(client, tp, s)

    async def _run_session(self, client: httpx.AsyncClient, tp: TenantPlan, session: Session) -> None:
        history: list[dict[str, str]] = []
        for step in session.steps:
            if step.pause_before_s > 0:
                await asyncio.sleep(step.pause_before_s)
            if self.past_end():
                return
            ok, user_content, reply = await self._run_step(client, tp, session, step, history)
            if not ok:
                return  # a refusal (context full) or an error ends the session, as a real client would
            history.append({"role": "user", "content": user_content})
            history.append({"role": "assistant", "content": reply})

    async def _run_step(
        self, client: httpx.AsyncClient, tp: TenantPlan, session: Session, step: Step, history: list[dict[str, str]]
    ) -> tuple[bool, str, str]:
        showcase: str | None = None
        if step.showcase is not None:
            # a real prompt: the file's text, the file's system message, a natural stop (no ignore_eos)
            pf = self._prompts[tp.name]
            prompt = next(p for p in pf.prompts if p.id == step.showcase)
            content = prompt.turns[step.i]
            system = [{"role": "system", "content": pf.system}] if pf.system else []
            messages = [*system, *history, {"role": "user", "content": content}]
            showcase = f"{prompt.id}#{step.i}"
        else:
            content = words(step.content_seed, step.user_tokens)
            messages = [*history, {"role": "user", "content": content}]
        body: dict[str, Any] = {
            "messages": messages,
            "max_tokens": step.max_tokens,
            "ignore_eos": showcase is None,
            "stream": step.stream,
        }
        if tp.shape:
            body["model"] = tp.shape
        headers = {
            "Authorization": f"Bearer {tp.api_key}",
            "X-Slotgw-Run": self.load_run_id,
            "X-Slotgw-Scenario": self.scenario_label,
            "X-Slotgw-Seed": str(self.plan.seed),
        }
        if showcase:
            headers["X-Slotgw-Showcase"] = showcase
        rec = ClientRecord(
            load_run=self.load_run_id,
            scenario=self.scenario_label,
            seed=self.plan.seed,
            tenant=tp.name,
            session=session.id,
            step=step.i,
            shape=tp.shape,
            stream=step.stream,
            t_send=now(),
            user_tokens=step.user_tokens,
            max_tokens=step.max_tokens,
            showcase=showcase,
        )
        self.counters.sent += 1
        self.counters.inflight += 1
        reply = ""
        try:
            if step.stream:
                reply = await self._stream(client, body, headers, rec)
            else:
                reply = await self._complete(client, body, headers, rec)
            rec.t_done = now()
        except httpx.HTTPError as e:
            rec.status, rec.error, rec.t_done = "error", f"{type(e).__name__}: {e}"[:200], now()
        except asyncio.CancelledError:
            rec.status, rec.t_done = "cancelled", now()
            self._record(rec)
            raise
        finally:
            self.counters.inflight -= 1
        self._record(rec)
        return rec.status == "ok", content, reply

    async def _stream(self, client: httpx.AsyncClient, body: dict[str, Any], headers: dict[str, str], rec: ClientRecord) -> str:
        parts: list[str] = []
        async with client.stream("POST", "/v1/chat/completions", json=body, headers=headers) as r:
            rec.t_first_byte = now()
            rec.http_status = r.status_code
            rec.request_id = r.headers.get("x-request-id")
            if r.status_code != 200:
                self._fail(rec, r.status_code, (await r.aread()).decode("utf-8", "replace"))
                return ""
            async for line in r.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                obj = json.loads(data)
                if "error" in obj:
                    rec.status, rec.error = "error", str(obj["error"].get("message"))[:200]
                    break
                choices = obj.get("choices") or []
                text = (choices[0].get("delta") or {}).get("content") if choices else None
                if text:
                    if rec.t_first_token is None:
                        rec.t_first_token = now()
                    rec.chunks += 1
                    parts.append(text)
                if obj.get("usage"):
                    rec.prompt_tokens = obj["usage"].get("prompt_tokens")
                    rec.output_tokens = obj["usage"].get("completion_tokens")
                if obj.get("slotgw"):
                    self._take_billing(rec, obj["slotgw"])
        return "".join(parts)

    async def _complete(self, client: httpx.AsyncClient, body: dict[str, Any], headers: dict[str, str], rec: ClientRecord) -> str:
        r = await client.post("/v1/chat/completions", json=body, headers=headers)
        rec.t_first_byte = now()
        rec.http_status = r.status_code
        rec.request_id = r.headers.get("x-request-id")
        if r.status_code != 200:
            self._fail(rec, r.status_code, r.text)
            return ""
        obj = r.json()
        rec.t_first_token = rec.t_first_byte  # a complete response arrives whole
        rec.chunks = 1
        usage = obj.get("usage") or {}
        rec.prompt_tokens = usage.get("prompt_tokens")
        rec.output_tokens = usage.get("completion_tokens")
        if obj.get("slotgw"):
            self._take_billing(rec, obj["slotgw"])
        return str(((obj.get("choices") or [{}])[0].get("message") or {}).get("content") or "")

    @staticmethod
    def _take_billing(rec: ClientRecord, slot: dict[str, Any]) -> None:
        rec.request_id = slot.get("request_id") or rec.request_id
        rec.throttled_ms = slot.get("throttled_ms")
        bill = slot.get("billing") or {}
        rec.billed_s = bill.get("billed_s")
        rec.shape_usd = bill.get("shape_usd")
        rec.per_token_usd = bill.get("per_token_usd")

    @staticmethod
    def _fail(rec: ClientRecord, status: int, text: str) -> None:
        code = None
        try:
            code = (json.loads(text).get("error") or {}).get("code")
        except (json.JSONDecodeError, AttributeError):
            pass
        rec.status = "refused" if status in (429, 503) else "error"
        rec.error = code or text[:200]

    def _record(self, rec: ClientRecord) -> None:
        row = asdict(rec)
        row["ttft_ms"] = rec.ttft_ms
        row["e2e_ms"] = rec.e2e_ms
        assert self._client_sink is not None
        self._client_sink.write(row)
        c = self.counters
        if rec.status == "ok":
            c.ok += 1
            c.output_tokens += rec.output_tokens or 0
            c.prompt_tokens += rec.prompt_tokens or 0
        elif rec.status == "refused":
            c.refused += 1
        elif rec.status == "cancelled":
            c.cancelled += 1
        else:
            c.errors += 1

    # ---- observers --------------------------------------------------------------------------

    async def _sample_loop(self, client: httpx.AsyncClient) -> None:
        assert self._sample_sink is not None
        while True:
            try:
                r = await client.get("/admin/replicas", timeout=5)
                t = now()
                body = r.json()
                sched = body.get("scheduler") or {}
                queues = sched.get("queues") or {}
                for rep in body.get("replicas", []):
                    h = rep.get("health") or {}
                    lg = rep.get("ledger") or {}
                    self._sample_sink.write(
                        {
                            "t": t,
                            "replica": rep.get("id"),
                            "kv_usage": h.get("kv_usage"),
                            "running": h.get("running"),
                            "waiting": h.get("waiting"),
                            "preemptions": h.get("preemptions"),
                            "client_inflight": self.counters.inflight,
                            "ledger_running": lg.get("running"),
                            "ledger_lazy_running": lg.get("lazy_running"),
                            "ledger_kv_held": lg.get("kv_held"),
                            "ledger_kv_usable": lg.get("kv_usable"),
                            "seats_now": lg.get("seats_now"),
                            "prefill_tokens_in_flight": lg.get("prefill_tokens_in_flight"),
                            "queue_interactive": queues.get("interactive"),
                            "queue_lazy": queues.get("lazy"),
                            "gw_preemptions": sched.get("preemptions"),
                            "gw_timeouts": sched.get("timeouts"),
                            "engine_active": lg.get("engine_active"),
                            "reserved_slots": lg.get("reserved_slots"),
                            "reserved_idle": lg.get("reserved_idle"),
                            "burst_running": lg.get("burst_running"),
                            "burst_rate_lent": lg.get("burst_rate_lent"),
                            "step_rate_now": lg.get("step_rate_now"),
                            "queue_burst": queues.get("burst"),
                            "gw_revocations": sched.get("revocations"),
                            "gw_bursts_admitted": sched.get("bursts_admitted"),
                        }
                    )
            except (httpx.HTTPError, ValueError) as e:
                log.warning("replica sample failed: %s", e)
            await asyncio.sleep(self.scenario.sample_interval_s)

    async def _progress_loop(self) -> None:
        while True:
            await asyncio.sleep(self.progress_every_s)
            c = self.counters
            log.info(
                "t=%5.0fs sent=%d ok=%d refused=%d errors=%d inflight=%d out_tokens=%d",
                self.elapsed(), c.sent, c.ok, c.refused, c.errors, c.inflight, c.output_tokens,
            )
