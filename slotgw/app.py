"""FastAPI application factory: replicas, ledgers, scheduler and background loops."""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI

from . import __version__
from .admission import ReplicaLedger, Scheduler
from .api import GatewayState, build_router
from .calibration import ThroughputProfile
from .config import Config, ReplicaConfig
from .meter import Meter
from .replicas.base import Replica
from .replicas.vllm import VllmReplica

log = logging.getLogger("slotgw")


def profile_for(cfg: ReplicaConfig, config_dir: Path | None = None) -> ThroughputProfile:
    if cfg.curve:
        return ThroughputProfile.from_curve_string(cfg.curve, source=f"{cfg.id}:inline")
    if cfg.calibration:
        path = Path(cfg.calibration)
        if not path.is_file() and config_dir is not None:
            path = config_dir.parent / cfg.calibration
        return ThroughputProfile.from_calibration(path, cfg.curve_context)
    raise RuntimeError(f"replica {cfg.id}: admission needs `calibration` or `curve` in replicas.yaml")


async def build_ledgers(config: Config, replicas: list[Replica], config_dir: Path | None = None) -> list[ReplicaLedger]:
    by_id = {cfg.id: cfg for cfg in config.replicas}
    ledgers = []
    for rep in replicas:
        cfg = by_id.get(rep.id)
        if cfg is None:
            raise RuntimeError(f"replica {rep.id} has no config entry")
        profile = profile_for(cfg, config_dir)
        kv_total, block_size = cfg.kv_tokens, 16
        cc = await rep.cache_config()
        if cc is not None:
            block_size = cc.block_size or block_size
            if kv_total is None:
                kv_total = cc.kv_tokens_total
        if kv_total is None:
            raise RuntimeError(f"replica {rep.id}: KV budget unknown (replica unreachable and no kv_tokens override)")
        ledg = ReplicaLedger(rep, profile, kv_total, block_size, config.gateway.admission, cfg.slot_plan)
        log.info(
            "ledger %s: KV %d tokens (usable %d), profile %s, seats at 15/30/60 tps = %d/%d/%d",
            rep.id, kv_total, ledg.kv_usable, profile.source, ledg.seats(15), ledg.seats(30), ledg.seats(60),
        )
        ledgers.append(ledg)
    return ledgers


def create_app(config: Config, meter: Meter, replicas: list[Replica] | None = None, config_dir: str | Path | None = None) -> FastAPI:
    cdir = Path(config_dir) if config_dir else None

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        client = httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=10.0))
        reps: list[Replica] = list(replicas) if replicas is not None else [
            VllmReplica(cfg, config.hf_model(cfg.model), client) for cfg in config.replicas
        ]
        scheduler: Scheduler | None = None
        loops: list[asyncio.Task] = []
        if config.gateway.admission.enabled:
            ledgers = await build_ledgers(config, reps, cdir)
            scheduler = Scheduler(ledgers, config.gateway.admission)
            loops.append(asyncio.create_task(_tick_loop(scheduler, config.gateway.admission.dispatch_tick_s)))
            loops.append(asyncio.create_task(_health_loop(ledgers, config.gateway.admission.health_interval_s)))
        app.state.gw = GatewayState(config=config, meter=meter, replicas=reps, scheduler=scheduler)
        log.info(
            "slotgw %s: %d shapes, %d tenants, %d replicas, pacing %s, admission %s",
            __version__, len(config.shapes), len(config.tenants), len(reps),
            "on" if config.gateway.pacing.enabled else "off", "on" if scheduler else "off",
        )
        try:
            yield
        finally:
            for t in loops:
                t.cancel()
            for t in loops:
                try:
                    await t
                except (asyncio.CancelledError, Exception):  # noqa: BLE001, S110 - shutting down
                    pass
            await client.aclose()
            meter.close()

    app = FastAPI(title="slotgw", version=__version__, lifespan=lifespan)
    app.include_router(build_router())
    return app


async def _tick_loop(scheduler: Scheduler, interval: float) -> None:
    while True:
        await asyncio.sleep(interval)
        try:
            scheduler.dispatch()
        except Exception:
            log.exception("dispatch failed")


async def _health_loop(ledgers: list[ReplicaLedger], interval: float) -> None:
    while True:
        await asyncio.sleep(interval)
        for ledg in ledgers:
            try:
                h = await ledg.replica.health()  # type: ignore[attr-defined]
                if h.ok != ledg.healthy:
                    log.warning("replica %s healthy=%s %s", ledg.id, h.ok, h.error or "")
                ledg.healthy = h.ok
            except Exception as e:  # noqa: BLE001 - a failing probe marks the replica unhealthy
                if ledg.healthy:
                    log.warning("replica %s health probe failed: %s", ledg.id, e)
                ledg.healthy = False
