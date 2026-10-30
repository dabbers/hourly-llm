"""FastAPI application factory."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI

from . import __version__
from .api import GatewayState, build_router
from .config import Config
from .meter import Meter
from .replicas.base import Replica
from .replicas.vllm import VllmReplica

log = logging.getLogger("slotgw")


def create_app(config: Config, meter: Meter, replicas: list[Replica] | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        client = httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=10.0))
        reps: list[Replica] = list(replicas) if replicas is not None else [
            VllmReplica(cfg, config.hf_model(cfg.model), client) for cfg in config.replicas
        ]
        app.state.gw = GatewayState(config=config, meter=meter, replicas=reps)
        log.info("slotgw %s: %d shapes, %d tenants, %d replicas", __version__, len(config.shapes), len(config.tenants), len(reps))
        try:
            yield
        finally:
            await client.aclose()
            meter.close()

    app = FastAPI(title="slotgw", version=__version__, lifespan=lifespan)
    app.include_router(build_router())
    return app
