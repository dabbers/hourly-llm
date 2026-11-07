from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from slotgw.app import create_app
from slotgw.config import BurstSettings, ReplicaConfig, load_config
from slotgw.meter import Meter
from slotgw.replicas.fake_server import create_fake_app

from .servers import ServerHandle

REPO = Path(__file__).resolve().parents[1]
FAKE_CURVE = "1:200,8:150,16:100,32:60"


@pytest.fixture(scope="session")
def fake_server():
    app = create_fake_app(curve=FAKE_CURVE, prefill_tps=20000.0)
    handle = ServerHandle(app).start()
    yield handle
    handle.stop()


@pytest.fixture(scope="session")
def meter_path(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("run") / "meter.sqlite"


@pytest.fixture(scope="session")
def gateway(fake_server, meter_path):
    config = load_config(REPO / "config")
    config = config.model_copy(
        update={
            "replicas": [
                ReplicaConfig(
                    id="fake-0", kind="fake", url=fake_server.url, model="small", curve=FAKE_CURVE,
                    slot_plan={"small.8k.30tps": 8},
                )
            ],
            # the pacing tests measure delivery at the shape's rate; burst would lend the fake engine's spare rate
            "gateway": config.gateway.model_copy(update={"burst": BurstSettings(enabled=False)}),
        }
    )
    meter = Meter(meter_path, run_id="test")
    app = create_app(config, meter)
    handle = ServerHandle(app).start()
    yield handle
    handle.stop()


@pytest.fixture
def meter_rows(meter_path):
    def _rows(**where):
        conn = sqlite3.connect(str(meter_path))
        conn.row_factory = sqlite3.Row
        sql = "SELECT * FROM requests"
        params: list = []
        if where:
            sql += " WHERE " + " AND ".join(f"{k} = ?" for k in where)
            params = list(where.values())
        rows = [dict(r) for r in conn.execute(sql, params).fetchall()]
        conn.close()
        return rows

    return _rows
