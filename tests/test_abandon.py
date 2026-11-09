"""A request whose client disconnects while it waits for admission leaves the queue unserved and
unbilled (DESIGN.md 4.1 step 5). Found in the phase-3 campaign (PHASE3.md section 6): a load generator
that gave up after 600 s on queued lazy requests had eight of them served to a closed socket."""

from __future__ import annotations

import asyncio
import json
import time

import httpx
import pytest

from slotgw.admission import Scheduler
from slotgw.config import AdmissionSettings

from .test_admission import S8_30, ledger
from .test_gateway import CHAT_KEY, MESSAGES

CURVE = "1:100,2:60,3:40,4:30"  # seats(30 tps) = 3


async def test_cancelled_waiter_leaves_the_queue_and_the_next_waiter_gets_the_seat():
    led = ledger(curve=CURVE)
    sched = Scheduler([led], AdmissionSettings())
    holds = [await sched.admit(f"c{i}", S8_30, 100) for i in range(3)]
    gone = asyncio.create_task(sched.admit("gone", S8_30, 100))
    await asyncio.sleep(0.02)
    stays = asyncio.create_task(sched.admit("stays", S8_30, 100))
    await asyncio.sleep(0.02)
    assert sched.queue_lengths()["interactive"] == 2
    gone.cancel()
    with pytest.raises(asyncio.CancelledError):
        await gone
    assert sched.queue_lengths()["interactive"] == 1
    assert sched.abandoned == 1 and sched.timeouts == 0
    sched.release(holds[0])
    await asyncio.wait_for(stays, 1)  # the freed seat goes to the waiter that is still there
    assert led.n_running == 3 and sched.queue_lengths()["interactive"] == 0


async def test_cancel_in_the_instant_of_dispatch_gives_the_seat_back():
    led = ledger(curve=CURVE)
    sched = Scheduler([led], AdmissionSettings())
    holds = [await sched.admit(f"c{i}", S8_30, 100) for i in range(3)]
    gone = asyncio.create_task(sched.admit("gone", S8_30, 100))
    await asyncio.sleep(0.02)
    sched.release(holds[0])  # dispatch hands the seat to the waiter synchronously ...
    assert led.n_running == 3
    gone.cancel()  # ... and its client is gone before the handler ever runs
    with pytest.raises(asyncio.CancelledError):
        await gone
    assert led.n_running == 2 and sched.abandoned == 1  # the seat went back
    await asyncio.wait_for(sched.admit("next", S8_30, 100), 1)
    assert led.n_running == 3


async def test_cancelled_reservation_waiter_leaves_the_reservation_queue():
    led = ledger(curve=CURVE)
    sched = Scheduler([led], AdmissionSettings())
    res = sched.reserve("chat", S8_30, 2)
    a = await sched.admit("a", S8_30, 100, tenant="chat")
    await sched.admit("b", S8_30, 100, tenant="chat")
    gone = asyncio.create_task(sched.admit("gone", S8_30, 100, tenant="chat"))
    await asyncio.sleep(0.02)
    assert len(res.queue) == 1
    gone.cancel()
    with pytest.raises(asyncio.CancelledError):
        await gone
    assert len(res.queue) == 0 and sched.abandoned == 1 and res.timeouts == 0
    sched.release(a)
    assert res.free_slot() is not None  # the slot waits for the next reserved request, not for the one that left


@pytest.fixture(scope="module")
def tight_gateway(fake_server, tmp_path_factory):
    """A gateway whose one replica seats three small.8k.30tps requests (ledger curve 1:100,2:60,3:40,4:30),
    so a fourth request queues; its own meter."""
    from slotgw.app import create_app
    from slotgw.config import BurstSettings, ReplicaConfig, load_config
    from slotgw.meter import Meter

    from .conftest import REPO
    from .servers import ServerHandle

    config = load_config(REPO / "config")
    config = config.model_copy(
        update={
            "replicas": [ReplicaConfig(id="tight-0", kind="fake", url=fake_server.url, model="small", curve=CURVE)],
            "gateway": config.gateway.model_copy(update={"burst": BurstSettings(enabled=False)}),
        }
    )
    meter_path = tmp_path_factory.mktemp("tight") / "meter.sqlite"
    handle = ServerHandle(create_app(config, Meter(meter_path, run_id="tight"))).start()
    handle.meter_path = meter_path  # type: ignore[attr-defined]
    yield handle
    handle.stop()


def _rows(meter_path, **where) -> list[dict]:
    import sqlite3

    conn = sqlite3.connect(str(meter_path))
    conn.row_factory = sqlite3.Row
    sql = "SELECT * FROM requests" + (" WHERE " + " AND ".join(f"{k} = ?" for k in where) if where else "")
    rows = [dict(r) for r in conn.execute(sql, list(where.values())).fetchall()]
    conn.close()
    return rows


async def test_queued_request_whose_client_disconnects_is_never_served_or_billed(tight_gateway, fake_server, monkeypatch):
    from slotgw import api

    monkeypatch.setattr(api, "DISCONNECT_POLL_S", 0.2)  # the gateway thread reads the module global at each poll
    gateway = tight_gateway

    async def engine_completed() -> int:
        async with httpx.AsyncClient(timeout=10) as c:
            text = (await c.get(fake_server.url + "/metrics")).text
        line = next(ln for ln in text.splitlines() if ln.startswith("vllm:request_success_total"))
        return int(float(line.split()[1]))

    done_before = await engine_completed()

    async def occupy(i: int) -> int:
        async with httpx.AsyncClient(base_url=gateway.url, timeout=90) as c:
            r = await c.post(
                "/v1/chat/completions",
                json={"model": "small.8k.30tps", "messages": MESSAGES + [{"role": "user", "content": f"occupy {i}"}],
                      "max_tokens": 400, "ignore_eos": True},
                headers=CHAT_KEY,
            )
            return r.status_code

    seats = [asyncio.create_task(occupy(i)) for i in range(3)]  # all three seats of the tight replica
    await asyncio.sleep(0.5)
    async with httpx.AsyncClient(base_url=gateway.url) as c:
        snap = (await c.get("/admin/replicas")).json()
    assert snap["replicas"][0]["ledger"]["running"] == 3, snap["replicas"][0]["ledger"]
    # The ninth request goes over a raw socket so that the disconnect is a real TCP close while it queues.
    body = json.dumps({"model": "small.8k.30tps", "messages": MESSAGES + [{"role": "user", "content": "abandoned"}],
                       "max_tokens": 5}).encode()
    head = (b"POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\nAuthorization: Bearer sk-local-chat\r\n"
            b"Content-Type: application/json\r\nContent-Length: " + str(len(body)).encode() + b"\r\n\r\n")
    _reader, writer = await asyncio.open_connection("127.0.0.1", gateway.port)
    writer.write(head + body)
    await writer.drain()
    await asyncio.sleep(0.5)  # queued: every seat is taken
    writer.close()
    await writer.wait_closed()

    assert all(code == 200 for code in await asyncio.gather(*seats))
    rows: list[dict] = []
    deadline = time.monotonic() + 8
    while time.monotonic() < deadline:
        rows = _rows(gateway.meter_path, abort_reason="client_disconnect_queued")
        if rows:
            break
        await asyncio.sleep(0.2)
    assert len(rows) == 1, f"the abandoned request was not metered as a queued disconnect: {_rows(gateway.meter_path)}"
    row = rows[0]
    assert row["status"] == "client_disconnect" and row["http_status"] == 499
    assert row["output_tokens"] == 0 and row["t_admitted"] is None and row["queued_ms"] >= 400
    assert len(_rows(gateway.meter_path, status="ok")) == 3
    assert await engine_completed() == done_before + 3  # the engine never saw the abandoned request
    async with httpx.AsyncClient(base_url=gateway.url) as c:
        snap = (await c.get("/admin/replicas")).json()
    assert snap["scheduler"]["abandoned"] == 1 and snap["scheduler"]["queues"]["interactive"] == 0
