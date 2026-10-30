"""Run ASGI apps as real uvicorn servers in background threads so tests exercise real HTTP
streaming between the gateway and its replicas."""

from __future__ import annotations

import socket
import threading
import time

import uvicorn


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class ServerHandle:
    def __init__(self, app, port: int | None = None):
        self.port = port or free_port()
        cfg = uvicorn.Config(app, host="127.0.0.1", port=self.port, log_level="warning", lifespan="on")
        self.server = uvicorn.Server(cfg)
        self.thread = threading.Thread(target=self.server.run, daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self) -> ServerHandle:
        self.thread.start()
        deadline = time.time() + 15
        while not self.server.started:
            if time.time() > deadline:
                raise RuntimeError("server did not start")
            if not self.thread.is_alive():
                raise RuntimeError("server thread died during startup")
            time.sleep(0.02)
        return self

    def stop(self) -> None:
        self.server.should_exit = True
        self.thread.join(10)
