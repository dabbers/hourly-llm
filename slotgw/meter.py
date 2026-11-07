"""Meter: one SQLite row per request with the timestamps and token counts that price it
both ways (docs/DESIGN.md section 8)."""

from __future__ import annotations

import csv
import sqlite3
import threading
from dataclasses import asdict, dataclass, fields
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS requests (
  run_id TEXT NOT NULL,
  request_id TEXT PRIMARY KEY,
  tenant TEXT NOT NULL,
  shape TEXT NOT NULL,
  lazy INTEGER NOT NULL,
  replica TEXT,
  delivery TEXT NOT NULL,
  class_ctx TEXT NOT NULL,
  t_arrival REAL NOT NULL,
  t_admitted REAL,
  t_first_token REAL,
  t_engine_done REAL,
  t_last_token REAL,
  status TEXT NOT NULL,
  http_status INTEGER NOT NULL,
  prompt_tokens INTEGER NOT NULL,
  output_tokens INTEGER NOT NULL,
  output_tokens_guaranteed INTEGER NOT NULL,
  output_tokens_burst INTEGER NOT NULL,
  throttled_ms REAL NOT NULL,
  queued_ms REAL NOT NULL,
  wasted_tokens INTEGER NOT NULL,
  ttft_ms REAL,
  e2e_ms REAL,
  revoked INTEGER NOT NULL,
  preempted_runs INTEGER NOT NULL,
  abort_reason TEXT,
  load_run TEXT,
  scenario TEXT,
  seed INTEGER,
  delivered_tokens INTEGER,
  reservation_id TEXT
);
CREATE TABLE IF NOT EXISTS reservations (
  run_id TEXT NOT NULL,
  reservation_id TEXT PRIMARY KEY,
  tenant TEXT NOT NULL,
  shape TEXT NOT NULL,
  slots INTEGER NOT NULL,
  replicas TEXT,
  t_start REAL NOT NULL,
  t_end REAL,
  price_per_second REAL NOT NULL,
  served INTEGER NOT NULL,
  timeouts INTEGER NOT NULL,
  load_run TEXT,
  scenario TEXT,
  seed INTEGER
);
"""


@dataclass
class RequestRecord:
    request_id: str
    tenant: str
    shape: str
    lazy: bool
    delivery: str  # stream | complete
    t_arrival: float
    replica: str | None = None
    class_ctx: str = "guaranteed"  # guaranteed | burst
    t_admitted: float | None = None
    t_first_token: float | None = None
    t_engine_done: float | None = None
    t_last_token: float | None = None
    status: str = "ok"  # ok | refused | upstream_error | client_disconnect | error
    http_status: int = 200
    prompt_tokens: int = 0
    output_tokens: int = 0
    output_tokens_guaranteed: int = 0
    output_tokens_burst: int = 0
    throttled_ms: float = 0.0
    queued_ms: float = 0.0
    wasted_tokens: int = 0
    revoked: bool = False
    preempted_runs: int = 0
    abort_reason: str | None = None
    load_run: str | None = None  # set from X-Slotgw-Run by the load generator
    scenario: str | None = None
    seed: int | None = None
    delivered_tokens: int = 0  # tokens the client actually received (billed quantity)
    reservation_id: str | None = None  # served from this reservation's slot (milestone 4)

    @property
    def ttft_ms(self) -> float | None:
        if self.t_first_token is None or self.t_admitted is None:
            return None
        return (self.t_first_token - self.t_admitted) * 1000.0

    @property
    def e2e_ms(self) -> float | None:
        if self.t_last_token is None:
            return None
        return (self.t_last_token - self.t_arrival) * 1000.0


_COLUMNS = [f.name for f in fields(RequestRecord)] + ["ttft_ms", "e2e_ms"]


@dataclass
class ReservationRecord:
    """One standing reservation: billed slots x (t_end - t_start) x price_per_second (DESIGN 8)."""

    reservation_id: str
    tenant: str
    shape: str
    slots: int
    t_start: float
    price_per_second: float
    replicas: str | None = None
    t_end: float | None = None
    served: int = 0
    timeouts: int = 0
    load_run: str | None = None
    scenario: str | None = None
    seed: int | None = None


_RES_COLUMNS = [f.name for f in fields(ReservationRecord)]


class Meter:
    """Append-only request log. Writes are tiny and synchronous; a lock makes it safe to call
    from the event loop and from worker threads."""

    def __init__(self, path: str | Path, run_id: str):
        self.path = Path(path)
        self.run_id = run_id
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(SCHEMA)
        have = {row[1] for row in self._conn.execute("PRAGMA table_info(requests)")}
        for col, typ in (
            ("load_run", "TEXT"), ("scenario", "TEXT"), ("seed", "INTEGER"), ("delivered_tokens", "INTEGER"),
            ("reservation_id", "TEXT"),
        ):
            if col not in have:  # meter files written before milestone 2
                self._conn.execute(f"ALTER TABLE requests ADD COLUMN {col} {typ}")
        self._conn.commit()

    def record(self, rec: RequestRecord) -> None:
        row = asdict(rec)
        row["ttft_ms"] = rec.ttft_ms
        row["e2e_ms"] = rec.e2e_ms
        row["lazy"] = int(rec.lazy)
        row["revoked"] = int(rec.revoked)
        cols = ["run_id"] + _COLUMNS
        values = [self.run_id] + [row[c] for c in _COLUMNS]
        sql = f"INSERT OR REPLACE INTO requests ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})"
        with self._lock:
            self._conn.execute(sql, values)
            self._conn.commit()

    def record_reservation(self, rec: ReservationRecord) -> None:
        row = asdict(rec)
        cols = ["run_id"] + _RES_COLUMNS
        values = [self.run_id] + [row[c] for c in _RES_COLUMNS]
        sql = f"INSERT OR REPLACE INTO reservations ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})"
        with self._lock:
            self._conn.execute(sql, values)
            self._conn.commit()

    def count(self) -> int:
        with self._lock:
            return int(self._conn.execute("SELECT COUNT(*) FROM requests").fetchone()[0])

    def export_csv(self, path: str | Path) -> int:
        with self._lock:
            cur = self._conn.execute("SELECT * FROM requests ORDER BY t_arrival")
            names = [d[0] for d in cur.description]
            rows = cur.fetchall()
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(names)
            w.writerows(rows)
        return len(rows)

    def close(self) -> None:
        with self._lock:
            self._conn.close()
