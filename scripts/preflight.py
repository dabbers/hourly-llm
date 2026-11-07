"""Refuse to start a paid run unless the gateway and every replica are healthy, idle and the ones
we think they are (docs/PHASE2.md section 4, G0).

    uv run python scripts/preflight.py --gateway http://127.0.0.1:8080 --expect-replicas 3 \
        --expect-model Qwen/Qwen2.5-7B-Instruct-AWQ [--not-run-id <previous>] \
        [--expect-calibration runpod-4090] [--runs-dir runs] [--min-free-gb 5] [--json]

Exit 0 and `PREFLIGHT OK run_id=<id>` when every check passes; otherwise one `PREFLIGHT FAIL
<check>: <detail>` line per failed check and exit 1. Checks: /healthz answers; the run id is not the
previous run's; exactly the expected number of replicas, each healthy with nothing running or
waiting and KV below 2%, ledger idle (engine_active 0, running equal to reserved_slots, so idle
reserved slots are the only holds allowed); every replica serves the expected HF model; every
ledger's calibration source contains the expected substring; scheduler queues empty; the tokenizer
agrees with loadgen's token-exact vocabulary within 2%; and the runs directory has room.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loadgen.content import words  # repo root on sys.path first, as scripts/calibrate.py does

KV_USAGE_MAX = 0.02
TOKENIZER_TOLERANCE = 0.02
SAMPLE_SEED, SAMPLE_TOKENS = 12345, 400


@dataclass
class Check:
    name: str
    ok: bool
    detail: str = ""


def _free_gb(path: Path) -> float:
    p = path.resolve()
    while not p.exists():
        if p.parent == p:
            break
        p = p.parent
    return shutil.disk_usage(p).free / 1e9


def run_checks(
    gateway: str,
    *,
    expect_replicas: int,
    expect_model: str,
    not_run_id: str | None = None,
    expect_calibration: str | None = None,
    runs_dir: str | Path = "runs",
    min_free_gb: float = 5.0,
    client: httpx.Client | None = None,
    kv_usage_max: float = KV_USAGE_MAX,
) -> tuple[list[Check], str | None]:
    """All checks against one gateway; returns (checks, run_id). Never raises on gateway errors."""
    checks: list[Check] = []
    c = client or httpx.Client(base_url=gateway, timeout=20)
    run_id: str | None = None

    try:
        r = c.get("/healthz")
        r.raise_for_status()
        health = r.json()
        run_id = str(health.get("run_id"))
        checks.append(Check("healthz", bool(health.get("ok")), f"run_id={run_id}"))
    except (httpx.HTTPError, ValueError) as e:
        checks.append(Check("healthz", False, f"{gateway}/healthz: {e}"))
        checks.append(Check("disk_free", *_disk(runs_dir, min_free_gb)))
        return checks, None

    if not_run_id is not None:
        checks.append(Check("run_id_fresh", run_id != not_run_id,
                            f"run_id={run_id}" + ("" if run_id != not_run_id else " equals the previous run's")))

    try:
        r = c.get("/admin/replicas")
        r.raise_for_status()
        snap = r.json()
    except (httpx.HTTPError, ValueError) as e:
        checks.append(Check("replicas", False, f"/admin/replicas: {e}"))
        snap = {"replicas": [], "scheduler": None}
    replicas: list[dict[str, Any]] = snap.get("replicas") or []
    checks.append(Check("replica_count", len(replicas) == expect_replicas,
                        f"{len(replicas)} replica(s), expected {expect_replicas}: "
                        + ", ".join(str(x.get("id")) for x in replicas)))

    for rep in replicas:
        rid = rep.get("id")
        h = rep.get("health") or {}
        problems: list[str] = []
        if not h.get("ok"):
            problems.append(f"unhealthy: {h.get('error')}")
        if h.get("running") != 0:
            problems.append(f"running={h.get('running')}")
        if h.get("waiting") != 0:
            problems.append(f"waiting={h.get('waiting')}")
        kv = h.get("kv_usage")
        if kv is None or kv >= kv_usage_max:
            problems.append(f"kv_usage={kv} (limit {kv_usage_max})")
        checks.append(Check(f"replica_idle[{rid}]", not problems, "; ".join(problems) or "healthy, nothing running"))

        lg = rep.get("ledger")
        if lg is None:
            checks.append(Check(f"ledger_idle[{rid}]", False, "no ledger (admission control off?)"))
        else:
            lp: list[str] = []
            if lg.get("engine_active") != 0:
                lp.append(f"engine_active={lg.get('engine_active')}")
            if lg.get("running") != lg.get("reserved_slots"):
                lp.append(f"running={lg.get('running')} reserved_slots={lg.get('reserved_slots')}")
            if not lg.get("healthy", True):
                lp.append("ledger marks the replica unhealthy")
            detail = "; ".join(lp) or f"idle (reserved_slots={lg.get('reserved_slots')}, kv_usable={lg.get('kv_usable')})"
            checks.append(Check(f"ledger_idle[{rid}]", not lp, detail))
            if expect_calibration:
                src = str(lg.get("profile") or "")
                checks.append(Check(f"calibration[{rid}]", expect_calibration in src,
                                    f"profile={src!r}, expected to contain {expect_calibration!r}"))

        checks.append(Check(f"model[{rid}]", rep.get("hf_model") == expect_model,
                            f"hf_model={rep.get('hf_model')!r}, expected {expect_model!r}"))

    sched = snap.get("scheduler")
    if sched is None:
        checks.append(Check("queues_empty", False, "no scheduler snapshot (admission control off?)"))
    else:
        queues = sched.get("queues") or {}
        checks.append(Check("queues_empty", all(v == 0 for v in queues.values()) and bool(queues), f"queues={queues}"))

    checks.append(_tokenizer_check(c))
    checks.append(Check("disk_free", *_disk(runs_dir, min_free_gb)))
    return checks, run_id


def _disk(runs_dir: str | Path, min_free_gb: float) -> tuple[bool, str]:
    free = _free_gb(Path(runs_dir))
    return free >= min_free_gb, f"{free:.1f} GB free at {runs_dir} (need {min_free_gb:g})"


def _tokenizer_check(c: httpx.Client) -> Check:
    """Same method as loadgen/runner.py `_tokenizer_check`: template overhead from a one-word
    message, then a 400-token sample must count to 400 + overhead within 2%."""
    try:
        r = c.post("/admin/tokenize", json={"messages": [{"role": "user", "content": "Hello"}]})
        r.raise_for_status()
        overhead = int(r.json()["count"]) - 1
        sample = words(SAMPLE_SEED, SAMPLE_TOKENS)
        r = c.post("/admin/tokenize", json={"messages": [{"role": "user", "content": sample}]})
        r.raise_for_status()
        count = int(r.json()["count"])
    except (httpx.HTTPError, KeyError, ValueError) as e:
        return Check("tokenizer", False, f"/admin/tokenize: {e}")
    expected = SAMPLE_TOKENS + overhead
    off = abs(count - expected) / SAMPLE_TOKENS
    return Check("tokenizer", off <= TOKENIZER_TOLERANCE,
                 f"{SAMPLE_TOKENS}-token sample counted {count} (overhead {overhead}, expected {expected}, off {off:.1%})")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="refuse to start a run unless the gateway and replicas are healthy and idle")
    p.add_argument("--gateway", default="http://127.0.0.1:8080")
    p.add_argument("--expect-replicas", type=int, required=True)
    p.add_argument("--expect-model", required=True, help="HF model id every replica must serve")
    p.add_argument("--not-run-id", default=None, help="the previous run's id; the gateway must have a different one")
    p.add_argument("--expect-calibration", default=None, help="substring every ledger's calibration source must contain")
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--min-free-gb", type=float, default=5.0)
    p.add_argument("--json", action="store_true", help="print the check results as JSON")
    args = p.parse_args(argv)

    checks, run_id = run_checks(
        args.gateway, expect_replicas=args.expect_replicas, expect_model=args.expect_model,
        not_run_id=args.not_run_id, expect_calibration=args.expect_calibration,
        runs_dir=args.runs_dir, min_free_gb=args.min_free_gb,
    )
    failed = [c for c in checks if not c.ok]
    if args.json:
        print(json.dumps({"ok": not failed, "run_id": run_id, "checks": [asdict(c) for c in checks]}, indent=1))
    else:
        for c in failed:
            print(f"PREFLIGHT FAIL {c.name}: {c.detail}")
        if not failed:
            print(f"PREFLIGHT OK run_id={run_id}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
