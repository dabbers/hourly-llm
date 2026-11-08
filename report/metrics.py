"""Metrics over meter rows (docs/DESIGN.md section 10). Everything derives from
runs/<run_id>/meter.sqlite plus the load generator's side files under runs/<run_id>/loads/*/."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from slotgw.billing import per_token_usd, shape_seconds
from slotgw.calibration import profile_for
from slotgw.config import Config, PriceReference, load_config

LONG_PREFILL_TOKENS = 3000  # a prompt at least this long counts as a "long prefill" (DESIGN A4)
FLOOR_TOLERANCE = 0.95  # delivered rate below 95% of the shape's tps is a floor violation
MIN_JUDGE_TOKENS = 30  # delivered tokens needed before a request's rate is judged
MIN_JUDGE_STREAM_S = 0.5

GROUP = ["scenario", "seed", "load_run", "tenant", "shape", "delivery"]


@dataclass
class LoadRun:
    id: str
    dir: Path
    meta: dict
    samples: pd.DataFrame | None


RESERVATION_COLUMNS = ["run_id", "reservation_id", "tenant", "shape", "slots", "replicas", "t_start", "t_end", "price_per_second", "served", "timeouts", "load_run", "scenario", "seed"]


@dataclass
class RunData:
    run_dir: Path
    run_id: str
    requests: pd.DataFrame
    loads: list[LoadRun]
    config: Config  # the run's own config snapshot when it has one
    fallback_config: Config | None = None  # the --config-dir config, for settings the snapshot lacks
    reservations: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=RESERVATION_COLUMNS))
    texts: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=TEXT_COLUMNS))


def read_meter(path: Path) -> pd.DataFrame:
    conn = sqlite3.connect(str(path))
    try:
        df = pd.read_sql_query("SELECT * FROM requests ORDER BY t_arrival", conn)
    finally:
        conn.close()
    if "reservation_id" not in df:  # meter files written before milestone 4
        df["reservation_id"] = None
    for col, default in (("cached_tokens", 0), ("billed_shape", None), ("showcase", None)):  # before phase 2
        if col not in df:
            df[col] = default
    return df


TEXT_COLUMNS = ["run_id", "request_id", "prompt_json", "output_text"]


def read_texts(path: Path) -> pd.DataFrame:
    """The completions (and showcase prompts) the gateway kept when store_text was on."""
    conn = sqlite3.connect(str(path))
    try:
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if "texts" not in tables:
            return pd.DataFrame(columns=TEXT_COLUMNS)
        return pd.read_sql_query("SELECT run_id, request_id, prompt_json, output_text FROM texts", conn)
    finally:
        conn.close()


def read_reservations(path: Path) -> pd.DataFrame:
    conn = sqlite3.connect(str(path))
    try:
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if "reservations" not in tables:
            return pd.DataFrame(columns=RESERVATION_COLUMNS)
        return pd.read_sql_query("SELECT * FROM reservations ORDER BY t_start", conn)
    finally:
        conn.close()


def _read_samples(path: Path) -> pd.DataFrame | None:
    if not path.is_file() or path.stat().st_size == 0:
        return None
    df = pd.read_csv(path)
    return df if not df.empty else None


def load_run_dir(run_dir: str | Path, config_dir: str | Path | None = None) -> RunData:
    run_dir = Path(run_dir)
    meter = run_dir / "meter.sqlite"
    if not meter.is_file():
        raise FileNotFoundError(f"no meter.sqlite in {run_dir}")
    df = read_meter(meter)
    loads: list[LoadRun] = []
    loads_dir = run_dir / "loads"
    if loads_dir.is_dir():
        for d in sorted(p for p in loads_dir.iterdir() if p.is_dir()):
            meta_path = d / "loadrun.json"
            meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {"load_run": d.name}
            loads.append(LoadRun(id=str(meta.get("load_run") or d.name), dir=d, meta=meta, samples=_read_samples(d / "samples.csv")))
    cfg_dir: Path | None = None
    for lr in loads:
        if (lr.dir / "config" / "shapes.yaml").is_file():
            cfg_dir = lr.dir / "config"  # the configuration the run actually used
            break
    fallback_dir = Path(config_dir) if config_dir else Path("config")
    fallback = load_config(fallback_dir) if (fallback_dir / "shapes.yaml").is_file() else None
    config = load_config(cfg_dir) if cfg_dir is not None else fallback
    if config is None:
        raise FileNotFoundError(f"no config snapshot in {run_dir} and no config at {fallback_dir}")
    run_id = str(df["run_id"].iloc[0]) if not df.empty else run_dir.name
    return RunData(
        run_dir=run_dir, run_id=run_id, requests=df, loads=loads, config=config, fallback_config=fallback,
        reservations=read_reservations(meter), texts=read_texts(meter),
    )


def enrich(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Per-request derived columns: both bills, delivery rates, floor violations."""
    df = df.copy()
    shapes = {s.name: s for s in config.shapes}
    pricing = config.pricing.token
    df["load_run"] = df["load_run"].fillna("adhoc")
    df["scenario"] = df["scenario"].fillna("adhoc")
    df["seed"] = df["seed"].fillna(0).astype(int)
    df["lazy"] = df["lazy"].astype(bool)
    # the shape whose tps and price applied: the request's own, or a context burst's burst class (post-M4)
    df["billed_shape"] = df["billed_shape"].where(df["billed_shape"].notna(), df["shape"])
    df["tps"] = df["billed_shape"].map(lambda n: shapes[n].tps if n in shapes else np.nan)
    df["price_per_second"] = df["billed_shape"].map(lambda n: shapes[n].price_per_second if n in shapes else np.nan)
    df["max_context"] = df["shape"].map(lambda n: shapes[n].max_context if n in shapes else np.nan)
    df["cached_tokens"] = pd.to_numeric(df["cached_tokens"], errors="coerce").fillna(0).astype(int).clip(lower=0)
    df["cached_tokens"] = np.minimum(df["cached_tokens"], df["prompt_tokens"])
    df["delivered_tokens"] = (df["output_tokens"] - df["wasted_tokens"]).clip(lower=0).astype(int)
    ok = df["status"] == "ok"
    df["ok"] = ok
    df["refused"] = df["status"] == "refused"
    df["revoked_run"] = df["status"] == "revoked"  # a context burst aborted for a guaranteed request
    df["error"] = ~ok & ~df["refused"] & ~df["revoked_run"]
    df["context_burst"] = df["class_ctx"] == "burst"
    df["reserved"] = df["reservation_id"].notna()
    df["showcase"] = df["showcase"].where(df["showcase"].notna(), None)

    secs = [
        shape_seconds(r.t_admitted, r.t_first_token, r.t_last_token, int(r.delivered_tokens), float(r.tps))
        if r.ok and not np.isnan(r.tps)
        else (0.0, 0.0, 0.0)
        for r in df.itertuples(index=False)
    ]
    df["wall_s"], df["bound_s"], df["billed_s"] = (np.array([s[i] for s in secs]) for i in range(3))
    # prefill charge: uncached prompt tokens in full, prefix-cache hits at the cache-read share (DESIGN 8)
    prompt_price = config.pricing.shape.prompt_usd_per_million / 1e6
    cache_ratio = config.pricing.cache_read_ratio
    charged_prompt = (df["prompt_tokens"] - df["cached_tokens"]) + df["cached_tokens"] * cache_ratio
    df["prompt_usd"] = np.where(ok & (df["delivered_tokens"] > 0), charged_prompt * prompt_price, 0.0)
    df["shape_usd"] = df["billed_s"] * df["price_per_second"].fillna(0) + df["prompt_usd"]
    df["bound_usd"] = df["bound_s"] * df["price_per_second"].fillna(0) + df["prompt_usd"]
    df["wall_usd"] = df["wall_s"] * df["price_per_second"].fillna(0) + df["prompt_usd"]
    df["burst_discount_usd"] = df["bound_usd"] - df["shape_usd"]
    df["per_token_usd"] = [
        per_token_usd(int(r.prompt_tokens), int(r.output_tokens), bool(r.lazy), pricing, int(r.cached_tokens)) if r.ok else 0.0
        for r in df.itertuples(index=False)
    ]

    df["stream_s"] = np.where(ok, df["t_last_token"] - df["t_first_token"], np.nan)
    engine_s = df["t_engine_done"] - df["t_first_token"]
    df["engine_tps"] = np.where(ok & (engine_s > 0.5), df["output_tokens"] / engine_s.replace(0, np.nan), np.nan)
    # Delivered rate over the whole stream, head start included: the bucket's head start can only
    # raise it, so a rate below the shape's tps means the engine fell behind the promise.
    judge = ok & (df["delivered_tokens"] >= MIN_JUDGE_TOKENS) & (df["stream_s"] > MIN_JUDGE_STREAM_S)
    df["delivered_tps"] = np.where(judge, df["delivered_tokens"] / df["stream_s"].replace(0, np.nan), np.nan)
    df["floor_violation"] = judge & (df["delivered_tps"] < FLOOR_TOLERANCE * df["tps"])
    df["violation_stream_s"] = np.where(df["floor_violation"], df["stream_s"], 0.0)
    df["throttled_s"] = df["throttled_ms"].fillna(0) / 1000.0
    t0 = df.groupby("load_run")["t_arrival"].transform("min")
    df["t_rel"] = df["t_arrival"] - t0
    return df


def _q(p: float):
    return lambda s: float(s.quantile(p)) if len(s) else np.nan


def per_tenant(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (scenario, seed, load_run, tenant, shape, delivery)."""
    df = df.assign(
        refused_context=df["refused"] & (df["abort_reason"] == "context_exceeds_shape"),
        refused_capacity=df["refused"] & df["abort_reason"].fillna("").str.startswith("capacity"),
    )
    all_rows = df.groupby(GROUP, dropna=False).agg(
        requests=("request_id", "count"),
        refused=("refused", "sum"),
        refused_context=("refused_context", "sum"),
        refused_capacity=("refused_capacity", "sum"),
        errors=("error", "sum"),
        revoked=("revoked_run", "sum"),
        context_bursts=("context_burst", "sum"),
        lazy=("lazy", "max"),
        tps=("tps", "first"),
        price_per_second=("price_per_second", "first"),
    )
    okdf = df[df["ok"]]
    ok_rows = okdf.groupby(GROUP, dropna=False).agg(
        ok=("request_id", "count"),
        reserved_requests=("reserved", "sum"),
        burst_tokens=("output_tokens_burst", "sum"),
        prompt_tokens=("prompt_tokens", "sum"),
        cached_tokens=("cached_tokens", "sum"),
        output_tokens=("output_tokens", "sum"),
        delivered_tokens=("delivered_tokens", "sum"),
        per_token_usd=("per_token_usd", "sum"),
        prompt_usd=("prompt_usd", "sum"),
        shape_usd=("shape_usd", "sum"),
        bound_usd=("bound_usd", "sum"),
        wall_usd=("wall_usd", "sum"),
        billed_s=("billed_s", "sum"),
        bound_s=("bound_s", "sum"),
        wall_s=("wall_s", "sum"),
        stream_s=("stream_s", "sum"),
        throttled_s=("throttled_s", "sum"),
        queued_p50_ms=("queued_ms", _q(0.5)),
        queued_p95_ms=("queued_ms", _q(0.95)),
        ttft_p50_ms=("ttft_ms", _q(0.5)),
        ttft_p95_ms=("ttft_ms", _q(0.95)),
        e2e_p50_ms=("e2e_ms", _q(0.5)),
        e2e_p95_ms=("e2e_ms", _q(0.95)),
        delivered_tps_mean=("delivered_tps", "mean"),
        engine_tps_mean=("engine_tps", "mean"),
        floor_judged=("delivered_tps", "count"),
        floor_violations=("floor_violation", "sum"),
        violation_stream_s=("violation_stream_s", "sum"),
    )
    out = all_rows.join(ok_rows, how="left")
    counts = ["ok", "reserved_requests", "burst_tokens", "prompt_tokens", "cached_tokens", "output_tokens", "delivered_tokens",
              "floor_judged", "floor_violations"]
    sums = ["per_token_usd", "prompt_usd", "shape_usd", "bound_usd", "wall_usd", "billed_s", "bound_s", "wall_s",
            "stream_s", "throttled_s", "violation_stream_s"]
    out[counts] = out[counts].fillna(0).astype(int)
    out[sums] = out[sums].fillna(0.0)
    out["burst_discount_usd"] = out["bound_usd"] - out["shape_usd"]
    with np.errstate(divide="ignore", invalid="ignore"):
        out["usd_per_1m_out_per_token"] = np.where(out["output_tokens"] > 0, out["per_token_usd"] / out["output_tokens"] * 1e6, np.nan)
        out["usd_per_1m_out_shape"] = np.where(out["delivered_tokens"] > 0, out["shape_usd"] / out["delivered_tokens"] * 1e6, np.nan)
        out["tokens_per_billed_s"] = np.where(out["billed_s"] > 0, out["delivered_tokens"] / out["billed_s"], np.nan)
        out["throttled_share"] = np.where(out["stream_s"] > 0, out["throttled_s"] / out["stream_s"], np.nan)
        out["floor_violation_share"] = np.where(out["stream_s"] > 0, out["violation_stream_s"] / out["stream_s"], np.nan)
        out["burst_share"] = np.where(out["output_tokens"] > 0, out["burst_tokens"] / out["output_tokens"], np.nan)
        out["cached_share"] = np.where(out["prompt_tokens"] > 0, out["cached_tokens"] / out["prompt_tokens"], np.nan)
    return out.reset_index()


def concurrency_series(ok: pd.DataFrame, t0: float, t1: float) -> np.ndarray:
    """Requests in flight (admitted, not yet delivered) per one-second bin from t0 to t1."""
    n = max(1, int(np.ceil(t1 - t0)) + 1)
    delta = np.zeros(n + 1)
    starts = np.clip(np.floor(ok["t_admitted"].to_numpy() - t0).astype(int), 0, n)
    ends = np.clip(np.ceil(ok["t_last_token"].to_numpy() - t0).astype(int), 0, n)
    np.add.at(delta, starts, 1)
    np.add.at(delta, ends, -1)
    return np.cumsum(delta)[:n]


def per_run(df: pd.DataFrame, loads: list[LoadRun], config: Config, reservations: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per load run: totals, both bills, GPU cost, concurrency and sampled replica state."""
    metas = {lr.id: lr for lr in loads}
    rb = reservation_bills(df, reservations, config) if reservations is not None and len(reservations) else pd.DataFrame()
    rows = []
    for load_run, sub in df.groupby("load_run", sort=False):
        ok = sub[sub["ok"]]
        t0 = float(sub["t_arrival"].min())
        t1 = float(max(ok["t_last_token"].max() if len(ok) else t0, sub["t_arrival"].max()))
        duration = max(0.0, t1 - t0)
        conc = concurrency_series(ok, t0, t1) if len(ok) else np.zeros(1)
        lr = metas.get(load_run)
        meta = lr.meta if lr else {}
        samples = lr.samples if lr else None
        row = {
            "scenario": sub["scenario"].iloc[0],
            "seed": int(sub["seed"].iloc[0]),
            "load_run": load_run,
            "started_at": meta.get("started_at"),
            "pacing": (meta.get("gateway_pacing") or {}).get("enabled", config.gateway.pacing.enabled),
            "planned_duration_s": meta.get("duration_s"),
            "duration_s": round(duration, 1),
            "requests": len(sub),
            "ok": len(ok),
            "refused": int(sub["refused"].sum()),
            "errors": int(sub["error"].sum()),
            "revoked": int(sub["revoked_run"].sum()),
            "context_bursts": int(sub["context_burst"].sum()),
            "reserved_requests": int(ok["reserved"].sum()),
            "output_tokens_burst": int(ok["output_tokens_burst"].sum()),
            "burst_share": float(ok["output_tokens_burst"].sum() / ok["output_tokens"].sum()) if ok["output_tokens"].sum() else np.nan,
            "burst_discount_usd": float(ok["bound_usd"].sum() - ok["shape_usd"].sum()),
            "reservation_usd": float(rb.loc[rb["load_run"] == load_run, "reservation_usd"].sum()) if len(rb) else 0.0,
            "tenant_usd": float(ok.loc[~ok["reserved"], "shape_usd"].sum())
            + (float(rb.loc[rb["load_run"] == load_run, "reservation_usd"].sum()) if len(rb) else 0.0),
            "prompt_tokens": int(ok["prompt_tokens"].sum()),
            "cached_tokens": int(ok["cached_tokens"].sum()),
            "cached_share": float(ok["cached_tokens"].sum() / ok["prompt_tokens"].sum()) if ok["prompt_tokens"].sum() else np.nan,
            "output_tokens": int(ok["output_tokens"].sum()),
            "delivered_tokens": int(ok["delivered_tokens"].sum()),
            "per_token_usd": float(ok["per_token_usd"].sum()),
            "shape_usd": float(ok["shape_usd"].sum()),
            "bound_usd": float(ok["bound_usd"].sum()),
            "wall_usd": float(ok["wall_usd"].sum()),
            "replicas_used": _gpus_used(ok),
            "gpu_cost_usd": config.pricing.provider.gpu_usd_per_hour * _gpus_used(ok) * duration / 3600.0,
            "output_tokens_per_gpu_hour": (ok["output_tokens"].sum() / (_gpus_used(ok) * duration) * 3600.0) if duration > 0 else np.nan,
            "concurrency_mean": float(conc.mean()),
            "concurrency_p95": float(np.percentile(conc, 95)),
            "concurrency_max": float(conc.max()),
            "throttled_s": float(ok["throttled_s"].sum()),
            "floor_violations": int(ok["floor_violation"].sum()),
            "kv_usage_mean": np.nan,
            "kv_usage_p95": np.nan,
            "engine_running_mean": np.nan,
            "preemptions": np.nan,
            "burst_rate_lent_mean": np.nan,
            "revocations_sampled": np.nan,
        }
        if samples is not None and "burst_rate_lent" in samples:
            lent = pd.to_numeric(samples["burst_rate_lent"], errors="coerce").dropna()
            rev = pd.to_numeric(samples["gw_revocations"], errors="coerce").dropna()
            row["burst_rate_lent_mean"] = float(lent.mean()) if len(lent) else np.nan
            row["revocations_sampled"] = float(rev.max()) if len(rev) else np.nan
        if samples is not None and "kv_usage" in samples:
            kv = pd.to_numeric(samples["kv_usage"], errors="coerce").dropna()
            running = pd.to_numeric(samples["running"], errors="coerce").dropna()
            pre = pd.to_numeric(samples["preemptions"], errors="coerce").dropna()
            row["kv_usage_mean"] = float(kv.mean()) if len(kv) else np.nan
            row["kv_usage_p95"] = float(kv.quantile(0.95)) if len(kv) else np.nan
            row["engine_running_mean"] = float(running.mean()) if len(running) else np.nan
            row["preemptions"] = float(pre.max() - pre.min()) if len(pre) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def reservation_bills(df: pd.DataFrame, reservations: pd.DataFrame | None, config: Config) -> pd.DataFrame:
    """Per reservation (load run, tenant): the standing bill, slots x duration x price per second,
    how much of it the tenant used, and the tenant's bill with the reservation (reservation plus
    on-demand shape bills for its unreserved requests) against per-request shape and per-token."""
    if reservations is None or reservations.empty:
        return pd.DataFrame()
    rows = []
    for _, r in reservations.iterrows():
        run = df[df["load_run"] == r["load_run"]] if pd.notna(r.get("load_run")) else df
        _t0, t1 = _run_window(run) if len(run) else (float(r["t_start"]), float(r["t_start"]))
        t_end = float(r["t_end"]) if pd.notna(r["t_end"]) else t1
        duration = max(0.0, t_end - float(r["t_start"]))
        slots = int(r["slots"])
        ok = run[run["ok"] & (run["tenant"] == r["tenant"])]
        served = ok[ok["reservation_id"] == r["reservation_id"]]
        used_s = float((served["t_engine_done"] - served["t_admitted"]).clip(lower=0).sum())
        reserved_s = slots * duration
        bill = reserved_s * float(r["price_per_second"])
        on_demand = float(ok.loc[ok["reservation_id"] != r["reservation_id"], "shape_usd"].sum())
        shape_all = float(ok["shape_usd"].sum())
        per_token = float(ok["per_token_usd"].sum())
        rows.append({
            "scenario": run["scenario"].iloc[0] if len(run) else r.get("scenario"),
            "load_run": r["load_run"],
            "tenant": r["tenant"],
            "reservation_id": r["reservation_id"],
            "shape": r["shape"],
            "slots": slots,
            "replicas": r["replicas"],
            "duration_s": duration,
            "price_per_second": float(r["price_per_second"]),
            "slot_usd_per_hour": float(r["price_per_second"]) * 3600.0,
            "served": int(r["served"]),
            "timeouts": int(r["timeouts"]),
            "reserved_slot_hours": reserved_s / 3600.0,
            "used_fill": used_s / reserved_s if reserved_s else np.nan,
            "idle_slot_hours": max(0.0, reserved_s - used_s) / 3600.0,
            "queued_p50_ms": float(served["queued_ms"].median()) if len(served) else np.nan,
            "queued_p95_ms": float(served["queued_ms"].quantile(0.95)) if len(served) else np.nan,
            "reservation_usd": bill,
            "on_demand_usd": on_demand,
            "tenant_usd": bill + on_demand,
            "shape_usd": shape_all,
            "per_token_usd": per_token,
            "tenant_vs_shape": (bill + on_demand) / shape_all if shape_all else np.nan,
            "tenant_vs_per_token": (bill + on_demand) / per_token if per_token else np.nan,
        })
    return pd.DataFrame(rows)


def _run_window(sub: pd.DataFrame) -> tuple[float, float]:
    ok = sub[sub["ok"]]
    t0 = float(sub["t_arrival"].min())
    t1 = float(max(ok["t_last_token"].max() if len(ok) else t0, sub["t_arrival"].max()))
    return t0, t1


def _num(s: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(s[col], errors="coerce") if col in s else pd.Series(dtype=float)


def per_replica(
    df: pd.DataFrame, loads: list[LoadRun], config: Config, fallback: Config | None = None
) -> pd.DataFrame:
    """Per (load_run, replica): seat-seconds held, fill rate against the slot plan, utilisation,
    revenue vs GPU cost, the break-even fill rate, and the sampled ledger/engine state. A slot plan
    is a reporting choice, so a run whose snapshot has none takes the current config's."""
    shapes = {s.name: s for s in config.shapes}
    plans = {r.id: r.slot_plan for r in (fallback.replicas if fallback else [])}
    plans.update({r.id: r.slot_plan for r in config.replicas if r.slot_plan})
    metas = {lr.id: lr for lr in loads}
    gpu_h = config.pricing.provider.gpu_usd_per_hour
    rows = []
    for load_run, run in df.groupby("load_run", sort=False):
        t0, t1 = _run_window(run)
        duration = max(0.0, t1 - t0)
        ok_all = run[run["ok"]]
        for replica, ok in ok_all.groupby("replica"):
            plan = plans.get(replica, {})
            slots = sum(plan.values())
            plan_price_s = sum(n * shapes[name].price_per_second for name, n in plan.items() if name in shapes)
            plan_tps = sum(n * shapes[name].tps for name, n in plan.items() if name in shapes)
            seat_s = float((ok["t_engine_done"] - ok["t_admitted"]).clip(lower=0).sum())
            row = {
                "scenario": run["scenario"].iloc[0],
                "load_run": load_run,
                "replica": replica,
                "duration_s": round(duration, 1),
                "ok": len(ok),
                "output_tokens": int(ok["output_tokens"].sum()),
                "prompt_tokens": int(ok["prompt_tokens"].sum()),
                "seat_seconds": seat_s,
                "slots": slots,
                "slot_plan": ";".join(f"{n}x{name}" for name, n in plan.items()),
                "fill_rate": seat_s / (slots * duration) if slots and duration else np.nan,
                "utilization": float(ok["output_tokens"].sum()) / (plan_tps * duration) if plan_tps and duration else np.nan,
                "shape_usd": float(ok["shape_usd"].sum()),
                "per_token_usd": float(ok["per_token_usd"].sum()),
                "gpu_cost_usd": gpu_h * duration / 3600.0,
                "shape_usd_per_hour": float(ok["shape_usd"].sum()) / duration * 3600.0 if duration else np.nan,
                "per_token_usd_per_hour": float(ok["per_token_usd"].sum()) / duration * 3600.0 if duration else np.nan,
                "plan_usd_per_hour_full": plan_price_s * 3600.0,
                "breakeven_fill_rate": gpu_h / (plan_price_s * 3600.0) if plan_price_s else np.nan,
                "ledger_running_mean": np.nan,
                "ledger_kv_held_share_mean": np.nan,
                "seats_now_mean": np.nan,
                "engine_running_mean": np.nan,
                "engine_kv_usage_mean": np.nan,
                "engine_kv_usage_p95": np.nan,
                "queue_interactive_mean": np.nan,
                "queue_interactive_max": np.nan,
                "queue_lazy_mean": np.nan,
                "gw_timeouts": np.nan,
                "gw_preemptions": np.nan,
            }
            lr = metas.get(load_run)
            if lr is not None and lr.samples is not None and "replica" in lr.samples:
                s = lr.samples[lr.samples["replica"] == replica]

                def num(col: str, s: pd.DataFrame = s) -> pd.Series:
                    return _num(s, col)

                if len(s):
                    row["engine_running_mean"] = float(num("running").mean())
                    row["engine_kv_usage_mean"] = float(num("kv_usage").mean())
                    row["engine_kv_usage_p95"] = float(num("kv_usage").quantile(0.95))
                    if "ledger_running" in s and num("ledger_running").notna().any():
                        row["ledger_running_mean"] = float(num("ledger_running").mean())
                        held, usable = num("ledger_kv_held"), num("ledger_kv_usable")
                        row["ledger_kv_held_share_mean"] = float((held / usable.replace(0, np.nan)).mean())
                        row["seats_now_mean"] = float(num("seats_now").mean())
                        row["queue_interactive_mean"] = float(num("queue_interactive").mean())
                        row["queue_interactive_max"] = float(num("queue_interactive").max())
                        row["queue_lazy_mean"] = float(num("queue_lazy").mean())
                        row["gw_timeouts"] = float(num("gw_timeouts").max())
                        row["gw_preemptions"] = float(num("gw_preemptions").max())
            rows.append(row)
    return pd.DataFrame(rows)


def _slot_plans(config: Config, fallback: Config | None) -> dict[str, dict[str, int]]:
    plans = {r.id: r.slot_plan for r in (fallback.replicas if fallback else [])}
    plans.update({r.id: r.slot_plan for r in config.replicas if r.slot_plan})
    return plans


def _gpus_used(ok: pd.DataFrame) -> int:
    """Replicas that served the run's requests: the GPU count behind the run's cost and its tokens per
    GPU-hour (1 when the meter has no replica ids, as in the single-replica milestones)."""
    n = int(ok["replica"].dropna().nunique()) if "replica" in ok.columns else 0
    return max(1, n)


def _run_traffic(df: pd.DataFrame, config: Config, fallback: Config | None) -> pd.DataFrame:
    """Per load run: token sums split interactive / lazy, GPU cost, both revenues and the measured
    utilisation against the slot plans of the replicas the run used."""
    shapes = {s.name: s for s in config.shapes}
    plans = _slot_plans(config, fallback)
    gpu_h = config.pricing.provider.gpu_usd_per_hour
    rows = []
    for load_run, run in df.groupby("load_run", sort=False):
        t0, t1 = _run_window(run)
        duration = max(0.0, t1 - t0)
        ok = run[run["ok"]]
        lazy = ok["lazy"].astype(bool)
        plan_tps = sum(
            n * shapes[name].tps
            for rep in ok["replica"].dropna().unique()
            for name, n in plans.get(rep, {}).items()
            if name in shapes
        )
        output_total = float(ok["output_tokens"].sum())
        rows.append({
            "scenario": run["scenario"].iloc[0],
            "seed": int(run["seed"].iloc[0]),
            "load_run": load_run,
            "duration_s": duration,
            "prompt_tokens_interactive": int(ok.loc[~lazy, "prompt_tokens"].sum()),
            "cached_tokens_interactive": int(ok.loc[~lazy, "cached_tokens"].sum()),
            "output_tokens_interactive": int(ok.loc[~lazy, "output_tokens"].sum()),
            "prompt_tokens_lazy": int(ok.loc[lazy, "prompt_tokens"].sum()),
            "cached_tokens_lazy": int(ok.loc[lazy, "cached_tokens"].sum()),
            "output_tokens_lazy": int(ok.loc[lazy, "output_tokens"].sum()),
            "replicas_used": _gpus_used(ok),
            "gpu_cost_usd": gpu_h * _gpus_used(ok) * duration / 3600.0,
            "shape_usd": float(ok["shape_usd"].sum()),
            "per_token_usd": float(ok["per_token_usd"].sum()),
            "utilization": output_total / (plan_tps * duration) if plan_tps and duration else np.nan,
        })
    return pd.DataFrame(rows)


def _references(config: Config, fallback: Config | None) -> list[PriceReference]:
    """The run's price references, or the current config's for a run snapshotted before they
    existed; the run's own (active) per-token prices are always one of them."""
    tok = config.pricing.token
    refs = list(config.pricing.references) or list(fallback.pricing.references if fallback else [])
    if not any(r.name == tok.reference for r in refs):
        refs.insert(0, PriceReference(name=tok.reference, model="the run's own token prices",
                                      input_usd_per_million=tok.input_usd_per_million,
                                      output_usd_per_million=tok.output_usd_per_million))
    return refs


def market(df: pd.DataFrame, config: Config, fallback: Config | None = None) -> pd.DataFrame:
    """Per (load run, price reference): what the run's traffic earns at that reference's per-token
    prices (lazy requests at the batch discount) against GPU cost and the run's own shape revenue,
    the gross margin, and the GPU price per hour at which that reference breaks even, at the
    measured utilisation and as if the slot plan ran at the target utilisation (cost per token
    scales with measured ÷ target). Prompt tokens served from the prefix cache are priced at the
    reference's cache-read price (its listed one, else cache_read_ratio × input); `revenue_usd_uncached`
    is what charging them in full would earn (the pre-phase-2 upper bound)."""
    traffic = _run_traffic(df, config, fallback)
    pr = config.pricing
    d = pr.token.batch_discount
    ratio = pr.token.cache_read_ratio
    target = pr.provider.target_utilization
    rows = []
    for t in traffic.itertuples(index=False):
        hours = t.duration_s / 3600.0
        scale = (target / t.utilization) if t.utilization and not np.isnan(t.utilization) else np.nan
        prompt_all = t.prompt_tokens_interactive + d * t.prompt_tokens_lazy
        cached_all = t.cached_tokens_interactive + d * t.cached_tokens_lazy
        for ref in _references(config, fallback):
            cache_price = ref.cache_read_price(ratio)
            rev_uncached = (
                t.prompt_tokens_interactive * ref.input_usd_per_million + t.output_tokens_interactive * ref.output_usd_per_million
                + d * (t.prompt_tokens_lazy * ref.input_usd_per_million + t.output_tokens_lazy * ref.output_usd_per_million)
            ) / 1e6
            rev = rev_uncached - cached_all * (ref.input_usd_per_million - cache_price) / 1e6
            rows.append({
                "scenario": t.scenario,
                "load_run": t.load_run,
                "reference": ref.name,
                "model": ref.model or ("median of " + ", ".join(ref.basket)),
                "active": ref.name == pr.token.reference,
                "input_usd_per_million": ref.input_usd_per_million,
                "output_usd_per_million": ref.output_usd_per_million,
                "cache_read_usd_per_million": cache_price,
                "cached_share": cached_all / prompt_all if prompt_all else np.nan,
                "revenue_usd_uncached": rev_uncached,
                "revenue_usd": rev,
                "revenue_usd_per_hour": rev / hours if hours else np.nan,
                "gpu_cost_usd": t.gpu_cost_usd,
                "shape_usd": t.shape_usd,
                "revenue_vs_gpu": rev / t.gpu_cost_usd if t.gpu_cost_usd else np.nan,
                "revenue_vs_shape": rev / t.shape_usd if t.shape_usd else np.nan,
                "gross_margin": 1.0 - t.gpu_cost_usd / rev if rev else np.nan,
                "utilization": t.utilization,
                "target_utilization": target,
                "gross_margin_at_target": 1.0 - t.gpu_cost_usd / (rev * scale) if rev and not np.isnan(scale) else np.nan,
                "breakeven_gpu_usd_per_hour": rev / hours if hours else np.nan,
                "breakeven_gpu_usd_per_hour_at_target": rev * scale / hours if hours and not np.isnan(scale) else np.nan,
            })
    return pd.DataFrame(rows)


def list_prices(df: pd.DataFrame, config: Config, fallback: Config | None = None) -> pd.DataFrame:
    """Per (load run, margin target): the per-token prices this provider would have to list to earn
    that gross margin on revenue from the run's traffic, at the measured and at the target
    utilisation, against the active market reference (the current config's when the run's snapshot
    predates price references). Prompt tokens count as output-equivalent tokens at that reference's
    input:output price ratio (the input price follows at that ratio) and lazy tokens at the batch
    discount, so one output price covers the whole bill."""
    traffic = _run_traffic(df, config, fallback)
    pr = config.pricing
    tok = pr.token if pr.references or not (fallback and fallback.pricing.references) else fallback.pricing.token
    ratio = tok.input_usd_per_million / tok.output_usd_per_million if tok.output_usd_per_million else 0.0
    cache_ratio = pr.cache_read_ratio  # a cached prompt token weighs cache_read_ratio x an uncached one
    d = tok.batch_discount
    target = pr.provider.target_utilization
    rows = []
    for t in traffic.itertuples(index=False):
        prompt_i = t.prompt_tokens_interactive - t.cached_tokens_interactive * (1.0 - cache_ratio)
        prompt_l = t.prompt_tokens_lazy - t.cached_tokens_lazy * (1.0 - cache_ratio)
        weighted = t.output_tokens_interactive + ratio * prompt_i + d * (t.output_tokens_lazy + ratio * prompt_l)
        scale = (t.utilization / target) if t.utilization and not np.isnan(t.utilization) else np.nan
        for m in pr.provider.margin_targets:
            out = t.gpu_cost_usd / (1.0 - m) / weighted * 1e6 if weighted else np.nan
            rows.append({
                "scenario": t.scenario,
                "load_run": t.load_run,
                "margin_target": m,
                "utilization": t.utilization,
                "target_utilization": target,
                "output_usd_per_million": out,
                "input_usd_per_million": out * ratio,
                "output_usd_per_million_at_target": out * scale,
                "input_usd_per_million_at_target": out * scale * ratio,
                "reference": tok.reference,
                "reference_output_usd_per_million": tok.output_usd_per_million,
                "vs_reference": out / tok.output_usd_per_million if tok.output_usd_per_million else np.nan,
                "vs_reference_at_target": out * scale / tok.output_usd_per_million if tok.output_usd_per_million else np.nan,
            })
    return pd.DataFrame(rows)


RESERVATION_SIZINGS = (("p50", 50.0), ("p95", 95.0), ("max", 100.0))


def _seat_weights(config: Config, fallback: Config | None, config_dir: Path | None) -> tuple[dict[str, float], str]:
    """Capacity weight of each shape in units of the slot plan's first shape: seats(plan tps) ÷
    seats(shape tps) on the planned replica's calibration curve (shape nesting, DESIGN 5.2), or the
    tps ratio when no curve can be loaded. Returns the weights and which basis was used."""
    shapes = {s.name: s for s in config.shapes}
    plans = _slot_plans(config, fallback)
    unit_tps = next((shapes[n].tps for p in plans.values() for n in p if n in shapes), None)
    if unit_tps is None:
        return {}, "none"
    profile = None
    margin = config.gateway.admission.tps_margin
    for cfg in [*(fallback.replicas if fallback else []), *config.replicas]:
        if plans.get(cfg.id):
            try:
                profile = profile_for(cfg, config_dir)
                break
            except (OSError, RuntimeError, ValueError, KeyError):
                continue
    if profile is None:
        return {n: s.tps / unit_tps for n, s in shapes.items()}, "tps ratio"
    unit_seats = profile.n_max(unit_tps, margin)
    weights = {}
    for n, s in shapes.items():
        seats = profile.n_max(s.tps, margin)
        weights[n] = unit_seats / seats if seats else np.nan
    return weights, profile.source


def reservations(df: pd.DataFrame, config: Config, fallback: Config | None = None, config_dir: Path | None = None) -> pd.DataFrame:
    """What-if (DESIGN 8, reservation tier): each tenant reserves N slots of its shape for the whole
    run and pays slot-seconds instead of per request. Demand is the tenant's queue-free occupancy:
    each request holds a slot from its arrival for the seat time it measured (admission to engine
    done); capacity refusals count at the tenant's median seat time, context refusals not at all.
    N is the p50, p95 and max of that occupancy at one-second resolution. Per sizing: the share of
    requests that would still wait (all N slots busy at arrival), the reservation's used fill and
    idle slot-hours (resellable as burst or lazy), the reservation bill at the shape's price per
    second (slot time covers prefill, so no prefill charge), the run's per-request shape and
    per-token bills, the GPU cost of the reserved slice of the card (N × capacity weight ÷ plan
    slots), and the slot prices per hour at break-even and at per-token revenue parity."""
    shapes = {s.name: s for s in config.shapes}
    plans = _slot_plans(config, fallback)
    weights, weight_basis = _seat_weights(config, fallback, config_dir)
    gpu_h = config.pricing.provider.gpu_usd_per_hour
    rows = []
    for load_run, run in df.groupby("load_run", sort=False):
        t0, t1 = _run_window(run)
        duration = max(1.0, t1 - t0)
        n_sec = int(np.ceil(duration)) + 1
        used_replicas = run.loc[run["ok"], "replica"].dropna().unique()
        plan_slots = sum(sum(plans.get(rep, {}).values()) for rep in used_replicas)
        gpu_run_h = gpu_h * max(1, len(used_replicas))  # the plan spans every replica the run used, so does the cost
        for (tenant, shape_name), sub in run.groupby(["tenant", "shape"], sort=False):
            shape = shapes.get(shape_name)
            if shape is None:
                continue
            ok = sub[sub["ok"]]
            cap = sub[sub["abort_reason"].fillna("").astype(str).str.startswith("capacity")]
            seat_s = (ok["t_engine_done"] - ok["t_admitted"]).clip(lower=0).to_numpy(dtype=float)
            median_seat = float(np.median(seat_s)) if len(seat_s) else 0.0
            starts = np.concatenate([ok["t_arrival"].to_numpy(dtype=float), cap["t_arrival"].to_numpy(dtype=float)])
            durs = np.concatenate([seat_s, np.full(len(cap), median_seat)])
            if len(starts) == 0:
                continue
            delta = np.zeros(n_sec + 1)
            i0 = np.clip((starts - t0).astype(int), 0, n_sec - 1)
            i1 = np.clip(np.ceil(starts + durs - t0).astype(int), 0, n_sec)
            np.add.at(delta, i0, 1)
            np.add.at(delta, i1, -1)
            occ = np.cumsum(delta)[:n_sec]
            weight = weights.get(shape_name, np.nan)
            for label, pct in RESERVATION_SIZINGS:
                n_slots = max(1, int(np.ceil(np.percentile(occ, pct))))
                would_queue = float(np.mean(occ[i0] - 1 >= n_slots))
                reserved_s = n_slots * duration
                used_s = float(np.minimum(occ, n_slots).sum())
                bill = reserved_s * shape.price_per_second
                cost_share = gpu_run_h * duration / 3600.0 * n_slots * weight / plan_slots if plan_slots and not np.isnan(weight) else np.nan
                per_token = float(ok["per_token_usd"].sum())
                rows.append({
                    "scenario": run["scenario"].iloc[0],
                    "seed": int(run["seed"].iloc[0]),
                    "load_run": load_run,
                    "tenant": tenant,
                    "shape": shape_name,
                    "sizing": label,
                    "slots": n_slots,
                    "occupancy_p50": float(np.percentile(occ, 50)),
                    "occupancy_p95": float(np.percentile(occ, 95)),
                    "occupancy_max": float(occ.max()),
                    "requests": len(starts),
                    "would_queue_share": would_queue,
                    "reserved_slot_hours": reserved_s / 3600.0,
                    "used_fill": used_s / reserved_s if reserved_s else np.nan,
                    "idle_slot_hours": (reserved_s - used_s) / 3600.0,
                    "reservation_usd": bill,
                    "shape_usd": float(ok["shape_usd"].sum()),
                    "per_token_usd": per_token,
                    "reservation_vs_shape": bill / float(ok["shape_usd"].sum()) if float(ok["shape_usd"].sum()) else np.nan,
                    "reservation_vs_per_token": bill / per_token if per_token else np.nan,
                    "capacity_weight": weight,
                    "weight_basis": weight_basis,
                    "cost_share_usd": cost_share,
                    "reservation_vs_cost": bill / cost_share if cost_share else np.nan,
                    "slot_usd_per_hour": shape.price_per_second * 3600.0,
                    "breakeven_slot_usd_per_hour": gpu_run_h * weight / plan_slots if plan_slots and not np.isnan(weight) else np.nan,
                    "revenue_parity_slot_usd_per_hour": per_token / (reserved_s / 3600.0) if reserved_s else np.nan,
                })
    return pd.DataFrame(rows)


SAMPLE_COLUMNS = [
    "scenario", "load_run", "tenant", "showcase", "prompt_id", "turn", "shape", "billed_shape", "class_ctx", "delivery",
    "status", "abort_reason", "t_rel", "prompt_tokens", "cached_tokens", "output_tokens", "delivered_tokens",
    "queued_ms", "ttft_ms", "stream_s", "throttled_s", "delivered_tps", "tps", "burst_share", "floor_violation",
    "shape_usd", "per_token_usd", "prompt_json", "output_text",
]


def samples(df: pd.DataFrame, texts: pd.DataFrame | None) -> pd.DataFrame:
    """The showcase requests (docs/PHASE2.md section 6): real prompts and the completions the gateway
    kept, with the constraints each ran under. One row per request, in arrival order; a multi-turn
    prompt is several rows (`turn`). Requests without stored text keep their metrics and empty text."""
    show = df[df["showcase"].notna()].copy()
    if show.empty:
        return pd.DataFrame(columns=SAMPLE_COLUMNS)
    if texts is not None and len(texts):
        show = show.merge(texts[["request_id", "prompt_json", "output_text"]], on="request_id", how="left")
    else:
        show["prompt_json"] = None
        show["output_text"] = None
    parts = show["showcase"].astype(str).str.split("#", n=1, expand=True)
    show["prompt_id"] = parts[0]
    show["turn"] = pd.to_numeric(parts[1], errors="coerce").fillna(0).astype(int) if parts.shape[1] > 1 else 0
    with np.errstate(divide="ignore", invalid="ignore"):
        show["burst_share"] = np.where(show["output_tokens"] > 0, show["output_tokens_burst"] / show["output_tokens"], np.nan)
    show = show.sort_values(["load_run", "t_arrival"])
    return show[SAMPLE_COLUMNS].reset_index(drop=True)


def refusals(df: pd.DataFrame) -> pd.DataFrame:
    """Refusals by reason per load run and tenant (429 context, 503 capacity by ledger rule)."""
    ref = df[df["refused"]].copy()
    if ref.empty:
        return pd.DataFrame(columns=["scenario", "load_run", "tenant", "reason", "n", "queued_p50_ms"])
    ref["reason"] = ref["abort_reason"].fillna("unknown")
    out = ref.groupby(["scenario", "load_run", "tenant", "reason"], dropna=False).agg(
        n=("request_id", "count"), queued_p50_ms=("queued_ms", "median")
    ).reset_index()
    return out


def across_seeds(pt: pd.DataFrame) -> pd.DataFrame:
    """Per (scenario, tenant): mean and coefficient of variation of each bill across seeds."""
    t = pt.groupby(["scenario", "seed", "load_run", "tenant"], dropna=False).agg(
        per_token_usd=("per_token_usd", "sum"),
        prompt_usd=("prompt_usd", "sum"),
        shape_usd=("shape_usd", "sum"),
        bound_usd=("bound_usd", "sum"),
        output_tokens=("output_tokens", "sum"),
        requests=("requests", "sum"),
        refused=("refused", "sum"),
    ).reset_index()
    g = t.groupby(["scenario", "tenant"], dropna=False)
    out = g.agg(
        seeds=("seed", "nunique"),
        per_token_usd_mean=("per_token_usd", "mean"),
        per_token_usd_std=("per_token_usd", "std"),
        prompt_usd_mean=("prompt_usd", "mean"),
        shape_usd_mean=("shape_usd", "mean"),
        shape_usd_std=("shape_usd", "std"),
        bound_usd_mean=("bound_usd", "mean"),
        output_tokens_mean=("output_tokens", "mean"),
        output_tokens_std=("output_tokens", "std"),
        requests_mean=("requests", "mean"),
        refused_mean=("refused", "mean"),
    ).reset_index()
    with np.errstate(divide="ignore", invalid="ignore"):
        out["per_token_usd_cv"] = out["per_token_usd_std"] / out["per_token_usd_mean"]
        out["shape_usd_cv"] = out["shape_usd_std"] / out["shape_usd_mean"]
        out["output_tokens_cv"] = out["output_tokens_std"] / out["output_tokens_mean"]
        out["shape_vs_token_ratio"] = out["shape_usd_mean"] / out["per_token_usd_mean"]
    return out


def interference(df: pd.DataFrame) -> pd.DataFrame:
    """Decode rate of streamed requests while a long prefill is in flight vs while none is."""
    rows = []
    for load_run, sub in df.groupby("load_run", sort=False):
        ok = sub[sub["ok"]]
        prefills = ok[ok["prompt_tokens"] >= LONG_PREFILL_TOKENS][["t_admitted", "t_first_token"]].dropna().to_numpy()
        streams = ok[(ok["delivery"] == "stream") & ok["engine_tps"].notna()]
        if streams.empty:
            continue
        s0 = streams["t_first_token"].to_numpy()
        s1 = streams["t_engine_done"].to_numpy()
        if len(prefills):
            overlap = ((s0[:, None] < prefills[:, 1][None, :]) & (s1[:, None] > prefills[:, 0][None, :])).any(axis=1)
        else:
            overlap = np.zeros(len(streams), dtype=bool)
        for tenant, grp in streams.assign(_ov=overlap).groupby("tenant"):
            for flag in (False, True):
                sel = grp[grp["_ov"] == flag]
                if sel.empty:
                    continue
                rows.append(
                    {
                        "scenario": sub["scenario"].iloc[0],
                        "load_run": load_run,
                        "tenant": tenant,
                        "during_long_prefill": flag,
                        "n": len(sel),
                        "engine_tps_mean": float(sel["engine_tps"].mean()),
                        "engine_tps_p10": float(sel["engine_tps"].quantile(0.1)),
                        "delivered_tps_mean": float(sel["delivered_tps"].mean()) if sel["delivered_tps"].notna().any() else np.nan,
                        "floor_violation_share": float(sel["floor_violation"].mean()),
                    }
                )
    return pd.DataFrame(rows)


def timeline(df: pd.DataFrame, loads: list[LoadRun], step_s: float = 5.0) -> dict[str, dict]:
    """Per load run: in-flight requests (meter) and, when sampled, the engine's running count and
    KV usage, all against seconds since the run's first arrival, downsampled to `step_s`."""
    metas = {lr.id: lr for lr in loads}
    out: dict[str, dict] = {}
    for load_run, sub in df.groupby("load_run", sort=False):
        ok = sub[sub["ok"]]
        if ok.empty:
            continue
        t0 = float(sub["t_arrival"].min())
        t1 = float(ok["t_last_token"].max())
        conc = concurrency_series(ok, t0, t1)
        k = max(1, int(step_s))
        n = len(conc) // k * k
        binned = conc[:n].reshape(-1, k).mean(axis=1) if n else conc
        xs = (np.arange(len(binned)) * k).tolist()
        entry: dict = {"t": xs, "inflight": binned.round(2).tolist()}
        lr = metas.get(load_run)
        if lr and lr.samples is not None and "t" in lr.samples:
            s = lr.samples.copy()
            s["t_rel"] = pd.to_numeric(s["t"], errors="coerce") - t0
            s = s[(s["t_rel"] >= 0) & (s["t_rel"] <= t1 - t0)]
            entry["sample_t"] = s["t_rel"].round(1).tolist()
            entry["running"] = pd.to_numeric(s["running"], errors="coerce").fillna(0).tolist()
            entry["kv_usage"] = (pd.to_numeric(s["kv_usage"], errors="coerce").fillna(0) * 100).round(2).tolist()
        out[load_run] = entry
    return out
