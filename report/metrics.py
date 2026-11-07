"""Metrics over meter rows (docs/DESIGN.md section 10). Everything derives from
runs/<run_id>/meter.sqlite plus the load generator's side files under runs/<run_id>/loads/*/."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from slotgw.billing import per_token_usd, shape_seconds
from slotgw.config import Config, load_config

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


@dataclass
class RunData:
    run_dir: Path
    run_id: str
    requests: pd.DataFrame
    loads: list[LoadRun]
    config: Config


def read_meter(path: Path) -> pd.DataFrame:
    conn = sqlite3.connect(str(path))
    try:
        return pd.read_sql_query("SELECT * FROM requests ORDER BY t_arrival", conn)
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
    if cfg_dir is None:
        cfg_dir = Path(config_dir) if config_dir else Path("config")
    config = load_config(cfg_dir)
    run_id = str(df["run_id"].iloc[0]) if not df.empty else run_dir.name
    return RunData(run_dir=run_dir, run_id=run_id, requests=df, loads=loads, config=config)


def enrich(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Per-request derived columns: both bills, delivery rates, floor violations."""
    df = df.copy()
    shapes = {s.name: s for s in config.shapes}
    pricing = config.pricing.token
    df["load_run"] = df["load_run"].fillna("adhoc")
    df["scenario"] = df["scenario"].fillna("adhoc")
    df["seed"] = df["seed"].fillna(0).astype(int)
    df["lazy"] = df["lazy"].astype(bool)
    df["tps"] = df["shape"].map(lambda n: shapes[n].tps if n in shapes else np.nan)
    df["price_per_second"] = df["shape"].map(lambda n: shapes[n].price_per_second if n in shapes else np.nan)
    df["max_context"] = df["shape"].map(lambda n: shapes[n].max_context if n in shapes else np.nan)
    df["delivered_tokens"] = (df["output_tokens"] - df["wasted_tokens"]).clip(lower=0).astype(int)
    ok = df["status"] == "ok"
    df["ok"] = ok
    df["refused"] = df["status"] == "refused"
    df["error"] = ~ok & ~df["refused"]

    secs = [
        shape_seconds(r.t_admitted, r.t_first_token, r.t_last_token, int(r.delivered_tokens), float(r.tps))
        if r.ok and not np.isnan(r.tps)
        else (0.0, 0.0, 0.0)
        for r in df.itertuples(index=False)
    ]
    df["wall_s"], df["bound_s"], df["billed_s"] = (np.array([s[i] for s in secs]) for i in range(3))
    df["shape_usd"] = df["billed_s"] * df["price_per_second"].fillna(0)
    df["bound_usd"] = df["bound_s"] * df["price_per_second"].fillna(0)
    df["wall_usd"] = df["wall_s"] * df["price_per_second"].fillna(0)
    df["burst_discount_usd"] = df["bound_usd"] - df["shape_usd"]
    df["per_token_usd"] = [
        per_token_usd(int(r.prompt_tokens), int(r.output_tokens), bool(r.lazy), pricing) if r.ok else 0.0
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
    all_rows = df.groupby(GROUP, dropna=False).agg(
        requests=("request_id", "count"),
        refused=("refused", "sum"),
        errors=("error", "sum"),
        lazy=("lazy", "max"),
        tps=("tps", "first"),
        price_per_second=("price_per_second", "first"),
    )
    okdf = df[df["ok"]]
    ok_rows = okdf.groupby(GROUP, dropna=False).agg(
        ok=("request_id", "count"),
        prompt_tokens=("prompt_tokens", "sum"),
        output_tokens=("output_tokens", "sum"),
        delivered_tokens=("delivered_tokens", "sum"),
        per_token_usd=("per_token_usd", "sum"),
        shape_usd=("shape_usd", "sum"),
        bound_usd=("bound_usd", "sum"),
        wall_usd=("wall_usd", "sum"),
        billed_s=("billed_s", "sum"),
        bound_s=("bound_s", "sum"),
        wall_s=("wall_s", "sum"),
        stream_s=("stream_s", "sum"),
        throttled_s=("throttled_s", "sum"),
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
    counts = ["ok", "prompt_tokens", "output_tokens", "delivered_tokens", "floor_judged", "floor_violations"]
    sums = ["per_token_usd", "shape_usd", "bound_usd", "wall_usd", "billed_s", "bound_s", "wall_s", "stream_s",
            "throttled_s", "violation_stream_s"]
    out[counts] = out[counts].fillna(0).astype(int)
    out[sums] = out[sums].fillna(0.0)
    out["burst_discount_usd"] = out["bound_usd"] - out["shape_usd"]
    with np.errstate(divide="ignore", invalid="ignore"):
        out["usd_per_1m_out_per_token"] = np.where(out["output_tokens"] > 0, out["per_token_usd"] / out["output_tokens"] * 1e6, np.nan)
        out["usd_per_1m_out_shape"] = np.where(out["delivered_tokens"] > 0, out["shape_usd"] / out["delivered_tokens"] * 1e6, np.nan)
        out["tokens_per_billed_s"] = np.where(out["billed_s"] > 0, out["delivered_tokens"] / out["billed_s"], np.nan)
        out["throttled_share"] = np.where(out["stream_s"] > 0, out["throttled_s"] / out["stream_s"], np.nan)
        out["floor_violation_share"] = np.where(out["stream_s"] > 0, out["violation_stream_s"] / out["stream_s"], np.nan)
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


def per_run(df: pd.DataFrame, loads: list[LoadRun], config: Config) -> pd.DataFrame:
    """One row per load run: totals, both bills, GPU cost, concurrency and sampled replica state."""
    metas = {lr.id: lr for lr in loads}
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
            "prompt_tokens": int(ok["prompt_tokens"].sum()),
            "output_tokens": int(ok["output_tokens"].sum()),
            "delivered_tokens": int(ok["delivered_tokens"].sum()),
            "per_token_usd": float(ok["per_token_usd"].sum()),
            "shape_usd": float(ok["shape_usd"].sum()),
            "bound_usd": float(ok["bound_usd"].sum()),
            "wall_usd": float(ok["wall_usd"].sum()),
            "gpu_cost_usd": config.pricing.provider.gpu_usd_per_hour * duration / 3600.0,
            "output_tokens_per_gpu_hour": (ok["output_tokens"].sum() / duration * 3600.0) if duration > 0 else np.nan,
            "concurrency_mean": float(conc.mean()),
            "concurrency_p95": float(np.percentile(conc, 95)),
            "concurrency_max": float(conc.max()),
            "throttled_s": float(ok["throttled_s"].sum()),
            "floor_violations": int(ok["floor_violation"].sum()),
            "kv_usage_mean": np.nan,
            "kv_usage_p95": np.nan,
            "engine_running_mean": np.nan,
            "preemptions": np.nan,
        }
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


def across_seeds(pt: pd.DataFrame) -> pd.DataFrame:
    """Per (scenario, tenant): mean and coefficient of variation of each bill across seeds."""
    t = pt.groupby(["scenario", "seed", "load_run", "tenant"], dropna=False).agg(
        per_token_usd=("per_token_usd", "sum"),
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
