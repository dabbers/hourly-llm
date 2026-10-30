"""Calibrate a replica (docs/DESIGN.md section 5.2):

* KV budget from /metrics (num_gpu_blocks x block_size).
* Prefill speed (TTFT) at each context length.
* Per-stream decode step rate at N concurrent streams, measured two ways:
    - pure:  inside the window where *all* N streams are decoding (no prefill in the batch) — this
             is the number the seat rule uses;
    - mixed: over each stream's whole decode phase, which includes neighbours' prefill chunks —
             this shows prefill interference (assumption A4).
* An explicit interference probe: one decoding stream, then a long prompt arrives; the victim's
  tokens/s before, during and after the aggressor's prefill.

Writes calibration/<replica>.yaml and prints which shapes in config/shapes.yaml the replica can
honour (n_max per shape).

Usage (host, against a running vLLM):
    uv run python scripts/calibrate.py --url http://localhost:8000 --replica-id local-2080ti
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import shutil
import statistics
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from slotgw.config import load_config
from slotgw.prom import first_labels, first_value, parse_prometheus

FILLER = (
    "The quarterly planning meeting covered capacity forecasts, the token bucket rollout, "
    "replica packing experiments, and the open question of how lazily queued batch work should be "
    "priced relative to interactive traffic. "
)


def nvidia_smi() -> dict | None:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        out = subprocess.run(
            [exe, "--query-gpu=name,memory.total,memory.used,memory.free,driver_version",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip().splitlines()[0]
    except Exception:  # noqa: BLE001
        return None
    name, total, used, free, driver = [p.strip() for p in out.split(",")]
    return {"name": name, "memory_total_mib": int(total), "memory_used_mib": int(used),
            "memory_free_mib": int(free), "driver": driver}


def gpu_sample() -> tuple[float, int] | None:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        out = subprocess.run(
            [exe, "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, check=True,
        ).stdout.strip().splitlines()[0]
        util, mem = [p.strip() for p in out.split(",")]
        return float(util), int(mem)
    except Exception:  # noqa: BLE001
        return None


async def tokenize(client: httpx.AsyncClient, url: str, model: str, messages: list[dict]) -> int:
    r = await client.post(f"{url}/tokenize", json={"model": model, "messages": messages,
                                                   "add_generation_prompt": True})
    r.raise_for_status()
    return int(r.json()["count"])


async def build_prompt(client: httpx.AsyncClient, url: str, model: str, target_tokens: int, nonce: str) -> list[dict]:
    """Grow filler text until the chat prompt is within ~2% of target_tokens. The nonce goes first so
    prefix caching cannot share prefill between streams or runs."""
    reps = max(1, target_tokens // 40)
    messages: list[dict] = []
    for _ in range(8):
        text = f"[{nonce}] " + FILLER * reps
        messages = [{"role": "user", "content": text + "\nContinue the notes."}]
        n = await tokenize(client, url, model, messages)
        if abs(n - target_tokens) <= max(16, target_tokens * 0.02):
            return messages
        reps = max(1, int(reps * target_tokens / max(n, 1)))
    return messages


async def one_stream(client: httpx.AsyncClient, url: str, model: str, messages: list[dict], max_tokens: int) -> dict:
    body = {"model": model, "messages": messages, "max_tokens": max_tokens, "ignore_eos": True,
            "temperature": 0.0, "stream": True,
            "stream_options": {"include_usage": True, "continuous_usage_stats": True}}
    t0 = time.perf_counter()
    token_times: list[float] = []
    completion = 0
    async with client.stream("POST", f"{url}/v1/chat/completions", json=body) as r:
        if r.status_code != 200:
            raise RuntimeError(f"{r.status_code}: {(await r.aread())[:300]!r}")
        async for line in r.aiter_lines():
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            obj = json.loads(data)
            usage = obj.get("usage") or {}
            choices = obj.get("choices") or []
            content = (choices[0].get("delta") or {}).get("content") if choices else None
            if usage.get("completion_tokens") is not None:
                completion = int(usage["completion_tokens"])
            if content:
                token_times.append(time.perf_counter())
    t_end = time.perf_counter()
    first = token_times[0] if token_times else None
    last = token_times[-1] if token_times else None
    mixed = ((len(token_times) - 1) / (last - first)) if (first and last and last > first and len(token_times) > 1) else None
    return {
        "t0": t0,
        "ttft_s": (first - t0) if first else None,
        "total_s": t_end - t0,
        "completion_tokens": completion,
        "token_times": token_times,
        "decode_tps_mixed": mixed,
    }


def rate_in_window(token_times: list[float], w0: float, w1: float) -> float | None:
    if w1 - w0 <= 0:
        return None
    return sum(1 for t in token_times if w0 <= t <= w1) / (w1 - w0)


def pure_window(streams: list[dict]) -> tuple[float, float] | None:
    """Window in which every stream has produced its first token and none has finished."""
    starts = [s["token_times"][0] for s in streams if s["token_times"]]
    ends = [s["token_times"][-1] for s in streams if s["token_times"]]
    if len(starts) != len(streams):
        return None
    w0, w1 = max(starts) + 0.25, min(ends)
    return (w0, w1) if w1 - w0 >= 1.0 else None


async def run_level(client, url, model, prompts: list[list[dict]], n: int, max_tokens: int) -> dict:
    stop = asyncio.Event()
    samples: list[tuple[float, int]] = []
    kv_peak = 0.0

    async def sampler():
        nonlocal kv_peak
        while not stop.is_set():
            s = gpu_sample()
            if s:
                samples.append(s)
            try:
                m = await client.get(f"{url}/metrics")
                kv = first_value(parse_prometheus(m.text), "vllm:kv_cache_usage_perc")
                if kv is not None:
                    kv_peak = max(kv_peak, kv)
            except Exception:  # noqa: BLE001, S110 - sampler must never kill the run
                pass
            await asyncio.sleep(0.5)

    task = asyncio.create_task(sampler())
    t0 = time.perf_counter()
    results = await asyncio.gather(*(one_stream(client, url, model, prompts[i % len(prompts)], max_tokens)
                                     for i in range(n)))
    wall = time.perf_counter() - t0
    stop.set()
    await task
    mixed = [r["decode_tps_mixed"] for r in results if r["decode_tps_mixed"]]
    ttft = [r["ttft_s"] for r in results if r["ttft_s"] is not None]
    total_tokens = sum(r["completion_tokens"] for r in results)
    win = pure_window(results)
    pure = []
    if win:
        pure = [v for v in (rate_in_window(r["token_times"], *win) for r in results) if v is not None]
    return {
        "n": n,
        "step_rate_pure_mean": round(statistics.fmean(pure), 2) if pure else None,
        "step_rate_pure_min": round(min(pure), 2) if pure else None,
        "pure_window_s": round(win[1] - win[0], 2) if win else 0.0,
        "step_rate_mixed_mean": round(statistics.fmean(mixed), 2) if mixed else None,
        "step_rate_mixed_min": round(min(mixed), 2) if mixed else None,
        "aggregate_tps": round(total_tokens / wall, 2),
        "ttft_p50_s": round(statistics.median(ttft), 3) if ttft else None,
        "ttft_max_s": round(max(ttft), 3) if ttft else None,
        "wall_s": round(wall, 2),
        "gpu_util_mean": round(statistics.fmean(u for u, _ in samples), 1) if samples else None,
        "gpu_mem_peak_mib": max(m for _, m in samples) if samples else None,
        "kv_usage_peak": round(kv_peak, 4),
    }


async def interference_probe(client, url, model, victim_prompt: list[dict], aggressor_prompt: list[dict],
                             victim_tokens: int) -> dict:
    """One stream decodes; a long prompt arrives mid-way. How much does the victim slow down?"""
    victim_task = asyncio.create_task(one_stream(client, url, model, victim_prompt, victim_tokens))
    await asyncio.sleep(3.0)
    t_a0 = time.perf_counter()
    agg = await one_stream(client, url, model, aggressor_prompt, 1)
    t_a1 = t_a0 + (agg["ttft_s"] or 0.0)
    victim = await victim_task
    tt = victim["token_times"]
    if not tt:
        return {"error": "victim produced no tokens"}
    before = rate_in_window(tt, tt[0] + 0.25, t_a0)
    during = rate_in_window(tt, t_a0, t_a1)
    after = rate_in_window(tt, t_a1 + 0.25, tt[-1])
    return {
        "aggressor_prefill_s": round(agg["ttft_s"], 3) if agg["ttft_s"] else None,
        "victim_tps_before": round(before, 2) if before else None,
        "victim_tps_during_prefill": round(during, 2) if during is not None else None,
        "victim_tps_after": round(after, 2) if after else None,
        "victim_tokens": victim["completion_tokens"],
    }


def n_max(curve: dict[int, float | None], tps: float, margin: float) -> int:
    best = 0
    for n, rate in sorted(curve.items()):
        if rate is not None and rate * (1 - margin) >= tps:
            best = n
    return best


async def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://localhost:8000")
    p.add_argument("--model", default=None, help="served model name; default: first of /v1/models")
    p.add_argument("--replica-id", default="local-2080ti")
    p.add_argument("--concurrency", default="1,2,4,8,12,16,24")
    p.add_argument("--contexts", default="2048,6144", help="prompt token lengths to calibrate at")
    p.add_argument("--output-tokens", type=int, default=256)
    p.add_argument("--tps-margin", type=float, default=0.10)
    p.add_argument("--config-dir", default="config")
    p.add_argument("--out", default=None)
    p.add_argument("--skip-interference", action="store_true")
    args = p.parse_args()

    levels = [int(x) for x in args.concurrency.split(",")]
    contexts = [int(x) for x in args.contexts.split(",")]
    out_path = Path(args.out or f"calibration/{args.replica_id}.yaml")

    async with httpx.AsyncClient(timeout=httpx.Timeout(1800.0, connect=10.0)) as client:
        models = (await client.get(f"{args.url}/v1/models")).json()["data"]
        model = args.model or models[0]["id"]
        max_model_len = models[0].get("max_model_len")
        try:
            version = (await client.get(f"{args.url}/version")).json().get("version")
        except Exception:  # noqa: BLE001
            version = None
        metrics = parse_prometheus((await client.get(f"{args.url}/metrics")).text)
        cache = first_labels(metrics, "vllm:cache_config_info") or {}
        num_blocks = int(cache["num_gpu_blocks"]) if cache.get("num_gpu_blocks", "None") != "None" else None
        block_size = int(cache["block_size"]) if cache.get("block_size") else None
        gpu = nvidia_smi()
        print(f"model={model} vllm={version} max_model_len={max_model_len} "
              f"num_gpu_blocks={num_blocks} block_size={block_size} gpu={gpu}")

        nonce = f"{random.randrange(1 << 30):x}"
        result: dict = {
            "replica": args.replica_id,
            "measured_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "url": args.url,
            "model": model,
            "vllm_version": version,
            "max_model_len": max_model_len,
            "gpu": gpu,
            "cache": {"num_gpu_blocks": num_blocks, "block_size": block_size,
                      "kv_tokens_total": (num_blocks * block_size) if (num_blocks and block_size) else None,
                      "raw": {k: v for k, v in cache.items() if k in (
                          "block_size", "num_gpu_blocks", "cache_dtype", "enable_prefix_caching",
                          "gpu_memory_utilization", "kv_cache_max_concurrency")}},
            "output_tokens_per_stream": args.output_tokens,
            "tps_margin": args.tps_margin,
            "method": "pure = per-stream tokens/s inside the window where all N streams are decoding; "
                      "mixed = over each stream's whole decode phase (includes neighbours' prefill)",
            "levels": {},
        }
        warm = await build_prompt(client, args.url, model, 256, nonce + "-warm")
        await one_stream(client, args.url, model, warm, 16)

        for ctx in contexts:
            print(f"\n== context {ctx} ==")
            prompts = [await build_prompt(client, args.url, model, ctx, f"{nonce}-{ctx}-{i}")
                       for i in range(max(levels))]
            per_level = {}
            for n in levels:
                r = await run_level(client, args.url, model, prompts, n, args.output_tokens)
                per_level[n] = r
                print(f"n={n:3d} pure={r['step_rate_pure_mean']} (min {r['step_rate_pure_min']}, "
                      f"window {r['pure_window_s']}s) mixed={r['step_rate_mixed_mean']} "
                      f"(min {r['step_rate_mixed_min']}) aggregate={r['aggregate_tps']} tps "
                      f"ttft_p50={r['ttft_p50_s']}s gpu_util={r['gpu_util_mean']} kv_peak={r['kv_usage_peak']}")
                if r["step_rate_mixed_mean"] is None:
                    break
            result["levels"][f"context_{ctx}"] = per_level

        if not args.skip_interference:
            print("\n== prefill interference probe ==")
            victim = await build_prompt(client, args.url, model, min(contexts), nonce + "-victim")
            aggressor = await build_prompt(client, args.url, model, max(contexts), nonce + "-aggressor")
            probe = await interference_probe(client, args.url, model, victim, aggressor, 1200)
            probe.update({"victim_context": min(contexts), "aggressor_context": max(contexts)})
            result["interference"] = probe
            print(probe)

        config = load_config(args.config_dir)
        feasible = {}
        for s in config.shapes:
            per_ctx = {}
            for ctx in contexts:
                lv = result["levels"][f"context_{ctx}"]
                curve = {n: (v["step_rate_pure_mean"] if v["step_rate_pure_mean"] is not None
                             else v["step_rate_mixed_mean"]) for n, v in lv.items()}
                per_ctx[f"context_{ctx}"] = n_max(curve, s.tps, args.tps_margin)
            kv_total = result["cache"]["kv_tokens_total"]
            by_kv = (kv_total // s.max_context) if kv_total else None
            feasible[s.name] = {
                "tps": s.tps, "max_context": s.max_context,
                "n_max_by_step_rate": per_ctx,
                "n_max_by_kv": by_kv,
                "sellable": all(v >= 1 for v in per_ctx.values()),
            }
        result["shapes"] = feasible

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(result, sort_keys=False), encoding="utf-8")
    print(f"\nwrote {out_path}")
    print("\nshape                    tps   ctx    n_max(pure step rate per ctx)   n_max(kv)  sellable")
    for name, f in feasible.items():
        print(f"{name:24s} {f['tps']:5.0f} {f['max_context']:6d}  {f['n_max_by_step_rate']}  "
              f"{f['n_max_by_kv']}  {f['sellable']}")


if __name__ == "__main__":
    asyncio.run(main())
