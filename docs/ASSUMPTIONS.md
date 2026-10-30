# Measured assumptions (phase 1, local RTX 2080 Ti)

Every number the capacity model relies on, with where it came from. Design rationale lives in
[DESIGN.md](DESIGN.md) section 5; this file is the evidence. Updated as calibration runs land.

## Replica `local-2080ti` (vLLM v0.32.0, Qwen/Qwen2.5-3B-Instruct-AWQ, fp16)

Startup facts, read from the vLLM log and `/metrics` on 2026-10-30:

| Fact | Value | Source |
|------|-------|--------|
| Free GPU memory at startup | 9.87 of 11.0 GiB (desktop apps hold the rest) | vLLM `gpu_worker` log |
| `--gpu-memory-utilization` | 0.82 → 9.02 GiB budget | compose default `VLLM_GPU_MEM` |
| Memory for weights + activations + CUDA graphs | 2.07 GiB consumed at load, ~2.8 GiB total non-KV | vLLM log |
| **KV cache memory** | **6.25 GiB** | vLLM log "Available KV cache memory" |
| **KV cache size** | **181,936 tokens** = 11,371 blocks × 16 | `vllm:cache_config_info` (`num_gpu_blocks`, `block_size`), also the gateway's `/admin/replicas` |
| KV bytes per token (implied) | 6.25 GiB ÷ 181,936 ≈ 36.9 KiB (model: 36 layers × 2 KV heads × 128 × 2 B × 2 = 36 KiB, plus alignment) | arithmetic; matches DESIGN §5.1 |
| Max concurrency at 16,384 tokens/request | 11.1× | vLLM log; = 181,936 ÷ 16,384 |
| Max concurrency at 8,192 tokens/request | 22.2× (by KV) | arithmetic |
| `--max-model-len` | 16,384 | compose default; covers the 16k shapes and 2× context bursts of 8k shapes |
| Attention backend | TRITON_ATTN (FlashAttention 2 refused: compute capability < 8) | vLLM log |
| Quantization kernel | MarlinLinearKernel (AWQ via Marlin sm75) | vLLM log |
| torch.compile | works; 22.7 s on first start, cached afterwards | vLLM log |
| CUDA graphs | enabled (default), memory profiled | vLLM log |
| Prefix caching | enabled (default) | `vllm:cache_config_info` `enable_prefix_caching="True"` |
| Cold start to healthy | ~200 s including the 2.69 GB model download; warm restarts faster | startup watcher |
| Single-request sanity | 9-token answer: TTFT 61 ms, e2e 242 ms through the gateway; 16-token answer: TTFT 27 ms, e2e 142 ms | gateway meter rows |

Consequences for the capacity model (DESIGN §5):

- **A1 KV reservation**: a `*.8k.*` request holds 8,192 tokens (512 blocks); a `*.16k.*` request holds
  16,384 (1,024 blocks). With `kv_margin = 0.05`, 172,839 tokens are admissible: **21 × 8k or 10 × 16k
  guaranteed requests** on this replica, before the seat rule.
- The seat rule (step rate at N concurrent streams) is the binding constraint for fast shapes; see the
  calibration section below once measured.

## Calibration (`calibration/local-2080ti.yaml`, 2026-10-30, 256 output tokens per stream)

**Pure** = per-stream decode tokens/s inside the window where all N streams are decoding and no
prefill is in the batch (the number the seat rule uses). **Mixed** = over each stream's whole decode
phase, which includes neighbours' prefill chunks. **Aggregate** = all output tokens ÷ wall time of
the level, prefill included (so it understates steady-state throughput at long contexts).

| N streams | 2k ctx pure | 2k mixed | 2k aggregate | 6k ctx pure | 6k mixed | 6k aggregate |
|-----------|-------------|----------|--------------|-------------|----------|--------------|
| 1  | 137 | 138 | 102 | 105 | 105 | 38  |
| 2  | 120 | 108 | 183 | 80  | 80  | 68  |
| 4  | 90  | 76  | 249 | 50  | 28  | 72  |
| 8  | 71  | 53  | 337 | 35  | 16  | 79  |
| 12 | 54  | 43  | 426 | 25  | 12  | 105 |
| 16 | 45  | 34  | 501 | 20  | 10  | 131 |
| 24 | 33  | 22  | 477 | 13  | 6   | 112 |

Prefill (TTFT at N=1): 2,048-token prompt 0.61 s (~3.4k tok/s); 6,144-token prompt 4.2 s
(~1.45k tok/s). Prefill cost grows super-linearly with length on the Triton attention backend.

**Prefill interference probe (A4):** a stream at 2k context decoding at 130 tok/s; a 6,144-token
prompt arrives; the victim drops to **1.7 tok/s for the 4.6 s** the prefill takes, then recovers to
125 tok/s. With the default 2,048-token prefill chunks every running stream is effectively paused
while a long prompt prefills.

**Seat ceilings** (`n_max`: largest N whose pure rate × 0.9 still meets the shape's tps):

| Shape tps | n_max at 2k ctx | n_max at 6k ctx | n_max by KV (8k / 16k shapes) |
|-----------|-----------------|-----------------|-------------------------------|
| 15 | 24 | 16 | 22 / 11 |
| 30 | 16 | 8  | 22 / 11 |
| 60 | 8  | 2  | 22 / 11 |

Consequences:

- Use the 6k curve for the 8k and 16k shapes (conservative): `small.8k.15tps` → 16 seats,
  `small.8k.30tps` → 8, `small.8k.60tps` → 2, `small.16k.30tps` → 8 (KV would allow 11). The seat
  rule, not KV, binds for the 30 and 60 tps shapes on this card; the catalog as configured is sellable.
- Shape rates are honest in steady-state decode only. Under mixed traffic a long summary prompt
  violates every running stream's floor for seconds. Options to measure in M3 (admission): smaller
  `--max-num-batched-tokens` (finer prefill/decode interleaving, at the cost of long-prompt TTFT),
  a per-replica cap on concurrent prefill tokens in admission, or both. Until then floor violations
  caused by prefill are reported, not prevented.
- Phase-2 Ampere+ GPUs with FlashAttention will have a flatter curve and far cheaper prefill; every
  replica class gets its own calibration file.

## Environment the numbers depend on

- Docker Desktop 4.2.0 with the WSL driver-store workaround (`scripts/vllm-entrypoint.sh`); WSL kernel
  5.15.167.4; Windows driver 610.88 (WSL user-mode 610.57.01).
- The GPU also drives the Windows desktop (WDDM). Free memory at vLLM start varies with what else is
  open; the calibration file records the free memory it saw. Close LM Studio before runs.
- Numbers are for this card and this model build; phase-2 replicas get their own calibration files.
