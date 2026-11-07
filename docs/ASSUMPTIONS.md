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

## Milestone 2 measurements (pacing, load generator, report)

**Prompt sizing (loadgen/content.py).** Through vLLM's `/tokenize` with the Qwen2.5 chat template on
2026-11-06: every word of the vocabulary is one token with a leading space, sentence periods are one
token, so 400 planned tokens measured 400–402 (ratio 1.000–1.005); the chat template adds a fixed
**29 tokens** per request (Qwen's default system prompt plus role markers). The fake replica's
estimator counts words and punctuation the same way (ratio 1.000). Prompt sizes in a plan are
therefore content tokens; `prompt_tokens` in the meter is content + 29 (+ history for multi-turn).

**KV budget when the desktop holds memory.** With LM Studio and the desktop holding 3.5 GiB, vLLM
runs at `VLLM_GPU_MEM=0.64`: KV cache 4.27 GiB = **124,256 tokens** (7,766 blocks), 7.6× at 16k
context, against 181,936 tokens at the usual 0.82. Step rates do not depend on the KV budget, so the
calibration above still applies; the milestone-2 GPU runs below used the smaller budget and the
report's KV-usage percentages are relative to it (`loadrun.json` records the budget of each run).

**Pacing precision.** `tests/test_pacing.py` asserts the steady-state delivered rate of a 30 tps shape
within ±5% of 30 tok/s while the fake engine runs at ~200 tok/s (measured client-side from per-token
timestamps), and that a complete response of 60 tokens is released about one second after its first
token. Fake-profile smoke runs (two seeds, 115 requests): every tenant's delivered rate at or above
its shape's rate, zero floor violations, throttled time 72% of chat streaming time and 98% of batch
(complete-mode) time, i.e. the fake engine is far faster than any shape sold.

### GPU run of `config/scenarios/mixed.yaml` (2026-11-06, gateway run `20261106-224910`, seeds 1 and 2)

18-minute timeline + 90 s drain per seed, no admission control (milestone 2), pacing on, KV budget
124,256 tokens (see above). Per seed: 628 / 602 requests, 594 / 575 ok, 34 / 27 refused (all agent
steps whose context outgrew the 8k shape, `429 context_exceeds_shape`), 0 errors, 0 vLLM
preemptions (A9 holds). Requests in flight (admitted → last token delivered) mean 6.6, p95 14; the
engine's own running count averaged 4.3 because delivery is paced after the engine finishes. KV
usage mean 12%, p95 36%: on this card compute, not KV, is the constraint.

| Tenant (shape) | ok / run | per-token USD | shape billed USD | CV per-token | CV shape | TTFT p50 / p95 | delivered tok/s (shape) | floor violations | throttled share |
|---|---|---|---|---|---|---|---|---|---|
| chat (8k.30tps, stream) | 282 / 293 | 0.142 / 0.149 | 0.033 / 0.033 | 3.5% | **0.8%** | 0.56 s / 3.2 s | 30.0 (30) | 52% | 31% |
| agent (8k.60tps, stream) | 218 / 186 | 0.178 / 0.150 | 0.044 / 0.039 | 12.3% | **8.5%** | 1.3 s / 3.8 s | 38.3 (60) | **92%** | 2% |
| notetaker (16k.30tps, complete) | 15 / 14 | 0.014 / 0.015 | 0.004 / 0.005 | 6.7% | 7.9% | 3.7–5.8 s / 6.7–8.3 s | 26.2 (30) | 63% | 36% |
| batch (8k.15tps.lazy, complete) | 79 / 82 | 0.018 / 0.017 | 0.010 / 0.010 | 1.3% | **0.3%** | 0.55 s / 2.7 s | 15.3 (15) | 10% | 93% |

CV = coefficient of variation of the tenant's bill across the two seeds (two "days"); more seeds
are needed for a tight estimate, but the direction is consistent: shape bills vary less than
per-token bills for every tenant except the note-taker's 15-request sample.

What the run shows:

- **Pacing holds where the engine keeps up**: chat delivered 30.0 tok/s on the 30 tps shape and
  batch 15.3 on 15 tps, with 31% and 93% of their streaming time spent throttled (the engine was
  ahead; complete responses are held until their virtual stream would end).
- **The 60 tps shape is not honoured under this load**: with ~6.6 requests in flight the engine's
  per-stream decode rate is ~40 tok/s (calibration: `n_max(60)` = 2 at 6k context), so 92% of agent
  streams violated the floor. The seat rule of §5.2 would have refused or queued most of them; this
  is the milestone-3 baseline.
- **Prefill interference (A4) dominates floor violations of the 30 tps shapes**: chat streams ran at
  68 tok/s engine rate with no long (≥3k) prefill in flight and 29 tok/s (p10 13.5) while one was;
  floor-violation share 4% vs 58%. TTFT p50 grew from 0.44 s (≤2k prompts) to 1.7 s (6–9k) and p95
  to 6.1 s, i.e. prefills queue behind each other.
- **The billing cap did its job**: uncapped wall-clock would have billed $0.130 per run; the capped
  shape bill was $0.090 (bound $0.094). Customers did not pay for the floor violations above, the
  provider did.
- **Prefill is unpriced under the placeholder shape prices.** The run processed 1.9M prompt tokens
  against 140k output tokens; per-token revenue was $0.35 per run, shape revenue $0.09, GPU cost
  $0.09 (at $0.30/h). The placeholder `price_per_second` was derived from the output-token
  reference only, so a 6k-token summary pays for its ~4 s of prefill at the same rate as 4 s of
  decode. Shape prices (or a per-prompt-token component) must cover prefill before the schemes are
  compared on revenue; that is a pricing input to milestone 3/4, not a property of time-slicing.

## Environment the numbers depend on

- Docker Desktop 4.2.0 with the WSL driver-store workaround (`scripts/vllm-entrypoint.sh`); WSL kernel
  5.15.167.4; Windows driver 610.88 (WSL user-mode 610.57.01).
- The GPU also drives the Windows desktop (WDDM). Free memory at vLLM start varies with what else is
  open; the calibration file records the free memory it saw. Close LM Studio before runs.
- Numbers are for this card and this model build; phase-2 replicas get their own calibration files.
