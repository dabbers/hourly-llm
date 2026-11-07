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

## Milestone 3 measurements (admission, prefill chunk size, overcommit, lazy requeue)

All on 2026-11-07, KV budget at `VLLM_GPU_MEM=0.64` (124,688 tokens with the 128-token chunk below,
118,453 usable after `kv_margin`), seat curve `context_6144` of the calibration file: 19 / 8 / 2
seats at 15 / 30 / 60 tps.

### Prefill chunk size (`--max-num-batched-tokens`, A4)

`scripts/calibrate.py --interference-only`: one 2k-context stream decodes alone, then a 6,144-token
prompt arrives; the victim's rate is measured before, during and after the aggressor's prefill. Raw
probes: `calibration/interference/local-2080ti-chunk<size>.json`. Only the first of the two repeats
counts: the second sends the same prompt and hits the prefix cache (prefill 0.09 s).

| chunk tokens | aggressor prefill | victim tok/s before → during → after |
|---|---|---|
| 2048 (vLLM default) | 4.78 s | 137 → **1.7** → 133 |
| 512 | 5.07 s | 138 → 3.2 → 134 |
| 256 | 5.24 s | 139 → 5.5 → 134 |
| 128 | 5.51 s | 138 → **9.8** → 133 |

The victim gets about one decode step per prefill chunk, so its rate during a prefill is roughly the
prefill rate divided by the chunk size (1,115 tok/s ÷ 128 ≈ 8.7, measured 9.8). A 30 tps floor would
need ~37-token chunks, and the per-step overhead already costs 15% of prefill speed at 128 (64 and
32 were not measured). Decision: 128 is the compose default (6× the victim rate for +15% prefill
time); A4 stays a documented limit of this card, not something chunking removes. The per-replica cap
on prefill tokens in flight (`SLOTGW_MAX_PREFILL_TOKENS`) bounds how many long prefills queue behind
each other, a TTFT lever rather than a decode-rate one, and was not measured on the GPU.

### GPU runs of `config/scenarios/mixed.yaml` with admission on (chunk 128)

Four gateway runs on the same 18-minute timeline as the milestone-2 run (`20261106-224910`: chunk
2048, no admission), which is the baseline column. `max_queue_s` is 30 s for the interactive shapes.
Seeds in each cell are separated by `;`.

| | M2 baseline (s1; s2) | overcommit 1.0 (s1; s2) `20261107-005416` | overcommit 1.5 (s1) `20261107-013112` | overcommit 2.0 (s1) `20261107-014935` | overcommit 1.0 + `requeue` (s1) `20261107-020804` |
|---|---|---|---|---|---|
| ok / refused context / refused seat | 594 / 34 / –; 575 / 27 / – | 520 / 31 / 7; 489 / 23 / 8 | 514 / 30 / 8 | 534 / 32 / 4 | 510 / 33 / 5 |
| output tokens (per GPU-hour) | 140k (459k); 136k (447k) | 115k (377k); 110k (362k) | 115k (378k) | 118k (386k) | 110k (364k) |
| engine running mean / max holds (with a 60 tps hold) | 4.3 / – | 2.05 / 8 (2); 2.09 / 8 (2) | 2.67 / 12 (3) | 3.44 / 13 (4) | 2.04 / 8 (2) |
| floor violations | 301; 267 | 59; 59 | 197 | 259 | 78 |
| violation share chat / agent / notetaker / batch | 52 / 92 / 63 / 10% | 9 / 16 / 13 / 0%; 8 / 15 / 50 / 9% | 14 / 76 / 20 / 6% | 23 / 92 / 33 / 3% | 9 / 24 / 13 / 14% |
| chat queue wait p50 / p95 | 0 | 8.8 / 21 s; 10.2 / 25 s | 10.4 / 22 s | 9.2 / 25 s | 9.8 / 25 s |
| per-token USD / shape USD (of which prefill) | 0.352 / 0.090 (0); 0.331 / 0.086 (0) | 0.318 / 0.244 (0.171); 0.295 / 0.227 (0.157) | 0.315 / 0.246 (0.169) | 0.327 / 0.257 (0.176) | 0.317 / 0.241 (0.170) |
| fill / utilisation (break-even fill 58%) | – | 28 / 44%; 28 / 42% | 35 / 44% | 44 / 45% | 27 / 42% |
| gateway preemptions / vLLM preemptions | – / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 21 / 0 |

What the runs show:

- **Admission never exceeded a ledger.** Replaying every admission against the calibration curve:
  holds filled the seat ceiling exactly and never passed it; at most 2 / 3 / 4 concurrent holds while
  a 60 tps request ran at overcommit 1.0 / 1.5 / 2.0, at most 8 / 12 / 13 otherwise. vLLM's own
  preemption counter stayed 0 (A9). KV usage mean 6%, p95 13%.
- **Floor violations fell 80%** (301 / 267 → 59 / 59 per run): chat 52% → 9%, agent 92% → 16%,
  batch 10% → 0–9%, note-taker 63% → 13% / 50% (14–15 requests). The residual is prefill
  interference: chat streams with no long prefill in flight had 0% violations at 85 tok/s engine
  rate; with one, 19% at 49 tok/s (p10 22), against 58% and 29 tok/s (p10 13.5) in the M2 run.
- **The price is queueing and throughput.** Output per run −18% (140k → 115k tokens; 459k → 377k
  per GPU-hour), interactive queue wait p50 7–10 s and p95 19–26 s, 7–8 requests per run (1.4%)
  refused `capacity:seat` after the full 30 s wait. The context refusals (23–31) are agent sessions
  outgrowing 8k, as in M2. The cause is shape nesting: `n_max(60)` = 2 at 6k context, so one agent
  request on the 60 tps shape shrinks the replica from 8 seats to 2 for its lifetime and idles six
  chat slots. On this card the 60 tps shape is the one that costs capacity; the honest options are
  to price it for the capacity it displaces or not to sell it here (Q3, milestone 4).
- **Overcommit buys nothing on this card.** 1.5× raised engine concurrency to 2.7 and 2.0× to 3.4,
  output tokens by 0% and +2%, queue waits not at all (p50 9–10 s), while agent violations returned
  to 76% and 92% (the no-admission level) and chat's rose to 14% and 23%. Decode is compute-bound
  here, so extra seats split the same step rate. `SLOTGW_TPS_OVERCOMMIT` stays 1.0.
- **`requeue` hurts the lazy tenant without helping the interactive ones.** 21 preemptions, batch
  completions 35 → 22, 2,270 of 7,899 batch output tokens discarded (29%), batch e2e p95 215 → 469 s;
  chat queue p50 9.8 s vs 8.8 s, seat refusals 5 vs 7, agent violations 24% vs 16% (noise either
  way). The seats interactive requests wait for are taken by the 60 tps rule, not by lazy work
  (lazy share ≤ 50%, lazy queue mean 1.1). Default stays `never`.
- **Economics with the prefill charge** ($0.10 per 1M prompt tokens): shape revenue $0.244 per run,
  of which $0.171 prefill and $0.073 runtime, against $0.318 per-token and $0.092 GPU cost: 77% of
  per-token revenue and 2.7× GPU cost (M2 without the charge: 26% and 1.0×). Fill 28% against a
  break-even of 58% on the runtime-only plan price: the runtime price alone ($0.24 per hour at this
  fill) does not pay for the GPU ($0.30 per hour), the prefill charge does. Fill counts each hold as
  one slot of the 8 × 30 tps plan, so a 60 tps hold worth four slots' capacity counts as one;
  utilisation (output tokens against the plan's 240 tok/s), 42–45%, is the better measure of how
  much of the card the plan sells.
- **Bill variance across seeds** (two seeds, overcommit 1.0; CV per-token vs shape): chat 0.26% vs
  0.11%, agent 10.9% vs 10.1%, batch 10.0% vs 12.0%, note-taker 6.6% vs 9.0%. The prefill charge
  makes the shape bill track prompt tokens the way the per-token bill does, so the milestone-2 gap
  narrowed; chat, the only tenant with a large sample, still varies less under shapes.

### Market reference prices (OpenRouter snapshot of 2026-11-06)

`slotgw-fetch-prices` snapshotted OpenRouter's public list into `config/price_references.yaml`.
Qwen2.5-3B is not listed; the comparables are the 3B-class open models, and `basket-3b` (their
median) is now the active reference, so shape prices derived from it are 4× below the gpt-4o-mini
placeholder the milestone-2/3 runs were billed at (30 tps: $4.5 per million seconds instead of $18;
prefill $0.0755 per 1M prompt tokens instead of $0.10).

| reference | $/1M in | $/1M out | M3 run revenue | × GPU cost | gross margin | break-even GPU $/h (44% / 70% util.) |
|---|---|---|---|---|---|---|
| gemma-3-4b | 0.050 | 0.100 | $0.094 | 1.03 | 3% | 0.31 / 0.50 |
| llama-3.2-3b | 0.051 | 0.335 | $0.121 | 1.32 | 24% | 0.40 / 0.64 |
| **basket-3b** (median, active) | 0.0755 | 0.150 | $0.142 | 1.56 | 36% | 0.47 / 0.76 |
| ministral-3b | 0.100 | 0.100 | $0.177 | 1.94 | 48% | 0.58 / 0.94 |
| qwen-2.5-7b | 0.100 | 0.200 | $0.188 | 2.06 | 51% | 0.62 / 1.00 |
| gpt-4o-mini (old placeholder) | 0.150 | 0.600 | $0.314 | 3.45 | 71% | 1.03 / 1.67 |

Revenue is the mean over the five admission-on runs above (1.6–1.8M prompt and 110–118k output
tokens each, GPU cost $0.091–0.092 at $0.30 per hour, utilisation 42–45%); lazy requests at the
batch discount. Prices this provider would have to list to earn a gross margin on that traffic
(overcommit 1.0, seed 1; prompt tokens weighted at the basket's 1:2 input:output ratio):

| gross margin | $/1M in / out at 44% utilisation | × basket out price | at 70% utilisation | × basket |
|---|---|---|---|---|
| 0% | 0.048 / 0.096 | 0.64 | 0.030 / 0.060 | 0.40 |
| 25% | 0.064 / 0.128 | 0.85 | 0.040 / 0.080 | 0.53 |
| 50% | 0.097 / 0.192 | 1.28 | 0.060 / 0.120 | 0.80 |

What it says:

- **At market prices the card is viable on this traffic, with a caveat.** A $0.30-per-hour 2080 Ti
  serving the AWQ 3B model earns 1.56× its cost at the 3B basket's prices at the measured 44%
  utilisation (36% gross margin), 1.03× at the cheapest listing (Gemma 3 4B) and 3.45× at the
  gpt-4o-mini placeholder, which is why the earlier economics looked comfortable. To earn 25% it
  could list 15% below the basket today, or 47% below at 70% utilisation; it can afford a GPU up to
  $0.47 per hour at break-even ($0.76 at 70%).
- **The revenue is in prompt tokens.** 1.7M prompt tokens against 115k output tokens: at the
  basket, 90% of the per-token revenue is input. The same holds for shape bills once the prefill
  charge is at parity, and the runtime part alone (8 × 30 tps × $4.5 per million seconds = $0.13 per
  hour at 100% fill) can never pay for a $0.30-per-hour card: on prompt-heavy traffic a price per
  second of decode is the smaller half of the bill, and the prefill charge is not optional.
- **The caveat**: the meter counts every prompt token at the input price, but the agent profile's
  growing history is mostly prefix-cache hits inside vLLM, and providers that publish cache-read
  prices discount them (Ministral 3B 90%, gpt-4o-mini 50%). The margins above are upper bounds
  until cached tokens are metered and priced separately (DESIGN.md §13).
- **Variance is unchanged by the reference**: both bills scale with the same token counts, so the
  milestone-2/3 predictability findings hold at any reference; only the level moves.

## Environment the numbers depend on

- Docker Desktop 4.2.0 with the WSL driver-store workaround (`scripts/vllm-entrypoint.sh`); WSL kernel
  5.15.167.4; Windows driver 610.88 (WSL user-mode 610.57.01).
- The GPU also drives the Windows desktop (WDDM). Free memory at vLLM start varies with what else is
  open; the calibration file records the free memory it saw. Close LM Studio before runs.
- Numbers are for this card and this model build; phase-2 replicas get their own calibration files.
- The fake replica (tests, `--profile fake`) schedules tokens on absolute due times with a 16 ms
  minimum sleep and emits several tokens per chunk when behind: on Windows the asyncio clock has
  15.6 ms resolution and treats shorter sleeps as already due whenever the loop is busy, which made
  the fake 20x too fast in streaming mode before this fix. Inside Docker (Linux) sleeps are accurate.
