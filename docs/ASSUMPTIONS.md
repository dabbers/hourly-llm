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

### Reservation what-if on the milestone-3 runs

`reservations.csv` sizes, per tenant, a standing reservation of N slots for the whole 18-minute run
from the tenant's queue-free occupancy (each request holds a slot from arrival for the seat time it
measured), and bills it slot-seconds at the shape's price with no prefill charge. Overcommit 1.0,
seed 1, the snapshot's gpt-4o-mini-derived prices (a 30 tps slot $0.065 per hour; at the 3B basket
all slot prices are 4× lower). Capacity weight = seats(30 tps) ÷ seats(shape tps) on the calibration
curve, so a 60 tps slot displaces four 30 tps plan slots.

| tenant (shape) | occupancy p50 / p95 / max | p95 reservation: slots, would wait, used fill | reservation USD at p95 / at peak | per-request shape USD | per-token USD | slot $/h: price / break-even / per-token parity at p95 |
|---|---|---|---|---|---|---|
| chat (8k.30tps) | 1 / 4 / 9 | 4, 12%, 31% | 0.079 / 0.178 | 0.100 | 0.134 | 0.065 / 0.038 / 0.110 |
| agent (8k.60tps) | 1 / 3 / 5 | 3, 9%, 34% | 0.119 / 0.198 | 0.122 | 0.162 | 0.130 / 0.150 / 0.177 |
| notetaker (16k.30tps) | 0 / 1 / 1 | 1, 0%, 14% | 0.025 / 0.025 | 0.012 | 0.014 | 0.081 / 0.038 / 0.044 |
| batch (8k.15tps.lazy) | 0 / 1 / 2 | 1, 20%, 22% | 0.005 / 0.010 | 0.009 | 0.008 | 0.016 / 0.016 / 0.026 |

What it says:

- **A p95 reservation is cheaper for the steady tenants than paying per request** (chat 0.79×, agent
  0.97× of the per-request shape bill; 0.59× and 0.73× of per-token), and about one request in ten
  would wait for the tenant's own slots. Sized at the peak nobody waits and the bill is 1.3× per-token.
  The hour-mark note-taker is the opposite case: its one slot is 86% idle and costs 1.8× per-token, so
  bursty tenants want on-demand or burst, steady ones a reservation.
- **Two thirds of a p95 reservation sits idle** (used fill 31–34%; at p50 sizing 60% but seven in ten
  requests wait). That idle reserved capacity is the resale pool the burst manager (milestone 4) lends
  to burst and lazy work, and it is what the customer is paying for when their demand is not steady.
- **Parity slot prices do not cover the provider's cost.** At the snapshot prices a sold 30 tps slot
  earns 1.7× its cost share, but a 60 tps slot earns 0.86× because it displaces four plan slots: 60 tps
  reservations lose money at parity even when fully used. At the 3B basket every slot sells below cost
  ($0.016 against $0.038 per hour for 30 tps). A reservation product needs its own price, between
  break-even ($0.038 per hour for a 30 tps slot on a $0.30 card) and what the tenant's traffic earns
  per token ($0.11 per hour for chat at p95 sizing; about $0.05 at basket prices): a narrow window on
  this card, wider on a card with more seats per dollar.
- **Whole-run reservations straddle the scenario's quiet and peak phases**, which is most of the idle
  share; reservations with phase-length windows (business hours, hour marks) would fill better, and
  the sizing rule is the customer's lever, not the provider's.

## Milestone 4 measurements (tps burst, context bursts, reservations)

All on 2026-11-07 on the card and settings of milestone 3 (KV budget `VLLM_GPU_MEM=0.64`, 128-token
prefill chunks, overcommit 1.0, `requeue` off), billed at the `basket-3b` reference (prefill $0.0755
per 1M prompt tokens; a 30 tps slot $4.5 per million seconds = $0.0162 per hour, 16k shapes × 1.25),
`config/scenarios/mixed.yaml` seed 1, 18 minutes + 90 s drain. Reservations as the scenario states
them: chat 4 slots of `small.8k.30tps` (the what-if's p95 sizing), note-taker 1 slot of
`small.16k.30tps`, both for the whole run; the on-demand runs used `--no-reservations`. A smoke run
went first (`smoke.yaml`, burst on, one chat slot reserved from 30 s to 90 s): 102 / 102 ok, the
reservation created and released on time, 14 requests served from it, 0 timeouts. The burst (`grace`)
and burst + reservations runs share gateway run `20261107-054909` (same gateway settings, so compose
kept the container); the report splits them by load run. Report: `runs/20261107-053033/report-m4`.

| | burst off, on demand `20261107-053033` | burst (`grace`), on demand `20261107-054909` | burst (`abort`), on demand `20261107-064403` | burst + reservations `20261107-054909` | burst off + reservations `20261107-062541` |
|---|---|---|---|---|---|
| ok / refused (agent context / agent seat / chat seat) | 517 / 39 (32 / 3 / 4) | 554 / 39 (1 / 34 / 4) | 558 / 40 (0 / 34 / 6) | 461 / 38 (0 / 38 / 0) | 344 / 38 (0 / 38 / 0) |
| context bursts classified / ran / revoked | – | 37 / 3 / 0 | 32 / 0 / 0 | 0 | – |
| output tokens (per GPU-hour) | 114k (371k) | 128k (424k) | 131k (436k) | 120k (399k) | 73k (241k) |
| burst tokens (share of output); lent rate per running request, sampled mean | 0 | 40k (32%); 56 tok/s | 43k (33%); 56 | 71k (59%); 65 | 0 |
| engine streams mean / ledger holds mean (of which reserved) | 2.0 / 2.2 | 2.3 / 2.5 | 2.3 / 2.5 | 1.9 / 5.9 (5) | 1.1 / 5.2 (5) |
| delivered tok/s chat / agent / note-taker / batch (shapes 30 / 60 / 30 / 15) | 36 / 70 / 32 / 16 | 64 / 72 / 59 / 77 | 67 / 72 / 58 / 80 | 73 / – / 59 / 81 | 37 / – / 32 / 16 |
| throttled s | 2058 | 219 | 250 | 404 | 2390 |
| floor violations (share chat / agent / note-taker / batch) | 57 (14 / 18 / 22 / 0%) | 67 (18 / 28 / 15 / 0%) | 57 (16 / 24 / 0 / 0%) | 14 (10 / – / 9 / 0%) | 20 (10 / – / 20 / 0%) |
| chat queue p50 / p95; e2e p50 / p95 | 10.4 / 22.7 s; 17.0 / 29.5 s | 11.4 / 25.0; 16.8 / 30.1 | 13.3 / 25.2; 16.7 / 30.4 | 0.01 / 1.6; 3.4 / 9.4 | 0.01 / 2.2; 6.1 / 10.8 |
| agent queue p50 / p95; e2e p50 / p95 | 6.2 / 20.5 s; 11.7 / 23.8 s | 8.5 / 24.6; 14.0 / 28.9 | 9.9 / 24.9; 15.2 / 29.9 | nothing served | nothing served |
| batch ok; e2e p50 / p95 | 33; 33 / 265 s | 62; 8.3 / 281 | 70; 7.1 / 196 | 163; 6.1 / 71 | 47; 38 / 82 |
| per-token USD / shape USD (bound) / burst discount | 0.142 / 0.146 (0.148) / 0.002 | 0.152 / 0.153 (0.159) / 0.006 | 0.147 / 0.149 (0.155) / 0.007 | 0.084 / 0.086 (0.094) / 0.008 | 0.074 / 0.077 (0.078) / 0.001 |
| billed: reservations + on demand = revenue; × GPU cost ($0.090 per run) | 0.146; 1.58 | 0.153; 1.69 | 0.149; 1.65 | 0.026 + 0.018 = 0.043; 0.48 | 0.026 + 0.006 = 0.032; 0.35 |
| fill / utilisation | 27 / 43% | 32 / 49% | 31 / 50% | 25 / 46% | 15 / 28% |
| KV usage mean / p95 | 6 / 12% | 7 / 14% | 6 / 12% | 4 / 13% | 3 / 12% |

In the reservation runs the per-request shape bill is what the reserved tenants would have paid on
demand; the market section of the report prices each run's traffic at the reference's per-token
rates, so its reservation-run margins (−8% / −22%) are for the traffic, not for what was billed.

| reservation | slots × hours | served / timeouts | queue on own slots p50 / p95 | used fill / idle slot-hours | bill | per-request shape / per-token it would have paid |
|---|---|---|---|---|---|---|
| chat, burst on | 4 × 0.30 h | 283 / 0 | 13 ms / 1.6 s | 24% / 0.91 | $0.0194 | $0.0615 / $0.0643 (0.32× / 0.30×) |
| chat, burst off | 4 × 0.30 h | 282 / 0 | 13 ms / 2.2 s | 21% / 0.94 | $0.0194 | $0.0637 / $0.0639 (0.31× / 0.30×) |
| note-taker, burst on | 1 × 0.30 h | 15 / 0 | 18 / 26 ms | 12% / 0.26 | $0.0061 | $0.0064 / $0.0062 (0.95× / 0.97×) |
| note-taker, burst off | 1 × 0.30 h | 15 / 0 | 17 / 23 ms | 12% / 0.26 | $0.0061 | $0.0067 / $0.0062 (0.91× / 0.97×) |

What the runs show:

- **tps burst is free throughput.** The same GPU time produced 12–15% more output (114k → 128k /
  131k tokens; 371k → 424k / 436k per GPU-hour) with the engine running 2.3 streams instead of 2.0:
  requests finish sooner, so the closed-loop tenants (batch at concurrency 2, agent sessions) send
  the next one sooner. Chat was delivered at 64–67 tok/s on its 30 tps shape, batch at 77–80 on 15
  tps lazy, with batch completions 33 → 62 / 70 and its e2e p50 33 s → 7–8 s; throttled time fell
  from 2,058 s to 219–250 s per run. 32–33% of all output tokens were burst, and the lent rate per
  running request averaged 56 tok/s, the engine's spare at 2–3 streams. Chat's queue (p50 11–13 s)
  and TTFT did not move: queueing is admission's price, bursting only shortens generation. The clean
  number is in the reservation runs, where chat never queues: e2e p50 6.1 s without burst, 3.4 s with.
- **The burst discount is 4% of the bill and 30% of its runtime part.** Bound − billed rose from
  $0.002 to $0.006–0.007 on a $0.15 bill, because at parity prices 90% of the bill is prefill
  (1.7–1.8M prompt tokens at $0.0755 per 1M = $0.13; runtime bound ≈ $0.02). Giving the spare rate
  away costs the provider little on prompt-heavy traffic; it would cost a third of a decode-only bill.
- **Lending spends the pacer's smoothing margin.** Floor violations went 57 → 67 with 7% more
  requests (`abort` run: 57), the agent's share 18% → 28% / 24% and chat's 14% → 18% / 16%; in the
  interference split, agent streams overlapping a long prefill violated 27% of the time without burst
  and 41% / 36% with it, streams with no prefill in flight 1.8% in all three. Without burst a request
  paced at its tps banks the engine's surplus in its buffer and rides out a prefill stall from it;
  with burst the surplus is delivered as it arrives (one bucket depth of credit, never banked), so a
  stall lands on an empty buffer and the whole-stream average drops below the floor. A lending
  reserve (lend only above a margin kept banked) is the knob to try; it was not built.
- **Context bursts at the shape's own tps do not fit on this card.** 37 / 32 agent steps that had
  outgrown 8k were classified as context bursts instead of being refused. A burst keeps its shape's
  tps, so a 60 tps burst needs the replica at ≤ 1 other hold (`n_max(60)` = 2); the burst queue is
  served only when nothing else waits, and at peak 2–8 holds were running. 3 bursts ran in the `grace`
  run (8.7–14.8k prompts, 9–11 s each), none in the `abort` run; the other 34 + 32 waited the full 30
  s and were refused `capacity:seat`. Against the burst-off run, where the same steps got an instant
  `429` and the session ended, the agent lost 30 s per step for the same answer. No burst was ever
  revoked (`grace` and `abort` runs are identical in this respect), so the revocation policies were
  exercised only by the fake-replica tests, and whether revocation hurts guaranteed traffic is
  unanswered on hardware. The design change this asks for (not built): admit context bursts as best
  effort at the slowest tps class, which is what a request with no floor is, and give the burst queue
  a short wait (a few seconds), because a burst that cannot be placed now should fail fast.
- **Reservations give the reserved tenants a different product.** Chat on its four slots: queue p50
  13 ms and p95 1.6–2.2 s against 10–13 s / 23–25 s on demand, e2e p50 3.4 s (burst) / 6.1 s, 0
  timeouts, no seat refusals; the note-taker's 15 requests queued ≤ 26 ms. Used fill was 21–24% (chat)
  and 12% (note-taker): 0.91 of 1.20 chat slot-hours and 0.26 of 0.30 note-taker slot-hours idle,
  about what the what-if predicted (31% / 14%) for a whole-run reservation across quiet and peak.
  With burst on, the idle reserved rate went to the lazy tenant: batch completed 163 requests (62 on
  demand with burst, 33 without), 59% of all output tokens were burst, and chat's own delivered rate
  was 73 tok/s. Floor violations fell to 14 / 20 because the fast shape was absent (next bullet).
- **Standing holds of a slow shape lock the fast shape out.** Five reserved 30 tps slots count as
  seats whether idle or occupied, and the 60 tps ceiling is 2 seats, so no agent request could be
  placed at any time: 38 / 38 refused `capacity:seat` after 30 s, 18 of them while nothing at all
  was running. That is the nesting rule applied to standing holds, as designed (the idle slots are
  the guarantee), and on this card it means a reservation of two or more slots of any shape removes
  the 60 tps shape from sale on that replica; the agent tenant's $0.08 per run, half the on-demand
  revenue, went with it. Reservations and the 60 tps shape need separate replicas here, or the 60 tps
  shape is not sold (RESULTS.md Q3).
- **At parity prices a reservation is a gift.** Chat paid $0.0194 for four slots (4 × 0.3 h × $0.0162)
  against $0.0615–0.0637 per request on demand (0.31–0.32×; 0.30× of per-token), the note-taker
  $0.0061 against $0.0064–0.0067 (0.91–0.95×). The reserved slice is 5 of 8 seats, $0.056 of the
  run's $0.090 GPU cost, and earned $0.0255: 0.45× its cost share. Revenue per run fell from $0.146–
  0.153 (1.6–1.7× GPU cost) to $0.043 / $0.032 (0.48× / 0.35×), the agent lockout and the cheap
  reservation in about equal parts; bursting over the idle slots recovered $0.011 of it through
  batch. For chat's reservation to match its per-request bill `reservation_price_factor` would be
  ≈ 3.2 (one over used fill, less the prefill charge it no longer pays); to cover the slice's cost
  share, ≥ 2.3 ($0.0375 per slot-hour = $0.30 ÷ 8 seats, the what-if's $0.038). The knob exists;
  the config keeps 1.0 so the report shows the parity case.

## Phase 2 assumptions set before any pod exists (G0, 2026-11-07)

Measured values replace these in stage A (docs/PHASE2.md section 4); until then they are inputs,
not findings.

- **Cache-read price.** Prompt tokens vLLM serves from its prefix cache are priced at the active
  reference's cache-read price when it lists one, else `pricing.token.cache_read_ratio` = 0.25 × the
  input price, under both schemes and in the shape bill's prefill charge. The 7B basket lists one:
  the median of Llama 3.1 8B ($0.025), Ministral 8B ($0.015) and Granite 8B ($0.015) is **$0.015 per
  1M cached tokens** against $0.08 input, a 0.19 ratio; Qwen 2.5 7B lists none. 0.25 is a middle
  value between the 0.1× of Anthropic and DeepSeek and the 0.5× of OpenAI; most open-model hosts on
  OpenRouter list no cache-read price at all, which means they charge hits in full, and the report
  keeps that case visible as the "uncached" revenue column. The 2080 Ti runs have no cached-token
  counts (the meter did not keep them), so their margins stay upper bounds.
- **7B price basket** (`config/phase2/price_references.yaml`, OpenRouter list of 2026-11-07):
  `basket-7b` = median of qwen/qwen-2.5-7b-instruct ($0.10 / $0.20 per 1M in / out),
  meta-llama/llama-3.1-8b-instruct ($0.05 / $0.08), mistralai/ministral-8b-2512 ($0.15 / $0.15) and
  ibm-granite/granite-4.2-8b ($0.06 / $0.25) = **$0.08 in / $0.175 out**. At parity a 30 tps slot is
  $0.175 × 30 / 1e6 per second = **$0.0189 per hour** (the 3B basket's was $0.0162). Mistral 7B
  Instruct is no longer listed, hence Ministral 8B.
- **GPU price.** $0.89 per pod-hour (RunPod Secure Cloud on-demand RTX 4090, October 2026 price
  snapshots); storage adds about $0.004 per pod-hour for a 20 GB volume. A stopped pod keeps its
  volume at $0.20 per GB-month and nothing else.
- **4090 step-rate curve, a guess.** The fake replicas used for the rehearsal run
  `1:130, 8:110, 16:95, 32:70, 48:50, 64:35` tokens/s per stream with a 280k-token KV budget
  (`FAKE_RATE_CURVE_4090`, `FAKE_KV_TOKENS_4090` in compose.yaml), the author's estimate for a 7B
  AWQ model on a 4090 with FlashAttention. On that curve a replica holds about 65 seats at 30 tps
  and 34 at 60 tps, eight times the 2080 Ti's, so `mixed-4090.yaml` keeps `rate_multiplier: 1.0`
  only as a placeholder; stage A's calibration sets the curve, the shapes and the multiplier.
- **Reservation price factor 2.3** (`SLOTGW_RESERVATION_FACTOR`): the 2080 Ti's break-even for the
  reserved slice; carried over as the starting point, and the report prints the bill at other
  factors since it is linear in it.
- **Context bursts run as the slowest non-lazy shape** (`context_tps_class: slowest`, queue 5 s):
  the post-M4 design change, verified by the fake-replica tests (tests/test_phase2.py) and not yet
  on hardware.

## Phase 2 measurements: stage A, one RTX 4090 (2026-11-08)

Pod: RunPod Secure Cloud, EUR-IS-2, RTX 4090 24 GB, 36 vCPU, RunPod PyTorch image with vLLM 0.32.0
installed into a venv on the 20 GB volume; `Qwen/Qwen2.5-7B-Instruct-AWQ`, fp16 activations, Marlin
AWQ kernels, FlashAttention 2, `--max-model-len 16384 --max-num-seqs 64 --gpu-memory-utilization 0.90
--max-num-batched-tokens 512`, prefix caching on. KV cache **271,152 tokens** (16,947 blocks of 16),
21.1 GB of the 24 GB in use. `scripts/calibrate.py`, 256 output tokens per stream, concurrency
1–48 at 2k and 6k context, margin 0.10; the whole calibration took two minutes.

| streams | 2k pure tok/s | 2k mixed | 2k aggregate | 6k pure | 6k mixed | 6k aggregate | 6k TTFT p50 |
|---|---|---|---|---|---|---|---|
| 1 | 162 | 163 | 143 | 154 | 155 | 109 | 0.66 s |
| 2 | 158 | 154 | 274 | 144 | 133 | 210 | 0.42 |
| 4 | 152 | 138 | 480 | 136 | 104 | 309 | 0.47 |
| 8 | 146 | 117 | 766 | 113 | 72 | 400 | 0.56 |
| 16 | 121 | 88 | 1,089 | 82 | 44 | 454 | 0.90 |
| 24 | 101 | 71 | 1,413 | 64 | 33 | 565 | 0.38 |
| 32 | 90 | 63 | 1,702 | 54 | 27 | 644 | 0.36 |
| 48 | 71 | 65 | 1,778 | – (does not fit the KV) | 22 | 498 | 0.63 |

Against the 2080 Ti (105 → 35 tok/s from 1 to 8 streams at 6k, aggregate peak 131 tok/s): single
stream 1.5× faster, eight streams 3.2× faster, aggregate 4.9× higher, and the card stays
compute-rich much deeper into the batch (the 2080 Ti halved its per-stream rate by 4 streams, the
4090 by 24).

**Seat ceilings** (`n_max`, pure rate × 0.9 ≥ tps, at the 6k level the gateway uses; KV bound for
the shape's whole context):

| shape tps | n_max by step rate at 6k | n_max by KV (8k / 16k) | seats sold per replica |
|---|---|---|---|
| 15 | ≥ 48 (every tested level) | 33 / 16 | 33 / 16 |
| 30 | 32 (interpolated ≈ 48) | 33 / 16 | 32 / 16 |
| 60 | 16 (interpolated ≈ 21) | 33 / 16 | 16–21 |
| 120 | 1–2 | | not sold |

The KV budget, not the step rate, is what caps an 8k-context replica at 33 seats; the 2080 Ti's
8-seat plan becomes **32 per replica, 96 across three**. A 60 tps request costs about two 30 tps
seats of ceiling here, not six as on the 2080 Ti, so the nesting penalty that drove the M4 lockouts
shrinks by a factor of three.

**Prefill interference (A4):** a 6,144-token prompt prefills in **0.65 s** (4.2–5.5 s on the 2080
Ti). The decoding neighbour drops from 164 to 32 tok/s for that time and recovers to 165. A 30 tps
stream with its one-second bucket (30 tokens banked) rides out a 0.65 s stall at 32 tok/s without a
floor violation; at 2080 Ti prefill times it could not.

**Gateway smoke on the one replica** (`smoke-4090.yaml`, two minutes, burst on, text capture on,
runs `20261108-003551`): 150 requests, 148 ok, 2 refused (the two long agent showcase prompts whose
late turns exceed even the 16k burst cap: `429 context_exceeds_shape`, as designed), 0 errors, 6
context bursts admitted as the 15 tps class, 0 floor violations, 792k output tokens per GPU-hour.
With the card otherwise idle every stream was delivered at 139–153 tok/s on top of its shape: burst
share 68%. TTFT p50 59 ms (chat), 115 ms (agent), 328 ms (note-taker, 3–4k prompts). **Prefix-cache
hits: 48% of all prompt tokens** (agent 71%, chat 68%, batch 2%, note-taker 0.4%; vLLM's own
counters showed 61% over the earlier smoke), which the meter only sees with
`--enable-prompt-tokens-details`. Priced at the 7B basket's cache-read rate ($0.015 against $0.08
per 1M input), the traffic's per-token revenue is $0.0123 instead of $0.0194 uncached: **the M3 and
M4 margins, computed with every prompt token at full price, were 36% too high for this
multi-turn mix**. The 36 showcase requests kept prompt and completion; the 7B's answers follow the
instructions (a four-day itinerary revised under new constraints, a JSON classification with the
right fields) and are the first entries for the write-up's samples.

Stage A also settled three operational facts recorded in RUNBOOK.md: the vLLM image's start
command never ran on RunPod (container idle 26 min), so pods run RunPod's PyTorch image and vLLM is
installed over SSH; the 20 GB volume is too small for the 18 GB venv plus the 5.3 GB model cache, so
the cache and the engine log live on the container disk; and vLLM 0.32's FlashInfer sampler needs
`ninja`/`nvcc` the image lacks, so `VLLM_USE_FLASHINFER_SAMPLER=0` (attention stays on
FlashAttention). Spend for stage A: see PHASE2.md section 3.

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
