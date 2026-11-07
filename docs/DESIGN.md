# Time-sliced inference billing PoC — design and milestone plan

Status: **v0.4, 2026-10-29 — agreed baseline for milestone 1.** Shapes are per-request execution
contracts (you run requests *against* a shape), billed by wall-clock runtime; mixed shapes pack into
free capacity; streamed and complete delivery share one model; lazy (batch) shapes fill idle capacity
at a lower price. Decisions and their rationale are in §12.

---

## 1. Goal

Sell LLM inference as fixed **shapes** on a model, e.g. `model1.8k.100tps` = up to 8k context and
at least 100 output tokens/s on one fixed model, **billed per wall-clock second a request runs at that
shape**, instead of per token. The shape's rate is a guaranteed minimum, so the bill is bounded before
the request is sent; spare capacity makes it faster and cheaper, never slower or dearer. The provider plans each vLLM replica's capacity in shape-sized **slots** and fills them
with whatever mix of shapes fits: a `model1.8k.100tps` request can run inside an empty
`model1.16k.100tps` or `model1.8k.300tps` slot. Shapes are **burstable**: idle capacity is lent to
running requests as extra tokens/s or extra context at no charge, and is revocable at any time so
every request's shape is honoured first. Requests are **streamed** (real-time chat, agent loops) or
**complete** (summaries, batch jobs), mixed on the same replica, with the same shape semantics.

**What we optimize for**

1. **Customer predictability**: the shape's guaranteed rate bounds both time and cost
   (`≤ (TTFT + output_tokens ÷ tps) × price_per_second`) before the request is sent; the API surface
   stays plain OpenAI so bills compare apples to apples with per-token pricing.
2. **Token output per reservation of capacity**: fill empty slots with smaller shapes and lend idle
   capacity as burst, so each GPU-hour and each sold shape-second yields as many tokens as possible
   without breaking anyone's shape.

Provider viability (break-even fill rate, overcommit) is the constraint, not the objective.

**Mental model: a VPS host.**

| Cloud concept | Slot concept | Behaviour |
|---------------|--------------|-----------|
| vCPU share of a time-sliced CPU | shape `tps` | Guaranteed output tokens/s = a share of the replica's decode steps. Overcommittable like vCPUs, at the cost of "steal". |
| Reserved RAM | shape `max_context` | KV reservation while the request runs. Strict, never overcommitted: vLLM preemption is swapping and hurts everyone. |
| On-demand instance billed per second, existing only while the job runs | request at a shape | Admitted when its shape fits, billed per wall-clock second while running, released on completion. |
| Spot / preemptible instance | lazy shape (§3) | Cheaper per second; queued until capacity is idle; may be requeued if an interactive shape needs the seat (policy). |
| Instance-type nesting (a small job on a big host) | shape nesting | Admission is by resources, so any shape runs in any free capacity, including a larger shape's slot. |
| Burstable CPU / balloon memory | burst | Idle neighbours' tokens/s and unreserved KV are lent free and reclaimed when a shape needs them. No credits. |
| Hypervisor placement | admission + scheduler | Place a request (or reservation) on a replica with free KV and a seat whose step rate honours its tps. |
| Steal time / CPU-ready time | floor violations, throttled time, queue time | Time a request wanted its shape's rate and did not get it. |

The PoC must answer:

| # | Question | Where the report answers it |
|---|----------|-----------------------------|
| Q1 | At what replica fill rate (and overcommit ratio) does shape pricing break even for the provider? | §10 provider economics: analytic break-even utilization + empirical revenue/cost per run, per overcommit setting |
| Q2 | Which workload profiles get cheaper / more predictable bills under shapes, which are better off per-token? | §10 per-tenant bills under per-token (with batch discount) and shape pricing from the same run; bounded-cost vs actual; variance across seeds ("days") |
| Q3 | How much does throttling hurt each profile's latency? | §10 per-tenant p50/p95 TTFT and e2e latency, throttled time, with vs without pacing |
| Q4 | How much does bursting improve tenant latency and replica utilization, and does revoking bursts ever hurt guaranteed traffic? | §10 bursting-on vs bursting-off runs (same seed): latency, burst share, revocations, floor violations, KV/GPU utilization |

---

## 2. Phase-1 environment (measured 2026-10-29)

| Item | Finding | Implication |
|------|---------|-------------|
| GPU | RTX 2080 Ti, 11264 MiB, Turing (SM 7.5), driver 610.88, CUDA 13.3 UMD, WDDM mode | fp16 only; **~1.9 GB already used by desktop apps** (Chrome, LM Studio, Edge WebView) → ~9.3 GB free. vLLM refuses to start if `gpu_memory_utilization × total` exceeds free memory, so we size from *free* memory and close LM Studio during runs. |
| WSL2 | Debian12 (default, running), kernel 5.15.167.4; `nvidia-smi` works inside WSL2 | GPU passthrough into WSL2 is confirmed. |
| Docker | Docker Desktop **4.2.0 (Nov 2021)**, engine 20.10.10, compose v2.1.1 | Five years old. GPU reservations in compose and the WSL2 nvidia runtime have changed since. |
| WSL VM | 15.6 GiB RAM / 24 CPUs (default = half of 32 GB host), no `.wslconfig` | Enough for a 3B/7B AWQ model. Add `.wslconfig` with `memory=24GB` if model load or compile swaps. |
| Host | Ryzen 9 3900X, 32 GB RAM, 452 GB free disk; Python 3.11.3 and 3.14 on PATH, no `uv` | Dev tooling: install `uv`, use Python 3.12 in containers. |

**Runtime decision (Docker vs WSL2).** Measured, not assumed: Docker Desktop 4.2.0 still passes the
GPU through (`docker run --gpus all nvidia/cuda:13.0.1-base` sees the 2080 Ti), and its compose v2.1.1
accepts the GPU reservation and health-gated `depends_on` (it only rejects the top-level `name:` key).
The WSL2 distro turned out to be a bare Debian 13 with no Python, compiler or passwordless sudo, so the
native path would need the user to install packages. Decision: **everything runs in compose**, vLLM
from the official `vllm/vllm-openai:v0.32.0` image on the `gpu` profile, the fake replica on the `fake`
profile, one `VLLM_URL` either way. Upgrading Docker Desktop is optional.

One real obstacle, measured: `nvidia-smi` worked in containers but `cuInit` failed with
"Error 500: named symbol not found". Windows drivers 555+ ship `libnvdxgdmal.so.1` in the WSL driver
store and NVIDIA container toolkits older than 1.14.4 (Docker Desktop 4.2's is from 2021) do not
mount it. `scripts/vllm-entrypoint.sh` plus two read-only bind mounts of `/usr/lib/wsl/{lib,drivers}`
in `compose.yaml` fix it (torch then sees the 2080 Ti with ~10 GB free); on a normal Linux host the
script is a plain `vllm serve`. Upgrading Docker Desktop would remove the need for it.

### 2.1 vLLM on Turing (SM 7.5) — verification

Web research on 2026-10-29 against vLLM source, docs and issues; sources and verification levels are in
[VLLM_TURING_VERIFICATION.md](VLLM_TURING_VERIFICATION.md). **Verdict: viable with specific flags, with a
performance caveat.** Nobody has publicly confirmed this exact combination (Turing + Docker Desktop +
v0.32.0), so M1 starts with a smoke test.

| Topic | Finding | What we do |
|-------|---------|------------|
| Release | vLLM **v0.32.0** (18 Oct 2026). Image `vllm/vllm-openai:v0.32.0` = CUDA 13.0, torch 2.13, Python 3.12 | Pin the version in compose and in the WSL2 script |
| SM 7.5 support | Docs minimum is CC 7.5 (Volta dropped); Dockerfile builds `7.5 8.0 …`; torch 2.13 cu130 wheels include sm_75; no deprecation notice found | OK as-is |
| Host driver | CUDA 13 images need an R580+ Windows driver; we have 610.88 | OK, no `-cu129` tag needed |
| Attention backend | FlashAttention needs SM80; FlashInfer is temporarily gated off on SM75 (broken); V1 auto-selects **TRITON_ATTN**. Triton has no tensor-core MMA on SM<80 → on a T4, prefill is 6–15× slower than FlashInfer and decode 1.2–3× slower (issue #60821; fix PR #55380 unmerged) | Accept in phase 1 and measure (calibration captures it; phase 2 GPUs are Ampere+). Escape hatch: pin `v0.23.0`, the last image where FlashInfer + fp8 KV worked on SM75 — see Q12 |
| Quantization | AWQ and GPTQ supported on Turing; AWQ-GEMM checkpoints run through the Marlin sm75 kernel (merged Dec 2025), fp16 activations | Use AWQ checkpoints |
| KV / dtype | fp8 KV cache rejected by TRITON_ATTN below SM89; bf16 unsupported (auto falls back to fp16) | fp16 KV, `--dtype half`; never pass `--kv-cache-dtype fp8` (A6 confirmed) |
| CUDA graphs / compile | No capability gate; some T4 users run `--enforce-eager` | Default on; `--enforce-eager` is the first fallback |
| Memory | V1 aborts if free VRAM at start < `utilization × total` | `--gpu-memory-utilization 0.80–0.85`; close LM Studio |
| Docker Desktop + WSL2 | `--gpus all` only (no toolkit inside WSL); `--ipc=host`; vLLM disables pinned host memory on WSL by default | compose: `gpus: all`, `ipc: host`, HF cache volume |
| Models | `Qwen/Qwen2.5-7B-Instruct-AWQ`: Apache-2.0, 5.57 GB. `Qwen/Qwen2.5-3B-Instruct-AWQ`: **Qwen Research licence**, 2.69 GB. Both ungated, AWQ 4-bit g128, fp16 | Licence is part of Q6 |

Proposed M1 server flags:
`--model Qwen/Qwen2.5-3B-Instruct-AWQ --dtype half --max-model-len 16384 --max-num-seqs 32 --gpu-memory-utilization 0.82`
(`--max-model-len` must cover the largest *burst* context, not just the largest shape).

---

## 3. Definitions

| Term | Definition |
|------|------------|
| **Shape (SKU)** | `{name, model, max_context, tps, price_per_second, lazy}`; name = `model.ctx.tps[.lazy]`, e.g. `small.8k.30tps`. A per-request execution contract: prompt + completion ≤ `max_context`, output delivered at ≥ `tps` tokens/s, priced per wall-clock second of runtime. `tps` is per request (one stream); a tenant wanting more throughput runs more requests. |
| **Lazy shape** | A shape with `lazy: true` (the "batched instance type"): lower `price_per_second`, long queue tolerance (`max_queue_s`, e.g. 3600), lowest admission priority, admitted only into capacity no interactive request is waiting for, and limited to `lazy_max_share` of a replica's seats. `lazy_preempt = never | requeue` decides whether an interactive request may evict a running lazy one (evicted runs are not billed, like spot instances). Same OpenAI API; nothing special on the wire. |
| **Request** | Runs against one shape: the tenant's default shape (from the API key) or an explicit `shape` field in the request body. Lifecycle: arrival → queue → admitted (resources held) → running (prefill, decode, delivery) → done. Billed in shape-seconds (§8). |
| **Delivery mode** | `stream: true` sends tokens as SSE at the shape's rate; `stream: false` returns one complete response, released at the moment the paced stream would have delivered its last token. Same timing and billing either way; only the wire format differs. KV and seat are freed when the **engine** finishes, which can be before delivery completes. |
| **Slot** | The provider's capacity unit: the resources one request of a shape holds while running, `{kv: ceil(max_context / block_size) × block_size, seat: 1 at step rate ≥ tps}`. A replica's **slot plan** (e.g. "12 × small.8k.30tps + 2 × small.16k.30tps") is how sellable capacity and fill rate are counted; admission itself is by resources, which is what lets a smaller shape run in a bigger slot. |
| **Guaranteed traffic** | Tokens delivered within the shape (context ≤ `max_context`, rate ≤ `tps`). |
| **Burst traffic** | Tokens delivered above the shape's `tps`, or requests above `max_context` ("context burst"). Free, revocable. |
| **Replica** | One vLLM server (or an in-process fake) with a KV budget (tokens) and a calibrated throughput profile. |
| **Fill rate** | Occupied slot-seconds ÷ sellable slot-seconds (replica-seconds × slots in the plan); reported next to utilization (tokens served ÷ tokens the plan could serve). |

Shape nesting, concretely: with resource-based admission a `small.8k.30tps` request fits wherever 8k
of KV and one seat at ≥ 30 tps are free, which includes an idle `small.16k.30tps` slot. A
`small.8k.60tps` request needs a seat at ≥ 60 tps, so it is admitted only while the replica's running
count stays within `n_max(60)` (§5.2); the fastest running shape sets the concurrency ceiling.

---

## 4. Architecture

```
                 +---------------------------- gateway (one Python process) ----------------------------+
  tenants  --->  | auth -> shape -> shape check -> admission/queue -> pacer (token bucket) -> stream   |
  (loadgen)      |                                    |                    ^                             |
                 |                               scheduler <-- replica ledgers (KV, seats, tps)         |
                 |                                    |              burst manager (lends tps / ctx)     |
                 |                                 meter  <---------- events (SQLite per run) ----------|
                 +-------------+------------------------------+-----------------------------------------+
                               | OpenAI API + /metrics        |
                        +------v------+                +------v------+        +-------------+
                        | replica 0   |                | replica 1   |  ...   | fake replica|
                        | vLLM (GPU)  |                | vLLM (pod)  |        | (in-process)|
                        +-------------+                +-------------+        +-------------+
```

One gateway process is the control plane (queues, ledgers, buckets, meter). Replicas are remote HTTP
endpoints; adding a RunPod pod in phase 2 is a line in `replicas.yaml`.

### 4.1 Request path

1. `POST /v1/chat/completions` (also `/v1/completions`, `/v1/models`) with `Authorization: Bearer <key>`.
2. **Auth**: key → tenant, default shape, optional `max_concurrent` per key.
3. **Shape**: `shape` field in the body overrides the default; unknown shape → `400`.
4. **Shape check**: `prompt_tokens` (gateway tokenizes with the model tokenizer) + `max_tokens` ≤ `max_context`
   → guaranteed context; otherwise burst context if the burst manager finds free KV, else `429 context_exceeds_shape`.
5. **Admission**: `scheduler.admit(shape, kv_needed)` picks a replica whose ledger fits (§5.2 and §5.4).
   No fit → wait in the priority queue (interactive before lazy, FIFO within class) up to the shape's
   `max_queue_s`, then `503 capacity` + `Retry-After` (metered as a refusal). Queue time is never billed.
6. **Forward** to the replica (httpx streaming), with vLLM request `priority` (guaranteed=0, burst=10)
   so the engine's own queue also prefers guaranteed work (if the vLLM version supports priority scheduling).
7. **Pace**: upstream tokens go into a per-request buffer; the pacer releases them at `tps` plus whatever
   burst rate is currently lent. Each released token is tagged `guaranteed` or `burst`; time with a
   non-empty buffer and an empty bucket is **throttled time**. Streamed requests get each token as it is
   released; complete requests get the whole response when the last token is released.
8. **Finish/disconnect**: release the slot when the engine finishes, meter the request when delivery
   finishes; abort upstream if the client went away.
9. **Lazy preemption** (policy `requeue` only): when an interactive request would otherwise wait and a
   lazy request holds the seat it needs, the lazy request is aborted upstream, its run is discarded
   unbilled, and it goes back to the head of the lazy queue.

### 4.2 Replica interface

```python
class Replica(Protocol):
    id: str
    model: str
    kv_tokens_total: int                 # num_gpu_blocks * block_size, read at startup
    block_size: int
    profile: ThroughputProfile           # calibrated step-rate curve (section 5.2)
    async def health(self) -> ReplicaHealth   # kv usage %, running, waiting, preemptions, from /metrics
    async def stream(self, req: ChatRequest, priority: int) -> AsyncIterator[TokenChunk]
    async def abort(self, request_id: str) -> None
```

Implementations: `VllmReplica` (HTTP) and `FakeReplica` (in-process; configurable KV total and
step-rate curve, emits filler tokens at the computed rate). Fakes power unit tests and local
multi-replica scheduling tests.

---

## 5. Capacity model — what makes the shapes honest

### 5.1 KV cache sizing

Per-token KV bytes = `2 (K and V) × n_layers × n_kv_heads × head_dim × 2 bytes (fp16)`.
vLLM allocates `gpu_memory_utilization × VRAM − weights − profiled activation peak` to KV in
blocks of `block_size` tokens and reports `num_gpu_blocks` (startup log "GPU KV cache size: N tokens",
and the `vllm:cache_config_info` metric). The gateway reads `kv_tokens_total = num_gpu_blocks × block_size`
from the replica, never from a formula.

Candidate models (AWQ int4, fp16 KV). Budget assumes ~9.2 GB usable (`gpu_memory_utilization≈0.82`
with desktop apps running) minus ~0.9 GB activations/CUDA graphs:

| Model | layers × kv_heads × head_dim | KV bytes/token | Weights (AWQ) | ≈KV budget | ≈KV tokens | 8k slots by KV |
|-------|------------------------------|----------------|---------------|------------|------------|----------------|
| Qwen2.5-3B-Instruct-AWQ | 36 × 2 × 128 | 36 KiB | 2.69 GB on disk (tied embeddings) | ~5.6 GB | ~155k | ~19 |
| Qwen2.5-7B-Instruct-AWQ | 28 × 4 × 128 | 56 KiB | 5.57 GB on disk (untied 152k-vocab head) | ~2.7 GB | ~49k | ~6 |
| Llama-3.2-3B-Instruct (AWQ) | 28 × 8 × 128 | 112 KiB | ~2.3 GB | ~6.0 GB | ~55k | ~6 |

Recommendation: **Qwen2.5-3B-Instruct-AWQ** for phase 1 (2 KV heads → cheapest KV per token →
enough slots per replica for packing to be interesting). 7B is a config switch, not a code change.
These are estimates; M1 replaces them with measured `num_gpu_blocks`.

**KV rule (admission):** a running request holds `ceil(max_context / block_size) × block_size` KV
tokens of its shape, for guaranteed context; a context burst holds its actual rounded
length. Admit only if `Σ held ≤ kv_tokens_total × (1 − kv_margin)`, `kv_margin = 0.05`. KV is never
overcommitted.

### 5.2 Throughput calibration and seats

In a continuous-batching engine every running sequence advances one token per decode step, so
per-stream output tps = step rate, and step rate falls as the batch grows. We cannot make one
sequence slower than another inside vLLM; we can only bound **how many** sequences run.
`scripts/calibrate.py` measures, per replica, `step_rate[n]` for n in {1, 2, 4, 8, 12, 16, 24, 32}
concurrent streams at two context lengths (≈2k and ≈6k) and writes `calibration/<replica>.yaml`.

**Seat rule (admission):** `n_max(t) = max n such that step_rate[n] × (1 − tps_margin) ≥ t`. A replica
admits a request of shape tps `t` only if, afterwards, `n_running ≤ n_max(max tps among running requests
including this one)`. The fastest running shape sets the ceiling, which is the price of honouring it.

**Overcommit (the vCPU lever).** `tps_overcommit` (default 1.0 = strict) allows
`n_running ≤ n_max × tps_overcommit`. Above 1.0 the provider sells more floors than the replica can
honour if every seat streams at once; as with vCPU overcommit the cost is "steal" (floor violations),
which the report plots against the ratio.

**Feasible shapes are a calibration output.** A single request cannot exceed `step_rate[1]`, so a shape
with `tps > step_rate[1] × (1 − tps_margin)` cannot be sold on that replica; the gateway refuses to
route it there. On this card with a 3B model, shapes in the 15–60 tps range are plausible; `100tps`
and `300tps` shapes belong to phase-2 GPUs (or to tenant-aggregate bundles, see Q-B).

### 5.3 Assumptions (each is a config knob or a documented limit)

- **A1** KV is held worst case for the request's shape (`max_context`), not its actual length, so a short request in a big shape still pays the shape's KV. No KV overcommit.
- **A2** Prefix caching stays enabled (it helps growing-context profiles) but the budget never counts on it; cached blocks are evictable and count as free in vLLM too.
- **A3** `kv_margin` (5%) covers block rounding and transient allocations; `tps_margin` (10%) covers calibration noise.
- **A4** The shape's tps applies to steady-state decode. Prefill of long prompts (chunked prefill) slows concurrent decode steps. **Measured on the 2080 Ti (ASSUMPTIONS.md): a 6k-token prefill takes 4.6 s and drops a concurrently decoding stream from 130 to 1.7 tokens/s for that time.** Phase 1 reports the resulting floor violations; M3 measures the mitigations (smaller `--max-num-batched-tokens`, a per-replica cap on concurrent prefill tokens at admission, vLLM priority scheduling) and picks one.
- **A5** Pacing happens at the gateway (delivery), not inside the engine (§6). GPU work per token is the same with or without pacing; pacing changes *when* tokens are delivered and therefore the closed-loop arrival rate.
- **A6** fp16 KV cache; an fp8 KV cache is not assumed to work on Turing.
- **A7** One model per replica; shapes name the model; no cross-model packing.
- **A8** A request stays on its replica for its lifetime. No live migration or rebalancing in the PoC.
- **A9** vLLM's own preemption counter must stay 0 during guaranteed-only runs. If it does not, A1/A3 are wrong and the report flags it.

### 5.4 Scheduler (bin-packing with shape affinity)

This is where "optimize the requests intelligently" lives, and it is pure policy over the ledgers:

1. **Fit**: candidate replicas are those passing health, with free KV for the shape, a seat under
   `n_max(max tps)` (§5.2) and, for lazy shapes, under `lazy_max_share`.
2. **Affinity**: because the fastest running shape caps a replica's seat count, mixing a `60tps` request
   into a replica full of `15tps` lazy work shrinks that replica's capacity. Candidates are scored by the
   capacity they lose if this request is placed there (`n_max(before) − n_max(after)`), then by load
   (`Σ held KV / kv_total`); lowest score wins. With one replica this degenerates to plain admission;
   with several it naturally sorts fast interactive shapes and slow lazy shapes onto different replicas.
3. **Filler**: lazy requests are admitted only when no interactive request is waiting for that replica,
   so idle capacity is filled without delaying paying interactive traffic.

Fake replicas make all of this testable without a second GPU.

---

## 6. Pacing and throttling

Per request: a token bucket with rate `tps` and depth `tps × bucket_depth_s` (default 1 s, so a fresh
request can deliver one second of tokens immediately). The burst manager may add a second, revocable
rate on top (§7).

The gateway consumes the upstream stream as fast as vLLM produces it, buffers, and releases to the
client at bucket pace. Consequences, stated plainly:

- The engine still generates at its natural step rate; the tenant sees exactly the shape's rate (or more, when burst is lent).
- Buffered-but-undelivered tokens on client disconnect are wasted GPU work (metered as `wasted_tokens`).
- The shape's rate is protected by admission (§5.2), not by the bucket.
- Complete (non-streamed) requests run through the same bucket virtually: the response is released when
  the bucket would have delivered the last token, so a `8k.15tps` summary of 300 tokens returns ~20 s
  after its first token unless burst is lent, exactly like its streamed twin. Without this, every
  non-streamed request would be an unmetered burst.
- Mixed types are the real test of the guarantees: a 8k-token prefill for a summary slows the decode
  steps of concurrent chat streams (A4). The report splits every metric by delivery mode and measures
  decode step rate while long prefills are in flight.

Implementation (milestone 2, `slotgw/pacing.py`): the bucket is pure arithmetic over timestamps,
`schedule(n, t_now) -> t_deliver`. The stream path runs a producer task that drains the engine into
an unbounded buffer and a consumer that sleeps until each chunk's `t_deliver`; the complete path
runs every chunk through the same arithmetic and releases the response at the virtual last-token
time. **Throttled time** is the wall-clock time delivery was held by the bucket beyond
`max(chunk arrival, previous delivery)`: zero when the engine is slower than the shape, the engine's
lead otherwise. The bucket starts full, so the first `tps × bucket_depth_s` tokens leave as fast
as the engine makes them (a "head start"); the report judges the delivered rate over the whole
stream (head start included, so it can only help) against the shape's tps. Knobs: `config/gateway.yaml` `pacing.enabled` (false = milestone-1 passthrough) and
`pacing.bucket_depth_s` (default 1.0). On client disconnect the producer is cancelled, which aborts
the upstream request; buffered tokens are `wasted_tokens`.

Alternative considered: **chunked generation** (request `max_tokens=k` repeatedly, relying on prefix
caching) would make GPU time itself proportional to the shape's rate. It costs a re-prefill per chunk and
breaks stop sequences/sampling continuity. Not proposed for the PoC; noted as a phase-2 experiment.

---

## 7. Burst manager

Per replica, a **ledger** of three lendable resources:

| Resource | Total | Held by shapes | Lendable to bursts |
|----------|-------|----------------|--------------------|
| KV tokens | `kv_total × (1 − kv_margin)` | `Σ shape KV of running requests` | remainder |
| Seats | `n_max(fastest running shape) × tps_overcommit` | `n_running` | remainder (context bursts need a seat too) |
| Output tps | `step_rate[n_running] × n_running` | `Σ tps` of running requests | remainder, shared max-min fair across requests wanting more |

Priority classes, highest first: **guaranteed interactive** (within a non-lazy shape), **guaranteed
lazy** (within a lazy shape), **burst** (above any shape). Revocation goes in the opposite order: burst
rate first, then burst context, then (policy `requeue` only) lazy seats. Policy, in order of cost:

1. **tps burst** — lent rate recomputed on every ledger change and on a 250 ms tick; revocation is
   instant (next tick) and costs nothing, because the engine was generating at that speed anyway.
   Under wall-clock billing (§8) a tps burst shortens the tenant's runtime and therefore its bill; the
   report records that "burst discount" as revenue the provider gave away for latency.
2. **context burst** (request over `max_context`) — admitted only from the unreserved remainder, sent with
   low vLLM priority. **Proposed revocation policy: stop admitting new bursts the moment a guaranteed
   admission needs the space; let in-flight bursts finish up to `burst_grace_s` (default 15 s); abort
   after that.** Rationale: guaranteed requests' KV is held at admission, so an in-flight burst never
   endangers a running request's shape; the only thing it can delay is admitting a *new* request on that
   replica, which the scheduler can place elsewhere or queue briefly. Aborting wastes finished work.
   `burst.revoke_policy = finish | grace | abort` is a config knob so the report can compare all three.

Metrics: burst tokens by kind, revocations (tps throttled, bursts refused, bursts aborted), guaranteed
admissions delayed by bursts, and **floor violations** (any 1 s window where a guaranteed request with a
non-empty buffer received < `tps`). `burst.enabled=false` turns the whole module off.

---

## 8. Meter and billing

SQLite file per run (`runs/<run_id>/meter.sqlite`), exported to CSV. One table:

`requests(run_id, tenant, request_id, shape, lazy (bool), replica, delivery (stream|complete),
class_ctx (guaranteed|burst), t_arrival, t_admitted, t_first_token, t_engine_done, t_last_token,
status, prompt_tokens, output_tokens, output_tokens_guaranteed, output_tokens_burst, throttled_ms,
queued_ms, wasted_tokens, ttft_ms, e2e_ms, revoked (bool), preempted_runs, abort_reason)`

One run prices two ways from the same rows:

| Scheme | Quantity | Price |
|--------|----------|-------|
| Per-token (reference) | `prompt_tokens`, `output_tokens`; lazy requests get the reference batch discount | `$/1M input`, `$/1M output`, `batch_discount` |
| Shape (wall-clock) | **billed seconds** = `min(t_last_token − t_admitted, (t_first_token − t_admitted) + output_tokens ÷ shape.tps)` | `shape.price_per_second` |

Billed runtime is wall-clock from admission to the last token delivered (or the complete response
released), **excluding queue time**, and **capped at the shape's promise**: if the engine misses the
guaranteed rate the customer never pays for the provider's slowness; if burst makes it faster the bill
shrinks. That cap is what makes the bill predictable (an upper bound known in advance) while keeping the
quantity honest. The report also shows the uncapped wall-clock and the bound so the gap is visible.
Preempted lazy runs are discarded and unbilled; only the completed run counts.

Placeholder shape prices are derived from the per-token reference so that `$/output token at the
guaranteed rate` (= `price_per_second ÷ tps`) matches the reference for interactive shapes and the
discounted reference for lazy shapes; `pricing.yaml` can override every number.

Every response carries the bill it would get under both schemes in its `slotgw.billing` object
(`billed_s`, `bound_s`, `wall_s`, `shape_usd`, `bound_usd`, `per_token_usd`, `delivered_tokens`), and
`slotgw.throttled_ms`, so a customer can audit a request without the meter (`slotgw/billing.py`).
The report recomputes the same numbers from the meter rows with the config snapshot of the run.

Token classification: each delivered token draws from the guaranteed bucket first; if that bucket is
empty and burst rate is lent, the token is burst. Prompt tokens are guaranteed if the request is within
shape, else burst.

---

## 9. Load generator

Python asyncio client of the gateway. One scenario YAML = tenants (API keys, default shape, profile,
arrival process) + a timeline. Reproducible: `--seed` feeds a master `numpy`
Generator; each tenant derives a child seed, so adding a tenant does not perturb others.

| Profile | Prompt tokens | Output tokens | Delivery | Pattern | Shape choice (tenant's) |
|---------|---------------|---------------|----------|---------|-------------------------|
| Note-taker | 3k–8k transcript | 150–400 | complete | clustered at "hour marks" (Gaussian σ≈2 min around each mark), some mid-hour | `16k.30tps` (or `8k` + context bursts, a scenario variant) |
| Agentic loop | starts 1–2k, grows 0.5–1.5k per step, 5–15 sequential steps | 100–400 | streamed | sessions Poisson; 1–5 s tool pause between steps | `8k.60tps` (latency-sensitive, fast shape) |
| Chat | 200–800 growing per turn, 3–10 turns | 50–300 | streamed | session arrivals diurnal (work-hours phases), think time 5–30 s | `8k.30tps` |
| Batch | 500–2k | 200–600 | complete | closed loop, fixed concurrency K, no latency target | `8k.15tps.lazy` (cheapest; queued until capacity is idle) |

Shape choice is where the pricing model differentiates: latency-insensitive work buys the slow lazy
shape and leaves fast seats to others. A note-taker variant on `16k.30tps.lazy` is a scenario too, since
summaries rarely need to be instant.

Arrival processes: `poisson(rate)`, `bursty` (on/off Markov-modulated Poisson), `diurnal` (rate
schedule over the timeline). Time is **not** compressed for generation; instead the timeline is
built from phases (e.g. quiet 3 min → ramp 2 min → peak 8 min with an "hour mark" → quiet 5 min)
and `hour_length_s` (default 600) defines where note-taker spikes land. "Days" = the same scenario
with different seeds.

Content: prompts are synthetic text sized in tokens; outputs use `ignore_eos=true` and `max_tokens`
so output lengths are exactly the sampled values (reproducible load, independent of what the model
says). Prompt text comes from a vocabulary of common English words that are single tokens in
GPT-style BPE tokenizers (one token per word, one per sentence period, `loadgen/content.py`); the
load generator measures the real ratio and the chat-template overhead through the gateway's
`/admin/tokenize` at the start of every run and records them in `loadrun.json` (measured values in
ASSUMPTIONS.md). Multi-turn profiles send the real conversation history (previous user turns and
the model's actual replies), so prefix caching sees realistic prefixes and the context grows as it
would for a real client.

Reproducibility is a **plan**: `loadgen/plan.py` turns scenario + seed into every session, arrival
offset, step size, pause and content seed before the first request (written to `plan.json` with a
digest). Execution follows the plan; only wall-clock send times of dependent steps vary with server
latency. The gateway records the load run, scenario and seed from request headers
(`X-Slotgw-Run/Scenario/Seed`), so one gateway run holds several seeded load runs under
`runs/<gateway_run>/loads/<load_run>/` (plan, `client.csv` with client-observed timings, `samples.csv`
with the engine's running count and KV usage sampled through the gateway, a config snapshot).

A session ends at the first refusal or error, as a real client's would. With the agent profile as
specified (context growing 0.5–1.5k per step over 5–15 steps) most agent sessions outgrow an 8k
shape around step 5, so milestone 2 reports `429 context_exceeds_shape` refusals for that tenant;
milestone 4's context bursts are the fix under test, and a scenario variant with smaller growth is
one line in the YAML.

---

## 10. Report

`report/build.py runs/<run_id>...` → `per_request.csv`, `per_tenant.csv`, `summary.csv`,
`across_seeds.csv`, `interference.csv` (`per_replica.csv` from milestone 3) and `report.html`
(self-contained: charts are inline SVG generated by `report/charts.py`, no JavaScript and no vendored
library, light and dark themes). `pricing.yaml` holds reference per-token prices (input/output per
1M, batch discount), `price_per_second` per shape, and `$/GPU-hour` per replica.

Per tenant and run:
- **Predictability**: bill under per-token and under shapes (billed, uncapped wall-clock, and the shape bound); effective $/1M output tokens under each; coefficient of variation of each bill across seeds ("days"); floor-violation time (steal); burst discount (bound − billed).
- **Token output per capacity**: tokens per billed second vs the shape's nominal `tps` (burst uplift).
- **Experience**: p50/p95 TTFT and e2e latency, throttled time (total and % of streaming time), queue time, refusals (`503`/`429`), revocations, lazy preemptions and the tokens they wasted. Every metric is also split by delivery mode (streamed vs complete) and by lazy vs interactive.
- **Interference**: decode step rate of streamed requests while long prefills (complete-mode summaries) are in flight, vs without them.

Per replica: KV usage (mean/p95 from `/metrics`), KV held vs used, seats in use, step rate, GPU
utilization (nvidia-smi sampler), preemptions, fill rate (slot-plan occupancy) vs utilization (tokens
served ÷ plan capacity), achieved overcommit ratio, tokens served per GPU-hour, shape mix.

Provider economics: revenue under each scheme, GPU cost, analytic break-even fill rate
`f* = gpu_$/h ÷ (Σ_slots price_per_second × 3600)` for the slot plan, and the fill rate at which shape
revenue overtakes per-token revenue for the measured workload mix, per overcommit setting.

---

## 11. Milestones

| M | Deliverable | Acceptance |
|---|-------------|------------|
| **M1** vLLM + gateway passthrough + meter | Step 0: smoke test of pinned vLLM on the 2080 Ti via the WSL2 script; `docker compose up` starts the gateway (and vLLM when Docker Desktop is upgraded); `/v1/chat/completions` passes through in both `stream: true` and `stream: false` modes with API-key→tenant/shape mapping; meter writes requests with token counts, timings and delivery mode; `scripts/calibrate.py` produces the KV numbers and step-rate curve and the feasible-shape list; `docs/ASSUMPTIONS.md` records measured values | curl through gateway streams; `meter.sqlite` token counts match vLLM `usage`; calibration YAML exists; unit tests for auth + meter pass |
| **M2** token bucket + loadgen + report v1 | per-request pacing with throttled-time accounting; 4 profiles + 3 arrival processes, seeded; report with per-token and shape bills (billed, uncapped, bound), variance across seeds, latency, throttle time | two runs with the same seed produce identical request sequences; pacing holds delivered tps within ±5% of the shape rate; report HTML renders from CSV |
| **M3** admission + multi-replica scheduler | `Replica` interface, `VllmReplica`, `FakeReplica`; KV/seat ledgers with shape nesting; priority queues and `503`; lazy shapes with `lazy_max_share` and the `requeue` policy; shape-affinity scoring; overcommit knob; replica metrics, fill rate and break-even in report | tests with 1 real + 3 fake replicas: admission never exceeds a ledger; a small shape is placed into a larger free slot; fast and lazy shapes separate across replicas; lazy work never delays a waiting interactive request; vLLM preemptions stay 0 under guaranteed-only load |
| **M4** burst manager | tps and context bursts, priority classes, revocation policies, bursting-on vs off runs in report; written answers to Q1–Q4 in `docs/RESULTS.md` | revocation under injected guaranteed load restores shape rates within one tick; report shows burst share, revocations, floor violations, utilization with/without bursting |

Each milestone ends with: tests green, `docker compose up` works from a clean checkout, docs updated.

---

## 12. Decisions (2026-10-29)

| Decision | Choice | Why |
|----------|--------|-----|
| Billing unit | Wall-clock runtime of the API call, admission → last token delivered, queue time excluded, capped at the shape's promise | User's call ("wall clock for API runtime"). The cap keeps the bound that makes bills predictable and stops the provider charging for its own slowness. |
| Shape tps | Per request, one stream at ≥ `tps`; more throughput = more requests | Simplest honest admission (one seat per request) and one bill per request. Intelligence goes into the scheduler (§5.4), not into the contract. |
| Reserved tier | Dropped | It was a "hold a slot and pay whether used or not" option, useful only for a flat-bill comparison; not part of the user's model. Can return later as a thin layer. |
| Batched work | Plain OpenAI API (`stream: false`) on **lazy shapes**: cheaper per second, lazily queued, filler for idle capacity, optional `requeue` preemption | User: keep OpenAI API shapes for apples-to-apples comparison; a special batched instance type with lower cost per second because it is lazily queued. Compared against the per-token reference *with* its batch discount. |
| Delivery modes | Streamed and complete, same pacing and billing | User: mixed types on one replica. |
| Runtime | Compose for everything; vLLM native in WSL2 for M1, compose GPU service once Docker Desktop is upgraded | User: whichever is easier. |
| Context accounting | prompt + completion per request; over-shape = context burst or `429` | Default, unobjected. |
| Pacing | Gateway delivery pacing; admission protects the rate | Default, unobjected (§6). |
| Bursts | tps + context, revocation refuse-new → 15 s grace → abort, no credits | Default, unobjected (§7). |
| Model | Qwen2.5-3B-Instruct-AWQ (Qwen Research licence; PoC use) | Most slots per replica (§5.1). |
| Prices | Placeholders derived from a per-token reference (§8), all overridable | No numbers given. |
| Runs | 15–30-minute phase-built scenarios, 10-minute "hour", 5 seeds as days | Default, unobjected. |
| Stack | Python 3.12, FastAPI, httpx, SQLite, pandas, Chart.js, `uv`, pytest, compose | Default, unobjected. |
| vLLM | v0.32.0, Triton attention accepted on this card; `v0.23.0` as escape hatch | §2.1. |
| Overcommit | `tps_overcommit` knob, runs at 1.0 / 1.5 / 2.0; KV never overcommitted | The vCPU lever (§5.2). |
| Git | Repository initialised at M1 start | Default, unobjected. |
| Report charts (M2) | Inline SVG generated in Python instead of Chart.js | Self-contained offline HTML without vendoring a JS library; the CSVs carry the numbers, native `<title>` tooltips carry hover values. |
| Bucket head start (M2) | `bucket_depth_s = 1.0`: the first second of tokens leaves unpaced | Per §6; a floor violation is a delivered rate below 95% of the promise over the whole stream, which the head start can only help, and the billing cap already bounds what the customer pays. |
| Agent context overflow (M2) | Sent as specified; the gateway refuses with 429 and the session ends | Keeps the profile honest to the brief and gives milestone 4's context bursts a measured baseline instead of hiding the problem in the load generator. |

---

## 13. Risks

- Turing gets the Triton attention backend without tensor-core MMA (§2.1): prefill of 3k–8k note-taker prompts will take seconds and will visibly slow concurrent decode (A4); feasible shapes on this card are slow. Calibration decides, names stay config, and the v0.23.0 escape hatch exists (Q12).
- No public report of vLLM v0.32.0 on Turing under Docker Desktop/WSL2; the M1 smoke test is the real confirmation. Fallbacks in order: `--enforce-eager`, the `v0.23.0` image, native WSL2 install.
- Desktop apps share the GPU (WDDM). Free memory varies run to run; the calibration script records it, and runs should close LM Studio.
- Prefill interference (A4) is not noise on this card, it is a pause: one long prompt stalls every running stream for seconds. Without a mitigation in M3 the 30/60 tps shapes are honest only while no long prompt is prefilling.
- Wall-clock billing needs honest timestamps: queue time must be excluded and the cap needs TTFT, so arrival, admission, first-token, engine-done and delivered are separate meter columns.
- Prefill is GPU time that the placeholder shape prices do not cover: `price_per_second` was derived from the output-token reference, so a long prompt bills only its TTFT seconds at the decode rate. Measured in the milestone-2 GPU run (ASSUMPTIONS.md): 1.9M prompt tokens vs 140k output tokens, shape revenue a quarter of per-token revenue, roughly GPU cost. Milestone 3 must price prefill (a prompt-token component, or a price per second that includes the measured prefill share) before the schemes are compared on provider revenue.
- Under wall-clock billing every tps burst is revenue given away; the report quantifies it so the provider can decide how much latency goodwill to buy.
- Python gateway overhead at a few hundred tokens/s across tens of streams is fine; at phase-2 scale (3–4 replicas) still fine; beyond that a Go rewrite would be the answer.

---

## 14. Repository layout and stack

```
vLLM/
  compose.yaml                 # vllm (gpu profile), gateway, fake replicas (fake profile), loadgen/report as run-once services
  config/{shapes,tenants,replicas,pricing,gateway}.yaml, config/scenarios/*.yaml
  calibration/<replica>.yaml   # measured KV + step-rate curves + feasible shapes
  slotgw/                      # gateway: api.py, config.py, pacing.py, billing.py, meter.py, clock.py, replicas/
  loadgen/                     # scenario.py, arrivals.py, plan.py, content.py, runner.py, main.py
  report/                      # metrics.py, charts.py (inline SVG), html.py, build.py
  scripts/calibrate.py
  tests/                       # unit + integration (gateway and fake replica as real uvicorn servers)
  runs/<run_id>/               # meter.sqlite, loads/<load_run>/ (plan, client.csv, samples.csv, config), report/
  docs/DESIGN.md (design + decisions), ASSUMPTIONS.md (measured), RESULTS.md, VLLM_TURING_VERIFICATION.md
```

Stack: Python 3.12, FastAPI/uvicorn, httpx, pydantic (config), numpy (RNG), pandas (report),
tokenizers (shape check), pytest + pytest-asyncio, ruff. Everything runs with `docker compose up`
(profiles `gpu` / `fake`), plus `docker compose run --rm loadgen --scenario X --seed N` and
`docker compose run --rm report runs/...` (both gateways answer to the network alias `gateway`).
