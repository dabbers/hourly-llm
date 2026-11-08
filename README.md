# slotgw — time-sliced LLM inference billing PoC

Sell inference as fixed **shapes** (`small.8k.30tps` = up to 8k context, at least 30 output
tokens/s) billed per second a request runs, pack mixed shapes onto shared vLLM replicas, lend idle
capacity as burst, and measure whether that beats per-token billing. Design, decisions and
assumptions: [docs/DESIGN.md](docs/DESIGN.md). Measured numbers: [docs/ASSUMPTIONS.md](docs/ASSUMPTIONS.md).
vLLM-on-Turing verification: [docs/VLLM_TURING_VERIFICATION.md](docs/VLLM_TURING_VERIFICATION.md).

## Run

GPU (local RTX 2080 Ti via Docker Desktop, pulls `vllm/vllm-openai:v0.32.0` and the model once):

```bash
docker compose --profile gpu up --build -d
```

No GPU (fake replica with a step-rate curve that approximates the 2080 Ti):

```bash
docker compose --profile fake up --build -d
```

Then talk OpenAI to the gateway; the `model` field names the **shape**:

```bash
curl http://localhost:8080/v1/chat/completions -H "Authorization: Bearer sk-local-chat" -H "Content-Type: application/json" -d "{\"model\":\"small.8k.30tps\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":32,\"stream\":true}"
```

Tokens are delivered at the shape's rate (a one-second head start, then 30 tokens/s here), and every
response carries its bill under both schemes in `slotgw.billing`. Every request lands in
`runs/<run_id>/meter.sqlite` with the timestamps and token counts that price it per token and per
shape-second.

Generate load (one seed = one "day"; same seed = same request sequence) and build the report:

```bash
docker compose run --rm loadgen --scenario config/scenarios/mixed.yaml --seed 1
```

```bash
docker compose run --rm loadgen --scenario config/scenarios/mixed.yaml --seed 2
```

```bash
docker compose run --rm report runs/<run_id>
```

`<run_id>` is printed by the load generator (`gateway_run_id`) and by `GET /healthz`. The report
(`runs/<run_id>/report/report.html` plus CSVs) compares per-token and shape bills (billed, uncapped,
bound), their variance across seeds, latency, throttled time, floor violations, a timeline,
prefill interference, bursting (burst tokens, discount, revocations), reservations and their bills,
replica fill and break-even, market reference prices and margin, and a reservation what-if. `config/scenarios/smoke.yaml` is a two-minute version of the same workload.

Settings that are worth varying per run are environment overrides, so one run differs from the next
by a prefix, and `--label` keeps them apart in the report:

```bash
SLOTGW_TPS_OVERCOMMIT=1.5 docker compose --profile gpu up -d gateway-gpu
```

```bash
docker compose run --rm loadgen --scenario config/scenarios/mixed.yaml --seed 1 --label oc1.5
```

A scenario can give tenants standing reservations (`reservation: {slots: 4}` in `mixed.yaml`); the
load generator creates them through `POST /admin/reservations` at the start (or per named phase) and
releases them at the end. `--no-reservations` runs the same scenario on demand, so bursting and
reservations can be switched per run:

```bash
SLOTGW_BURST=false docker compose --profile gpu up -d gateway-gpu
```

```bash
docker compose run --rm loadgen --scenario config/scenarios/mixed.yaml --seed 1 --no-reservations --label burstoff
```

| Variable | Default | What |
|---|---|---|
| `SLOTGW_TPS_OVERCOMMIT` | 1.0 | seats = calibrated `n_max` × this (the vCPU lever) |
| `SLOTGW_LAZY_PREEMPT` | never | `requeue` lets an interactive request evict a running lazy (complete-mode) request |
| `SLOTGW_MAX_PREFILL_TOKENS` | 0 | cap on prompt tokens prefilling at once per replica; 0 = off |
| `SLOTGW_BURST` | true | burst manager: lend idle step rate to running requests and admit over-context requests into free KV |
| `SLOTGW_BURST_REVOKE` | grace | `finish` / `grace` (15 s) / `abort`: what happens to a context burst when a guaranteed request needs its capacity |
| `SLOTGW_BURST_CTX_CLASS` | slowest | a context burst runs as the model's slowest non-lazy shape (seat, pacing, floor, price); `own` keeps the request's shape (milestone-4 behaviour) |
| `SLOTGW_BURST_QUEUE_S` | 5 | how long a context burst waits for a seat before `503 capacity` (never longer than its shape's `max_queue_s`) |
| `SLOTGW_STORE_TEXT` | false | keep every completion (and showcase prompts) in the meter's `texts` table for the report's Output samples |
| `SLOTGW_PRICE_REFERENCE` | basket-3b | per-token market reference from `config/price_references.yaml` (`gpt-4o-mini`, `llama-3.2-3b`, …, or `custom`); shape prices derive from it |
| `SLOTGW_PROMPT_USD_PER_MILLION` | auto | prefill charge added to shape bills; auto = the reference's input price, 0 = runtime only |
| `VLLM_MAX_NUM_BATCHED_TOKENS` | 128 | vLLM prefill chunk size (vLLM's own default is 2048); smaller = less decode interference |
| `VLLM_GPU_MEM` | 0.82 | vLLM memory fraction (lower it when other apps hold VRAM) |

Stop with `docker compose --profile gpu down` (or `--profile fake`).

### Phase 2: three replicas on RunPod

The plan, decisions, budget and gates are in `docs/PHASE2.md`; the operator's commands in
`docs/RUNBOOK.md`. The phase-2 configuration lives in `config/phase2/` (three pod replicas, the 7B
price basket, text capture on) with the scenario `config/scenarios/mixed-4090.yaml` and the run
matrices `config/phase2/matrix-ondemand.yaml` and `matrix-resv.yaml`. Everything can be rehearsed without a pod on three fake
replicas with a 4090-like curve:

```bash
docker compose --profile fake3 up --build -d
```

```bash
docker compose run --rm loadgen --scenario config/scenarios/mixed-4090.yaml --seed 1 --config-dir config/phase2 --no-reservations
```

A scenario tenant can carry a `showcase:` spec (a file of real prompts under `config/prompts/`, one
session every few minutes); with text capture on, the report's Output samples section shows their
completions with the constraints they ran under. `reservation: {replica: ...}` pins a tenant's
reservation to one replica and `--reservation-replica ID|none` overrides it per run; a replica's
`exclude_shapes` keeps a shape off it (the "reservations on their own replica" setting).

## Develop

```bash
uv sync --extra dev
uv run pytest
uv run slotgw-fake-replica --port 8001       # fake vLLM on :8001
uv run slotgw --config-dir config            # gateway on :8080, expects VLLM_URL (default http://localhost:8000)
uv run slotgw-loadgen --scenario config/scenarios/smoke.yaml --seed 1 [--label tag]
uv run slotgw-report runs/<run_id> [runs/<other_run_id> ...] [--out DIR]
uv run slotgw-fetch-prices                     # refresh config/price_references.yaml from OpenRouter (no key)
uv run python scripts/calibrate.py --url http://localhost:8000 --replica-id local-2080ti
```

Knobs: `config/shapes.yaml` (catalog; prices derive from the active reference unless set),
`config/tenants.yaml` (API keys are local test values), `config/pricing.yaml` (active price reference,
GPU cost, margin targets, prefill charge), `config/price_references.yaml` (OpenRouter snapshot),
`config/gateway.yaml`
(pacing, admission margins, overcommit, lazy share and preemption, prefill cap, burst manager), `config/replicas.yaml`
(replica URLs, calibration curve, slot plan), `config/scenarios/*.yaml` (profiles, arrivals, phases).

## Layout

| Path | What |
|------|------|
| `slotgw/` | gateway: `api.py` (OpenAI surface), `admission.py` (ledgers, queues, scheduler), `calibration.py` (step-rate profiles), `pacing.py` (token bucket), `billing.py`, `prices.py` (market price snapshot), `meter.py`, `config.py`, `replicas/` (vLLM client, fake server) |
| `loadgen/` | `scenario.py`, `arrivals.py` (poisson, diurnal, bursty, hour marks), `plan.py` (seeded plan), `content.py`, `runner.py` |
| `report/` | `metrics.py`, `charts.py` (inline SVG), `html.py`, `build.py` |
| `config/` | shapes, tenants, replicas, pricing, gateway, scenarios |
| `calibration/` | measured KV budget and step-rate curves per replica |
| `scripts/calibrate.py` | produces the calibration file and the feasible-shape list |
| `runs/` | per-run meter databases, load-run side files, CSVs and reports (git-ignored) |
| `docs/` | design, decisions, assumptions, verification |

## Docker Desktop on WSL2

Windows NVIDIA drivers 555+ need `libnvdxgdmal.so.1` inside containers, and the container toolkit
bundled with older Docker Desktop releases (4.2 was tested) does not mount it: `nvidia-smi` works but
CUDA init fails with "named symbol not found". `compose.yaml` bind-mounts `/usr/lib/wsl/{lib,drivers}`
and `scripts/vllm-entrypoint.sh` puts them on the library path. On a Linux host, or with a current
Docker Desktop, that is a no-op.

## Status

Milestones 1–4 built and verified on the local GPU: vLLM + gateway + meter, per-request pacing +
seeded load generator + report, admission control + multi-replica scheduler + prefill-interference
mitigation (measured defaults: 128-token prefill chunks, overcommit 1.0, lazy requeue off), the
burst manager (tps and context bursts, revocation policies) and reservations as standing holds,
with bursting on/off and reservation runs in the report. The answers to the design questions are in
`docs/RESULTS.md`; the measurements behind them in `docs/ASSUMPTIONS.md` ("Milestone 4
measurements"); what the runs showed should change next (context bursts at the slowest tps class, a
lending reserve, a reservation price above parity) is in `docs/DESIGN.md` sections 11–13. Next:
phase 2 on three RunPod replicas, planned in `docs/PHASE2.md` (budget cap $150, stage gates, output
capture for the write-up); not started.
