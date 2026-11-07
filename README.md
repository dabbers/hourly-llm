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
bound), their variance across seeds, latency, throttled time, floor violations, a timeline and
prefill interference. `config/scenarios/smoke.yaml` is a two-minute version of the same workload.

Stop with `docker compose --profile gpu down` (or `--profile fake`).

## Develop

```bash
uv sync --extra dev
uv run pytest
uv run slotgw-fake-replica --port 8001       # fake vLLM on :8001
uv run slotgw --config-dir config            # gateway on :8080, expects VLLM_URL (default http://localhost:8000)
uv run slotgw-loadgen --scenario config/scenarios/smoke.yaml --seed 1
uv run slotgw-report runs/<run_id>
uv run python scripts/calibrate.py --url http://localhost:8000 --replica-id local-2080ti
```

Knobs: `config/shapes.yaml` (catalog and prices), `config/tenants.yaml` (API keys are local test
values), `config/pricing.yaml` (per-token reference, GPU cost), `config/gateway.yaml` (pacing on/off,
bucket depth), `config/scenarios/*.yaml` (profiles, arrival processes, phases).

## Layout

| Path | What |
|------|------|
| `slotgw/` | gateway: `api.py` (OpenAI surface), `pacing.py` (token bucket), `billing.py`, `meter.py`, `config.py`, `replicas/` (vLLM client, fake server) |
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

Milestone 1 (vLLM + gateway passthrough + meter) and milestone 2 (per-request pacing, seeded load
generator, first report) built and verified on the local GPU. Next: milestone 3 (admission control,
multi-replica scheduler, prefill-interference mitigation). See `docs/DESIGN.md` section 11.
