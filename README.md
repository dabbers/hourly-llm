# slotgw — time-sliced LLM inference billing PoC

Sell inference as fixed **shapes** (`small.8k.30tps` = up to 8k context, at least 30 output
tokens/s) billed per second a request runs, pack mixed shapes onto shared vLLM replicas, lend idle
capacity as burst, and measure whether that beats per-token billing. Design, decisions and
assumptions: [docs/DESIGN.md](docs/DESIGN.md). vLLM-on-Turing verification:
[docs/VLLM_TURING_VERIFICATION.md](docs/VLLM_TURING_VERIFICATION.md).

## Run

GPU (local RTX 2080 Ti via Docker Desktop, pulls `vllm/vllm-openai:v0.32.0` and the model once):

```bash
docker compose --profile gpu up --build
```

No GPU (fake replica with a configurable step-rate curve):

```bash
docker compose --profile fake up --build
```

Then talk OpenAI to the gateway; the `model` field names the **shape**:

```bash
curl http://localhost:8080/v1/chat/completions -H "Authorization: Bearer sk-local-chat" -H "Content-Type: application/json" -d "{\"model\":\"small.8k.30tps\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":32,\"stream\":true}"
```

Every request lands in `runs/<run_id>/meter.sqlite` with the timestamps and token counts that price
it per token and per shape-second.

## Develop

```bash
uv sync --extra dev
uv run pytest
uv run slotgw --config-dir config            # gateway on :8080, expects VLLM_URL (default http://localhost:8000)
uv run slotgw-fake-replica --port 8001       # fake vLLM on :8001
uv run python scripts/calibrate.py --url http://localhost:8000 --replica-id local-2080ti
```

## Layout

| Path | What |
|------|------|
| `slotgw/` | gateway: `api.py` (OpenAI surface), `config.py`, `meter.py`, `replicas/` (vLLM client, fake server) |
| `config/` | shapes, tenants, replicas, pricing, scenarios |
| `calibration/` | measured KV budget and step-rate curves per replica |
| `scripts/calibrate.py` | produces the calibration file and the feasible-shape list |
| `loadgen/`, `report/` | milestone 2 |
| `runs/` | per-run meter databases, CSVs and reports (git-ignored) |
| `docs/` | design, decisions, assumptions, verification |

## Docker Desktop on WSL2

Windows NVIDIA drivers 555+ need `libnvdxgdmal.so.1` inside containers, and the container toolkit
bundled with older Docker Desktop releases (4.2 was tested) does not mount it: `nvidia-smi` works but
CUDA init fails with "named symbol not found". `compose.yaml` bind-mounts `/usr/lib/wsl/{lib,drivers}`
and `scripts/vllm-entrypoint.sh` puts them on the library path. On a Linux host, or with a current
Docker Desktop, that is a no-op.

## Status

Milestone 1 (vLLM + gateway passthrough + meter) built and verified on the local GPU. Measured
capacity numbers: [docs/ASSUMPTIONS.md](docs/ASSUMPTIONS.md). Next: milestone 2 (token bucket, load
generator, first report). See `docs/DESIGN.md` section 11.
