#!/usr/bin/env bash
# Stage N on the local GPU (RESULTS Q18): the same scenario and seed under three gateway settings.
#   scripts/stage_n_local.sh "1 2 3" "n-fifo n-cap n-fair"
# vLLM must be up (docker compose --profile gpu up -d vllm). The gateway is recreated for every run with
# the arm's settings and its own run id, so each run has its own runs/<id>/meter.sqlite.
set -u
cd "$(dirname "$0")/.."
SEEDS=${1:-"1 2 3"}
ARMS=${2:-"n-fifo n-cap n-fair"}
SCENARIO=${SCENARIO:-config/scenarios/fairshare-2080ti.yaml}
CAP=${CAP:-2048}
mkdir -p runs/stage-n
LOG=runs/stage-n/chain.log
for seed in $SEEDS; do
  for arm in $ARMS; do
    case $arm in
      n-fifo) cap=0;    new=false; fair=false ;;
      n-cap)  cap=$CAP; new=true;  fair=false ;;
      n-fair) cap=$CAP; new=true;  fair=true ;;
      *) echo "unknown arm $arm"; exit 2 ;;
    esac
    run_id="$arm-s$seed-$(date -u +%Y%m%d-%H%M%S)"
    echo "$(date -u +%H:%M:%S) START $run_id cap=$cap new=$new fair=$fair" | tee -a "$LOG"
    RUN_ID=$run_id SLOTGW_MAX_PREFILL_TOKENS=$cap SLOTGW_INGEST_NEW_TOKENS=$new SLOTGW_INGEST_FAIR_SHARE=$fair \
      docker compose --profile gpu up -d --force-recreate --no-deps gateway-gpu >>"$LOG" 2>&1 || { echo "FAIL gateway $run_id" | tee -a "$LOG"; exit 1; }
    for i in $(seq 1 60); do curl -fsS http://localhost:8080/healthz >/dev/null 2>&1 && break; sleep 2; done
    curl -fsS http://localhost:8080/healthz >/dev/null 2>&1 || { echo "FAIL gateway not healthy $run_id" | tee -a "$LOG"; exit 1; }
    docker compose run --rm loadgen --scenario "$SCENARIO" --seed "$seed" --no-reservations --label "$arm" >>"$LOG" 2>&1 \
      || { echo "FAIL loadgen $run_id" | tee -a "$LOG"; exit 1; }
    echo "$(date -u +%H:%M:%S) END $run_id" | tee -a "$LOG"
    echo "$run_id" >> runs/stage-n/runs.txt
  done
done
echo "$(date -u +%H:%M:%S) DONE" | tee -a "$LOG"
