#!/usr/bin/env bash
# Stage O on the local GPU (RESULTS Q19): the stage N traffic under the slot-ceiling model.
#   scripts/stage_o_local.sh "1 2 3" "o-slots o-bucket o-prio o-over"
# Every tenant runs on its own reserved slots (flat slot-hour billing), bursting is off (the output rate is a
# ceiling). o-bucket adds the input bucket, o-prio adds tier priority in the engine, o-over sells each tenant
# its p95 slots on an oversold card with both.
set -u
cd "$(dirname "$0")/.."
SEEDS=${1:-"1 2 3"}
ARMS=${2:-"o-slots o-bucket o-prio o-over"}
mkdir -p runs/stage-o
LOG=runs/stage-o/chain.log
for seed in $SEEDS; do
  for arm in $ARMS; do
    scenario=config/scenarios/fairshare-2080ti-slots.yaml; over=1.0
    case $arm in
      o-slots)  bucket=false; prio=false ;;
      o-bucket) bucket=true;  prio=false ;;
      o-prio)   bucket=true;  prio=true ;;
      o-over)   bucket=true;  prio=true; over=2.0; scenario=config/scenarios/fairshare-2080ti-slots-p95.yaml ;;
      *) echo "unknown arm $arm"; exit 2 ;;
    esac
    run_id="$arm-s$seed-$(date -u +%Y%m%d-%H%M%S)"
    echo "$(date -u +%H:%M:%S) START $run_id bucket=$bucket prio=$prio overcommit=$over" | tee -a "$LOG"
    RUN_ID=$run_id SLOTGW_BURST=false SLOTGW_INGEST_NEW_TOKENS=true SLOTGW_INGEST_BUCKET=$bucket SLOTGW_TIER_PRIORITY=$prio SLOTGW_TPS_OVERCOMMIT=$over \
      docker compose --profile gpu up -d --force-recreate --no-deps gateway-gpu >>"$LOG" 2>&1 || { echo "FAIL gateway $run_id" | tee -a "$LOG"; exit 1; }
    for i in $(seq 1 60); do curl -fsS http://localhost:8080/healthz >/dev/null 2>&1 && break; sleep 2; done
    curl -fsS http://localhost:8080/healthz >/dev/null 2>&1 || { echo "FAIL gateway not healthy $run_id" | tee -a "$LOG"; exit 1; }
    docker compose run --rm loadgen --scenario "$scenario" --seed "$seed" --reservations-required --label "$arm" >>"$LOG" 2>&1 \
      || { echo "FAIL loadgen $run_id" | tee -a "$LOG"; exit 1; }
    echo "$(date -u +%H:%M:%S) END $run_id" | tee -a "$LOG"
    echo "$run_id" >> runs/stage-o/runs.txt
  done
done
echo "$(date -u +%H:%M:%S) DONE" | tee -a "$LOG"
