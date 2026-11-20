#!/usr/bin/env bash
# Stage P on the local GPU (RESULTS Q20): slot-hour billing without an output ceiling.
#   scripts/stage_p_local.sh "1 2 3" "o-slots p-slots p-fair o-slots-open p-slots-open p-prio o-over p-over p-plain"
# vLLM must be up (docker compose --profile gpu up -d vllm). The gateway is recreated for every run.
# o-* arms pace each stream at its tier's rate (the stage O arms, re-run the same day); p-* arms do not.
# Bursting is off everywhere. *-open arms send each chat turn at a fixed planned time (scheduled turns).
set -u
cd "$(dirname "$0")/.."
SEEDS=${1:-"1 2 3"}
ARMS=${2:-"o-slots p-slots p-fair o-slots-open p-slots-open p-prio o-over p-over p-plain"}
mkdir -p runs/stage-p
LOG=runs/stage-p/chain.log
for seed in $SEEDS; do
  for arm in $ARMS; do
    scenario=config/scenarios/fairshare-2080ti-slots.yaml
    pacing=true; bucket=false; prio=false; over=1.0; cap=0; fair=false; gate=false; new=true; timing=closed; res=--reservations-required
    case $arm in
      o-slots)      ;;
      p-slots)      pacing=false ;;
      p-fair)       pacing=false; cap=2048; fair=true; gate=true ;;
      o-slots-open) timing=scheduled ;;
      p-slots-open) pacing=false; timing=scheduled ;;
      p-prio)       pacing=false; prio=true ;;
      o-over)       bucket=true; prio=true; over=2.0; scenario=config/scenarios/fairshare-2080ti-slots-p95.yaml ;;
      p-over)       pacing=false; bucket=true; prio=true; over=2.0; scenario=config/scenarios/fairshare-2080ti-slots-p95.yaml ;;
      p-plain)      pacing=false; new=false; scenario=config/scenarios/fairshare-2080ti.yaml; res=--no-reservations ;;
      *) echo "unknown arm $arm"; exit 2 ;;
    esac
    run_id="$arm-s$seed-$(date -u +%Y%m%d-%H%M%S)"
    echo "$(date -u +%H:%M:%S) START $run_id pacing=$pacing bucket=$bucket prio=$prio overcommit=$over cap=$cap fair=$fair gate=$gate timing=$timing" | tee -a "$LOG"
    RUN_ID=$run_id SLOTGW_PACING=$pacing SLOTGW_BURST=false SLOTGW_INGEST_NEW_TOKENS=$new SLOTGW_INGEST_BUCKET=$bucket \
      SLOTGW_TIER_PRIORITY=$prio SLOTGW_TPS_OVERCOMMIT=$over SLOTGW_MAX_PREFILL_TOKENS=$cap SLOTGW_INGEST_FAIR_SHARE=$fair \
      SLOTGW_RESERVED_PREFILL_GATE=$gate \
      docker compose --profile gpu up -d --force-recreate --no-deps gateway-gpu >>"$LOG" 2>&1 || { echo "FAIL gateway $run_id" | tee -a "$LOG"; exit 1; }
    for i in $(seq 1 60); do curl -fsS http://localhost:8080/healthz >/dev/null 2>&1 && break; sleep 2; done
    curl -fsS http://localhost:8080/healthz >/dev/null 2>&1 || { echo "FAIL gateway not healthy $run_id" | tee -a "$LOG"; exit 1; }
    docker compose run --rm loadgen --scenario "$scenario" --seed "$seed" $res --turn-timing "$timing" --label "$arm" >>"$LOG" 2>&1 \
      || { echo "FAIL loadgen $run_id" | tee -a "$LOG"; exit 1; }
    echo "$(date -u +%H:%M:%S) END $run_id" | tee -a "$LOG"
    echo "$run_id" >> runs/stage-p/runs.txt
  done
done
echo "$(date -u +%H:%M:%S) DONE" | tee -a "$LOG"
