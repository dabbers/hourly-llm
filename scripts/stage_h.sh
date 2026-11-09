#!/usr/bin/env bash
# stage_h.sh [--self <pod letter>]: the chains that follow the campaign sweep, run without an operator at the
# keyboard (docs/PHASE3.md sections 6 and 9, docs/RUNBOOK.md C2 "Stage H"). Runs ON the pod in tmux session
# `stageh`, started by `pod.sh chain <pod> stage-h`, which copies this script, scripts/engines_restart.sh and
# the matrices to the pod first.
#
#   1. waits until runs/phase2/chain.status says DONE; the chain is then sleeping out its --stop-grace-min
#      before stopping the pod, and killing its tmux session cancels that stop (runchain.sh finish());
#   2. for each step of STEPS below: restarts every engine with engine 0's argument line and the step's
#      --max-num-batched-tokens, GPU 0 last with a warm compile cache (START_ORDER "1 2 0", STAGGER 1, as
#      before stage I), checks every engine's KV budget from /metrics (within 1 % of each other, and of the
#      step's expected total where one is given), one retry; then chains the step's matrix. Steps before the
#      last wait for their chain to end; the last chain carries --stop-pods and stops the pod itself.
#
#   steps (2026-11-09): chunk 256 -> matrix-chunk256.yaml; chunk 2048 -> matrix-chunk2048.yaml (stage H);
#   chunk 512 restored, KV 289,056 expected -> matrix-stage-j.yaml (stage J, --stop-pods --stop-grace-min 20).
#
# If the campaign chain ends FAILED (watchdog, interrupted) nothing is started: its own --stop-pods applies.
# If the engines do not come back, or their KV budgets disagree, after the retry, the pod is stopped
# (RUNPOD_API_KEY from .env, as runchain.sh does) and the status says why. A chain that ends FAILED does not
# stop the later steps: they are independent measurements, and each matrix's own preflight is the final gate.
# Status: runs/phase2/stageh.status (WAITING | RUNNING ... | HANDED-OVER ... | ABORTED ... | FAILED ...);
# log: runs/phase2/stageh.log. A rehearsal overrides CHAIN_DIR (runs/phase2), RUNCHAIN (bash
# scripts/runchain.sh), RESTART (bash scripts/engines_restart.sh), KV_CHECK=skip, ARGS_FILE
# (/workspace/vllm-0.args), ENGINES (the /workspace/vllm-*.args count), POLL_S (15) and SETTLE_MAX_S (120).
set -euo pipefail

usage() { sed -n '2,/^set -euo/p' "$0" | grep '^#' | sed 's/^# \{0,1\}//'; }

SELF=a
while [ $# -gt 0 ]; do
  case "$1" in
    --self) SELF=${2:?--self needs a pod letter}; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "stage_h.sh: unknown argument $1" >&2; usage; exit 2 ;;
  esac
done

cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH" HF_HOME=${HF_HOME:-/root/hf}
P2=${CHAIN_DIR:-runs/phase2}
RUNCHAIN=${RUNCHAIN:-bash scripts/runchain.sh}
RESTART=${RESTART:-bash scripts/engines_restart.sh}
ARGS_FILE=${ARGS_FILE:-/workspace/vllm-0.args}
POLL_S=${POLL_S:-15}; SETTLE_MAX_S=${SETTLE_MAX_S:-120}
CHAIN_LOG=$P2/chain.log; CHAIN_STATUS=$P2/chain.status
LOG=$P2/stageh.log; STATUS=$P2/stageh.status
# chunk | matrix | expected KV total per engine (- = spread only) | runchain arguments; the last step stops the pod
STEPS=(
  "256|config/phase3/matrix-chunk256.yaml|-|--max-hours 2"
  "2048|config/phase3/matrix-chunk2048.yaml|-|--max-hours 2"
  "512|config/phase3/matrix-stage-j.yaml|289056|--max-hours 2 --stop-pods --stop-grace-min 20"
)
mkdir -p "$P2"

log() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "$LOG"; }
set_status() { echo "$*" >"$STATUS"; log "STATUS $*"; }
chain_alive() { tmux has-session -t chain 2>/dev/null; }
chain_status() { cat "$CHAIN_STATUS" 2>/dev/null || true; }
chain_result() {  # <chain.log line count before the chain started>: the last STATUS that chain wrote, if any
  tail -n +"$(($1 + 1))" "$CHAIN_LOG" 2>/dev/null | grep -o 'STATUS .*' | tail -n 1 || true
}

stop_pod() {  # what runchain.sh does at the end of a --stop-pods chain
  if [ -z "${RUNPOD_API_KEY:-}" ] && ! grep -qs '^\(export \)\?RUNPOD_API_KEY=' .env; then
    log "RUNPOD_API_KEY is not on this pod (environment or .env); pod left running"; return 0
  fi
  [ -f "$P2/pods.json" ] || { log "$P2/pods.json missing; pod left running"; return 0; }
  log "stopping pod $SELF"
  uv run python scripts/runpod_pods.py stop --only "$SELF" 2>&1 | tee -a "$LOG" || log "stopping $SELF failed"
}

engines_count() {
  local n; n=${ENGINES:-$(ls /workspace/vllm-*.args 2>/dev/null | wc -l || true)}
  [ "$n" -gt 0 ] 2>/dev/null || n=3
  echo "$n"
}

kv_check() {  # <n engines> <expected total | ->: every engine's KV budget from /metrics within 1 % of each other (and of the total)
  local n=$1 expect=$2 i m nb bs tok min=0 max=0 all=""
  [ "${KV_CHECK:-}" = skip ] && { log "KV check skipped (rehearsal)"; return 0; }
  for i in $(seq 0 $((n - 1))); do
    m=$(curl -s -m 5 "localhost:$((8000 + 10 * i))/metrics" | grep '^vllm:cache_config_info' | head -n 1 || true)
    nb=$(printf '%s' "$m" | grep -o '[{,]num_gpu_blocks="[0-9]*"' | head -n 1 | grep -o '[0-9]*' || true)
    bs=$(printf '%s' "$m" | grep -o '[{,]block_size="[0-9]*"' | head -n 1 | grep -o '[0-9]*' || true)
    if [ -z "$nb" ] || [ -z "$bs" ]; then log "KV check: engine $i reports no cache_config_info"; return 1; fi
    tok=$((nb * bs)); all="$all engine$i=$tok"
    if [ "$min" -eq 0 ] || [ "$tok" -lt "$min" ]; then min=$tok; fi
    if [ "$tok" -gt "$max" ]; then max=$tok; fi
  done
  log "KV budgets:$all (spread $(( (max - min) * 1000 / max )) per mille${expect:+, expected ${expect}})"
  if [ $(( (max - min) * 100 )) -gt "$max" ]; then log "KV check: the engines disagree by more than 1 %"; return 1; fi
  if [ "$expect" != "-" ]; then
    if [ $(( (expect - min) * 100 )) -gt "$expect" ] || [ $(( (max - expect) * 100 )) -gt "$expect" ]; then
      log "KV check: an engine is more than 1 % away from the expected $expect"; return 1
    fi
  fi
  return 0
}

restart_engines() {  # <chunk> <expected KV total | ->: every engine with engine 0's argument line and the chunk, GPU 0 last, KV checked
  local chunk=$1 expect=$2 args n order attempt
  args=$(sed -E "s/--max-num-batched-tokens [0-9]+/--max-num-batched-tokens $chunk/" "$ARGS_FILE")
  case "$args" in *"--max-num-batched-tokens $chunk"*) ;; *) args="$args --max-num-batched-tokens $chunk" ;; esac
  n=$(engines_count)
  if [ "$n" -gt 1 ]; then order="$(seq -s ' ' 1 $((n - 1))) 0"; else order=0; fi
  for attempt in 1 2; do
    log "restarting $n engine(s) with chunk $chunk, order $order (attempt $attempt): $args"
    if ENGINES=$n ARGS=$args START_ORDER=$order STAGGER=1 $RESTART >>"$LOG" 2>&1; then
      if kv_check "$n" "$expect"; then log "engines back with chunk $chunk"; return 0; fi
      log "engine restart with chunk $chunk: engines healthy but the KV check failed (attempt $attempt)"
    else
      log "engine restart with chunk $chunk failed (attempt $attempt); see $LOG and /root/vllm-*.log"
    fi
  done
  return 1
}

start_chain() {  # <matrix> [runchain args...]: in tmux session `chain`, as chain_switch.sh does
  local q; printf -v q '%q ' "$@" --self "$SELF"
  tmux new-session -d -s chain "export PATH=\"$PATH\" HF_HOME=\"$HF_HOME\"; $RUNCHAIN $q"
  log "chain started in tmux session chain: $RUNCHAIN $q"
}

# ---- 1. the campaign chain ends ------------------------------------------------------------------------
for f in "$ARGS_FILE"; do [ -f "$f" ] || { set_status "FAILED $f is missing; nothing started"; exit 2; }; done
for step in "${STEPS[@]}"; do
  IFS='|' read -r _ matrix _ _ <<<"$step"
  [ -f "$matrix" ] || { set_status "FAILED $matrix is missing; nothing started"; exit 2; }
done
set_status "WAITING for the campaign chain to end DONE (now: $(chain_status)); then ${#STEPS[@]} step(s)"
while :; do
  st=$(chain_status)
  case "$st" in
    DONE*) break ;;
    FAILED*) set_status "ABORTED the campaign chain ended '$st'; nothing started (its own --stop-pods applies)"; exit 1 ;;
  esac
  chain_alive || { set_status "ABORTED the chain session is gone with status '$st'; nothing started"; exit 1; }
  sleep "$POLL_S"
done
log "the campaign chain is DONE: $st"
# runchain.sh's finish() stops the gateway, logs "pods stop in N min" and only then installs the trap that
# turns a kill into "pod stop cancelled" (a kill before that would overwrite chain.status with FAILED interrupted).
done_line=$( (grep -n 'STATUS DONE' "$CHAIN_LOG" || true) | tail -n 1 | cut -d: -f1); done_line=${done_line:-1}
deadline=$((SECONDS + SETTLE_MAX_S))
until tail -n +"$done_line" "$CHAIN_LOG" | grep 'pods stop in' >/dev/null; do
  chain_alive || break
  if [ "$SECONDS" -ge "$deadline" ]; then log "no 'pods stop in' line within $SETTLE_MAX_S s (a chain without --stop-pods?)"; break; fi
  sleep 2
done
sleep 5
if chain_alive; then tmux kill-session -t chain; log "campaign chain session killed: its pod stop is cancelled"; else log "the campaign chain session had already ended"; fi
# tmux's HUP reaches the chain's bash, which runs its trap only once the foreground `sleep <grace>` ends: end it
# now, so the chain logs "pod stop cancelled" and exits at once instead of lingering as an orphan for the grace period.
for pid in $(pgrep -f 'runchain' || true); do pkill -P "$pid" sleep 2>/dev/null || true; done
for _ in $(seq 1 10); do pgrep -f 'runchain' >/dev/null || break; sleep 1; done
if tail -n +"$done_line" "$CHAIN_LOG" | grep 'pod stop cancelled' >/dev/null; then log "chain.log confirms: pod stop cancelled"; else log "warning: no 'pod stop cancelled' line in chain.log yet ($(pgrep -fc 'runchain' || true) runchain process(es) left)"; fi
tmux kill-session -t gateway 2>/dev/null && log "gateway session killed" || true
for _ in $(seq 1 30); do pgrep -f slotgw-loadgen >/dev/null || break; sleep 1; done
sleep 3
if tail -n +"$done_line" "$CHAIN_LOG" | grep 'stopping pod' >/dev/null; then
  set_status "ABORTED the campaign chain is stopping the pod; nothing started"; exit 1
fi

# ---- 2. the steps ---------------------------------------------------------------------------------------
results=""
total=${#STEPS[@]}; k=0
for step in "${STEPS[@]}"; do
  k=$((k + 1))
  IFS='|' read -r chunk matrix expect chain_args <<<"$step"
  if ! restart_engines "$chunk" "$expect"; then
    set_status "FAILED step $k/$total: the engines did not come back with chunk $chunk and a uniform KV budget (${results:-no earlier step}); stopping the pod"
    stop_pod; exit 1
  fi
  mark=$(wc -l <"$CHAIN_LOG")
  # shellcheck disable=SC2086  # chain_args is a word list on purpose
  start_chain "$matrix" $chain_args
  if [ "$k" -lt "$total" ]; then
    set_status "RUNNING step $k/$total: $matrix with chunk $chunk (${results:-first step})"
    while chain_alive; do sleep "$POLL_S"; done
    res=$(chain_result "$mark"); res=${res:-no status written}
    log "step $k/$total ($matrix) ended: $res"
    results="${results:+$results; }$matrix: $res"
  else
    until [ -n "$(chain_result "$mark")" ]; do chain_alive || break; sleep "$POLL_S"; done
    res=$(chain_result "$mark"); res=${res:-no status written}
    set_status "HANDED-OVER to the last chain $matrix with chunk $chunk ($res; earlier: ${results:-none}); that chain stops the pod 20 min after its DONE"
  fi
done
