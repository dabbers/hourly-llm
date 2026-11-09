#!/usr/bin/env bash
# stage_h.sh [--self <pod letter>]: phase 3 stage H, the vLLM prefill-chunk sweep, chained onto the end of
# the campaign chain without an operator at the keyboard (docs/PHASE3.md section 6, docs/RUNBOOK.md C2
# "Stage H"). Runs ON the pod in tmux session `stageh`, started by `pod.sh chain <pod> stage-h`, which
# copies this script, scripts/engines_restart.sh and the two matrices to the pod first.
#
#   1. waits until runs/phase2/chain.status says DONE; the chain is then sleeping out its --stop-grace-min
#      before stopping the pod, and killing its tmux session cancels that stop (runchain.sh finish());
#   2. restarts every engine with engine 0's argument line and --max-num-batched-tokens 256, GPU 0 last
#      with a warm compile cache (START_ORDER "1 2 0", STAGGER 1, as before stage I); one retry;
#   3. chains config/phase3/matrix-chunk256.yaml (--max-hours 2) and waits for that chain to end;
#   4. restarts with --max-num-batched-tokens 2048 and chains config/phase3/matrix-chunk2048.yaml with
#      --stop-pods --stop-grace-min 20: the last chain of the campaign stops the pod itself.
#
# If the campaign chain ends FAILED (watchdog, interrupted) nothing is started: its own --stop-pods applies.
# If the engines do not come back after the retry the pod is stopped (RUNPOD_API_KEY from .env, as
# runchain.sh does) and the status says why. A chunk256 chain that ends FAILED does not stop the 2048
# half: the two halves are independent measurements.
# Status: runs/phase2/stageh.status (WAITING | RUNNING ... | HANDED-OVER ... | ABORTED ... | FAILED ...);
# log: runs/phase2/stageh.log. A rehearsal overrides CHAIN_DIR (runs/phase2), RUNCHAIN (bash
# scripts/runchain.sh), RESTART (bash scripts/engines_restart.sh), ARGS_FILE (/workspace/vllm-0.args),
# ENGINES (the /workspace/vllm-*.args count), POLL_S (15) and SETTLE_MAX_S (120).
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
MATRIX_256=config/phase3/matrix-chunk256.yaml; MATRIX_2048=config/phase3/matrix-chunk2048.yaml
CHAIN_LOG=$P2/chain.log; CHAIN_STATUS=$P2/chain.status
LOG=$P2/stageh.log; STATUS=$P2/stageh.status
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

restart_engines() {  # <chunk>: every engine with engine 0's argument line and --max-num-batched-tokens <chunk>, GPU 0 last
  local chunk=$1 args n order attempt
  args=$(sed -E "s/--max-num-batched-tokens [0-9]+/--max-num-batched-tokens $chunk/" "$ARGS_FILE")
  case "$args" in *"--max-num-batched-tokens $chunk"*) ;; *) args="$args --max-num-batched-tokens $chunk" ;; esac
  n=${ENGINES:-$(ls /workspace/vllm-*.args 2>/dev/null | wc -l || true)}
  [ "$n" -gt 0 ] 2>/dev/null || n=3
  if [ "$n" -gt 1 ]; then order="$(seq -s ' ' 1 $((n - 1))) 0"; else order=0; fi
  for attempt in 1 2; do
    log "restarting $n engine(s) with chunk $chunk, order $order (attempt $attempt): $args"
    if ENGINES=$n ARGS=$args START_ORDER=$order STAGGER=1 $RESTART >>"$LOG" 2>&1; then
      log "engines back with chunk $chunk"; return 0
    fi
    log "engine restart with chunk $chunk failed (attempt $attempt); see $LOG and /root/vllm-*.log"
  done
  return 1
}

start_chain() {  # <matrix> [runchain args...]: in tmux session `chain`, as chain_switch.sh does
  local q; printf -v q '%q ' "$@" --self "$SELF"
  tmux new-session -d -s chain "export PATH=\"$PATH\" HF_HOME=\"$HF_HOME\"; $RUNCHAIN $q"
  log "chain started in tmux session chain: $RUNCHAIN $q"
}

# ---- 1. the campaign chain ends ------------------------------------------------------------------------
for f in "$MATRIX_256" "$MATRIX_2048" "$ARGS_FILE"; do
  [ -f "$f" ] || { set_status "FAILED $f is missing; nothing started"; exit 2; }
done
set_status "WAITING for the campaign chain to end DONE (now: $(chain_status))"
while :; do
  st=$(chain_status)
  case "$st" in
    DONE*) break ;;
    FAILED*) set_status "ABORTED the campaign chain ended '$st'; stage H not started (its own --stop-pods applies)"; exit 1 ;;
  esac
  chain_alive || { set_status "ABORTED the chain session is gone with status '$st'; stage H not started"; exit 1; }
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
  set_status "ABORTED the campaign chain is stopping the pod; stage H not started"; exit 1
fi

# ---- 2+3. chunk 256 ------------------------------------------------------------------------------------
if ! restart_engines 256; then
  set_status "FAILED the engines did not come back with chunk 256; stopping the pod"; stop_pod; exit 1
fi
set_status "RUNNING chunk256 chain ($MATRIX_256)"
mark=$(wc -l <"$CHAIN_LOG")
start_chain "$MATRIX_256" --max-hours 2
while chain_alive; do sleep "$POLL_S"; done
res256=$(chain_result "$mark"); res256=${res256:-no status written}
log "chunk256 chain ended: $res256"

# ---- 4. chunk 2048, the last chain of the campaign: it stops the pod -------------------------------------
if ! restart_engines 2048; then
  set_status "FAILED the engines did not come back with chunk 2048 (chunk256 chain: $res256); stopping the pod"; stop_pod; exit 1
fi
set_status "RUNNING chunk2048 chain ($MATRIX_2048; chunk256 chain: $res256)"
mark=$(wc -l <"$CHAIN_LOG")
start_chain "$MATRIX_2048" --max-hours 2 --stop-pods --stop-grace-min 20
until [ -n "$(chain_result "$mark")" ]; do chain_alive || break; sleep "$POLL_S"; done
res2048=$(chain_result "$mark"); res2048=${res2048:-no status written}
set_status "HANDED-OVER to the chunk2048 chain ($res2048; chunk256 chain: $res256); that chain stops the pod 20 min after its DONE"
