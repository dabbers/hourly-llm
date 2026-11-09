#!/usr/bin/env bash
# chain_switch.sh <after label-sN> <matrix> [runchain args...]
#
# Replace the running chain between two runs without losing a paid minute. Runs ON THE POD (started
# by `pod.sh chain <pod> switch ...` in tmux session `switch`): waits until the chain's log says the
# per-run report of run <label-sN> is written (the last thing the chain does for a run), kills the
# chain and gateway tmux sessions, waits for any load generator to be gone and starts
# `scripts/runchain.sh <matrix> [args...]` in a new `chain` session.
#
# Killing the chain between runs is safe: the finished run is already in chain.runs; the chain's
# signal handler records "FAILED ... interrupted" in chain.status for the seconds until the new chain
# writes its own status, and with END_REASON empty it does not stop the pods. The new matrix must
# list only the runs still to do, with fresh_runs off so chain.runs keeps the earlier run ids.
# If the next run's gateway or load generator had already started in the second between the report
# line and the kill, they die with the chain; the new chain's preflight starts from a clean gateway.
# Logs to runs/phase2/chain-switch.log.
set -euo pipefail
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH" HF_HOME=/root/hf
AFTER=${1:?after: label-sN of the run to let finish}; MATRIX=${2:?matrix}; shift 2
P2=runs/phase2; LOG=$P2/chain-switch.log
log() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "$LOG"; }
[ -f "$MATRIX" ] || { log "no such matrix: $MATRIX"; exit 1; }
printf -v q '%q ' "$MATRIX" "$@"
# chain.log is appended across chains, so only lines written after this watcher started count: a label-sN
# that an earlier chain already ran would otherwise match at once (2026-11-09 12:02 UTC: stage I's switch
# fired on the first chain's resv-split-s6 report and killed stage I eight seconds in).
start=$(wc -l <"$P2/chain.log" 2>/dev/null || echo 0)
log "waiting for run $AFTER to finish (its report written after line $start of chain.log), then: runchain.sh $q"
pat="report written to runs/${AFTER}-[0-9]{8}-[0-9]{6}/report/report\.html"
until tail -n +"$((start + 1))" "$P2/chain.log" 2>/dev/null | grep -Eq "$pat"; do
  tmux has-session -t chain 2>/dev/null || { log "chain session gone before $AFTER finished; not switching"; exit 1; }
  sleep 1
done
log "run $AFTER finished; killing the chain between runs"
tmux kill-session -t chain 2>/dev/null || true
tmux kill-session -t gateway 2>/dev/null || true
for _ in $(seq 1 30); do pgrep -f slotgw-loadgen >/dev/null || break; sleep 1; done
if pgrep -f slotgw-loadgen >/dev/null; then log "a load generator is still running; not starting the new chain"; exit 1; fi
sleep 3
tmux new-session -d -s chain "export PATH=\"$PATH\" HF_HOME=\"$HF_HOME\"; bash scripts/runchain.sh $q"
log "new chain started in tmux session chain: $q"
