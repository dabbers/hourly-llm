#!/usr/bin/env bash
# Rehearse a chain matrix on three fake 4090-like replicas without a pod or Docker (docs/PHASE3.md
# section 5, docs/RUNBOOK.md section C2). Needs bash, tmux, curl, python3 and uv: on Windows run it
# inside WSL with the project's Linux venv outside the repo (UV_PROJECT_ENVIRONMENT=~/slotgw-venv,
# `uv sync --extra dev` once). Everything lands under runs/rehearsal (CHAIN_DIR for runchain.sh).
#
#   bash scripts/rehearsal.sh up                      # three fakes on :8001-:8003 in tmux + runs/rehearsal/phase2.env
#   bash scripts/rehearsal.sh chain <matrix.yaml>     # up, then runchain.sh in tmux session `chain` (--max-hours 1)
#   bash scripts/rehearsal.sh status                  # chain.status, finished and failed runs, tmux sessions
#   bash scripts/rehearsal.sh down                    # kill the chain, the gateway and the fakes
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
CURVE=${REHEARSAL_CURVE:-"1:154,2:144,4:136,8:113,16:82,24:64,32:54"}  # the 6k-context curve measured on pod A
DIR=runs/rehearsal
mkdir -p "$DIR/logs"

up() {
  local i=1 letter port
  for letter in a b c; do
    port=$((8000 + i)); i=$((i + 1))
    tmux kill-session -t "fake-$letter" 2>/dev/null || true
    tmux new-session -d -s "fake-$letter" \
      "FAKE_MODEL=Qwen/Qwen2.5-7B-Instruct-AWQ FAKE_RATE_CURVE=$CURVE FAKE_KV_TOKENS=271152 FAKE_PREFILL_TPS=20000 uv run slotgw-fake-replica --port $port >> $DIR/logs/fake-$letter.log 2>&1"
  done
  cat >"$DIR/phase2.env" <<EOF
REPLICA_A_URL=http://127.0.0.1:8001
REPLICA_B_URL=http://127.0.0.1:8002
REPLICA_C_URL=http://127.0.0.1:8003
REPLICA_A_ID=runpod-4090-a
REPLICA_B_ID=runpod-4090-b
REPLICA_C_ID=runpod-4090-c
REPLICA_CURVE=$CURVE
EOF
  for port in 8001 8002 8003; do
    for _ in $(seq 1 90); do curl -sf -m 2 "http://127.0.0.1:$port/health" >/dev/null 2>&1 && break; sleep 1; done
    curl -sf -m 2 "http://127.0.0.1:$port/v1/models" >/dev/null || { echo "fake on :$port did not come up; see $DIR/logs" >&2; exit 1; }
  done
  echo "three fakes up on :8001-:8003; gateway env in $DIR/phase2.env"
}

down() {
  for s in chain gateway fake-a fake-b fake-c; do tmux kill-session -t "$s" 2>/dev/null || true; done
  echo "rehearsal down"
}

case "${1:-}" in
  up) up ;;
  chain)
    [ -f "${2:-}" ] || { echo "rehearsal.sh chain <matrix.yaml>" >&2; exit 2; }
    up
    tmux kill-session -t chain 2>/dev/null || true
    rm -f "$DIR/chain.status"
    tmux new-session -d -s chain "CHAIN_DIR=$DIR bash scripts/runchain.sh $2 --max-hours ${REHEARSAL_MAX_HOURS:-1} >> $DIR/chain.console 2>&1"
    sleep 3
    echo "chain started in tmux session chain; status: bash scripts/rehearsal.sh status" ;;
  status)
    echo "status: $(head -n 1 "$DIR/chain.status" 2>/dev/null || echo none)"
    echo "--- runs"; cat "$DIR/chain.runs" 2>/dev/null || true
    echo "--- failed"; cat "$DIR/chain.failed" 2>/dev/null || true
    echo "--- tmux"; tmux ls 2>/dev/null || echo "no tmux server" ;;
  down) down ;;
  *) sed -n '2,/^set -euo/p' "$0" | grep '^#' | sed 's/^# \{0,1\}//'; exit 2 ;;
esac
