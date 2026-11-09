#!/usr/bin/env bash
# engines_restart.sh: restart the vLLM engines of this pod, one per GPU (RUNBOOK C2). Runs ON the pod;
# `pod.sh vllm <pod> restart [args...]` feeds it to `bash -s` over ssh, scripts/stage_h.sh runs it in place.
#
#   ENGINES=3 [ARGS='<argument line for engine 0>'] [START_ORDER='1 2 0'] [STAGGER=1] bash scripts/engines_restart.sh
#
# ARGS, when given, is the whole argument line written for engine 0 (it must contain --port 8000); engine i
# gets --port 8000+10*i in /workspace/vllm-i.args. Without ARGS the engines restart with their current files.
# START_ORDER starts the engines in that order (default 0 1 2 ...); STAGGER=1 waits for each engine's /health
# (up to 15 min) before starting the next: GPU 0 last with a warm compile cache gave every engine the same KV
# budget before stage I (2026-11-09), where a cold start had left engine 0 with 6 % less. Exit 0 once every
# engine answers /health, 1 otherwise (see /root/vllm-*.log).
ENGINES=${ENGINES:?ENGINES: the number of engines (one per GPU)}
ARGS=${ARGS:-}
if [ -n "$ARGS" ]; then
  case "$ARGS" in  # without the port the sed below is a no-op and every engine would bind 8000 (2026-11-09: 15 min lost)
    *"--port 8000"*) ;;
    *) echo "restart: the argument line must contain '--port 8000' (engine i gets --port 8000+10*i); nothing restarted" >&2; exit 1 ;;
  esac
  for i in $(seq 0 $((ENGINES - 1))); do
    echo "$ARGS" | sed "s/--port 8000/--port $((8000 + 10 * i))/" > "/workspace/vllm-$i.args"
    echo "wrote /workspace/vllm-$i.args: $(cat "/workspace/vllm-$i.args")"
  done
fi
for i in $(seq 0 $((ENGINES - 1))); do tmux kill-session -t "vllm-$i" 2>/dev/null || true; done
pkill -f vllm.entrypoints 2>/dev/null || true
sleep 3
ORDER=${START_ORDER:-$(seq -s ' ' 0 $((ENGINES - 1)))}
for i in $ORDER; do
  tmux new-session -d -s "vllm-$i" "/workspace/run-vllm.sh $i"
  echo "engine $i started: $(cat "/workspace/vllm-$i.args")"
  if [ "${STAGGER:-0}" = 1 ]; then
    for _ in $(seq 1 180); do curl -sf -m 3 "localhost:$((8000 + 10 * i))/health" >/dev/null && break; sleep 5; done
    if curl -sf -m 3 "localhost:$((8000 + 10 * i))/health" >/dev/null; then echo "engine $i healthy"; else echo "engine $i not healthy in 15 min; see /root/vllm-$i.log" >&2; exit 1; fi
  fi
done
for _ in $(seq 1 180); do
  ok=0
  for i in $(seq 0 $((ENGINES - 1))); do curl -sf -m 3 "localhost:$((8000 + 10 * i))/health" >/dev/null && ok=$((ok + 1)); done
  if [ "$ok" -eq "$ENGINES" ]; then echo "all $ENGINES engine(s) healthy"; exit 0; fi
  sleep 5
done
echo "not every engine became healthy in 15 min; see /root/vllm-*.log" >&2
exit 1
