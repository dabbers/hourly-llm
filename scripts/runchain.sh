#!/usr/bin/env bash
# The phase-2 matrix, run ON the gateway pod inside tmux (docs/PHASE2.md sections 3-4; started with
# `pod.sh chain <pod> start`, which copies pods.json and, for --stop-pods, .env to the pod first).
#
#   bash scripts/runchain.sh config/phase2/matrix-ondemand.yaml [--max-hours 5] [--stop-pods] [--stop-grace-min 30] [--self b]
#
# Matrix file:
#   scenario: config/scenarios/mixed-4090.yaml     # loadgen scenario
#   config_dir: config/phase2                      # gateway, loadgen and report config directory
#   report_out: runs/phase2/report                 # report rebuilt here after every run
#   expect_replicas: 3                             # preflight (default 3)
#   expect_model: Qwen/Qwen2.5-7B-Instruct-AWQ     # preflight (default)
#   expect_calibration: runpod-4090                # preflight: substring of every ledger's calibration source (optional)
#   runs:
#     - label: burstoff                            # loadgen --label and the run id prefix
#       seeds: [1, 2]                              # one fresh gateway + one loadgen run per seed
#       env: {SLOTGW_BURST: "false"}               # gateway environment for these runs (booleans become true/false)
#       loadgen_args: ["--no-reservations"]
#     - label: burst
#       seeds: [1, 2]
#       env: {SLOTGW_BURST: "true", SLOTGW_BURST_REVOKE: grace}
#       loadgen_args: ["--no-reservations"]
#
# Per (run, seed): stop any gateway; start one in tmux session `gateway` with run id
# <label>-s<seed>-<UTC yyyymmdd-HHMMSS> and environment runs/phase2/phase2.env + the run's env; wait
# for /healthz (120 s); scripts/preflight.py; slotgw-loadgen; then slotgw-report over every run so
# far (a report failure is logged, not fatal). The first failed preflight or loadgen writes FAILED
# and stops the chain. A watchdog ends the chain after --max-hours. With --stop-pods and
# RUNPOD_API_KEY available (environment or .env), pods are stopped when the chain ends with DONE,
# FAILED or the watchdog: the other pods first, this one last (--self <letter>, which pod.sh chain
# passes; else the pods.json entry matching RUNPOD_POD_ID or the hostname; else a). With --stop-grace-min N
# the stop waits N minutes first (the chain session stays alive, chain.status already says DONE/FAILED)
# so the operator can `pod.sh pull` the runs while the pods are up; `pod.sh chain <pod> stop` during the
# wait cancels the stop. A manual kill (pod.sh chain stop) stops the gateway only and leaves the pods running.
#
# Files under runs/phase2: chain.log (everything, UTC timestamps), chain.status (one line:
# RUNNING <label> s<seed> <run id> | DONE <n runs> | FAILED <label> s<seed> <reason>), chain.runs
# (one run id per line), logs/<label>-s<seed>.log (loadgen), logs/gateway-<run id>.log.
set -euo pipefail

usage() { sed -n '2,/^set -euo/p' "$0" | grep '^#' | sed 's/^# \{0,1\}//'; }

MATRIX=""; MAX_HOURS=5; STOP_PODS=0; SELF=""; STOP_GRACE_MIN=0
while [ $# -gt 0 ]; do
  case "$1" in
    --max-hours) MAX_HOURS=${2:?--max-hours needs a number}; shift 2 ;;
    --stop-pods) STOP_PODS=1; shift ;;
    --self) SELF=${2:?--self needs a pod letter}; shift 2 ;;
    --stop-grace-min) STOP_GRACE_MIN=${2:?--stop-grace-min needs a number}; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    -*) echo "runchain.sh: unknown option $1" >&2; usage; exit 2 ;;
    *) MATRIX=$1; shift ;;
  esac
done
if [ -z "$MATRIX" ] || [ ! -f "$MATRIX" ]; then echo "runchain.sh: matrix file missing: '$MATRIX'" >&2; usage; exit 2; fi

cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
P2=runs/phase2
mkdir -p "$P2/logs"
LOG=$P2/chain.log; STATUS=$P2/chain.status; RUNS=$P2/chain.runs; WATCHDOG_MARK=$P2/.watchdog
GATEWAY=http://127.0.0.1:8080
touch "$RUNS"

log() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "$LOG"; }
set_status() { echo "$*" >"$STATUS"; log "STATUS $*"; }

# ---- matrix -> one line per (run, seed), fields separated by \x1f (never collapses like tabs) ----
PLAN=$(uv run python - "$MATRIX" <<'PY'
import shlex, sys, yaml
m = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
SEP = "\x1f"
def s(v):
    return str(v).lower() if isinstance(v, bool) else str(v)
print(SEP.join(["HEADER", m["scenario"], m.get("config_dir", "config/phase2"), m.get("report_out", "runs/phase2/report"),
                str(m.get("expect_replicas", 3)), m.get("expect_model", "Qwen/Qwen2.5-7B-Instruct-AWQ"),
                m.get("expect_calibration") or ""]))
for run in m["runs"]:
    env = " ".join(shlex.quote(f"{k}={s(v)}") for k, v in (run.get("env") or {}).items())
    args = " ".join(shlex.quote(s(a)) for a in (run.get("loadgen_args") or []))
    for seed in run.get("seeds") or [1]:
        print(SEP.join(["RUN", str(run["label"]), str(seed), env, args]))
PY
)
IFS=$'\x1f' read -r _ SCENARIO CONFIG_DIR REPORT_OUT EXPECT_REPLICAS EXPECT_MODEL EXPECT_CAL <<<"$(printf '%s\n' "$PLAN" | head -n 1)"
N_TOTAL=$(printf '%s\n' "$PLAN" | grep -c '^RUN' || true)
[ "$N_TOTAL" -gt 0 ] || { echo "runchain.sh: $MATRIX has no runs" >&2; exit 2; }

N_DONE=0; PREV_RUN_ID=""; RUN_ID=""; CUR_LABEL=""; CUR_SEED=""; END_REASON=""; LG_PID=""

stop_gateway() {
  if tmux has-session -t gateway 2>/dev/null; then
    tmux kill-session -t gateway 2>/dev/null || true
    log "gateway session stopped"
  fi
  for _ in 1 2 3 4 5 6 7 8 9 10; do
    curl -sf -m 2 "$GATEWAY/healthz" >/dev/null 2>&1 || return 0
    sleep 1
  done
  pkill -f 'slotgw --port 8080' 2>/dev/null || true
}

start_gateway() {  # <run id> [KEY=VALUE...]
  local run_id=$1; shift
  local script=$P2/gateway-start.sh
  {
    echo '#!/usr/bin/env bash'
    echo 'export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"'
    printf 'cd %q || exit 1\n' "$PWD"
    echo 'set -a; [ -f runs/phase2/phase2.env ] && . runs/phase2/phase2.env; set +a'
    for kv in "$@"; do printf 'export %q\n' "$kv"; done
    printf 'exec uv run slotgw --port 8080 --config-dir %q --runs-dir runs --run-id %q >> %q 2>&1\n' \
      "$CONFIG_DIR" "$run_id" "$P2/logs/gateway-$run_id.log"
  } >"$script"
  chmod +x "$script"
  tmux new-session -d -s gateway "$script"
  log "gateway starting: run id $run_id, env: ${*:-none}"
}

wait_healthz() {  # <seconds>
  local deadline=$((SECONDS + $1)) out rid
  while [ "$SECONDS" -lt "$deadline" ]; do
    if out=$(curl -sf -m 3 "$GATEWAY/healthz" 2>/dev/null); then
      rid=$(printf '%s' "$out" | python3 -c 'import json,sys; print(json.load(sys.stdin)["run_id"])')
      log "gateway up: run_id $rid"
      [ "$rid" = "$RUN_ID" ] || log "warning: gateway run_id $rid differs from requested $RUN_ID"
      return 0
    fi
    sleep 2
  done
  tail -n 30 "$P2/logs/gateway-$RUN_ID.log" >>"$LOG" 2>/dev/null || true
  return 1
}

build_report() {
  local dirs=() id
  while read -r id; do [ -n "$id" ] && dirs+=("runs/$id"); done <"$RUNS"
  [ ${#dirs[@]} -gt 0 ] || return 0
  log "report over ${#dirs[@]} run(s) -> $REPORT_OUT"
  if ! uv run slotgw-report "${dirs[@]}" --out "$REPORT_OUT" --config-dir "$CONFIG_DIR" >>"$LOG" 2>&1; then
    log "report failed (continuing); see $LOG"
  fi
}

self_letter() {  # this pod's letter: --self, else the pods.json entry whose id is RUNPOD_POD_ID or the hostname; a if unknown
  if [ -n "$SELF" ]; then echo "$SELF"; return; fi
  uv run python -c 'import json, os, socket
pods = json.load(open("runs/phase2/pods.json"))["pods"]
ids = {os.environ.get("RUNPOD_POD_ID", ""), socket.gethostname()}
me = [p["name"].rsplit("-", 1)[-1] for p in pods if p["id"] in ids]
print(me[0] if me else "a")' 2>/dev/null || echo a
}

other_letters() {  # every pod letter in pods.json except this pod's
  local me=$1
  uv run python -c 'import json, sys; me = sys.argv[1]; print(",".join(sorted(p["name"].rsplit("-", 1)[-1] for p in json.load(open("runs/phase2/pods.json"))["pods"] if p["name"].rsplit("-", 1)[-1] != me)))' "$me" 2>/dev/null || true
}

stop_pods() {
  if [ -z "${RUNPOD_API_KEY:-}" ] && ! grep -qs '^\(export \)\?RUNPOD_API_KEY=' .env; then
    log "--stop-pods: RUNPOD_API_KEY is not available on this pod (environment or .env); pods left running"
    return 0
  fi
  if [ ! -f "$P2/pods.json" ]; then
    log "--stop-pods: $P2/pods.json missing on this pod; pods left running"
    return 0
  fi
  local me others; me=$(self_letter); others=$(other_letters "$me")
  if [ -n "$others" ]; then
    log "stopping pods $others"
    uv run python scripts/runpod_pods.py stop --only "$others" 2>&1 | tee -a "$LOG" || log "stopping $others failed"
  fi
  log "stopping pod $me (this one) last"
  uv run python scripts/runpod_pods.py stop --only "$me" 2>&1 | tee -a "$LOG" || log "stopping $me failed"
}

fail() {  # <label> <seed> <reason>
  set_status "FAILED $1 s$2 $3"
  END_REASON=failed
  exit 1
}

finish() {
  local rc=$?
  trap - EXIT
  if [ -n "${WATCHDOG_PID:-}" ]; then pkill -P "$WATCHDOG_PID" 2>/dev/null || true; kill "$WATCHDOG_PID" 2>/dev/null || true; fi
  stop_gateway
  if [ "$STOP_PODS" = 1 ] && [ -n "$END_REASON" ]; then
    if [ "$STOP_GRACE_MIN" -gt 0 ] 2>/dev/null; then
      log "pods stop in $STOP_GRACE_MIN min (pull the runs now; pod.sh chain stop cancels the stop)"
      trap 'log "pod stop cancelled"; exit "$rc"' INT TERM HUP
      sleep "$((STOP_GRACE_MIN * 60))"
      trap - INT TERM HUP
    fi
    stop_pods
  fi
  log "chain exit $rc (${END_REASON:-interrupted})"
  exit "$rc"
}

on_signal() {
  if [ -n "$LG_PID" ]; then  # bash only runs a trap once the foreground command ends, so loadgen runs in the background
    pkill -TERM -P "$LG_PID" 2>/dev/null || true
    kill -TERM "$LG_PID" 2>/dev/null || true
    log "loadgen (pid $LG_PID) killed"
  fi
  if [ -f "$WATCHDOG_MARK" ]; then
    END_REASON=watchdog
    set_status "FAILED ${CUR_LABEL:-?} s${CUR_SEED:-?} watchdog: --max-hours $MAX_HOURS passed"
  else
    END_REASON=""
    set_status "FAILED ${CUR_LABEL:-?} s${CUR_SEED:-?} interrupted"
  fi
  exit 130
}

trap finish EXIT
trap on_signal INT TERM HUP
rm -f "$WATCHDOG_MARK"
( sleep "$(awk "BEGIN { print int($MAX_HOURS * 3600) }")"; touch "$WATCHDOG_MARK"; kill -TERM $$ 2>/dev/null ) &
WATCHDOG_PID=$!
disown "$WATCHDOG_PID"

log "chain start: $MATRIX, $N_TOTAL run(s), scenario $SCENARIO, config $CONFIG_DIR, max $MAX_HOURS h, stop-pods=$STOP_PODS"

while IFS=$'\x1f' read -r kind label seed env_q args_q; do
  [ "$kind" = RUN ] || continue
  CUR_LABEL=$label; CUR_SEED=$seed
  eval "run_env=($env_q)"
  eval "lg_args=($args_q)"
  RUN_ID="${label}-s${seed}-$(date -u +%Y%m%d-%H%M%S)"
  log "=== run $((N_DONE + 1))/$N_TOTAL: $label seed $seed -> $RUN_ID (loadgen args: ${lg_args[*]:-none})"

  stop_gateway
  start_gateway "$RUN_ID" ${run_env[@]+"${run_env[@]}"}
  wait_healthz 120 || fail "$label" "$seed" "gateway did not answer /healthz in 120 s"

  if ! uv run python scripts/preflight.py --gateway "$GATEWAY" --expect-replicas "$EXPECT_REPLICAS" \
      --expect-model "$EXPECT_MODEL" --runs-dir runs \
      ${PREV_RUN_ID:+--not-run-id "$PREV_RUN_ID"} ${EXPECT_CAL:+--expect-calibration "$EXPECT_CAL"} 2>&1 | tee -a "$LOG"; then
    fail "$label" "$seed" "preflight"
  fi

  set_status "RUNNING $label s$seed $RUN_ID"
  LG_LOG=$P2/logs/$label-s$seed.log
  uv run slotgw-loadgen --scenario "$SCENARIO" --seed "$seed" --label "$label" --config-dir "$CONFIG_DIR" \
    --gateway "$GATEWAY" --runs-dir runs ${lg_args[@]+"${lg_args[@]}"} >"$LG_LOG" 2>&1 &
  LG_PID=$!
  rc=0
  wait "$LG_PID" || rc=$?  # `wait` returns at once on a trapped signal, a foreground command would not
  LG_PID=""
  if [ "$rc" -ne 0 ]; then
    tail -n 20 "$LG_LOG" >>"$LOG"
    fail "$label" "$seed" "loadgen exit $rc (see $LG_LOG)"
  fi
  tail -n 15 "$LG_LOG" >>"$LOG"
  echo "$RUN_ID" >>"$RUNS"
  N_DONE=$((N_DONE + 1)); PREV_RUN_ID=$RUN_ID
  log "run $RUN_ID ok ($N_DONE/$N_TOTAL)"
  stop_gateway
  build_report
done <<<"$PLAN"

set_status "DONE $N_DONE"
END_REASON=done
exit 0
