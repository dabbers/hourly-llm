#!/usr/bin/env bash
# SSH wrapper around runs/phase2/pods.json for the phase-2 pods (docs/RUNBOOK.md). Runs on this
# machine (Git Bash or Linux); <pod> is the letter a, b or c from the pod's name.
#
#   pod.sh create|status|wait-ready|env-file|stop|start|terminate [args]   -> scripts/runpod_pods.py
#   pod.sh ssh <pod> [command...]        ssh as root (interactive without a command)
#   pod.sh push <pod>                    copy the repo tree to /workspace/slotgw (tar over ssh)
#   pod.sh pull <pod> [run_id]           fetch runs/<run_id> (default: all of runs/) into local runs/
#   pod.sh setup <pod>                   install uv, `uv sync --frozen`, check `uv run slotgw --help`
#   pod.sh vllm <pod> status|log|restart [vllm args...]
#   pod.sh gateway <pod> start [KEY=VALUE...]|stop|log
#   pod.sh calibrate <pod> [calibrate.py args...]
#   pod.sh smoke <pod> [scenario]        smoke-4090.yaml (or smoke.yaml) through the pod's gateway; exit code + last 20 lines
#   pod.sh chain <pod> start [matrix.yaml] [--max-hours N] [--stop-pods] | status | log | stop
#   pod.sh key <pod>                     copy the local .env to the pod, mode 600 (`chain start --stop-pods` needs it)
#
# Environment: RUNPOD_SSH_KEY (default ~/.ssh/id_ed25519), VERBOSE=1 prints every ssh command,
# SLOTGW_PHASE2_DIR (default runs/phase2), LINES for log tails (default 60).
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && (pwd -W 2>/dev/null || pwd))  # pwd -W: a Windows path Git Bash and Windows Python both accept
P2="${SLOTGW_PHASE2_DIR:-$REPO/runs/phase2}"
PODS="$P2/pods.json"
SSH_KEY="${RUNPOD_SSH_KEY:-$HOME/.ssh/id_ed25519}"
REMOTE=/workspace/slotgw
# ssh commands do not read .bashrc: put uv (installer: ~/.local/bin, older: ~/.cargo/bin) on PATH
PRE='export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH" HF_HOME=/root/hf'

usage() { sed -n '2,/^set -euo/p' "$0" | grep '^#' | sed 's/^# \{0,1\}//'; }
die() { echo "pod.sh: $*" >&2; exit 1; }

if command -v uv >/dev/null 2>&1; then UV=(uv); elif command -v py >/dev/null 2>&1; then UV=(py -3.11 -m uv); else die "uv not found"; fi
pods_py() { (cd "$REPO" && "${UV[@]}" run python scripts/runpod_pods.py "$@"); }

IP=""; PORT=""; ID=""
resolve() {  # sets IP, PORT (ssh) and ID for pod letter $1 from pods.json
  [ -f "$PODS" ] || die "$PODS not found: create pods first (pod.sh create --count 1)"
  local out
  out=$(cd "$REPO" && "${UV[@]}" run python - "$PODS" "$1" <<'PY'
import json, sys
path, letter = sys.argv[1], sys.argv[2]
for p in json.load(open(path, encoding="utf-8"))["pods"]:
    if p["name"].rsplit("-", 1)[-1].lower() == letter.lower():
        print(p.get("public_ip") or "-", p.get("ssh_port") or "-", p["id"])
        break
else:
    sys.exit(f"pod {letter!r} is not in {path}")
PY
  ) || exit 1
  out=${out//$'\r'/}
  read -r IP PORT ID <<<"$out"
  if [ "$IP" = "-" ] || [ "$PORT" = "-" ]; then die "pod $1 has no public ip / ssh port yet: run pod.sh status (or wait-ready)"; fi
}

rssh() {  # rssh <pod> [command...]; stdin passes through to ssh
  local pod=$1; shift
  resolve "$pod"
  local opts=(-o StrictHostKeyChecking=accept-new -o "UserKnownHostsFile=$P2/known_hosts" -o ServerAliveInterval=30 -i "$SSH_KEY" -p "$PORT")
  if [ "${VERBOSE:-0}" = 1 ]; then echo "+ ssh ${opts[*]} root@$IP $*" >&2; fi
  ssh "${opts[@]}" "root@$IP" "$@"
}

check_pod() {
  [ -f "$PODS" ] || die "$PODS not found: create pods first (pod.sh create --count 1)"
  [[ "${1:-}" =~ ^[a-j]$ ]] || die "expected a pod letter (a, b, c), got '${1:-}'"
}

cmd_push() {
  echo "pushing $REPO -> $1:$REMOTE"
  tar -C "$REPO" --exclude=.git --exclude=.venv --exclude=runs --exclude=.pytest_cache --exclude=.ruff_cache \
      --exclude=__pycache__ --exclude=calibration/interference --exclude=.env -cf - . \
    | rssh "$1" "mkdir -p $REMOTE && tar -C $REMOTE --no-same-owner -xf - && echo 'pushed to $REMOTE'"
  if [ -f "$P2/phase2.env" ]; then  # the gateway on the pod sources it; runs/ itself is not pushed
    rssh "$1" "mkdir -p $REMOTE/runs/phase2 && cat > $REMOTE/runs/phase2/phase2.env && echo 'pushed runs/phase2/phase2.env'" <"$P2/phase2.env"
  fi
}

cmd_pull() {
  local src=runs
  if [ -n "${2:-}" ]; then src="runs/$2"; fi
  echo "fetching $1:$REMOTE/$src -> $REPO/runs"
  rssh "$1" "cd $REMOTE && if [ -e '$src' ]; then tar --exclude=pods.json -cf - '$src'; else echo 'missing on pod: $src' >&2; exit 1; fi" \
    | tar -C "$REPO" -xvf -
}

cmd_setup() {
  rssh "$1" bash -s <<'EOF'
set -e
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
cd /workspace/slotgw
uv sync --frozen --no-dev
uv run slotgw --help >/dev/null && echo "setup ok: uv $(uv --version | cut -d' ' -f2), uv run slotgw --help works"
EOF
}

cmd_vllm() {
  local pod=$1 action=${2:-status}
  shift; if [ $# -gt 0 ]; then shift; fi
  case "$action" in
    status)
      rssh "$pod" "curl -s -m 5 localhost:8000/health && echo ' <- /health'; curl -s -m 5 localhost:8000/v1/models; echo; tmux ls 2>/dev/null || echo 'no tmux sessions'; nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null || true" ;;
    log) rssh "$pod" "grep -v 'it/s\]' /root/vllm.log | tail -n ${LINES:-60}" ;;
    restart)
      if [ $# -gt 0 ]; then
        printf '%s\n' "$*" | rssh "$pod" "cat > /workspace/vllm.args && echo 'wrote /workspace/vllm.args'"
      fi
      rssh "$pod" bash -s <<'EOF'
tmux kill-session -t vllm 2>/dev/null || true
pkill -f vllm.entrypoints 2>/dev/null || true
sleep 3
tmux new-session -d -s vllm /workspace/run-vllm.sh
echo "vllm started: $(cat /workspace/vllm.args)"
for _ in $(seq 1 180); do
  if curl -sf -m 3 localhost:8000/health >/dev/null; then echo "vllm healthy"; exit 0; fi
  sleep 5
done
echo "vllm did not become healthy in 15 min; see /root/vllm.log" >&2
exit 1
EOF
      ;;
    install)
      # The template path (RUNBOOK stage A): the pod runs RunPod's own PyTorch image, which already
      # has sshd, so vLLM is installed here into a venv on the persistent volume and run in tmux
      # exactly as pod-bootstrap.sh would. Idempotent: a second call only restarts the engine.
      local version=${VLLM_VERSION:-0.32.0} model=${MODEL:-Qwen/Qwen2.5-7B-Instruct-AWQ} chunk=${VLLM_MAX_NUM_BATCHED_TOKENS:-512}
      rssh "$pod" "VLLM_VERSION='$version' MODEL='$model' CHUNK='$chunk' bash -s" <<'EOF'
set -u
# Layout learned in stage A: the venv lives on the 20 GB volume (18 GB, persists across restarts); the
# model cache and the log live on the container disk (/root), because the volume ran out of space with
# the 5.3 GB cache on it and the engine died writing. The model re-downloads in about a minute after a
# restart. VLLM_USE_FLASHINFER_SAMPLER=0: the image has no ninja/nvcc and vLLM 0.32 would JIT-build
# FlashInfer sampling kernels at start-up and die; attention stays on FlashAttention.
export DEBIAN_FRONTEND=noninteractive HF_HOME=/root/hf
mkdir -p /root/hf
command -v tmux >/dev/null || (apt-get update -qq && apt-get install -y -qq tmux >/dev/null) || echo "tmux install failed" >&2
if [ ! -x /workspace/venv/bin/python ]; then python3 -m venv /workspace/venv && echo "venv created"; fi
if ! /workspace/venv/bin/python -c "import vllm" 2>/dev/null; then
  echo "installing vllm==$VLLM_VERSION (several minutes)"
  /workspace/venv/bin/pip install -q --upgrade pip >/dev/null
  /workspace/venv/bin/pip install -q "vllm==$VLLM_VERSION" 2>&1 | tail -3
fi
/workspace/venv/bin/python -c "import vllm, torch; print('vllm', vllm.__version__, 'torch', torch.__version__, 'cuda', torch.cuda.is_available())"
if [ ! -f /workspace/vllm.args ]; then
  echo "--model $MODEL --dtype half --max-model-len 16384 --max-num-seqs 64 --gpu-memory-utilization 0.90 --scheduling-policy priority --max-num-batched-tokens $CHUNK --enable-prefix-caching --enable-prompt-tokens-details --port 8000 --host 0.0.0.0" > /workspace/vllm.args
  echo "wrote /workspace/vllm.args"
fi
cat > /workspace/run-vllm.sh <<'RUN'
#!/usr/bin/env bash
# model cache and log on the container disk; FlashInfer sampler off (see pod.sh vllm install)
export HF_HOME=/root/hf VLLM_USE_FLASHINFER_SAMPLER=0
mkdir -p /root/hf
cd /root || exit 1
# shellcheck disable=SC2046  # vllm.args is a whitespace-separated argv on purpose
exec /workspace/venv/bin/python -m vllm.entrypoints.openai.api_server $(cat /workspace/vllm.args) >>/root/vllm.log 2>&1
RUN
chmod +x /workspace/run-vllm.sh
grep -q 'HF_HOME=' /root/.bashrc 2>/dev/null || echo 'export HF_HOME=/root/hf' >>/root/.bashrc
df -h /workspace / | tail -2
tmux kill-session -t vllm 2>/dev/null || true
pkill -f vllm.entrypoints 2>/dev/null || true
sleep 2
tmux new-session -d -s vllm /workspace/run-vllm.sh
echo "vllm started: $(cat /workspace/vllm.args)"
for _ in $(seq 1 240); do
  if curl -sf -m 3 localhost:8000/health >/dev/null; then echo "vllm healthy"; curl -s localhost:8000/v1/models | head -c 300; echo; exit 0; fi
  sleep 5
done
echo "vllm did not become healthy in 20 min; see /root/vllm.log" >&2
grep -v "it/s\]" /root/vllm.log | tail -20 >&2
exit 1
EOF
      ;;
    *) die "vllm <pod> status|log|restart [args...]|install" ;;
  esac
}

wait_gateway() {
  rssh "$1" bash -s <<'EOF'
for _ in $(seq 1 60); do
  if out=$(curl -sf -m 3 localhost:8080/healthz); then
    printf '%s' "$out" | python3 -c 'import json,sys; d=json.load(sys.stdin); print("gateway up: run_id", d["run_id"], "| burst", d["burst"]["enabled"], d["burst"]["revoke_policy"], "| admission", d["admission"]["enabled"])'
    exit 0
  fi
  sleep 2
done
echo "gateway did not answer /healthz in 120 s" >&2
tail -n 30 /workspace/slotgw/runs/phase2/logs/gateway.log >&2
exit 1
EOF
}

cmd_gateway() {
  local pod=$1 action=${2:-}
  shift; if [ $# -gt 0 ]; then shift; fi
  case "$action" in
    start)
      {
        echo '#!/usr/bin/env bash'
        echo "$PRE"
        echo "cd $REMOTE || exit 1"
        echo 'mkdir -p runs/phase2/logs'
        echo 'set -a; [ -f runs/phase2/phase2.env ] && . runs/phase2/phase2.env; set +a'
        for kv in "$@"; do printf 'export %q\n' "$kv"; done
        echo 'exec uv run slotgw --port 8080 --config-dir config/phase2 --runs-dir runs >> runs/phase2/logs/gateway.log 2>&1'
      } | rssh "$pod" "mkdir -p $REMOTE/runs/phase2/logs && cat > $REMOTE/runs/phase2/gateway-start.sh && chmod +x $REMOTE/runs/phase2/gateway-start.sh && (tmux kill-session -t gateway 2>/dev/null || true) && tmux new-session -d -s gateway $REMOTE/runs/phase2/gateway-start.sh && echo 'gateway session started'"
      wait_gateway "$pod"
      ;;
    stop) rssh "$pod" "tmux kill-session -t gateway 2>/dev/null && echo 'gateway stopped' || echo 'no gateway session'" ;;
    log) rssh "$pod" "tail -n ${LINES:-60} $REMOTE/runs/phase2/logs/gateway.log" ;;
    *) die "gateway <pod> start [KEY=VALUE...]|stop|log" ;;
  esac
}

cmd_calibrate() {
  local pod=$1; shift
  local extra=""
  if [ $# -gt 0 ]; then printf -v extra '%q ' "$@"; fi
  rssh "$pod" "$PRE; cd $REMOTE && uv run python scripts/calibrate.py --url http://127.0.0.1:8000 --replica-id runpod-4090-$pod --config-dir config/phase2 --concurrency 1,2,4,8,16,24,32,48 --contexts 2048,6144 --out calibration/runpod-4090-$pod.yaml $extra"
  echo "fetching calibration/ from $pod"
  rssh "$pod" "cd $REMOTE && tar -cf - calibration" | tar -C "$REPO" -xvf -
}

cmd_smoke() {  # smoke <pod> [scenario]; default: the phase-2 smoke if the tree has one, else smoke.yaml
  local scenario=${2:-}
  if [ -z "$scenario" ]; then
    if [ -f "$REPO/config/scenarios/smoke-4090.yaml" ]; then scenario=config/scenarios/smoke-4090.yaml; else scenario=config/scenarios/smoke.yaml; fi
  fi
  rssh "$1" bash -s -- "$scenario" <<'EOF'
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
cd /workspace/slotgw || exit 1
mkdir -p runs/phase2/logs
echo "smoke: $1 through http://127.0.0.1:8080 (config/phase2)"
uv run slotgw-loadgen --scenario "$1" --seed 1 --config-dir config/phase2 --gateway http://127.0.0.1:8080 --label smoke > runs/phase2/logs/smoke.log 2>&1
rc=$?
echo "loadgen exit code: $rc (0 = the error counter was 0)"
tail -n 20 runs/phase2/logs/smoke.log
exit $rc
EOF
}

cmd_key() {
  [ -f "$REPO/.env" ] || die "$REPO/.env not found"
  rssh "$1" "umask 077 && mkdir -p $REMOTE && cat > $REMOTE/.env && echo 'copied .env to $REMOTE/.env (mode 600)'" <"$REPO/.env"
}

cmd_chain() {
  local pod=$1 action=${2:-status}
  shift; if [ $# -gt 0 ]; then shift; fi
  case "$action" in
    start)
      local matrix=config/phase2/matrix.yaml
      if [ $# -gt 0 ] && [[ "$1" != -* ]]; then matrix=$1; shift; fi
      # the chain stops pods through runpod_pods.py, which needs pods.json (and the key, for --stop-pods)
      rssh "$pod" "mkdir -p $REMOTE/runs/phase2 && cat > $REMOTE/runs/phase2/pods.json" <"$PODS"
      if [[ " $* " == *" --stop-pods "* ]]; then
        if [ -f "$REPO/.env" ]; then cmd_key "$pod"; else echo "warning: no $REPO/.env; without RUNPOD_API_KEY on the pod the chain cannot stop the pods" >&2; fi
      fi
      local q
      printf -v q '%q ' "$matrix" "$@"
      rssh "$pod" "$PRE; cd $REMOTE && (tmux kill-session -t chain 2>/dev/null || true) && tmux new-session -d -s chain \"bash scripts/runchain.sh $q\" && echo 'chain started in tmux session chain (pod.sh chain $pod status|log)'"
      ;;
    status)
      rssh "$pod" "cat $REMOTE/runs/phase2/chain.status 2>/dev/null || echo 'no chain.status yet'; echo '--- runs so far:'; cat $REMOTE/runs/phase2/chain.runs 2>/dev/null || true; tmux has-session -t chain 2>/dev/null && echo '--- chain session: running' || echo '--- chain session: none'" ;;
    log) rssh "$pod" "tail -n ${LINES:-60} $REMOTE/runs/phase2/chain.log" ;;
    stop) rssh "$pod" "tmux kill-session -t chain 2>/dev/null && echo 'chain session killed (pods left running)' || echo 'no chain session'; tmux kill-session -t gateway 2>/dev/null && echo 'gateway stopped' || true" ;;
    *) die "chain <pod> start [matrix] [--max-hours N] [--stop-pods]|status|log|stop" ;;
  esac
}

[ $# -ge 1 ] || { usage; exit 1; }
cmd=$1; shift
case "$cmd" in
  -h|--help|help) usage ;;
  create|status|wait-ready|env-file|stop|start|terminate) pods_py "$cmd" "$@" ;;
  ssh) check_pod "${1:-}"; rssh "$@" ;;
  push|pull|setup|vllm|gateway|calibrate|smoke|chain|key) check_pod "${1:-}"; "cmd_$cmd" "$@" ;;
  *) die "unknown command '$cmd' (pod.sh --help)" ;;
esac
