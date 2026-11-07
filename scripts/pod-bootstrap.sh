#!/usr/bin/env bash
# Container start script for every phase-2 pod (docs/PHASE2.md, docs/RUNBOOK.md).
#
# scripts/runpod_pods.py passes this file's content as the single argument of `bash -lc`
# (dockerEntrypoint ["bash", "-lc"], dockerStartCmd [<this file>]), so it has to be self-contained.
# The vllm/vllm-openai image has no sshd, so it is installed here; the account's SSH public key
# arrives in $PUBLIC_KEY. vLLM runs in a detached tmux session from /workspace/vllm.args, which is
# written once and kept on the persistent volume: edit it and `pod.sh vllm <pod> restart` to
# change the engine arguments. Safe to run again after a pod restart; every step checks first.
set -u
LOG=/workspace/bootstrap.log
mkdir -p /workspace
log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$LOG"; }

export DEBIAN_FRONTEND=noninteractive
need=""
for pkg in openssh-server tmux curl rsync; do
  dpkg -s "$pkg" >/dev/null 2>&1 || need="$need $pkg"
done
if [ -n "$need" ]; then
  log "installing$need"
  # shellcheck disable=SC2086  # $need is a word list on purpose
  if ! (apt-get update -qq && apt-get install -y -qq $need) >>"$LOG" 2>&1; then
    log "apt-get failed; see $LOG"
  fi
fi

mkdir -p /root/.ssh && chmod 700 /root/.ssh
touch /root/.ssh/authorized_keys
if [ -n "${PUBLIC_KEY:-}" ] && ! grep -qxF "$PUBLIC_KEY" /root/.ssh/authorized_keys; then
  echo "$PUBLIC_KEY" >>/root/.ssh/authorized_keys
fi
chmod 600 /root/.ssh/authorized_keys
if service ssh start >>"$LOG" 2>&1; then log "sshd up"; else log "sshd did not start; see $LOG"; fi

export HF_HOME=/workspace/hf
mkdir -p "$HF_HOME"
grep -q 'HF_HOME=' /root/.bashrc 2>/dev/null || echo 'export HF_HOME=/workspace/hf' >>/root/.bashrc

if [ ! -f /workspace/vllm.args ]; then
  cat >/workspace/vllm.args <<ARGS
--model ${MODEL:-Qwen/Qwen2.5-7B-Instruct-AWQ} --dtype half --max-model-len 16384 --max-num-seqs 64 --gpu-memory-utilization 0.90 --scheduling-policy priority --max-num-batched-tokens ${VLLM_MAX_NUM_BATCHED_TOKENS:-512} --enable-prefix-caching --port 8000 --host 0.0.0.0
ARGS
  log "wrote /workspace/vllm.args"
fi

cat >/workspace/run-vllm.sh <<'RUN'
#!/usr/bin/env bash
export HF_HOME=/workspace/hf
cd /workspace || exit 1
# shellcheck disable=SC2046  # vllm.args is a whitespace-separated argv on purpose
exec python3 -m vllm.entrypoints.openai.api_server $(cat /workspace/vllm.args) >>/workspace/vllm.log 2>&1
RUN
chmod +x /workspace/run-vllm.sh

if tmux has-session -t vllm 2>/dev/null; then
  log "vllm session already running"
else
  tmux new-session -d -s vllm /workspace/run-vllm.sh
  log "vllm started: $(cat /workspace/vllm.args)"
fi
log "bootstrap done"
sleep infinity
