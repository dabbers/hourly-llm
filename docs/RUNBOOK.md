# Phase 2 runbook: three RTX 4090 pods on RunPod

The operator's command sequence for [docs/PHASE2.md](PHASE2.md) sections 3, 4 and 7. Every paid
stage starts only on an explicit go; the scripts never print the API key. All commands run from
the repository root on this machine (Git Bash on Windows, or Linux); `pod.sh` talks to the pods
over SSH, `runpod_pods.py` to the RunPod REST API.

| Script | Runs where | Does |
|---|---|---|
| `scripts/runpod_pods.py` | here | create / status / wait-ready / env-file / stop / start / terminate pods through the REST API; `start --zero-gpu` and `ports` through the GraphQL API (recovery when a stopped pod's host has no free GPU) |
| `scripts/pod.sh` | here | SSH wrapper: push the tree, set up uv, drive vLLM, the gateway, calibration, smoke, the chain; pull artifacts |
| `scripts/pod-bootstrap.sh` | in every pod (experimental `create --bootstrap` path only) | container start command for the vLLM image: sshd, HF cache on the volume, vLLM in tmux; did not run on RunPod in stage A |
| `scripts/runchain.sh` | the gateway pod (B in this cluster), in tmux | the matrix: fresh gateway per run, preflight, loadgen, report; stops the pods when done, itself last |
| `scripts/preflight.py` | the gateway pod | refuses a run unless every replica is healthy, idle and the expected one |

State lives under `runs/phase2/` (gitignored): `pods.json` (pod ids, public ips, port mappings;
no secrets), `phase2.env` (replica URLs for the gateway), `known_hosts`, and, pulled from the gateway pod,
`chain.log`, `chain.status`, `chain.runs`, `logs/`, `report/`.

## 0. Prerequisites (once)

1. RunPod account with the prepaid credit loaded (the hard cap; PHASE2.md section 3).
2. Your SSH public key added under RunPod account settings. RunPod injects it into every pod as
   `PUBLIC_KEY`; `pod-bootstrap.sh` writes it to `authorized_keys`. The matching private key is
   `~/.ssh/id_ed25519` unless you set `RUNPOD_SSH_KEY`:

```bash
export RUNPOD_SSH_KEY=~/.ssh/id_ed25519
```

3. The API key in `.env` at the repository root (gitignored), nothing else prints or copies it
   except `pod.sh key` / `pod.sh chain <pod> start --stop-pods`, which put it on the gateway pod so the
   chain can stop the pods:

```bash
echo 'RUNPOD_API_KEY=...' > .env
```

4. Check the toolchain and that the key is found (the dry run prints the request bodies and never
   calls the API):

```bash
py -3.11 -m uv run ruff check scripts tests && py -3.11 -m uv run pytest tests/test_runpod_pods.py tests/test_preflight.py -q
```

```bash
bash scripts/pod.sh create --count 1 --dry-run
```

On Linux `pod.sh` uses `uv` directly; on Windows it uses `py -3.11 -m uv`. `VERBOSE=1` makes it
print every ssh command it runs.

## A. Stage A: one pod (gate G1)

Create pod `slotgw-a` (RTX 4090, Secure Cloud, global networking, 30 GB container disk, 20 GB
volume at `/workspace`, ports 8000/tcp and 22/tcp). The pod runs RunPod's own PyTorch image, which
has sshd and the `PUBLIC_KEY` handling built in; vLLM is installed afterwards over SSH into a venv on
the volume (`pod.sh vllm a install`). The first attempt on 2026-11-07 used the vLLM image with
`pod-bootstrap.sh` as the start command and the container sat idle for 26 minutes with neither sshd
nor vLLM running, so that path (`create --bootstrap`) is kept only as an experiment. Letters continue
from what `pods.json` already holds, so a later `create --count 2` makes `b` and `c`. Global
networking only exists in some data centres (PHASE2.md section 2); pin one with
`--datacenter EU-CZ-1` if creation is refused.

```bash
bash scripts/pod.sh create --count 1
```

Wait for the public ip and the port mappings (a few minutes: the image pull), then show ip, ssh
port, vLLM port, price and internal DNS. `status` also refreshes `pods.json`; run it again after any
restart, the port mappings change:

```bash
bash scripts/pod.sh wait-ready --ports-only --timeout 900
```

```bash
bash scripts/pod.sh status
```

Install vLLM on the pod and start it in tmux (pip install of vLLM into `/workspace/venv`, then the
model download of ~5.3 GB into `/root/hf` on the container disk; 15–20 minutes the first time, a restart afterwards). The venv (18 GB) fills most of the 20 GB volume, so the model cache and the engine log live on the container disk and the model re-downloads after a stop. `VLLM_USE_FLASHINFER_SAMPLER=0` is set because the image has no ninja/nvcc to JIT-build FlashInfer sampling kernels. `--enable-prompt-tokens-details` makes vLLM report `prompt_tokens_details.cached_tokens` per request (off by default in 0.32: without it the meter's cached share reads 0 while the engine's metrics show 60% prefix-cache hits).
The venv and the model cache live on the volume, so a stopped and restarted pod only needs the same
command again, which then just restarts the engine. `VLLM_VERSION`, `MODEL` and
`VLLM_MAX_NUM_BATCHED_TOKENS` in the environment override the defaults (0.32.0, the 7B AWQ model, 512):

```bash
bash scripts/pod.sh vllm a install
```

Copy the tree (everything except `.git`, `.venv`, `runs/`, caches, `calibration/interference`,
`.env`) and install the project on the pod:

```bash
bash scripts/pod.sh push a
```

```bash
bash scripts/pod.sh setup a
```

Check the engine: `/health`, `/v1/models` (the served id must be `Qwen/Qwen2.5-7B-Instruct-AWQ`),
tmux sessions and the GPU:

```bash
bash scripts/pod.sh vllm a status
```

```bash
bash scripts/pod.sh vllm a log
```

To change engine arguments, pass the whole new argument line; it is written to
`/workspace/vllm.args` and the engine restarted:

```bash
bash scripts/pod.sh vllm a restart --model Qwen/Qwen2.5-7B-Instruct-AWQ --dtype half --max-model-len 16384 --max-num-seqs 64 --gpu-memory-utilization 0.90 --scheduling-policy priority --max-num-batched-tokens 512 --enable-prefix-caching --enable-prompt-tokens-details --port 8000 --host 0.0.0.0
```

Calibrate (concurrency 1–48 at 2k and 6k context, plus the interference probe; 10–20 minutes) and
fetch `calibration/runpod-4090-a.yaml` back. From it the shape catalog, `config/phase2/` and
`config/scenarios/mixed-4090.yaml` are sized (PHASE2.md G1) before stage B:

```bash
bash scripts/pod.sh calibrate a
```

Start the gateway on the pod with only replica A of the phase-2 configuration (`config/phase2/`
lists all three pods; `SLOTGW_REPLICA_IDS` restricts it, otherwise the gateway would try to read the
KV budget of pods B and C and fail), run the smoke scenario (`config/scenarios/smoke-4090.yaml`;
pass another path as a second argument) through it and look at the result (exit code 0 =
`errors=0`). The smoke pins the chat reservation to replica A, which exists in stage A too:

```bash
bash scripts/pod.sh gateway a start SLOTGW_REPLICA_IDS=runpod-4090-a
```

```bash
bash scripts/pod.sh smoke a
```

Pull every run directory from the pod (or one: `pull a <run_id>`), then stop the pod as soon as
the calibration is on this machine. A stopped pod keeps its volume (model cache) for
$0.20/GB/month and costs no GPU time:

```bash
bash scripts/pod.sh pull a
```

```bash
bash scripts/pod.sh stop
```

Interactive shell when something needs a look:

```bash
bash scripts/pod.sh ssh a
```

## B. Stage B: the cluster (gate G2)

Start pod `a` again (or create it anew if it was terminated), then create `b` and `c`. After a
start, `status` is what picks up the new port mappings:

```bash
bash scripts/pod.sh start
```

```bash
bash scripts/pod.sh create --count 2
```

```bash
bash scripts/pod.sh wait-ready --timeout 1200
```

First pick the gateway pod. On RunPod two pods that land on the same physical host (same public ip
in `status`) cannot reach each other at all, neither over global networking nor through the public
port mapping, while pods on different hosts talk fine. Stage B hit this: pods `a` and `c` shared a
host, so the gateway runs on `b`, which reaches both. Choose a pod whose public ip is unique among
the three and pass it as `--gateway`; the rest of this runbook writes `b` where the gateway runs.
Write the replica URLs the gateway uses: the gateway pod's own engine as `http://127.0.0.1:8000`,
the others as `http://<id>.runpod.internal:8000`, plus `REPLICA_<X>_ID`.
`config/phase2/replicas.yaml` reads them as `${REPLICA_B_URL}`; the gateway sources
`runs/phase2/phase2.env` at start. Replica traffic goes over RunPod global networking, never the
HTTP proxy (it cuts requests at 100 s):

```bash
bash scripts/pod.sh env-file --gateway b
```

Push and set up all three pods (`b` and `c` only need the tree for a local calibration or a
restart; the gateway, meter and load generator live on `a`). `push` also copies
`runs/phase2/phase2.env` when it exists, so pushing `a` after `env-file` gives its gateway the
replica URLs:

```bash
for p in a b c; do bash scripts/pod.sh push $p && bash scripts/pod.sh setup $p; done
```

Check every replica from pod A over the internal network (B and C must answer on
`<id>.runpod.internal:8000`), start the gateway with the three replicas and run the smoke scenario.
`/admin/replicas` must list all three healthy; the smoke run must place work on all three:

```bash
bash scripts/pod.sh ssh a 'for u in $(grep _URL /workspace/slotgw/runs/phase2/phase2.env | cut -d= -f2); do echo -n "$u -> "; curl -s -m 5 $u/health && echo ok || echo FAIL; done'
```

```bash
bash scripts/pod.sh gateway a start
```

```bash
bash scripts/pod.sh ssh a "curl -s localhost:8080/admin/replicas | python3 -m json.tool | head -60"
```

```bash
bash scripts/pod.sh smoke a
```

Pull the artifacts and build the report here to prove the round trip, then stop the pods (gate
G2 also wants the watchdog seen stopping pods: run a one-run matrix with `--stop-pods`, as in
stage C, if there is budget for the extra half hour):

```bash
bash scripts/pod.sh pull a
```

```bash
py -3.11 -m uv run slotgw-report runs/<gateway_run_id> --out runs/phase2/report --config-dir config/phase2
```

```bash
bash scripts/pod.sh stop
```

## C. Stage C: the matrix (gate G3)

Start the pods, refresh `pods.json`, make sure every pod serves the model, and start the chain on
the gateway pod: B in this cluster, because pods A and C share a host and cannot reach each other
(stage B). `pod.sh chain b start` copies `pods.json` to the pod and, because of `--stop-pods`, the
local `.env` so the chain can stop the pods when it ends; it also passes `--self b` so the chain
stops its own pod last. The matrix format is in the header of `scripts/runchain.sh`. The matrix
comes in two parts: `config/phase2/matrix-ondemand.yaml` (burst off and on, three seeds) first,
then, once `mixed-4090.yaml`'s reservation slots are set from those runs' what-if occupancy,
`config/phase2/matrix-resv.yaml` (the two reservation settings, two seeds). `matrix-continue.yaml`
is what actually ran second: the three burst runs and the two reservation settings in one chain,
after the first chain stopped at burst seed 1 (section E):

```bash
bash scripts/pod.sh start && bash scripts/pod.sh wait-ready && bash scripts/pod.sh env-file --gateway b && bash scripts/pod.sh push b
```

```bash
bash scripts/pod.sh chain b start config/phase2/matrix-ondemand.yaml --max-hours 3 --stop-pods --stop-grace-min 30
```

The chain runs unattended: per (run, seed) it starts a fresh gateway with the run's environment
and a fresh run id, runs `preflight.py` (3 healthy idle replicas, the expected model and
calibration, empty queues, tokenizer within 2%, free disk), runs the load generator, then rebuilds
the report over every run so far. The first failed preflight or loadgen stops the chain with
`FAILED`; the watchdog ends it after `--max-hours`; in both cases and on `DONE` the pods are
stopped (the other pods first, the gateway pod last), after `--stop-grace-min` minutes if given,
so the runs can be pulled while the pods are up. Watch it:

```bash
bash scripts/pod.sh chain b status
```

```bash
bash scripts/pod.sh chain b log
```

`chain.status` holds one line: `RUNNING <label> s<seed> <run id>`, `DONE <n>` or `FAILED <label>
s<seed> <reason>`; `chain b status` prints it first, then the finished run ids, then whether the
tmux session is alive (a script that polls it reads the first line, not the last). A run takes
about 19.5 minutes (18 minutes of timeline plus the drain) and a gateway restart, preflight and
report build about 40 s. Killing the chain by hand stops the gateway but leaves the pods running;
during the grace wait it cancels the pod stop:

```bash
bash scripts/pod.sh chain b stop
```

Pull while the pods run (inside the grace period), then stop them yourself; the chain's stop after
the grace period is then a no-op:

```bash
bash scripts/pod.sh pull b && bash scripts/pod.sh stop
```

A stopped pod's volume is readable only after a `start` (new port mappings, `env-file` again). A
stopped pod stays bound to its host; if another tenant took the GPU meanwhile, `start` fails with
"not enough free GPUs on the host machine" (it did on 2026-11-08, for every retry). Then
`start --zero-gpu` resumes the pod CPU-only (RunPod's `podResumeZeroGpu`, $0.445/h here) with the
volume mounted; vLLM cannot run, ssh can. The zero-GPU pod came back with only its 8000/tcp mapping,
so `ports 8000/tcp,22/tcp` (GraphQL `podEditJob`, the container restarts, about a minute) gave it an
ssh port again; RunPod's ssh proxy (`ssh.runpod.io`) refused the key, so it is no fallback here:

```bash
bash scripts/pod.sh start --only b && bash scripts/pod.sh wait-ready --timeout 600 && bash scripts/pod.sh pull b && bash scripts/pod.sh stop
```

```bash
py -3.11 -m uv run python scripts/runpod_pods.py start --zero-gpu --only b && sleep 60 && bash scripts/pod.sh status
```

```bash
py -3.11 -m uv run python scripts/runpod_pods.py ports 8000/tcp,22/tcp --only b && sleep 60 && bash scripts/pod.sh status && bash scripts/pod.sh pull b && bash scripts/pod.sh stop --only b
```

The lesson for the next cluster is in PHASE2.md section 3: pull the runs before the pods stop (the
chain's `--stop-grace-min` exists for that), or put `/workspace` on a network volume, which any new
pod in the datacenter can mount.

`pull b` brings `runs/<run id>/` for every run and `runs/phase2/{chain.log,chain.status,chain.runs,logs,report}`
(and `runs/phase2/aside/`, where a run that was set aside is kept). Rebuild the report here over the
run ids in `runs/phase2/chain.runs`:

```bash
py -3.11 -m uv run slotgw-report $(sed 's#^#runs/#' runs/phase2/chain.runs) --out runs/phase2/report --config-dir config/phase2
```

A failed run is rerun from the reserve: fix the cause, remove the run's label/seed from the matrix
(or make a one-run matrix), and start the chain again; `chain.runs` keeps the earlier run ids, so
the report keeps covering them. Before a reservation setting runs, check that the scenario's
reservations fit: start a gateway (`pod.sh gateway b start`), `POST /admin/reservations` with the
scenario's tenant, shape, slots and `replica`, expect 201 (409 `reservation_capacity` means the
slots do not fit that replica's KV), delete it, stop the gateway. A reservation the gateway refuses
during a run is logged by the load generator as a warning and the run continues without it.

## C2. Phase 3: the campaign (docs/PHASE3.md)

The campaign is one long chain (49 runs, `config/phase3/matrix-campaign.yaml`, stages D–G and I)
followed by the engine sweep (stage H, two short chains with vLLM restarts in between). Before any
paid minute, every new chain feature and gateway setting is rehearsed on three fake replicas from WSL
(`scripts/rehearsal.sh`; tmux, curl, python3 and uv installed there, the Linux venv outside the repo):

```bash
bash scripts/rehearsal.sh chain config/phase3/matrix-rehearsal.yaml
```

```bash
bash scripts/rehearsal.sh status
```

The rehearsal matrix runs the two-minute smoke under each new setting, one run with a per-run
`scenario:` override and one run that fails preflight on purpose, which `continue_on_fail` must skip
(`chain.failed` gets a line, the status ends `DONE 7 ok 1 failed`); `report_mode: last` writes each
run's report into `runs/<run id>/report`.

**Bring-up.** A stopped pod is bound to its host and starts only while the host has a free GPU;
`hosts` says, read-only, which pods can start right now (and the balance):

```bash
py -3.11 -m uv run python scripts/runpod_pods.py hosts
```

On 2026-11-09 none of the three phase-2 pods could start, and three fresh one-GPU pods landed on one
host with no network path between them (no global-networking interface, `*.runpod.internal` did not
resolve, the public port mappings were not reachable from inside), so the campaign runs on **one pod
with three GPUs** (`create --gpu-count 3`, $2.67/h, the same price as three pods): `vllm install`
starts one engine per GPU (engine i on GPU i, port 8000+10·i because RunPod's image has nginx on 8001,
`/workspace/vllm-i.args`, tmux `vllm-i`,
`/root/vllm-i.log`), `env-file` lists them as replicas A, B and C on localhost, and the gateway, meter,
load generator and chain run on the same pod. Nothing crosses RunPod's inter-pod network any more.
Everything else is as in stage B:

```bash
bash scripts/pod.sh create --count 1 --gpu-count 3 --volume 30 --disk 40 && bash scripts/pod.sh wait-ready --ports-only --timeout 900
```

```bash
bash scripts/pod.sh vllm a install
```

```bash
bash scripts/pod.sh env-file --gateway a && bash scripts/pod.sh push a && bash scripts/pod.sh setup a && bash scripts/pod.sh key a && bash scripts/pod.sh vllm a status && bash scripts/pod.sh smoke a
```

`vllm a restart <args>` writes the argument line to every engine (engine i gets `--port 8000+10·i`) and
restarts them all; `vllm a status` shows each engine's health, model and args, and `nvidia-smi` per GPU.

**The chain.** 19 hours of watchdog for about 17 hours of runs; the pods stop an hour after the
chain ends, so a failure can be caught and the stop cancelled (`chain <gw> stop`):

```bash
bash scripts/pod.sh chain <gw> start config/phase3/matrix-campaign.yaml --max-hours 19 --stop-pods --stop-grace-min 60
```

Poll every 20 minutes and pull each finished run as it appears, so a lost pod costs at most the run
in flight (`pull <gw> <run id>` brings `runs/<run id>/` with its own report; `chain <gw> status`
prints the status line first, then the finished ids):

```bash
bash scripts/pod.sh chain <gw> status
```

```bash
bash scripts/pod.sh pull <gw> <run id>
```

**Stage H.** After `DONE`, inside the grace hour: cancel the stop, restart every engine with the new
chunk size (the whole argument line; `vllm <p> status` echoes it), chain `matrix-chunk256.yaml`
without `--stop-pods`, restart with 2048, chain `matrix-chunk2048.yaml` with `--stop-pods
--stop-grace-min 30`, pull:

```bash
bash scripts/pod.sh chain <gw> stop
```

```bash
for p in a b c; do bash scripts/pod.sh vllm $p restart --model Qwen/Qwen2.5-7B-Instruct-AWQ --dtype half --max-model-len 16384 --max-num-seqs 64 --gpu-memory-utilization 0.90 --scheduling-policy priority --max-num-batched-tokens 256 --enable-prefix-caching --enable-prompt-tokens-details --port 8000 --host 0.0.0.0; done
```

```bash
bash scripts/pod.sh chain <gw> start config/phase3/matrix-chunk256.yaml --max-hours 2
```

**The campaign report** is built here over every run of phases 2 and 3 (confidence intervals,
paired tests, sweeps; `config/phase3/campaign.yaml` names the settings and pairs):

```bash
py -3.11 -m uv run slotgw-campaign $(sed 's#^#runs/#' runs/phase2/chain.runs runs/phase3/chain.runs) --manifest config/phase3/campaign.yaml --out runs/phase3/campaign --config-dir config/phase2
```

(run directories are named by the gateway run id, `<label>-s<seed>-<timestamp>`; the two `chain.runs`
files list them)

Teardown as in section D once every run is local (`terminate --yes` deletes the one pod and its volume); then rotate the API key.

## D. Teardown

`stop` keeps the pod: the container disk is erased, the volume (model cache, 20 GB at
$0.20/GB/month, about $4/month for three pods) stays, no GPU is charged, and `start` brings the pod
back with new port mappings. `terminate` deletes the pod and its volume and costs nothing
afterwards; the next `create` downloads the model again (~10 minutes per pod).

```bash
bash scripts/pod.sh stop
```

```bash
bash scripts/pod.sh terminate --yes
```

`status --all` lists every pod on the account, in case one is not in `pods.json`; `stop --all`
stops them all:

```bash
py -3.11 -m uv run python scripts/runpod_pods.py status --all
```

After phase 2 the API key copied to the gateway pod is gone with the pod (`stop` erases the
container disk, but `/workspace/slotgw/.env` sits on the volume until `terminate`); rotate it in
RunPod's settings anyway (PHASE2.md section 7) and delete `.env`.

## E. When something is off

| Symptom | Where to look |
|---|---|
| `wait-ready` stuck on "waiting for ip/ports" | the pod is still being scheduled; RunPod console shows the reason (no 4090 in that data centre: create in another with `--datacenter`) |
| `wait-ready` stuck on "vLLM not ready" for > 15 min | `pod.sh vllm a log` (engine log at /root/vllm.log on the pod): image pull, model download, or an engine argument the card rejects |
| ssh refused | `pod.sh status` (port mappings changed after a restart); the key in RunPod settings must match `RUNPOD_SSH_KEY` |
| the gateway cannot reach one replica (`unhealthy`, startup fails with "KV budget unknown") while another pod can | the two pods share a physical host (same public ip in `status`): run the gateway on a pod with a unique public ip (`env-file --gateway <pod>`), or terminate and recreate the pod in another data centre |
| `preflight` fails on `model[...]` | the pod serves another model: `pod.sh vllm <pod> status`; fix `/workspace/vllm.args` and `pod.sh vllm <pod> restart` |
| `preflight` fails on `replica_idle` / `ledger_idle` | a previous run is still streaming or the gateway was not restarted; `pod.sh gateway a stop`, wait, start again |
| `preflight` fails on `tokenizer` | the served model's tokenizer differs from the Qwen2.5 family the load generator sizes against (loadgen/content.py VOCAB) |
| chain `FAILED ... loadgen exit 1` | `runs/phase2/logs/<label>-s<seed>.log` on the pod (`pod.sh pull b` fetches it): the error counter was non-zero (refusals and `revoked` context bursts are not errors; connection failures and unexpected stream errors are) |
| report build failed in the chain | logged in `chain.log`, not fatal; rebuild here after `pull b` |
| `start` fails: "There are not enough free GPUs on the host machine" (REST and GraphQL `podResume` alike, even with `gpuCount: 0`) | the host's GPU went to someone else while the pod was stopped; `runpod_pods.py start --zero-gpu --only <pod>` resumes it CPU-only to pull the volume; or migrate the pod from the RunPod console (new id and ip); or wait and retry |
| a zero-GPU pod shows one port mapping (8000) and `pod.sh ssh` says no ssh port | the resume dropped 22/tcp from the pod's ports; `runpod_pods.py ports 8000/tcp,22/tcp --only <pod>` re-adds it (the container restarts, mappings appear after about a minute), then `status`, `pull`, `stop` |
| chain session dies at once, `chain log` shows nothing new; `bash -n scripts/runchain.sh` on the pod says `syntax error near unexpected token $'in\r'` | a script reached the pod with CRLF line endings (a Windows editor, or Python's `pathlib.write_text`, which writes CRLF on Windows); `sed -i 's/\r$//' scripts/*.sh` here, push again. `.gitattributes` keeps `.sh`/`.py`/`.yaml` LF in git, not in files written outside git |
| chain `FAILED burst s1 loadgen exit 1` with `errors=15` and every error `context burst revoked` | fixed 2026-11-08 (loadgen counts them as `revoked`); on an older tree the run itself is complete and only misclassified: set it aside and rerun it, or keep it (the report reads the meter, where the status is `revoked`) |
