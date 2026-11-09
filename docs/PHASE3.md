# Phase 3 plan: the confidence campaign (milestone 6)

Status: APPROVED 2026-11-08 (the owner: the whole campaign as one approval, pods recreated where
their hosts have no free GPU). Local work (section 5) built and rehearsed 2026-11-08/09; the spend
log in section 7 says what has run.

Phase 2 (`PHASE2.md`) answered Q5–Q7 with two or three runs per setting. The owner wants the
remaining RunPod credit (**$62.35** on 2026-11-08 21:40 UTC, loaded only for this project) spent on
an exhaustive probe: the same runs repeated until the differences are statistically grounded, plus
the edges and scenarios phase 2 did not touch, so the write-up can say with confidence whether
shape billing gives tenants a more predictable bill without costing the provider margin, and where
it adds margin. Everything below follows the phase-2 rules: error-free paid runs, output captured
for readers, every stage's artifacts committed, nothing paid without the owner's go.

## 1. What the write-up has to be able to say

Three claims, each with the measurement that supports or refutes it:

| Claim | Measured by | Needs |
|---|---|---|
| **Predictable bills.** A tenant's bill under shapes varies less day to day than under per-token, a request's bill never exceeds its quote, and the bill does not rise when the cluster is busy | coefficient of variation of each tenant's daily bill across seeds, under both schemes; the per-request bill ÷ quote distribution (`shape_usd / bound_usd`, always ≤ 1 by construction; the discount is the burst); the slope of bill per 1M output tokens against cluster utilisation across the load sweep (per-token: flat by definition; shapes: flat or falling if the claim holds) | six or more seeds per setting; a load sweep |
| **Margin kept.** At the same workload, shape revenue ÷ GPU cost is not below per-token revenue ÷ GPU cost, and the margin curve against utilisation sits at or above per-token's | revenue, GPU cost, gross margin and output per GPU-hour per run with confidence intervals; the measured margin curve at four load levels (phase 2 has one point, 41–54 % utilisation); break-even utilisation measured, not derived | replication; the load sweep |
| **Margin added.** Bursting, overcommit and reservations raise revenue per GPU-hour, and what each costs tenants (floor violations, queueing, refusals) is quantified | paired differences setting vs baseline on the same seeds: bursting on/off, overcommit 1.0/1.25/1.5, reservations split/shared/spill/right-sized, pacing on/off (the plain best-effort deployment as control), prefill chunk 256/512/2048, prefill cap | the lever sweeps; the engine sweep |

Everything is paired by seed: the same seed is the same request sequence, so "burst on minus
burst off at seed 3" removes the day's randomness and the tests are on the differences.

## 2. Where things stand, and the decisions

| Item | State 2026-11-08 21:40 UTC | Decision |
|---|---|---|
| Credit | $62.35; volumes cost $0.017/h | spend up to ~$52 on the campaign, keep ~$10 for reruns and recovery; the optional phase-2 experiments (fp16, A100) only from what is left at the end, with a separate go |
| Pod A (`9g5ortg8uish1z`, EUR-IS-2) | stopped; its host has a free GPU | start it |
| Pods B (`9cz2sqjg40w0vl`, US-NC-1) and C (`59fpze6tlfdp2y`, EUR-IS-2) | stopped; both hosts report no free GPU; their data is pulled, their volumes hold only the vLLM venv | at campaign start, try to start them once; if refused, terminate them and create two fresh 4090 pods (letters b and c are reused, so `config/phase2/replicas.yaml` and the calibration files keep working); a fresh pod needs ~20 min of vLLM install and model download (~$0.30) |
| Host GPUs over the evening | `runpod_pods.py hosts` (read-only, new): A free at 21:40 UTC, C free at 22:20, none of the three at 22:45 | the hosts' free GPUs come and go within the hour; the bring-up probes once and recreates whatever is refused rather than waiting |
| **Topology** (decided 2026-11-09 00:40 UTC) | three fresh one-GPU pods created in EUR-IS-2 landed on one host with the same public ip and no network path between them: no global-networking interface, `*.runpod.internal` unresolvable, public port mappings unreachable from inside (phase 2's cluster had worked over global networking) | **one pod with three RTX 4090s** ($2.67/h, the same as three pods): one vLLM engine per GPU on ports 8000, 8010 and 8020 (8001 is RunPod's nginx), the gateway reaches its replicas on localhost, nothing depends on RunPod's inter-pod network for 20 unattended hours. Differences from phase 2 to keep in mind when pairing seeds 4–6 with 1–3: the three engines share one host's CPUs and PCIe, and the gateway-to-replica path has no WAN latency (in phase 2 two of the three replicas were reached from another continent) |
| Gateway pod | pods on one host cannot reach each other | the pod with a unique public ip runs the gateway, meter, loadgen and chain (`env-file --gateway <letter>`, chain `--self`) |
| Chain shape | a stopped pod may not restart for hours (the host rents the GPU out) | **one long chain** (~17 h) with the pods up throughout, then one short chain for the engine sweep; pods are stopped by the chain with a 60-minute grace so a failure can be caught and the stop cancelled; the runs are pulled while the chain runs |
| Cluster | three 4090 replicas, 32 seats each at 30 tps, 7B AWQ, chunk 512 | unchanged for every run except the engine sweep (section 3, stage H) |
| Run length | the phase-2 day: 18 min of load + 90 s drain + gateway restart ≈ 20.5 min ≈ **$0.91** at 3 × $0.89/h | unchanged, so every run is comparable with the ten phase-2 runs |

## 3. The campaign

53 runs in six stages, in this order (highest value first, so a budget stop still leaves a coherent
data set). Every stage reuses `config/scenarios/mixed-4090.yaml` unless a variant is named; seeds
1–3 already exist for `burstoff` and `burst`, 1–2 for the two reservation settings.

| Stage | Setting (label) | What changes | Seeds | Runs | Cost |
|---|---|---|---|---|---|
| **D. Replication** | `burstoff`, `burst` | as phase 2 (1), (2) | 4, 5, 6 | 6 | $5.5 |
| | `resv-split`, `resv-shared` | as phase 2 (3), (4) | 3, 4, 5, 6 | 8 | $7.3 |
| **E. Reservation edges** | `resv-spill` | setting (3) with `admission.reservation_spill` on: a reserved tenant whose slots are busy competes on demand (the Q6 fix) | 1–3 | 3 | $2.7 |
| | `resv-p95` | reservations sized to the measured peak (chat 44, note-taker 7 slots), no placement rule: what right-sizing costs the on-demand tenants | 1–2 | 2 | $1.8 |
| | `abort` | setting (2) with `SLOTGW_BURST_REVOKE=abort`: revocations cancel within a tick instead of a 15 s grace | 1–2 | 2 | $1.8 |
| **F. Load sweep** | `load0.5-burstoff`, `load0.5-burst` | `rate_multiplier` 5.5 (about 20–27 % utilisation, near the derived break-even) | 1–2 | 4 | $3.6 |
| | `load1.5-burstoff`, `load1.5-burst` | 16.5 (60–80 %) | 1–2 | 4 | $3.6 |
| | `load2-burstoff`, `load2-burst` | 22: saturation; queueing and refusals are the result, not a failure | 1–2 | 4 | $3.6 |
| **G. Levers** | `oc1.25`, `oc1.5` | `SLOTGW_TPS_OVERCOMMIT`: seats = calibrated × 1.25 / 1.5, burst on (the vCPU lever against the floor) | 1–2 | 4 | $3.6 |
| | `prefillcap` | `SLOTGW_MAX_PREFILL_TOKENS=4096` per replica, burst on: the gateway-side mitigation for the Q5 mechanism | 1–2 | 2 | $1.8 |
| | `nopace` | pacing off, admission on, burst off: tokens leave as fast as the engine makes them, the plain deployment that per-token billing assumes; the control for "what does slicing cost in throughput" | 1–2 | 2 | $1.8 |
| **I. Workload edges** | `outheavy-burstoff`, `outheavy-burst` | `mixed-4090-outheavy.yaml`: prompt : output ≈ 1.5 : 1 instead of 15 : 1 (long answers, short prompts), rate scaled so peak occupancy matches the base scenario; tests whether the floor finding is a property of prompt-heavy traffic | 1–2 | 4 | $3.6 |
| | `flash-burstoff`, `flash-burst` | `mixed-4090-flash.yaml`: a two-minute surge at 2.5 × the peak rate inside the peak phase; admission, queueing and bills under a flash crowd | 1–2 | 4 | $3.6 |
| **H. Engine sweep** | `chunk256`, `chunk2048` | vLLM `--max-num-batched-tokens` 256 (half) and 2048 (vLLM's default: what a naive deployment gets), burst on; needs an engine restart on all three pods between chains | 1–2 | 4 | $3.6 |
| | | | | **53** | **$48.2** |

Plus bring-up (start or create pods, vLLM up, push, cluster smoke: ~40 min, ~$1.8), the engine
restarts between chains (~$0.5) and the grace hour at the end (3 pods idle, $2.7 at most; cancelled
once the pull is done). **Expected total ≈ $52**, leaving ≈ $10.

If the reserve is needed, trim from the end: stage H first, then `flash`, then `load1.5`. Stages
D–G are the core and are not trimmed. If ≥ $8 is left when the campaign is committed, the owner can
choose fp16 vs AWQ (~$1.5, one pod) or the A100 cross-check (~$4); both are separate goes.

Time: 49 runs ≈ 16.7 h in the main chain (`--max-hours 19`), then 1.5 h for stage H. About 20 h of
wall clock from the first paid minute; the chain runs unattended on the gateway pod and survives this
machine going down.

## 4. How the numbers are analysed

A new campaign report (`slotgw-campaign`, `report/campaign.py` + `report/stats.py`) over every run
directory, with a manifest that maps each label to its family and sweep value. It is built on the
existing per-run and per-tenant tables and does not change the phase-2 report.

- **Per setting:** mean, standard deviation, n and a 95 % t-interval of every run-level metric
  (revenue, per-token revenue, GPU cost, gross margin, output per GPU-hour, floor violations and
  violation share, refusals by reason, revocations, context bursts, queue p50/p95 and TTFT p50/p95
  per tenant, cached share, burst discount) and of every tenant-level bill.
- **Paired comparisons** on common seeds: difference with 95 % CI, paired t-test p, Wilcoxon
  signed-rank p (n ≥ 6, so stage D makes it possible), Hedges' g. Headline table with Holm
  correction across its rows; the rest reported as estimates with intervals. Significance is p < 0.05
  two-sided; an effect whose interval excludes zero but whose size would not change a decision is
  reported as such.
- **Predictability:** the across-day CV of each tenant's bill under each scheme, paired by tenant
  and setting; the per-request bill ÷ quote distribution; and across the load sweep the regression of
  bill per 1M output tokens on utilisation, per scheme.
- **Margin curve:** revenue ÷ GPU cost, gross margin and output per GPU-hour against measured
  utilisation (seat-hours used ÷ offered) at the four load levels, both schemes, with intervals; the
  utilisation where shape revenue meets GPU cost, read from the fitted curve, against the 26 %
  derived in Q7.
- **Levers:** for each lever the paired gain in revenue per GPU-hour and the paired cost to tenants
  (floor violation share, queue p95, refusals), as a table and a scatter.
- **Edges:** the out-heavy and flash scenarios compared with the base scenario at the same seeds;
  the engine sweep as a three-point curve (256 / 512 / 2048) of floor violations, TTFT and output per
  GPU-hour.

Output capture stays on (`SLOTGW_STORE_TEXT=true`); `docs/SAMPLES.md` gets a section from the
out-heavy and flash runs so readers see long answers under load.

## 5. Local work before the first paid minute (free, tested, committed)

1. `scripts/runchain.sh`: per-run `scenario:` override; `report_mode: last | all | none` (the chain
   rebuilt the phase-2 report over every run after every run; over fifty runs that is minutes each
   time); `continue_on_fail: true` (a failed run is logged in `chain.failed` and the chain moves on
   instead of idling three pods until someone notices; status ends `DONE <ok> ok <k> failed`);
   `fresh_runs: true` (start `chain.runs` empty for the campaign); `CHAIN_DIR` so a rehearsal keeps
   its files apart from `runs/phase2`. Built.
2. `config/phase2/gateway.yaml`: `reservation_spill: ${SLOTGW_RESERVATION_SPILL:-false}` and
   `pacing.enabled: ${SLOTGW_PACING:-true}`. Preflight needed no change (it checks replicas, model,
   calibration, empty queues, tokenizer and disk, none of which these settings touch). Built.
3. Scenarios: `mixed-4090-load0.5 / 1.5 / 2.yaml` (only `rate_multiplier` differs),
   `mixed-4090-outheavy.yaml`, `mixed-4090-flash.yaml`, `mixed-4090-resv-p95.yaml`, generated from
   the base file by exact substitution. A plan replay (each request = 1 s head start + context /
   20k tok/s + max_tokens / tps, sessions back to back) sized them: base peak 75 streams mean / 97
   p95 (chat 42, agent 18, note-taker 3, batch 12); out-heavy at rate 14 gives 74 / 87 with a
   prompt : output ratio of 1.9 : 1 against the base's 21 : 1 in the same replay; the flash at 2 ×
   gives 83 / 100 during the surge and 107 mean for the two minutes after it (chat sessions started
   in the surge keep going), so 2.5 × was dropped as too far past saturation; 2 × load gives 144 /
   177 against 96 seats, deep saturation by design. A unit test shows the 44 + 7 unpinned
   reservation slots of `resv-p95` spread over the three replicas (and a pinned 44 is refused). Built.
4. `config/phase3/matrix-campaign.yaml` (49 runs, section 3 order), `matrix-chunk256.yaml`,
   `matrix-chunk2048.yaml`, `matrix-rehearsal.yaml`, `campaign.yaml` (the label → family/value
   manifest). Built.
5. `report/stats.py`, `report/campaign.py`, the `slotgw-campaign` entry point, tests; run over the
   ten phase-2 runs as the first campaign report (n = 2–3, intervals wide: the pipeline check).
6. Rehearsal on the fake replicas (`scripts/rehearsal.sh`, from WSL: tmux, curl, python3 and uv there,
   the Linux venv outside the repo): every new gateway setting (spill, unpinned reservation, pacing
   off, overcommit, prefill cap, abort) through the two-minute smoke in one chain, plus a per-run
   scenario override and one run that fails preflight on purpose for `continue_on_fail`.
7. `docs/RUNBOOK.md` section C2 (rehearsal, the `hosts` probe and the bring-up decision rule, the
   long chain, incremental pulls, the engine sweep, teardown); `scripts/runpod_pods.py hosts`
   (read-only: balance and whether each pod's host has a free GPU); this file's spend log. Built.

## 6. Operations

Bring-up (owner's go starts the clock): probe the hosts; start A; try B and C once; terminate and
recreate whichever refuses (`create --count N`), `vllm install` on new pods, `push`/`setup` on all,
`env-file --gateway <unique ip>`, `key <gateway>`, cluster smoke (`smoke-4090`, must place work on
all three replicas), then
`pod.sh chain <gw> start config/phase3/matrix-campaign.yaml --max-hours 19 --stop-pods --stop-grace-min 60`.

While it runs: a poll every 20 minutes reads `chain status` (first line) and pulls each finished
run id that is not yet local, so a lost pod costs at most the run in flight. The campaign report is
rebuilt locally from the pulled runs, never on the pod.

Stage H: after `DONE`, during the grace hour, `pod.sh chain <gw> stop` cancels the stop; `pod.sh
vllm <p> restart ... --max-num-batched-tokens 256` on all three; chain `matrix-chunk256.yaml`
without `--stop-pods`; restart with 2048; chain `matrix-chunk2048.yaml` with `--stop-pods
--stop-grace-min 30`; pull.

Teardown: with every run local and the report built, terminate the three pods (volumes included:
~$0.40/day otherwise) on the owner's word, then the owner rotates the RunPod API key in `.env`.

Error-free means what PHASE2.md §4 says: `errors = 0` in every load run, zero engine preemptions,
preflight OK. Refusals, revocations and queueing in the saturation and flash runs are results.
A run that fails is rerun from the reserve only if the stage would otherwise have fewer than two
seeds.

## 7. Spend log

| When (UTC) | What | Pod-hours | Cost |
|---|---|---|---|
| 2026-11-08 06:28 – 11-09 00:20 | three stopped phase-2 pods' volumes until terminated | – | ~$0.30 |
| 2026-11-09 00:23–00:36 | three fresh one-GPU pods (EUR-IS-2): same host, no network between them; terminated | 3 × 0.22 | ~$0.58 |
| 2026-11-09 00:39 → | one pod with three 4090s (`qka0zggy47mw7e`): bring-up, campaign | | $2.67/h |
| | | | |

Balance at plan time: $62.35 of $80 (phase 2 spent $17.61; see PHASE2.md §3).

## 8. What the owner decides

1. **Go for the campaign as one approval** (sections 3 and 6, ≈ $52, ≈ 20 h), or stage by stage.
2. **Recreating pods B and C** if their hosts still have no GPU at start (their volumes hold nothing
   that is not local; the alternative is waiting for the hosts, with no way to know how long).
3. **The optional phase-2 experiments** (fp16 vs AWQ, A100) from what is left: asked again at the
   end with the actual balance.
