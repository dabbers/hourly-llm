# Phase 3 plan: the confidence campaign (milestone 6)

Status: RUNNING since 2026-11-09 01:20 UTC (approved 2026-11-08 by the owner as one approval, pods
recreated where their hosts had no free GPU; topology changed to one pod with three GPUs, section 2).
Local work (section 5) built, tested and rehearsed; the spend log in section 7 says what has run.

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
| **Owner's adjustments** (2026-11-09, after reading the phase-2 brief) | the brief's four findings: the floor is not holding (check the metric first), on-demand slot billing tracks per-token billing so predictability must come from reservations, the configured reservations are under-sized, margins depend on configured prices | (1) floor metric split before the lever runs are judged: the **decode floor** is what `floor_violation` already measures (`delivered_tokens ÷ (t_last_token − t_first_token)` against 95 % of the shape's tps: queue and TTFT excluded, so the phase-2 misses are decode-side, the prefill mechanism of Q5), and a **start bound** (queue + TTFT) is reported beside it; (2) `nopace` moved to the head of the second chain, so the control exists even if the budget stops early; (3) `resv-p95-spill` added (p95 sizing + spill, seeds 1–2, ~$1.8): the reservation configuration that would ship; (4) the write-up leads with the bill CV of reserved tenants against the same tenants under per-token pricing, and with break-even utilisation and output per GPU-hour before any revenue ÷ cost ratio |
| Gateway pod | pods on one host cannot reach each other | the pod with a unique public ip runs the gateway, meter, loadgen and chain (`env-file --gateway <letter>`, chain `--self`) |
| Chain shape | a stopped pod may not restart for hours (the host rents the GPU out) | **one long chain** (~17 h) with the pods up throughout, then one short chain for the engine sweep; pods are stopped by the chain with a 60-minute grace so a failure can be caught and the stop cancelled; the runs are pulled while the chain runs |
| Cluster | three 4090 replicas, 32 seats each at 30 tps, 7B AWQ, chunk 512 | unchanged for every run except the engine sweep (section 3, stage H) |
| Run length | the phase-2 day: 18 min of load + 90 s drain + gateway restart ≈ 20.5 min ≈ **$0.91** at 3 × $0.89/h | unchanged, so every run is comparable with the ten phase-2 runs |

## 3. The campaign

55 runs in six stages, in this order (highest value first, so a budget stop still leaves a coherent
data set). Every stage reuses `config/scenarios/mixed-4090.yaml` unless a variant is named; seeds
1–3 already exist for `burstoff` and `burst`, 1–2 for the two reservation settings.

| Stage | Setting (label) | What changes | Seeds | Runs | Cost |
|---|---|---|---|---|---|
| **D. Replication** | `burstoff`, `burst` | as phase 2 (1), (2) | 4, 5, 6 | 6 | $5.5 |
| | `resv-split`, `resv-shared` | as phase 2 (3), (4) | 3, 4, 5, 6 | 8 | $7.3 |
| **E. Reservation edges** | `resv-spill` | setting (3) with `admission.reservation_spill` on: a reserved tenant whose slots are busy competes on demand (the Q6 fix) | 1–3 | 3 | $2.7 |
| | `resv-p95` | reservations sized to the measured peak (chat 44, note-taker 7 slots), no placement rule: what right-sizing costs the on-demand tenants | 1–2 | 2 | $1.8 |
| | `resv-p95-spill` | `resv-p95` with `admission.reservation_spill` on: p95 sizing and spill together, the reservation configuration that would actually ship (owner's addition, 2026-11-09) | 1–2 | 2 | $1.8 |
| | `abort` | setting (2) with `SLOTGW_BURST_REVOKE=abort`: revocations cancel within a tick instead of a 15 s grace | 1–2 | 2 | $1.8 |
| **F. Load sweep** | `load0.5-burstoff`, `load0.5-burst` | `rate_multiplier` 5.5 (about 20–27 % utilisation, near the derived break-even) | 1–2 | 4 | $3.6 |
| | `load1.5-burstoff`, `load1.5-burst` | 16.5 (60–80 %) | 1–2 | 4 | $3.6 |
| | `load2-burstoff`, `load2-burst` | 22: saturation; queueing and refusals are the result, not a failure | 1–2 | 4 | $3.6 |
| **G. Levers** | `oc1.25`, `oc1.5` | `SLOTGW_TPS_OVERCOMMIT`: seats = calibrated × 1.25 / 1.5, burst on (the vCPU lever against the floor) | 1–2 | 4 | $3.6 |
| | `prefillcap` | `SLOTGW_MAX_PREFILL_TOKENS=4096` per replica, burst on: the gateway-side mitigation for the Q5 mechanism | 1–2 | 2 | $1.8 |
| | `nopace` | pacing off, admission on, burst off: tokens leave as fast as the engine makes them, the plain deployment that per-token billing assumes; the control for "what does slicing cost in throughput" (runs first in the second chain, owner's order change 2026-11-09) | 1–2 | 2 | $1.8 |
| **I. Workload edges** | `outheavy-burstoff`, `outheavy-burst` | `mixed-4090-outheavy.yaml`: prompt : output ≈ 1.5 : 1 instead of 15 : 1 (long answers, short prompts), rate scaled so peak occupancy matches the base scenario; tests whether the floor finding is a property of prompt-heavy traffic | 1–2 | 4 | $3.6 |
| | `flash-burstoff`, `flash-burst` | `mixed-4090-flash.yaml`: a two-minute surge at 2.5 × the peak rate inside the peak phase; admission, queueing and bills under a flash crowd | 1–2 | 4 | $3.6 |
| **H. Engine sweep** | `chunk256`, `chunk2048` | vLLM `--max-num-batched-tokens` 256 (half) and 2048 (vLLM's default: what a naive deployment gets), burst on; needs an engine restart on all three pods between chains | 1–2 | 4 | $3.6 |
| **I** | `resv-split` s3–6, `resv-spill` s1–3 (re-runs) | the seven pinned-reservation runs that ran without the note-taker's 4 slots (section 6: replica A's KV deficit on the campaign pod); after the second chain, engines restarted first (GPU 0 last), KV preflight (`expect_kv_total` 289,056, `kv_spread_max` 1 %, `continue_on_fail` off) so a short engine stops the chain before any run, `--reservations-required` so a refusal costs seconds | split 3–6, spill 1–3 | 7 | $6.0 |
| | | | | **62** | **$56.0** |

Plus bring-up (start or create pods, vLLM up, push, cluster smoke: ~40 min, ~$1.8), the engine
restarts between chains (~$0.5) and the grace hour at the end (3 pods idle, $2.7 at most; cancelled
once the pull is done). **Expected total ≈ $60**, leaving ≈ $2 by the plan's per-run cost (the 07:17 balance trajectory, $0.86 a run, says ≈ $6).

If the reserve is needed, trim from the end: stage H first, then `flash`, then `load1.5`. Stages
D–G are the core and are not trimmed. If ≥ $8 is left when the campaign is committed, the owner can
choose fp16 vs AWQ (~$1.5, one pod) or the A100 cross-check (~$4); both are separate goes.

Time: 51 runs ≈ 16.5 h over two chains (`matrix-campaign.yaml`, runs 1–14 of stage D, `--max-hours 19`;
then `matrix-campaign-2.yaml`, 37 runs with `nopace` first, started between runs by `chain switch`,
`--max-hours 14`), then ≈ 2.4 h for stage I (seven re-runs after an engine restart) and 1.5 h for stage H. About 20 h of
wall clock from the first paid minute; the chain runs unattended on the gateway pod and survives this
machine going down. As run: the second chain was stopped at 11:18 UTC on 2026-11-09 after 17 of its 37 runs
(section 6, the batch tenant's timeout); stage I went first, then the 22 remaining sweep runs
(`matrix-campaign-2-resume.yaml`, ≈ 7.5 h) by `chain switch`, and stage H only if the balance allows. Stage H ran after the owner's top-up (chunk 256 and 2048, two seeds each,
21:56–23:52 UTC), then the chunk-512 restore and stage J (section 9 B: four runs, 00:14–01:30 UTC on 2026-11-10); the pod
stopped itself at 01:50. In all 68 runs were executed on the campaign pod and 59 are in the report (the seven
pinned-reservation runs that stage I replaced and the two timed-out `load1.5-burstoff` runs stay on disk).

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
  derived in Q7. The line is fitted per load series (bursting off, bursting on: only the load changes
  between the points) beside a pooled line over every run of every setting, which is a campaign-wide
  tendency, not a controlled curve (`margin_curve.csv`, `family` and `series` columns; added 2026-11-09).
- **Floor, in two parts** (owner, 2026-11-09): the **decode floor** is the delivered rate from first to
  last token against 95 % of the shape's tps (`floor_violation`, unchanged: queue and TTFT are not in
  it), and the **start bound** is queue wait + TTFT (`start_s` = first token − arrival) against a
  2 s placeholder for interactive shapes (`start_violation`; lazy shapes have no start promise), with
  its p50/p95 so a reader can apply another threshold. Both are recomputed from the meter, so
  phase 2 gets them for free, and the lever runs (`prefillcap`, `chunk256`, `oc1.25`/`oc1.5`) are judged
  on each separately.
  First recompute (2026-11-09 05:20 UTC, 21 runs: phase 2 and the campaign's stage D so far), mean over
  seeds, interactive shapes: `burstoff` agent misses the decode floor on 41 % of judged requests and the
  2 s start bound on 22 % (start p95 5.0 s), chat 20 % / 20 % (p95 4.8 s), note-taker 22 % / 32 %;
  `burst` agent 56 % / 49 % (p95 8.2 s), chat 35 % / 46 % (p95 7.5 s): bursting worsens both, the
  start bound more. Under reservations the start bound is the lockout of Q6: reserved chat misses it
  on 70–80 % of requests with a start p95 of 31 s (`max_queue_s` is 30) while its decode floor share
  is 36–51 %. The two halves are not the same requests: the decode misses are the prefill
  interference of Q5, the start misses are queueing (admission and reservation queues) plus prefill
  time, and the lever runs must move the first without inflating the second.
- **Report order** (owner, 2026-11-09): `campaign.html` opens with a Headline section: the reserved tenants' bill CV
  across seeds under slot billing against the same tenants under per-token pricing over the same runs
  (`reserved_bills.csv`, with the on-demand baseline the manifest pairs the label with), then output tokens
  per GPU-hour and utilisation per setting and the break-even utilisation per scheme, before any revenue ÷
  cost ratio. A reservation-integrity warning (`reservations.csv`: every reservation the load generator
  asked for and the gateway's answer, per seed) comes before all of it when a label's runs mix a granted
  and a refused reservation. The start bound joined the headline metrics (Holm family 7 pairs × 7 metrics).
  `runs.csv` records every replica's KV budget per run (`kv_profile`); `capacity.csv` splits the capacity-bound
  metrics per profile and the page marks every label that pools two (section 6, the owner's second addition).
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

Changing the plan mid-chain: the chain reads its matrix once, so `pod.sh chain <gw> switch
<label-sN> <matrix> [args]` arms `scripts/chain_switch.sh` on the pod, which waits for that run's
per-run report (the chain's last act for a run), kills the chain and gateway between runs and starts
the new matrix; nothing paid is lost, the new matrix lists only the runs still to do with
`fresh_runs: false` so `chain.runs` stays complete, and the chain's signal handler does not stop the
pod. Armed 2026-11-09 05:10 UTC for `resv-shared-s6` → `matrix-campaign-2.yaml --max-hours 14
--stop-pods --stop-grace-min 60` (the owner's order change and `resv-p95-spill`); fired 05:44:53, the
second chain's first run (nopace s1) passed preflight at 05:45.

**The note-taker's reservation (found 2026-11-09 07:30 UTC, from the first `reserved_bills` table).** Every
phase-3 run that pins reservations to replica A (`resv-split` s3–s6, `resv-spill` s1–s3) ran with chat's
24 slots and *without* the note-taker's 4: the gateway answered the second `POST /admin/reservations` with
`409 reservation_capacity: 4 x small.16k.30tps does not fit (kv; 3 placed)` and the load generator logged a
warning and went on. The unpinned runs (`resv-shared`, all seeds) got both. Cause: on the campaign pod the
engine on GPU 0 (replica A) profiled 271,152 KV tokens (257,594 usable) against 289,056 (274,603) on GPUs 1
and 2 and on every phase-2 pod, the same `--gpu-memory-utilization 0.90` and the same free memory at start;
vLLM's memory profile on that engine measured 1.16 GiB of peak activation against 0.21 GiB on the other
two (it was started first, with a cold compile cache), and the KV budget is fixed at engine start. Chat's
24 × 8k plus the note-taker's 4 × 16k need 262k of replica A's usable KV: 274.6k held it in phase 2,
257.6k does not. Consequences and response: (1) `report/campaign.py` now tabulates every reservation the
load generator asked for (`reservations.csv`) and warns at the top of the page when a label mixes granted
and refused seeds; `reserved_bills` keeps a tenant only when its reservation held in every run of the
label, so the note-taker is absent from `resv-split`/`resv-spill` until the re-runs land. (2) The load
generator gained `--reservations-required`: a refused t=0 reservation ends the run with exit 3 before any
load is sent (tested against the in-process gateway), so a run whose setting *is* the reservation can
never again run on demand with a warning. (3) Stage I (`matrix-rerun-resv.yaml`): after the second chain
is DONE, the engines are restarted with the RUNBOOK C2 argument line (a warm restart should give GPU 0
the full 289k; the first run's gateway log shows `ledger runpod-4090-a: KV ...`), then the seven runs are
repeated with the flag, pairs first (`resv-split` s3, `resv-spill` s1–s3, then `resv-split` s4–s6), ≈ 2.4 h
and ≈ $6 within the approved budget; the campaign report keeps the latest start per (label, seed), so
they supersede the broken runs, which stay on disk and in `chain.runs`. Stage H follows. The runs still
in the second chain are unaffected: `resv-p95` and `resv-p95-spill` are unpinned and spread over three
replicas, everything else runs without reservations.

**Owner's decision (2026-11-09 08:55 UTC): A, re-run**, "reservation predictability is the headline claim, and
two clean seeds for one label and zero for the other won't support it", with two additions. (1) **KV preflight
before stage I.** `scripts/preflight.py` checks every replica's KV budget: `--expect-kv-total` (the ledger's
`kv_total` within 1 % of 289,056) and `--kv-spread-max` (the replicas within 1 % of each other); `runchain.sh`
passes the matrix's `expect_kv_total` / `kv_spread_max`, and `matrix-rerun-resv.yaml` carries both with
`continue_on_fail: false`, so a short engine stops the chain at the first preflight (about two minutes of pod
time) instead of failing seven runs at reservation time. `pod.sh vllm <p> status` prints each engine's KV
from `/metrics`, and `START_ORDER="1 2 0" STAGGER=1 pod.sh vllm <p> restart ...` starts GPU 0 last, after
GPUs 1 and 2 are healthy and the compile cache is warm (the stage-A calibration pods, whose first engine also
started cold, profiled the same 271,152: `calibration/runpod-4090-*.yaml`; every phase-2 run had 289,056 on
all three replicas). The stage H matrices carry `kv_spread_max: 0.01` only, because the chunk size moves the
absolute budget. (2) **KV per replica per run in the report.** The load generator already stores the gateway's
`/admin/replicas` snapshot in `loadrun.json`; `report/campaign.py` now reads it into `runs.csv` (`kv_profile`,
`kv_total_min`), writes `capacity.csv` (per label and KV profile: refusal rate, start-bound violation rate,
start p95, utilisation, with the agent tenant's planned steps beside them as the load covariate), counts
`kv_profiles` per label in `settings.csv`, and the page carries a "Capacity per replica" card in the Overview
and a capacity note in the Headline for every label pooling two profiles (`burstoff`, `burst`, `resv-split`,
`resv-shared`: seeds 1–3 on the phase-2 pods, 3–6 on the campaign pod). The quick check the owner asked for,
stage D seeds 4–6 against 1–3 on the runs pulled by 08:50 UTC: refusal rate `burst` 2.8 % → 0.9 %, `burstoff`
4.5 % → 2.5 %; start-bound violation rate `burst` 64 % → 32 %, `burstoff` 28 % → 13 %; start p95 `burst` 9.2 s
→ 6.3 s, `burstoff` 5.8 s → 4.0 s; utilisation 54 % → 38 % and 48 % → 39 %; the agent tenant's planned steps
3,401 → 1,616. The shift is large and towards *less* pressure, which is the seeds' lighter load, not the 2 %
less cluster KV; the capacity effect cannot be separated from the seed effect in this design, so the write-up
reads refusals and the start bound per KV profile (and paired, per seed) rather than pooled across the two
pods, as the owner asked. Paired comparisons are unaffected: both arms of a pair share the seed and the engines.

**The batch tenant's timeout (found 2026-11-09 11:15 UTC, from `chain.failed`).** `load1.5-burstoff` s1 and s2
ran to the end, but the load generator exited 1 with 3 and 12 errors out of 7,992 and 7,908 requests, so the
chain recorded them as FAILED and went on (`continue_on_fail`). Every error is a client-side `ReadTimeout`
after 600 s on the batch tenant's `small.8k.15tps.lazy` requests (complete delivery, so the timeout spans
queue plus generation); none is a gateway error: the gateway answered 200, 429 (`context_exceeds_shape`) and
503 (`capacity:seat`) only, and its meter has no error rows. At 1.5 × the phase-2 arrival rate the interactive
queue stayed deep through the 480 s peak (up to 275 waiting), and lazy work is admitted only into capacity no
interactive request is waiting for (DESIGN.md 4.1), so the lazy queue (the batch tenant's 12 closed-loop
workers plus a showcase) stood at 12–14 from about t = 330 s to t = 1,030 s; lazy requests waited up to 585 s
(s1) and 624 s (s2) before admission, inside the shape's 3,600 s contract, and those past the load generator's
blanket 600 s read timeout were counted as errors. The gateway served and billed them regardless: in s2 eight
batch requests were admitted after their client had gone (2,789 output tokens delivered to a closed socket),
because a request waiting in the queue is not watched for a disconnect; that is a finding for the write-up.
The gateway fix was built the same afternoon while the sweep ran (DESIGN.md 4.1 step 5 and section 12,
"Abandoned queued requests": a queued request whose client disconnects leaves the queue unserved and
unbilled, metered as `client_disconnect` with reason `client_disconnect_queued`), but it was not pushed to
the pod: the campaign runs on the gateway without it, and no paid run is affected once the load generator
waits out the shape's queue contract. The chain was stopped at 11:18 UTC with `load1.5-burst` s1 twelve minutes in (killed; run again later):
every later load-1.5 and load-2 run would have failed the same way, and a timeout that does not match the
contract is a flaw in the measuring instrument, not a result. Fix: `/v1/models` now lists each shape's
`max_queue_s`, and the load generator sets every request's read timeout to that plus a 600 s generation
allowance (630 s for the interactive shapes, 4,200 s for the lazy ones; `loadrun.json` records
`request_timeouts_s`), so a request still queued at the drain deadline is cancelled and counted as
`cancelled`, never as an error; tested against the in-process gateway (`tests/test_loadgen.py`,
`tests/test_gateway.py`). Nothing else in the gateway changed; the pod ran the code of 2026-11-08 until this
push, which also brought the load generator's `--reservations-required` and the chain's KV preflight. Order
change: with the chain stopped and a push due anyway, the engines were restarted at once with GPU 0 last, so
stage I runs first and the 22 remaining sweep runs follow it (`matrix-campaign-2-resume.yaml`, armed with
`chain switch` after `resv-split-s6`), all with 289,056 KV tokens on every replica; only the 31 phase-3 runs
finished before 11:18 carry the short replica A. The two failed runs are pulled and kept
(`runs/load1.5-burstoff-s1-20261109-102828`, `runs/load1.5-burstoff-s2-20261109-104720`) as the record of
lazy starvation at 1.5 × load; the campaign report takes the re-runs (latest start per label and seed).

**First look, 2026-11-09 03:30 UTC, six runs in (burstoff and burst, seeds 4–6).** Seeds 4–6 offer
less load than seeds 1–3. The agent tenant's arrivals are an on/off process (`bursty`: mean on 120 s
at 33/min, off 180 s at 5.5/min after the ×11 multiplier) with only three or four cycles in an
18-minute run, so its planned sessions swing by seed: 319/300/404 for seeds 1–3, 144/157/204 for
4–6, while chat (529–607 sessions) and the note-taker (178–235) are steady. Burstoff therefore bills
$0.92–1.15 in seeds 4–6 against $1.29–1.42 in 1–3, burst $1.31–1.50 against $1.70–1.80, and
utilisation runs 33–44% against 48–55%. Within a seed the comparison holds: burst over burstoff is
+42/+26/+31% in seeds 4–6 (phase 2: +35/+31/+26%); over six seeds, +$0.377 [+0.314, +0.439], paired
t p < 0.0001, Wilcoxon p = 0.031 (the smallest value six pairs allow), Holm-adjusted 0.0003. The
pairing with phase 2 is intact: the seed-3 plan the pod generated for run 7 is identical to phase 2's
(534/404/180 sessions) and the load generator is unchanged since 9fd22cc. Consequences: (1) the
campaign report carries `plan_open_steps` (the plan's open-loop requests) per run and per setting as
the load covariate, and the write-up compares settings within seeds, never seeds 4–6 against 1–3 as
if they were one load; (2) the lighter seeds are a gain, not a loss: they add points at 33–44%
utilisation to the margin curve, which phase 2 could only extrapolate towards break-even; (3) the
floor-violation worsening under bursting, significant at three seeds (p 0.024), is not at six
(+9 points, CI −4 to +23, p 0.14): at the lighter load the difference is a few points either way
(seed 5: 23% bursting vs 32% not). The pod is healthy: 32 vCPUs at a load average of 2, GPUs at
61–67 °C with no throttle reason, zero preemptions, no failed run, ~19 min per run (DONE expected
≈ 16:45 UTC).

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

**Stage H without a top-up (2026-11-09 18:13 UTC).** The owner's call at 18:05: "run what we can until
zero, then I can top up where we need". Balance $15.08 at 18:03, which reaches zero at about 23:40 UTC at
$2.68/h; the sweep's 22 runs end about 21:25 and stage H (two engine restarts, four runs) needs about 100 min
more, so it ends about 23:05 with the pod stopping itself about 23:25: inside the balance if nothing slips,
with no room for a retry. So that no paid minute waits for an operator, stage H is chained onto the sweep by
`scripts/stage_h.sh` (RUNBOOK C2 "Stage H"), armed at 18:12:51 in tmux session `stageh` on the pod: it waits
for the sweep's DONE, cancels the chain's pod stop, restarts the engines per chunk size (GPU 0 last, one
retry), runs `matrix-chunk256.yaml` and then `matrix-chunk2048.yaml`, then (section 9, decided 18:40) restores
the campaign's chunk of 512 with the KV budgets checked and runs `matrix-stage-j.yaml`, the last chain, which
stops the pod 20 min after its DONE. Rehearsed in WSL against a fake chain, a fake runchain and a fake restart (one simulated restart
failure, retried), including the prompt cancel of the chain's grace-period stop. The engine restart body moved
from `pod.sh` into `scripts/engines_restart.sh` unchanged, so the driver and `pod.sh vllm restart` run the same
code. Risk accepted with the owner's call: if the balance reaches zero before the last run is pulled, RunPod
stops the pod, the run in flight is lost and the pod cannot be restarted on its host (no free GPU there); a
top-up of about $10 before about 22:30 removes it.

**As run (2026-11-09 21:17–21:56 UTC).** The sweep ended `DONE 22` at 21:17:43 with 0 request errors in every
run (58 phase-3 runs, 405,606 requests, 0 errors; every run local with its meter). The driver killed the chain
session at 21:17:57 and `chain.log` confirmed "pod stop cancelled". The chunk-256 restart's first attempt came
back healthy but failed the KV check: engine 1 reported 271,216 KV tokens against 289,680 on engines 0 and 2
(spread 6.3 %, limit 1 %). The second attempt (21:37–21:55) gave 289,680 on all three (18,105 blocks of 16;
624 tokens more than chunk 512's 289,056, the smaller chunk's smaller activation peak) and `matrix-chunk256.yaml`
started at 21:55:51 (`--max-hours 2`). Without the check the chunk-256 runs would have run with one replica 6 %
short of KV, the confound the preflight was carried over to catch; the retry cost 19 min (≈ $0.85). This is the
third occurrence of the same pattern (the stage-A calibration pods' first engine, GPU 0 at the campaign pod's
bring-up, now engine 1 at a chunk change): the first engine started with a cold compile cache profiles about 6 %
less KV, a warm restart gives every engine the same budget, and the cause of the larger profiled peak is not
investigated. Promoted to a finding in its own right (ASSUMPTIONS "Phase 3 measurements", RESULTS Q12, DESIGN
sections 12–13): vLLM's KV budget is not deterministic across engine starts, so capacity sold by KV needs a
margin of that order or a post-start gate like this one. Prediction, written before the event: the chunk-2048
restart changes the compiled shapes again, so its first attempt should fail the KV check the same way and the
second pass, about 19 min and $0.85 later; the chunk-512 restore for stage J has a warm cache and should pass
first time. **Outcome (23:14 UTC): confirmed.** The chunk-256 chain ended `DONE 2` at 22:33:41 (both seeds, 14,091
requests, 0 errors, 0 preemptions); the chunk-2048 restart's first attempt came back healthy at 22:53:48 with
engine 1 at 270,896 KV tokens against 285,584 on engines 0 and 2 (spread 5.1 %), the KV check failed, and the
second attempt gave 285,584 on all three at 23:14:28 (17,849 blocks; 3,472 tokens below chunk 512's 289,056,
the larger chunk's larger activation peak); `matrix-chunk2048.yaml` started at 23:14:28. Fourth occurrence, first
engine started every time, retry 20 min (≈ $0.90). **Second outcome (00:14 UTC on 2026-11-10): confirmed too.**
The chunk-2048 chain ended `DONE 2` at 23:52:16 (15,296 requests, 0 errors, 0 preemptions); the chunk-512 restore's
first attempt (23:52–00:14, three engines started one after another) gave 289,056 KV tokens on all three against
the expected 289,056, as predicted for a warm compile cache; `matrix-stage-j.yaml` started at 00:14:30 with
`--stop-pods --stop-grace-min 20`, run 1/4 `resv-p95-spill s3` PREFLIGHT OK with the KV checks; the driver wrote
HANDED-OVER and exited. Two predictions written before the events, two confirmed: the cold first start comes up
short, the warm one does not. Operator note: `pod.sh chain <gw> status` shows the driver's WAITING
line until the new chain starts, because `stageh.status` is rewritten only at WAITING, RUNNING (after the chain
start), HANDED-OVER, ABORTED and FAILED; the restart's progress is in `runs/phase2/stageh.log`.

**Stage J (00:14–01:30 UTC on 2026-11-10).** The four runs (`resv-p95-spill` s3 at 00:14, `resv-p95` s3 at 00:33,
`resv-p95-spill` s4 at 00:52, `resv-p95` s4 at 01:11) ended `DONE 4` at 01:30 with 0 request errors (7,185 / 7,248 /
6,953 / 6,953 requests) and both reservations granted in each; every run was pulled as it finished and the pod stopped
itself at about 01:50, as armed. The final report was built at 01:45 over the 78 run directories with meters (69 kept by
the manifest: the ten phase-2 runs and 59 of the 68 phase-3 runs; 482,123 requests) and copied to
`docs/reports/phase3-campaign/`. Not pulled: the pod-side `chain.log` and `stageh.log`, because the pod had stopped
before the copy was attempted; what they held is in this section, in section 7 and in `runs/phase3/poll.log`. The pod
was terminated at 01:56 UTC with every run local; the balance at the end was $9.16.

## 7. Spend log

| When (UTC) | What | Pod-hours | Cost |
|---|---|---|---|
| 2026-11-08 06:28 – 11-09 00:20 | three stopped phase-2 pods' volumes until terminated | – | ~$0.30 |
| 2026-11-09 00:23–00:36 | three fresh one-GPU pods (EUR-IS-2): same host, no network between them; terminated | 3 × 0.22 | ~$0.58 |
| 2026-11-09 00:39–01:20 | one pod with three 4090s (`qka0zggy47mw7e`, EU-RO-1; the EUR-IS-2 request had no 3-GPU instance): bring-up; vLLM's pip install onto the network volume took ~25 min, then one engine per GPU (KV 271,152 tokens each, as in phase 2), push, setup, cluster smoke (148 ok, 2 refused, 0 errors, work on all three engines) | 0.68 × 3 | ~$1.82 |
| 2026-11-09 01:20 → | the campaign chain: `matrix-campaign.yaml`, 49 runs, `--max-hours 19 --stop-pods --stop-grace-min 60`; runs pulled every 10 min; ~19 min per run | | $2.67/h |
| 2026-11-09 05:05 | balance $49.92 after 12 runs: $12.41 spent since the plan ($62.33) | | |
| 2026-11-09 05:44:53 → | `chain switch` fired 3 s after resv-shared s6's report: second chain `matrix-campaign-2.yaml`, 37 runs, `--max-hours 14 --stop-pods --stop-grace-min 60`, nopace s1 first; no run lost, `chain.runs` kept its 14 ids | | $2.67/h |
| 2026-11-09 07:17 | balance $43.88 after 18 runs (nopace s1–2 and resv-spill s1–2 done, 0 errors each; resv-spill s3 in flight): $6.04 for the last 6 runs, $18.45 since the plan; 37 runs ≈ $32 left, so ≈ $11 remains at the end | | |
| 2026-11-09 07:30 | found: `resv-split` s3–s6 and `resv-spill` s1–s3 ran without the note-taker's reservation (409, replica A's KV deficit; section 6). Stage I added: seven re-runs after an engine restart, ≈ $6; the chain goes on unchanged | | |
| 2026-11-09 08:48 | balance $39.87 after 23 runs (resv-p95 s1–2 and resv-p95-spill s1–2 got both reservations; abort s1 in flight; 0 errors). Decision memo for the owner: `docs/DECISION-stage-I.md` (re-run the seven or skip; default re-run at DONE) | | $2.68/h |
| 2026-11-09 08:55 | owner: **A, re-run** (memo answered), plus a KV preflight before stage I and KV per replica per run in the report (section 6); stage I cost unchanged ≈ $6; 24 runs pulled, abort s2 in flight, 0 errors | | |
| 2026-11-09 11:15 | found: `load1.5-burstoff` s1–s2 FAILED on 3 and 12 client `ReadTimeout`s (600 s) on the batch tenant's lazy requests, which waited up to 624 s in the lazy queue through the 1.5 × peak (contract 3,600 s); gateway errors 0 (section 6, the batch tenant's timeout). Chain stopped 11:18 with `load1.5-burst` s1 at 12 min; 31 runs local (29 clean) | | |
| 2026-11-09 11:25 | balance $33.17 after 31 runs. Load generator fixed (request timeout = the shape's `max_queue_s` + 600 s), pushed with the KV preflight and `--reservations-required`. Engine restart GPU 0 last: the first attempt (11:23) had no `--port 8000` in the line, engine 1 bound 8000, 15 min lost; `restart` now refuses such a line | ≈ $0.7 lost | $2.68/h |
| 2026-11-09 11:58 | second restart: all three engines **289,056** KV tokens (GPU 0 last, warm compile cache): the cold-first-start explanation holds | | |
| 2026-11-09 12:03 | stage I started (`matrix-rerun-resv.yaml --max-hours 3`, PREFLIGHT OK incl. `kv_budget`/`kv_uniform`; replica A ledger KV 289,056, usable 274,603). `chain switch` armed: after `resv-split-s6` → `matrix-campaign-2-resume.yaml --max-hours 9 --stop-pods --stop-grace-min 60` (22 runs ≈ 7.5 h ≈ $20). A first arming at 12:02 fired at once on the *first* chain's `resv-split-s6` report line in the shared `chain.log` and killed stage I eight seconds in (≈ $0.05); `chain_switch.sh` now reads only lines written after it was armed. Forecast: stage I DONE ≈ 14:25, resume chain DONE ≈ 22:00, pod stop ≈ 23:00 UTC with ≈ $2 left, so stage H (≈ $4.3) needs a top-up of ≈ $10 or is skipped | | |
| 2026-11-09 14:03 | balance $25.80; stage I 6/7 done, every re-run so far with both reservations granted (201/201) and 0 request errors | | |
| 2026-11-09 14:16 | **stage I DONE 7/7**: all seven re-runs got both reservations and finished with 0 request errors (`reservation_busy` refusals of chat requests in `resv-split`: 138–210 per seed, as in phase 2; 0 in `resv-spill`). The switch fired on the *new* `resv-split-s6` report line at 14:16:43 and started the resume chain at 14:16:47: run 1/22 `load1.5-burstoff s1` PREFLIGHT OK with the KV checks, per-shape request timeouts in force (lazy 4,200 s), 0 errors at t=190 s. Forecast: 22 runs DONE ≈ 21:20, pod self-stop ≈ 22:20 UTC with ≈ $3 left; stage H still unfunded | | |
| 2026-11-09 16:02 | balance $20.44; sweep 5/22 done, all with 0 request errors and 0 cancelled (load1.5-burstoff s1/s2 re-runs 637 and 578 refused; load1.5-burst s1/s2 638 and 564 refused, 2 and 1 revoked; load2-burstoff s1 978 refused), every run local with its meter; run 6/22 load2-burstoff s2 running. Forecast unchanged: DONE ≈ 21:30, pod self-stop ≈ 22:30 UTC with ≈ $3 left | | |
| 2026-11-09 18:13 | balance $15.08 (18:03); sweep 12/22 done, 0 request errors in every run so far (run 13/22 prefillcap s1 running); owner: run what we can until zero, top up where needed → stage H chained onto the sweep by `scripts/stage_h.sh` (armed 18:12:51); forecast sweep DONE ≈ 21:25, stage H DONE ≈ 23:05, pod self-stop ≈ 23:25, balance zero ≈ 23:40 | | |
| 2026-11-09 18:16 | owner topped up $15 ("to handle lulls where you need my response but I am asleep/busy"): balance $29.41, zero ≈ 05:15 UTC on 2026-11-10; stage H now ends with ≈ $15 left (sweep DONE ≈ 21:25, stage H DONE ≈ 23:05, pod self-stop ≈ 23:25), so a slow engine restart or a retry no longer costs the last run | | +15.00 |
| 2026-11-09 18:31 | owner: "Ship it, with a few adjustments" (section 9): stage J (`matrix-stage-j.yaml`, resv-p95-spill + resv-p95 seeds 3–4, ≈ $4) chained after stage H as the driver's third step with chunk 512 restored and the KV budgets checked from `/metrics` and again by the matrix preflight (`expect_kv_total` 289,056, `continue_on_fail` off); driver re-armed 18:30:46 (sweep 13/22 done, run 14 prefillcap s2 running). Forecast: sweep DONE ≈ 21:25, stage H ≈ 23:05, stage J DONE ≈ 00:40, pod self-stop ≈ 01:00 UTC on 2026-11-10 with ≈ $11 left; then write-up, terminate, rotate | | |
| 2026-11-09 21:56 | balance $19.81. **Sweep DONE 22** at 21:17:43, 0 request errors in every run (58 phase-3 runs, 405,606 requests); the driver cancelled the pod stop at 21:17:57. Chunk-256 restart: attempt 1 failed the KV check (engine 1 271,216 vs 289,680 KV tokens), attempt 2 uniform at 289,680; `matrix-chunk256.yaml` started 21:55:51. Forecast: chunk 256 DONE ≈ 23:15, chunk 2048 DONE ≈ 00:55, stage J DONE ≈ 02:35, pod self-stop ≈ 02:55 UTC on 2026-11-10 with ≈ $6 left (each further failed restart attempt ≈ $0.85) | ≈ $0.85 lost | |
| 2026-11-09 23:19 | balance $16.01. **Chunk 256 DONE 2** at 22:33 (two seeds, 0 errors; the chunk matrices are two runs each, so the 21:56 forecast was 40 min too long per chunk). Chunk-2048 restart: attempt 1 failed the KV check as predicted (engine 1 270,896 vs 285,584), attempt 2 uniform at 285,584; `matrix-chunk2048.yaml` started 23:14:28. Forecast: chunk 2048 DONE ≈ 23:53, chunk-512 restore (warm cache, one attempt) ≈ 00:14, stage J (4 runs) DONE ≈ 01:30, pod self-stop ≈ 01:50 UTC on 2026-11-10 with ≈ $9 left. First read of chunk 256 against `burst` (seeds 1–2): floor misses 44 → 53 %, start p95 6.7 → 15.8 s, output per GPU-hour −8 %, margin −3 points: the smaller chunk makes the floor worse, the opposite of the hypothesis (section 9 C's premise) | ≈ $0.90 lost | |
| 2026-11-10 00:16 | balance $13.56. **Chunk 2048 DONE 2** at 23:52 (0 errors). Chunk-512 restore passed the KV check first time (289,056 × 3, warm cache, as predicted); **stage J started 00:14:30** (4 runs, pod stop 20 min after its DONE); the stage-H driver handed over and exited. First read of chunk 2048 against `burst` (seeds 1–2): floor misses 44 → 35 % (p 0.05), delivered tps +12–18 %, output per GPU-hour +4 %, margin +1.4 points, TTFT p95 +22–43 %, start bound unchanged: the larger chunk helps the floor and costs the first token, the mirror of 256; neither replaces the prefill cap. Forecast: stage J DONE ≈ 01:31, pod self-stop ≈ 01:51 UTC with ≈ $9 left; then the final report, the write-up pass, terminate, rotate | | |
| 2026-11-10 01:50 | **Stage J DONE 4** at 01:30 (0 request errors in all four runs, 28,339 requests; both reservations granted each time). The pod stopped itself at ≈ 01:50; the final report was built at 01:45 (69 runs kept, 482,123 requests) and copied to `docs/reports/phase3-campaign/`. Balance $9.16: phase 3 cost $68.19 ($62.35 at plan time + $15.00 top-up − $9.16), both phases $85.80 of the $150 cap | | |
| 2026-11-10 01:56 | pod `qka0zggy47mw7e` terminated with its volume (`pod.sh terminate --yes`) once every run was local and the final report had landed; nothing is left on RunPod. The RunPod API key in `.env` is the owner's to rotate | | |
| | | | |

Balance at plan time: $62.35 of $80 (phase 2 spent $17.61; see PHASE2.md §3).

## 8. What the owner decides

1. **Go for the campaign as one approval** (sections 3 and 6, ≈ $52, ≈ 20 h), or stage by stage.
2. **Recreating pods B and C** if their hosts still have no GPU at start (their volumes hold nothing
   that is not local; the alternative is waiting for the hosts, with no way to know how long).
3. **The optional phase-2 experiments** (fp16 vs AWQ, A100) from what is left: asked again at the
   end with the actual balance.

## 9. After the campaign: the scorecard, two more seeds, and the backfill question

Decided 2026-11-09 18:40 UTC. The 18:20 assessment put the data so far in one sentence: predictability works
for the reserved tenant, but at the current prices it is paid for by the on-demand tenants' waiting time and
by the provider's idle seats, and the decode floor is not reliable under load. The owner's direction on the
five ideas offered ("Ship it, with a few adjustments"), and the order of what follows:

**A. Offline re-pricing from the existing meters, scorecard first.** Pricing does not change scheduling, so
the campaign's meters already contain every price table's revenue; the risk is overfitting, adjusting until all
three parties look happy. So the scorecard and its pass criteria are fixed here, before any price is tried, and
the result is the whole trade-off frontier (each setting under a grid of reservation price factors and SLA-credit
policies: reserved tenants' bill CV and bill, on-demand tenants' service, provider margin and output per
GPU-hour), not one chosen table. The write-up says plainly that offline re-pricing holds behaviour fixed: real
tenants would change what they buy and what they send if prices changed, which these meters cannot show.

The scorecard. "Baseline" is on-demand slot billing with bursting (`burst`) on the same seeds; "per-token" is the
same runs billed per token at `basket-7b` parity; every comparison is paired by seed.

| Party | Metric | Pass |
|---|---|---|
| reserved tenant | bill CV across seeds | at most 2 %, and below the same tenant's per-token CV on the same runs |
| reserved tenant | start-bound violation share (queue + TTFT over 2 s) and decode-floor violation share | not above the baseline's for the same tenant |
| on-demand tenants | start p95 | at most 1.5 × the baseline's on the same seeds and inside the shape's queue contract; refusal rate within one point of the baseline's |
| provider | gross margin at the price table under test | at least the baseline's gross margin under the campaign's price table, and at least the per-token margin of the same runs |
| provider | output tokens per GPU-hour | at least 95 % of the baseline's |
| everyone | decode-floor violation share per tenant | not above the baseline's |

The reserved tenant's bill against its per-token bill is reported on the frontier, not passed or failed: it
is the price of predictability, and the owner reads it. A price table can move the provider's row and the
reserved tenant's bill; it cannot move the on-demand tenants' waiting time or anyone's floor, which is why C
exists. The SLA-credit policy (a request whose decode floor was missed is refunded its runtime price, in whole or
in part) is on the same grid, so the frontier also shows the sellable fallback if the floor cannot be fixed:
best-effort throughput with credits. Added 2026-11-09 22:50 UTC on the owner's word, after the flash-crowd and
saturation results: a second credit basis, the start bound (a completed interactive request whose queue + first
token exceeded 2 s is refunded its runtime price, in whole or in part), because under a surge slot pricing holds
the bills fixed and moves the cost onto latency, and the contract that fails there is the start, not the floor;
`frontier.csv` carries a `credit_basis` column (`none`, `floor`, `start`) and the grid is factor × credit × basis.

**First result (2026-11-09 18:55 UTC, 50 runs, the sweep still running).** The frontier is built
(`report/frontier.py`: `scorecard.csv`, `frontier.csv` and the "Scorecard and re-pricing frontier" section of
`campaign.html`; the manifest's `scorecard:` block names the baseline and says that `resv-p95` runs the
`mixed-4090` workload with other reservation sizes). Over the runs so far only the baseline itself passes every
criterion. The reserved tenants' CV row passes in every reservation setting but `resv-spill` (6.9–9.7 %). The
provider's margin row passes only from a reservation factor of 4× (p95 sizing, with or without spill: 59 %
against the baseline's 54 % on the same seeds) or 5× (the 24 + 4 reservations: 51 % against 48 %), where the
reserved tenant pays 1.4–2.0× (p95) or 1.1–1.6× (24 + 4) its per-token bill. And no price table moves the
service rows: the p95 reservations put the on-demand tenants' start p95 at 2.4–2.5× the baseline's (agent 17–18 s
against 7 s, batch 17–18 s against 10 s) against the 1.5× limit, so they fail the on-demand row at every price;
the 24 + 4 reservations pass the on-demand row (they hold fewer seats) but fail their own reserved tenants'
service (`reservation_busy` refusals, a 30 s start p95) and, pinned, the floor row. A full SLA credit costs 1–4
points of margin at the campaign factor (`burst` 48.1 → 44.4 %). Read: the price of predictability is set by the
provider's margin criterion at about 4× the shape's price per second, where the reserved tenant pays roughly 1.4×
per-token for a bill that moves under 1 %; the on-demand tenants' waiting time is the cost no price table can
move, which is what C would address. Final numbers after stage J.

**Final (2026-11-10 01:45 UTC, 69 runs, four seeds of the p95 reservations; RESULTS.md Q13).** 245 price tables (6 factors ×
{none, floor credit, start credit} × {0.5, 1.0}) over 24 settings. Two settings pass every row: `burst` itself and, at the
campaign's prices, `chunk2048`; no reservation setting passes at any price. The provider's margin row is met from 3.2 × for
the p95 reservations (52.2 % against the baseline's 50.3 % on the four common seeds; the reserved tenant's bill 1.24–1.73 ×
its per-token bill) and from 5 × for the 24 + 4 reservations (51.5 % against 48.1 %; 1.06–1.48 ×), `resv-spill` from 4 ×.
The on-demand rows fail at every price under the p95 reservations: start p95 2.0–2.1 × the baseline's (agent 16.6 s against
8.1 s, batch 14.5 s against 7.4 s) and the agent's refusals 9.8 % against 7.1 %. A full credit at the campaign's factor
costs `burst` 3.7 (floor basis) / 4.0 (start basis) points of margin, `load1.5-burst` 5.7 / 7.9, `load2-burst` 5.4 / 7.4,
`flash-burst` 4.7 / 5.0, `resv-p95-spill` 1.4 / 2.4; the credit refunds the runtime part of the bill, not the prompt
charge, which is why crediting 88 % of requests at saturation costs 8 points and not half the revenue. What four seeds
changed against the two-seed read: the margin threshold for p95 sizing moved from 4 × to 3.2 × and the on-demand start-p95
ratio from 2.4–2.5 × to 2.0–2.1 × (seed 4 plans about half the agent load); no conclusion moved.

**B. Stage J, four runs tonight (≈ $4).** Two more seeds (3 and 4) of the configuration that would ship,
`resv-p95-spill`, and of its pair `resv-p95`, interleaved by seed so a lost last run still leaves complete
pairs (`config/phase3/matrix-stage-j.yaml`). Chained after stage H by the same driver (`scripts/stage_h.sh`, a
third step): the engines are restarted with the campaign's chunk of 512, which is exactly where GPU 0 came back
6 % short before stage I, so the owner's condition carries over: every engine's KV budget is read from
`/metrics` after the restart (within 1 % of each other and of 289,056; one retry, then the pod is stopped), and
the matrix repeats the check as its preflight with `continue_on_fail` off: every replica reports its full budget
before the first run or the chain stops. Stage J is the last chain of the campaign and stops the pod 20 min
after its DONE.

**C. Reservation backfill: the plan and its pass criteria come before any code; decided after tonight.**
The idea: an idle reserved slot is lent to a preemptable lazy request (complete mode, as the `requeue` policy
already preempts) and taken back the instant a request of the reserved tenant arrives: the lazy run is evicted
and requeued, its partial output wasted and unbilled, as a revoked burst is. It attacks the provider's cost of
predictability (idle reserved seats, 7 % of output per GPU-hour under the p95 reservations) and gives the batch
tenant cheaper capacity. The risk, named by the owner: reclaiming a seat is not instant. Freeing KV and aborting a
request mid-prefill takes time, and that delay lands on the reserved tenant's start bound. So the plan carries a
new metric, reclaim latency (reserved request arrival to admission when a backfilled hold had to be evicted), and
these pass criteria, paired by seed against `resv-p95-spill`: the reserved tenants' start p95 and start-bound
violation share not worse, their decode-floor violation share not worse, their bill CV still at most 2 %, output
per GPU-hour up, the batch tenant's queue p95 down. If the reserved tenants' start or floor gets worse, C fails
even if utilisation improves. Cost: about a day of code and tests (scheduler, fake replicas, rehearsal) and two
or three paired runs; the pod stops tonight and its host has no free GPU, so a fresh pod would also re-run the
`resv-p95-spill` baseline on the same hardware (two pairs ≈ $4, bring-up ≈ $2). Decision after tonight's prefill
cap and chunk results: if they fix the floor, C is the natural next step; if they do not, the sellable version of
the scheme is best-effort throughput with SLA credits, which A already models, and C is dropped.

**C. Reservation backfill: the decision (2026-11-10, after stage J).** Recommendation: drop C for this project and
record it as the next experiment, with the pass criteria as written. Reasons, in order:

1. The premise moved. C was the answer to "the reserved seats sit idle and the on-demand tenants wait"; the
   campaign found the floor problem is admission's (the prefill cap fixes the 30 tps floors, the chunk size only
   trades latency for steadiness, Q10 and Q12), the on-demand tenants' waiting under p95 reservations comes from
   51 of 96 seats being held (Q8; start p95 2.0–2.1 × the baseline's at four seeds, not moved by any price table,
   Q13), and the contract that fails under load is the start bound (Q9, Q11). Backfill lends idle reserved seats to
   lazy requests: it returns idle seats to the provider and shortens the lazy queue, which are real, but it does
   not move the on-demand interactive tenants' start p95 (they are not lazy) and it puts the reserved tenants'
   start bound at the mercy of the reclaim latency, which is exactly the pass criterion it has to meet.
2. The budget. Balance at the pod's stop $9.16. C needs a day of code and tests first, so this pod cannot wait
   for it ($2.68 per hour idle) and cannot be restarted on its host once stopped (no free GPU); a fresh three-GPU
   pod is ≈ $1.8 of bring-up plus ≈ $5 for a two-seed pair against `resv-p95-spill` with the KV gate, ≈ $7 with no
   room for a retry. The owner's rule was to run C only if money is genuinely left over; $9.16 is not that.
3. The write-up does not need it. The thesis holds without C: predictability works for the reserved tenant and is
   paid for by the on-demand tenants' waiting and the provider's idle seats at the campaign's prices; the frontier
   shows what price returns the provider's margin and that no price returns the waiting; the sellable version is
   the start-bound contract with an SLA credit (Q13). C is the experiment that would test whether the idle seats
   can be recovered without breaking the reserved tenant's start bound; it belongs to the next project, with the
   criteria below unchanged.

Pass criteria, as fixed on 2026-11-09 before any code (unchanged, for the record): against `resv-p95-spill` on the
same seeds, the reserved tenants' start p95, start-bound share and floor share not worse; their bill CV ≤ 2 %;
output per GPU-hour up; the lazy tenant's queue p95 down; a reclaim-latency metric (time from the holder's
arrival to its first token when a lent seat has to be taken back) reported per request, with the lazy request's
preemption counted as its cost. If the reserved start p95 or floor misses get worse than the shipping
configuration, C fails even if utilisation improves.

What a yes would cost and take, if the owner wants it anyway: a day of code (a `backfill` admission mode: a lent
seat is a lazy request admitted into an idle reserved slot with a preemption hook on the holder's arrival; the
meter records lent seconds and reclaims), the rehearsal against fake replicas, a fresh pod, two seeds of
`resv-p95-spill` with backfill on, ≈ $7 and an evening.

**D and E (a small reservation for the bursty agent tenant; lazy aging at 1.5 × load) are skipped** unless
money is genuinely left over: neither changes the headline.

**Then the write-up, and nothing else runs.** The owner's thesis, as of this evening: predictability works for the
reserved tenant; at the current prices it is paid for by the on-demand tenants' waiting time and the provider's
idle seats; the decode floor is not reliable under load. If the prefill cap or the chunk size fixes the floor,
the result gets better; if not, the sellable version is best-effort throughput with SLA credits. After tonight's
runs, stage J and possibly C: stop running experiments and write it up. The pod is terminated and the RunPod API
key rotated as soon as the last report lands, rather than letting the pod sit (owner's instruction, 2026-11-09
18:40 UTC; `terminate --yes` deletes the pod and its volume once every run is local).

**Done (2026-11-10).** The final report landed at 01:45 UTC and the pod was terminated at 01:56 UTC. The write-up pass
is RESULTS.md Q8–Q13 and its preamble and closing section, ASSUMPTIONS.md "Phase 3 measurements", DESIGN.md sections
11–13, SAMPLES.md "Output samples from phase 3" and the README's status; the campaign report and its CSVs are in
`docs/reports/phase3-campaign/`. C is dropped (above). Nothing else runs.

