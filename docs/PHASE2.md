# Phase 2 plan: three replicas on RunPod (milestone 5)

Status: **done through G3, 2026-11-08.** G0 (local work) built 2026-11-07: the two M4 fixes, output
capture, placement rules (pinned reservations, excluded shapes), the phase-2 configuration under
`config/phase2/` with the 7B price basket, the scenario `mixed-4090.yaml`, the run matrices, the pod
and chain scripts (docs/RUNBOOK.md), and a rehearsal on three fake replicas. Stage A (one pod,
calibration, smoke) and stage B (three pods behind the gateway on pod B) ran 2026-11-07/08; stage C
ran all ten matrix runs 2026-11-08 01:03–05:57 UTC in two chains, every run error-free, and the pods
were stopped by the chain. Results: ASSUMPTIONS.md "Phase 2 measurements" (stages A–C), RESULTS.md
Q5–Q7, SAMPLES.md; the report over the ten runs is `runs/phase2/report/report.html` (not in git).
Spend: **$17.61 of the $80 loaded** (section 3); the stopped pods still hold their 20 GB volumes
($0.017 per hour for the three) until terminated. Budget cap for the whole phase: **$150**, set by
the project owner. The optional experiments of section 3 remain affordable within the remaining
credit and each needs the owner's go; the pods' model caches survive on their volumes, but their
hosts may have no free GPU (RUNBOOK section C), so a rerun may mean new pods and a 10-minute
model download each.

## 1. Why move, and what the 2080 Ti could not answer

Milestones 1–4 answered Q1–Q4 (docs/RESULTS.md) on one RTX 2080 Ti with a 3B AWQ model. Three of the
headline findings are shaped by that card rather than by the product (ASSUMPTIONS.md, calibration):

- The replica is tiny. Per-stream decode falls from 105 tok/s alone to 35 at eight streams at 6k
  context, so the replica holds 8 seats at 30 tps and 2 at 60 tps. Every lockout finding (one agent
  request idling six slots, five reservation slots shutting the 60 tps shape out, context bursts
  unplaceable) is the nesting rule acting on a 2-seat ceiling.
- Prefill is slow. A 6k prompt takes 4–5.5 s and a decoding neighbour drops below 10 tok/s for the
  duration; that stall is the source of most floor violations. Ampere and newer cards with
  FlashAttention prefill the same prompt in a fraction of a second.
- The model and the market tier are the cheapest there are, so the break-even and margin numbers are
  directional only. A sellable product is a 7–8B or larger model on a rented card.

Phase 2 therefore answers, on rented hardware with real replicas:

| Q | Question | How the report answers it |
|---|----------|---------------------------|
| Q5 | Which M4 findings were artifacts of the small replica? | Seat ceilings per shape on the new card; burst on/off and reservation runs repeated: floor violations, lockouts, context-burst placement, revocations |
| Q6 | Do the scheduler's placement rules hold with real replicas? | Shape affinity across three replicas; reservations on their own replica with the 60 tps shape elsewhere; ledgers never exceeded; vLLM preemptions 0 |
| Q7 | What are the economics of a sellable configuration? | Q1 recomputed with the pod's real hourly price and a 7B-class price basket; the reservation bill at factor 2.3, and at any other factor, since the bill is linear in it |
| S | What does the output look like under the constraints? | Real prompts' completions saved per run and shown in the report and in docs/SAMPLES.md (section 6) |

## 2. Decisions (assumed unless the owner objects)

| Decision | Choice | Why |
|---|---|---|
| Card | RTX 4090 24 GB, RunPod Secure Cloud, on demand ($0.89/h in the October 2026 price snapshot) | Cheapest card with FlashAttention and fp8 KV; three pods test the scheduler for the price of one A100. Alternative: A100 80 GB ($1.79/h) if fp16 weights and 32k contexts matter more than replica count |
| Replicas | 3 identical pods | Shape affinity and "reservations on their own replica" need at least three: one for reservations, one for the fast shape, one general |
| Placement rules | `exclude_shapes` per replica and a `replica` pin on reservations (G0) | The simplest rules that express "reservations on their own replica": setting 3 pins the reservations to A and keeps the 60 tps shape off A; setting 4 runs the same reservations unpinned |
| Model | `Qwen/Qwen2.5-7B-Instruct-AWQ` | Ungated on Hugging Face (no licence click, no token on the pods); same tokenizer family as the 3B, so the load generator's token-exact vocabulary stays exact; AWQ weights (~5.5 GB) leave ~16 GB of KV and keep single-stream decode fast enough for a 60 tps shape. fp16 7B on a 4090 (~15 GB of weights, an expected 60–70 tok/s alone) would make 60 tps unsellable. Alternative: fp16 `Qwen/Qwen2.5-7B-Instruct` with `--kv-cache-dtype fp8` and 60 tps dropped from the catalog |
| Price reference | New basket `basket-7b` from OpenRouter via `slotgw-fetch-prices` (Qwen2.5 7B, Llama 3.1 8B, Mistral 7B class) | The 3B basket is the wrong tier for a 7B product |
| GPU cost knob | `provider.gpu_usd_per_hour` = the pod's actual price | Replaces the $0.30 placeholder |
| Shapes | Re-derived from the new calibration; keep 15 / 30 / 60 tps at 8k and 16k, add 120 tps only if `n_max(120)` is at least 4 | Calibration decides sellable shapes (DESIGN.md section 5) |
| Reservation price | `reservation_price_factor` 2.3 for the runs; the report also prints the bill at 1.0 and 3.2 | 2.3 covers the slice's cost share on the 2080 Ti; the bill is slots × seconds × price × factor, so other factors need no run |
| Where the gateway runs | On one pod next to its vLLM, with the load generator and the meter; replicas reached over RunPod global networking (internal DNS), never the HTTP proxy. It was pod B: pods A and C landed on one host and cannot reach each other (stage B) | WAN latency must not land in TTFT or pacing; RunPod's HTTP proxy cuts requests at about 100 s, which would kill batch streams. Verified in stages A and B |
| Orchestration | From this machine over SSH: `scripts/pod.sh` (rsync the tree, start and stop services, pull artifacts) and `scripts/runchain.sh` (the matrix, inside tmux on pod 1) | No registry and no remote repository needed; a dropped WAN link does not kill a run |
| Spend control | The owner loads exactly $150 of prepaid credit (the hard cap); pods are created per stage and stopped by the chain's watchdog when it finishes or when a wall-clock cap passes; `RUNPOD_API_KEY` lives in `.env` and is never printed | Idle pods are the only way to waste this budget |
| Scenario | `config/scenarios/mixed-4090.yaml`: the mixed workload re-scaled after calibration so the peak phase targets about 70% fill of the three-replica plan; reservation slots re-sized from the on-demand run's p95 occupancy, as in the what-if | The current rates load 6–10 streams; three 4090s would sit near idle |

## 3. Budget ledger

Expected spend is well under the cap; the cap exists for reruns, not for the plan.

| Stage | Pods × hours | Expected | Notes |
|---|---|---|---|
| G0 local work | 0 | $0 | fixes, output capture, pod scripts, fake-replica rehearsal of the exact phase-2 scenario |
| A: one pod | 1 × 1.5 h | ~$1.5 | image pull, model download, smoke, calibration, interference probe, long-stream check |
| B: cluster bring-up | 3 × 1 h | ~$3 | three replicas behind the gateway, cluster smoke, artifact round trip, report build |
| C: the matrix | 3 × ~4 h | ~$11 | 4 settings × 2 seeds, plus a third seed for the two on-demand settings: 10 runs of ~22 min |
| Storage | network volume for the HF cache | ~$1 | avoids re-downloading the model per pod restart |
| **Planned total** | | **~$17** | |
| Reserve | | ~$63 of the $80 loaded | reruns after a failed gate first; then the optional experiments below, in this order, while credit lasts |

Optional experiments, each only after the planned matrix is complete and committed:

| Experiment | Pods × hours | Cost | What it adds |
|---|---|---|---|
| fp16 vs AWQ on the same pod | 1 × 1.5 h | ~$1.5 | `Qwen2.5-7B-Instruct` fp16 with fp8 KV calibrated on the 4090 next to the AWQ build: seat ceilings and output samples side by side, the quality question for the write-up |
| Third and fourth seed for the reservation settings | 3 × 1.5 h | ~$4 | tenant-level numbers for the note-taker and the reservation bills stop being single-sample |
| A100 80 GB cross-check | 1 × 2 h | ~$4 | calibration, interference probe and one burst-on run of the sized scenario on the reference product card; shows how much of Q5 is the 4090 rather than "a modern card" |
| 120 tps shape and 32k note-taker | 3 × 1.5 h | ~$4 | only if `n_max(120)` is at least 4 on the 4090: one on-demand run with the agent tenant on 120 tps and the note-taker on a 32k shape |
| `abort` revocation policy | 3 × 0.75 h | ~$2 | only if the `grace` run shows revocations: the same seed under `abort` |

Spend log (RunPod credit; $80 loaded):

| When (UTC) | What | Pod-hours | Cost |
|---|---|---|---|
| 2026-11-07 22:59–23:27 | stage A, first pod on the vLLM image with the bootstrap start command: container idle for 26 min (no sshd, no engine), terminated | 0.46 | $0.41 |
| 2026-11-07 23:28 – 11-08 06:28 | pod A (PyTorch image, vLLM over SSH): stage A calibration, interference probe and smoke; stage B; stage C as replica A; stopped by the chain | 6.99 | $6.22 |
| 2026-11-08 00:30–06:28 | pods B and C: stage B bring-up and cluster smoke; stage C (gateway, meter, load generator and chain on B); stopped by the chain 30 min after `DONE` | 2 × 5.96 | $10.61 |
| 2026-11-08 20:30–20:56 | pod B CPU-only (`podResumeZeroGpu`, $0.445/h) to pull the runs: the host had no free GPU after the stop | 0.43 | $0.20 |
| since 2026-11-08 06:28 | three stopped pods' 20 GB volumes, $0.20/GB/month | – | $0.017/h until terminated |
| **total** | RunPod balance 2026-11-08 20:40 UTC: $62.39 of $80 | 19.4 | **$17.61** |

Of the ~75 minutes of idle pods between the first chain's stop (02:18) and the continuation (03:45),
about $3.30 was waste; the rest of the overrun against the ~$17 plan is the two stage-A/B hours the
pods sat up between stages.

Matrix settings (all on `mixed-4090.yaml`): (1) burst off, on demand; (2) burst on (`grace`), on
demand; (3) burst on, reservations at factor 2.3, reservations placed on replica A and the 60 tps
shape kept off it; (4) burst on, reservations with no placement rule, to measure the lockout on a
large replica. The `abort` policy is skipped unless setting (2) shows revocations.

## 4. What "error-free" means, and the gates that enforce it

A run counts only if the load generator reports `errors=0` (refusals and revoked context bursts are results, errors are not),
the gateway and all replicas stayed healthy for the whole run, vLLM preemptions stayed 0 under
guaranteed load, and the artifacts (meter rows, load samples, plan) are complete and copied back.
Everything that can be tested without a GPU is tested before a pod exists.

**G0, local and free.** Done before any pod is created (built 2026-11-07; see DESIGN.md section 12 for the decisions):

- The two hardware-independent fixes from M4: meter `cached_tokens` with cache-read pricing in the
  report (otherwise every phase-2 margin is an upper bound), and context bursts admitted at the
  slowest tps class with a short burst queue. Unit tests and the fake-replica HTTP tests cover both.
- Output capture (section 6), tested against the fake replica and exercised once on the 2080 Ti (a
  free smoke run), which also yields 3B samples for the write-up.
- `scripts/preflight.py`, run before every paid run: it refuses to start unless every replica is
  healthy and idle (free KV equals total, queues empty), the served model id and calibration file
  match the replica config, the tokenizer check is within 2%, the gateway run id is fresh (the
  gateway is recreated per run, so two runs never share a run id again), the disk has room, and the
  pod clocks agree.
- `scripts/runchain.sh`: runs the matrix inside tmux on pod 1, stops at the first run that fails any
  check above, pulls artifacts and rebuilds the report after every run, and stops the pods when done
  or when a wall-clock cap passes. Rehearsed end to end against three fake replicas on this machine.
- Three fake replicas on the 2080 Ti curve run the phase-2 scenario skeleton from a clean checkout
  with `docker compose --profile fake`; tests and ruff green; committed.

**G1, stage A (one pod).** Go/no-go for the cluster:

- vLLM serves the model and the served model id matches; a 5-minute streaming request completes over
  the chosen network path; `smoke.yaml` passes with `errors=0`.
- `scripts/calibrate.py` produces `calibration/runpod-4090.yaml` and the interference probe; the
  feasible-shape list and seat ceilings go into ASSUMPTIONS.md.
- From the calibration: the shape catalog is fixed, `mixed-4090.yaml` is sized, the fake replicas'
  curve is set to the new card, and the sized scenario is rehearsed locally against three fakes
  (free) before stage B. The pod is stopped as soon as the calibration is on this machine.

**G2, stage B (three pods).** Go/no-go for the matrix:

- All three replicas healthy behind the gateway with matching calibration; `smoke.yaml` places work
  on all three; artifacts round-trip and the report builds from them; the watchdog stops the pods.

**G3, stage C (the matrix).** Preflight before every run; the chain halts on the first failure
rather than continuing unattended; one rerun per failed run comes out of the reserve. Passed
2026-11-08: ten runs, ten preflights OK, loadgen errors 0 in every run, vLLM preemptions 0 on every
replica, artifacts pulled and the report built. The one halt was the first burst run exiting 1 on
15 revoked context bursts the load generator still counted as errors; the run was complete, the
classification was fixed, and the run was repeated in the continuation chain (its original is kept
under `runs/phase2/aside/`).

## 5. Deliverables

`calibration/runpod-4090.yaml`; `config/replicas.yaml` with the three pod entries and
`config/scenarios/mixed-4090.yaml`; `config/price_references.yaml` with `basket-7b`;
`scripts/pod.sh`, `scripts/runchain.sh`, `scripts/preflight.py`; the output-capture code and
`docs/SAMPLES.md`; ASSUMPTIONS.md "Phase 2 measurements"; RESULTS.md Q5–Q7; DESIGN.md sections
11–13 updated; the M5 commits. Each paid stage ends with its artifacts committed before the next starts.

## 6. Output capture (requested 2026-11-07)

The meter stores token counts, not text, so nothing from milestones 1–4 can be shown. Phase 2 adds:

- A `texts` table in the meter (`request_id`, `output_text`, and `prompt_json` for showcase requests)
  written for every request when `SLOTGW_STORE_TEXT=true`, the phase-2 default. Synthetic prompts
  are reproducible from the seed and `plan.json`, so only their outputs are kept; a run's output text
  is 1–2 MB.
- A showcase lane in the load generator: each tenant can name a prompt file
  (`config/prompts/<profile>.yaml`) of real, hand-written prompts in its own style (chat questions,
  an agent task whose context grows step by step, a long transcript for the note-taker to summarise,
  review classification for batch). Showcase requests are sent at a low fixed rate (about one per
  three minutes per tenant), carry `X-Slotgw-Showcase: 1`, pass through the same shape, admission,
  pacing and billing as everything else, and are flagged in `per_request.csv`.
- Report section "Output samples": for each showcase request, the prompt, the completion, and the
  constraints it ran under (shape, delivered tok/s, TTFT, throttled time, burst share, whether its
  context was a burst or it was refused or revoked, and its bill under both schemes).
  `report/samples.md` is exported per run and curated into `docs/SAMPLES.md` for the write-up, with
  the 2080 Ti's 3B samples next to the 4090's 7B ones.

Pacing changes when tokens arrive, not which tokens; what the samples show is model quality at the
size that fits the card, and what a tenant sees when the context limit, a burst revocation or a
refusal bites.

## 7. What the owner does (credentials never pass through the assistant)

1. RunPod account with exactly $150 of prepaid credit.
2. SSH public key added under RunPod account settings (pods are driven over SSH).
3. `RUNPOD_API_KEY=...` in the repo's `.env` (gitignored); the scripts read it, nothing prints it.
4. A go for each paid stage (A, B, C); the assistant creates and stops the pods through the API.

The API key is a live credential that spends money: the assistant never copies, prints or types it,
and the owner should rotate it in RunPod's settings once phase 2 is finished.
