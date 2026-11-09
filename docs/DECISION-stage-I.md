# Decision needed: re-run the seven pinned-reservation runs (stage I)

**Written** 2026-11-09 08:50 UTC for the project owner.
**Answer needed by** about 17:25 UTC today, when the second chain reaches DONE. If no answer arrives by then, the re-runs go ahead as the default.
**What the decision costs** about $6 of the approved $60 phase-3 budget. Balance now $39.87.

**Decided** 2026-11-09 about 08:55 UTC by the owner: **A, re-run**, with two additions, both implemented
(PHASE3.md section 6): a KV preflight before stage I (every engine's KV budget checked after the restart,
the chain stops before any run if one is short; GPU 0 restarted last, after the other two have warmed the
compile cache), and KV per replica recorded per run in the campaign report, with the capacity-bound metrics
split per KV profile wherever phase-2 and phase-3 seeds are pooled.

## The question

Seven phase-3 runs ran without one of their two reservations. Do you want them re-run after the chain finishes (about $6, 2.4 hours), or should the report read those labels on the seeds they actually have?

- **A. Re-run (recommended).** `resv-split` seeds 3 to 6 and `resv-spill` seeds 1 to 3 repeat after an engine restart, with the load generator set to stop before sending any load if a reservation is refused. The re-runs supersede the broken runs in the report; the broken runs stay on disk.
- **B. Skip.** The chunk sweep (stage H) starts at DONE instead. The write-up reads `resv-split` on seeds 1 and 2 for the note-taker and treats `resv-spill` as "chat reserved, note-taker on demand".

## What happened

Each reservation run asks the gateway for two standing reservations at t=0: chat (24 slots of small.8k.30tps) and the note-taker (4 slots of small.16k.30tps). The pinned labels place both on replica A. On the campaign pod the gateway granted chat and refused the note-taker in every pinned run:

| label | seeds that got both | seeds refused the note-taker | gateway answer |
|---|---|---|---|
| resv-split | 1, 2 (phase 2, separate pods) | 3, 4, 5, 6 | 409 reservation_capacity: 4 x small.16k.30tps does not fit (kv; 3 placed) |
| resv-spill | none | 1, 2, 3 | the same |

The load generator logged a warning and ran the note-taker on demand. The runs finished with 0 request errors, so nothing in the chain's monitoring flagged it. It surfaced on 2026-11-09 07:30 UTC in the first reserved-bills table of the new campaign report: the note-taker row was missing.

## Why

The engine on GPU 0 (replica A) profiled 6 % less KV cache at bring-up than the other two engines and than every phase-2 pod:

| | GPU 0 (replica A) | GPUs 1 and 2, and every phase-2 pod |
|---|---|---|
| KV tokens profiled | 271,152 | 289,056 |
| usable after the gateway's margin | 257,594 | 274,603 |
| peak activation memory in vLLM's profile | 1.16 GiB | 0.21 GiB |

Same free GPU memory at start, same `--gpu-memory-utilization 0.90`, same model and arguments. Engine 0 was started first with a cold compile cache, and vLLM fixes the KV budget at engine start. Chat's 24 × 8k plus the note-taker's 4 × 16k need 262k usable tokens. 274.6k holds it; 257.6k does not. This is a property of that one engine start on that pod, not of the code, the matrix or the shapes.

## What it does to the results

- `resv-split` pools two configurations: both tenants reserved on seeds 1 and 2, chat only on 3 to 6. Its chat row in the headline table is valid, because chat's reservation held in all six seeds. Its note-taker row is gone, and its paired comparisons against `burst` are confounded on four of six seeds.
- `resv-spill` is not the configuration its name says. Spill was on, but only chat was reserved, so the spill-versus-split pairing on seeds 1 to 3 compares different reservation sets.
- Unaffected: `resv-shared` (six seeds, both reservations held), `resv-p95` and `resv-p95-spill` (two seeds each, both reservations granted, confirmed in their run records). The data for the configuration that would ship is intact. Every other label runs without reservations.

## What is already in place

- The campaign report tabulates every reservation asked for and the gateway's answer (`reservations.csv`) and warns at the top of the page when a label mixes granted and refused seeds or never got a reservation. Committed; the suite's 117 tests pass.
- The load generator has `--reservations-required`: a refused reservation at t=0 ends the run with exit 3 before any load is sent. Tested against the in-process gateway.
- `config/phase3/matrix-rerun-resv.yaml` runs the seven, pairs first, with that flag. The procedure is in RUNBOOK section C2.
- Not verified: that an engine restart gives GPU 0 the full 289k. If it does not, each re-run fails within seconds (about $0.10 in pod time) and I stop and report rather than change the configuration unasked.

## Budget and time

| | now (08:48 UTC) | at DONE (about 17:25 UTC) | after stage I | after stage H (end) |
|---|---|---|---|---|
| A. re-run | $39.87 | about $17 | about $10.5 | about $6.5 |
| B. skip | $39.87 | about $17 | not run | about $13 |

Rate $2.68/h, about $0.85 and 19 minutes per run. 23 of the 55 planned runs are done, abort s1 is running, 0 errors so far. Stage I adds 7 runs and about 2.4 hours. The pod stops itself after stage H under either option.

## Recommendation

A. Reservation predictability is the headline claim of the write-up, the design is paired by seed, and $6 keeps it inside the approved budget with about $6 to spare. Option B leaves the two reserved-tenant labels with two and zero clean seeds respectively.

## To answer

Reply **"Yes, re-run"** or **"Skip them"**. Either way the chain in progress is untouched.
