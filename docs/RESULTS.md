# Results: what the proof of concept measured

Answers to the four questions of DESIGN.md §1, from the GPU runs of milestones 2–4 (2026-11-06 and
2026-11-07), and to the three phase-2 questions of PHASE2.md §1 (Q5–Q7, 2026-11-08). Setup for
Q1–Q4: one RTX 2080 Ti serving Qwen2.5-3B-Instruct-AWQ through vLLM, the four-tenant `mixed.yaml`
scenario (chat on `small.8k.30tps`, agent on `small.8k.60tps`, note-taker on `small.16k.30tps`,
batch on `small.8k.15tps.lazy`), 18 minutes per run, seed 1 unless stated, GPU priced at $0.30 per
hour, shape and per-token prices at parity with the `basket-3b` market reference (the median of
four 3B-class OpenRouter listings: $0.0755 per 1M prompt tokens, $0.15 per 1M output tokens). Setup
for Q5–Q7: three RunPod RTX 4090 pods ($0.89 per pod-hour) serving Qwen2.5-7B-Instruct-AWQ, the
same four tenants in `mixed-4090.yaml` at 11× the arrival rates, ten runs over four settings, prices
at parity with `basket-7b` ($0.08 / $0.175 / $0.015 cache read per 1M) with prefix-cache hits priced
at the cache-read rate. The measurements, run ids and caveats are in ASSUMPTIONS.md; the reports in
`runs/<run id>/report*` and `runs/phase2/report/`; what the output looked like is in SAMPLES.md.
Earlier milestones were billed at a gpt-4o-mini placeholder 4× higher; where a number comes from
them it is marked.

The one-paragraph version: time-sliced shapes with admission control turn a card that broke its
floors on 90% of fast streams into one that holds them 80–90% of the time, at the cost of queueing
(chat waits 10–13 s at peak) and 18% of throughput. At market parity the card earns 1.6–1.7× its
cost on this traffic, but only because prompt tokens are charged; the per-second price of decode
alone cannot pay for a GPU. Bursting gives back 12–15% of the throughput for free and halves
response times, at a few points of floor compliance. Reservations deliver a genuinely different
product (no queue, 13 ms) and, at parity prices, lose money; the price they need is 2.3–3.2×
parity. The 60 tps shape should not be sold on this card: it costs six of eight seats whenever it
runs, and any reservation locks it out completely.

The phase-2 paragraph (Q5–Q7, three RTX 4090s, a 7B model): the capacity findings were the small
card talking. On 32-seat replicas the 60 tps shape places freely, context bursts run and get
revoked, and reservations and the fast shape coexist. The floor findings were not: prompt tokens
outnumber output 15 : 1, the engine is prefilling for someone almost all the time, and the 60 tps
promise is missed on 59–83% of requests at 40–58% utilisation; bursting buys 32% more output per
GPU-hour with twice the violations. A reservation sized below its tenant's peak caps the tenant.
At the 7B market basket with cache hits priced as cache reads, the cluster earns 1.6× its cost
with bursting off and 2.1× with it on (gross margin 36–53%, break-even near 26% utilisation), and
a reservation at 2.3× cost share undercharges a well-used slot by a third.

## Q1. At what fill rate (and overcommit) does shape pricing break even for the provider?

**The runtime price never breaks even on its own; with the prefill charge the card breaks even at
about 28% utilisation, and overcommit does not move it.**

| | measured | source |
|---|---|---|
| plan of the replica | 8 slots × 30 tps = 240 tok/s at 100% fill | calibration curve, `n_max(30)` = 8 at 6k context |
| runtime revenue at 100% fill, parity price | 8 × $0.0162 = $0.13 per hour against $0.30 GPU | break-even fill 231% (report `per_replica.csv`) |
| same at the gpt-4o-mini placeholder | $0.52 per hour, break-even fill 58% | M3 |
| revenue per run with prefill at parity, on demand | $0.146–0.153 = 1.58–1.69 × GPU cost at 43–50% utilisation | M4 runs E, F, I |
| of which prefill | ≈ 90% (1.7–1.8M prompt tokens vs 115–130k output) | M3 market section |
| utilisation at which revenue = GPU cost | ≈ 28% (43% ÷ 1.58), 29% with bursting | linear in traffic |
| gross margin at the 70% target utilisation | ≈ 60% | 1.58 × 70 ÷ 43 |
| overcommit 1.5 / 2.0 | output +0% / +2%, agent floor violations 16% → 76% / 92% | M3 |
| reservations at parity | revenue 0.35–0.48 × GPU cost; the reserved slice earned 0.45× its cost share | M4 runs G, H |

What it means:

- Decode seconds are the smaller half of the bill on prompt-heavy traffic. A 30 tps slot at the
  output-token parity price is worth $0.0162 per hour; eight of them fully sold are $0.13, less than
  half the card. The prefill charge (`shape.prompt_usd_per_million`, `auto` = the reference's input
  price) is what pays for the GPU, and it is not optional in the shape model.
- Break-even utilisation of 28% is comfortable, but it is an upper bound: the meter charges every
  prompt token at the input price while vLLM serves most of an agent's growing history from the
  prefix cache, and providers that publish cache-read prices discount them 50–90%. Metering
  `cached_tokens` is the open item that could halve the margin.
- Overcommit does not help on this card: decode is compute-bound, extra seats split the same step
  rate, and floor violations return to the no-admission level. `SLOTGW_TPS_OVERCOMMIT` stays 1.0.
- A reservation sold at the shape's parity price is a loss. The slot needs ≥ 2.3× parity ($0.0375
  per 30 tps slot-hour, the card's hourly price over its eight seats) to cover its cost share, and
  ≈ 3.2× to match what the chat tenant paid per request at 21–24% used fill. `reservation_price_factor`
  is the knob; the config keeps 1.0 so the report shows the parity case.

## Q2. Which profiles get cheaper or more predictable bills under shapes, which are better off per token?

**At parity with the prefill charge the two bills are within 3% for every interactive tenant; shapes
win on predictability for chat, lose on price for the lazy tenant, and reservations are the tool
for the steady tenant once priced above parity.**

Per run, M4 burst-on run on demand (burst-off in brackets where it differs):

| tenant (shape) | ok | per-token USD | shape USD | shape ÷ per-token | CV across two seeds, per-token vs shape (M3) |
|---|---|---|---|---|---|
| chat (8k.30tps) | 267 | 0.059 | 0.057 | 0.97 | 0.26% vs **0.11%** |
| agent (8k.60tps) | 210 | 0.081 | 0.082 | 1.01 | 10.9% vs 10.1% |
| note-taker (16k.30tps) | 15 | 0.006 | 0.006 | 1.0–1.1 | 6.6% vs 9.0% |
| batch (8k.15tps.lazy) | 62 (33) | 0.005 (0.003) | 0.007 (0.005) | 1.4–1.5 | 10.0% vs 12.0% |
| chat on a 4-slot reservation | 283 | 0.064 | 0.019 (reservation) | 0.30 | 0 (fixed) |
| note-taker on a 1-slot reservation | 15 | 0.006 | 0.006 (reservation) | 0.97 | 0 (fixed) |

What it means:

- Once the prefill charge is at parity the shape bill tracks prompt tokens the way the per-token
  bill does, so the price level is the same and the difference is in the runtime part: the shape
  bill is capped at `ttft + tokens ÷ tps`, which makes it insensitive to how busy the engine was
  (M2: the cap cut $0.04 off a $0.13 wall-clock bill when floors were broken) and, with bursting, a
  little cheaper when the engine was idle (the burst discount, 4% of the bill).
- Chat, the only tenant with a sample large enough to judge, varies less under shapes (CV 0.11% vs
  0.26%); the others are within noise at two seeds. Bursty tenants (agent sessions, hour-mark
  note-taker) vary 7–11% under either scheme because their own traffic does.
- The lazy tenant pays 40–50% more under the lazy shape than per token with the batch discount,
  and under bursting gets 2–5× the throughput for it (62–163 completions per run against 33). Lazy
  work is better off per token if it only cares about price; the lazy shape sells it a queue
  position and a floor it does not need.
- A reservation makes the steady tenant's bill fixed and its queue disappear (13 ms p50 against
  10–13 s on demand). At parity it was 0.3× the per-request bill, which is the provider's problem,
  not the tenant's; priced at 2.3–3.2× parity it costs the chat tenant roughly what it pays per
  request today, for a product that queues nowhere. The bursty tenants are worse off reserving:
  the note-taker's slot was 88% idle and still cost it 0.9–1.0× per request.

## Q3. How much does throttling hurt each profile's latency?

**Pacing to the floor costs 20–50% of generation time where the engine is faster than the shape;
admission's queueing costs far more (10–13 s at peak) and is the price of holding the floors; the
60 tps shape is not deliverable alongside other work on this card.**

| | no pacing, no admission (M2, s1) | pacing + admission, burst off (M4 E) | + burst (M4 F) | on a reservation, burst off / on (M4 H / G) |
|---|---|---|---|---|
| chat delivered tok/s (shape 30) | 30.0 | 36 | 64 | 37 / 73 |
| chat TTFT p50 / p95 | 0.56 / 3.2 s | 0.52 / 4.4 s | 0.54 / 3.3 s | 0.50 / 1.3 s; 0.50 / 1.9 s |
| chat queue p50 / p95 | 0 | 10.4 / 22.7 s | 11.4 / 25.0 s | 0.01 / 2.2 s; 0.01 / 1.6 s |
| chat e2e p50 / p95 | – | 17.0 / 29.5 s | 16.8 / 30.1 s | 6.1 / 10.8 s; 3.4 / 9.4 s |
| chat floor violations | 52% | 14% | 18% | 10% / 10% |
| agent delivered tok/s (shape 60); violations | 38; 92% | 70; 18% | 72; 28% | locked out |
| agent queue p50 / p95; refused | 0; 34 context | 6.2 / 20.5 s; 32 context + 3 seat | 8.5 / 24.6 s; 34 seat | 38 seat |
| batch delivered tok/s (shape 15); e2e p50 | 15.3; – | 16; 33 s | 77; 8.3 s | 16; 38 s / 81; 6.1 s |
| throttled time per run | – | 2,058 s | 219 s | 2,390 s / 404 s |
| output tokens per run | 140k | 114k | 128k | 73k / 120k |

What it means:

- Throttling itself is mild. With burst off the pacer held tokens for 2,058 s per run; chat, which
  would be served at 64–73 tok/s, got 30–36, so its answers took about twice as long to arrive as
  they could have (e2e p50 6.1 s against 3.4 s where no queue is in the way). Batch, held to 15 tps
  on a 80 tok/s engine, waited 4–5× longer per job. TTFT does not depend on pacing.
- Queueing is where the latency goes: 10–13 s p50 and 23–25 s p95 for chat at peak, 7–8 requests
  per run refused after the full 30 s. That is admission holding the seat ceiling so the floors are
  honoured, and the floors are honoured: chat violations 52% → 14%, agent 92% → 18%, with the
  residual being prefill interference (A4: a long prompt still stalls every stream on this card;
  streams that overlap one violate 20–27% of the time, others 1–2%).
- The 60 tps shape is the expensive one. `n_max(60)` = 2, so an agent request idles six of eight
  seats while it runs; the agent queues 6–10 s p50, 1–14% of its requests fail on seats, and in the
  reservation runs it was never admitted at all. It is deliverable in isolation and not sellable
  next to anything else on this card: price it for the capacity it displaces, serve it on its own
  replica, or do not list it.
- A reservation removes the queue entirely for its tenant (p50 13 ms, p95 ≤ 2.2 s, 0 timeouts)
  and leaves the floors where they were.

## Q4. How much does bursting improve latency and utilisation, and does revoking bursts hurt guaranteed traffic?

**tps burst: +12–15% output per GPU-hour, chat response time halved, batch jobs 4× faster, for a
4% discount on the bill and 10 more floor violations per run. Context burst as built does not fit
on this card, so revocation never happened on the GPU and its effect on guaranteed traffic is
unmeasured.**

| burst off → on (`grace` / `abort`), on demand, same seed | | 
|---|---|
| output tokens per GPU-hour | 371k → 424k / 436k (+14% / +18%) |
| engine streams mean | 2.0 → 2.3 |
| burst tokens (share of output); lent rate per running request | 40–43k (32–33%); 56 tok/s |
| delivered tok/s chat / agent / batch | 36 / 70 / 16 → 64 / 72 / 77 |
| chat e2e p50 where it does not queue (reservation runs) | 6.1 s → 3.4 s |
| batch completions; e2e p50 | 33 → 62 / 70; 33 s → 7–8 s |
| throttled time | 2,058 s → 219 / 250 s |
| burst discount (bound − billed) | $0.002 → $0.006–0.007 (4% of the bill, 30% of its runtime part) |
| floor violations (agent share; chat share) | 57 (18%; 14%) → 67 (28%; 18%) / 57 (24%; 16%) |
| context bursts classified / ran / revoked | – → 37 / 3 / 0 and 32 / 0 / 0 |
| with reservations on: burst share; batch completions | 59%; 163 (47 with burst off) |

What it means:

- Lending the spare step rate is free throughput. The engine gives every stream the same rate, so
  the rate above the shape's floor is already being generated; lending it shortens requests, and
  the closed-loop tenants send the next one sooner. Nothing was spent: KV stayed under 14%, the
  engine's stream count rose 15%, and revocation of a tps burst is just the next admission lowering
  the lent rate.
- Under wall-clock billing the burst is revenue given away, and on this traffic it is cheap: 4% of
  the bill, because 90% of the bill is prefill. On decode-heavy traffic the same bursting would
  discount a third of the bill; the report's burst discount column is the number to watch.
- The cost is to the floors. With burst the surplus is delivered instead of banked, so a prefill
  stall lands on an empty buffer: agent streams that overlapped a long prefill violated 41% of the
  time instead of 27%, and violations rose from 57 to 67 per run. A lending reserve (keep a margin
  banked, lend the rest) is the knob to build before bursting is on by default for the fast shapes.
- Context burst did not deliver. A context burst keeps its shape's tps and so, for the 60 tps
  shape, needs a near-empty replica; 34 of 37 waited 30 s and failed, which is worse for the tenant
  than the instant `429` of the burst-off run, and the 3 that ran were never in anyone's way. The
  revocation machinery (`grace`, `abort`, `finish`) is therefore verified only by the fake-replica
  tests, where `abort` cancels within one tick and `finish` lets the burst complete; whether revoking
  hurts guaranteed traffic on hardware is open. The design change: admit context bursts as best
  effort at the slowest tps class and fail them fast when they cannot be placed.
- Bursting and reservations are complementary: the 59% burst share of the reservation run is the
  idle reserved capacity resold to the lazy tenant, which recovered a quarter of the revenue the
  parity-priced reservation gave up, and the reserved tenant's own requests ran at 73 tok/s.

## Q5. Which M4 findings were artifacts of the small replica?

**The lockouts were; the floor findings were not. On three 4090s the 60 tps shape places freely,
context bursts run (689–962 per run) and get revoked (6–19 per run), and reservations no longer
shut the fast shape out. But the floors are broken by prefill on this card too, now by the
aggregate prefill stream rather than one slow prompt: the 60 tps shape misses its floor on 59–83%
of requests in every setting, and bursting still doubles the violations it buys its throughput
with.**

| M4 finding on the 2080 Ti | On three 4090s (ASSUMPTIONS.md, stage C) | Verdict |
|---|---|---|
| one 60 tps request idles six of eight seats | a 60 tps request costs about two of 32 seats; 1,628–2,675 agent steps per run served, 0 capacity refusals of guaranteed agent requests in any setting | artifact |
| context bursts unplaceable (34 of 37 refused after 30 s, 0 revoked) | 689–962 bursts per run served as the 15 tps class, queued 0.95 s median, TTFT 1.1–2.0 s; 45–247 refused at the 5 s burst cap; 6–19 revoked under `grace` after 200–360 delivered tokens | artifact, plus the G0 fix (slowest class, short queue) |
| five reserved slots lock the 60 tps shape out (38 / 38 refused) | reservations spread over the same replicas as the fast shape: 0 guaranteed agent refusals for capacity | artifact |
| floor violations from one slow 6k prefill (4–5.5 s) | a 6k prompt prefills in 0.65 s, but prompt tokens outnumber output 15 : 1 and 98–99.7% of requests overlap someone's prefill; per-stream decode 62–78 tok/s during it (burst off) against 122–154 clear; chat violates on 12–38%, the agent on 59–70% | real, different mechanism |
| burst on raises violations (57 → 67) via the emptied pacer buffer | 1,615 → 3,360 per run; chat 24% → 58% of requests, agent to 73–83%, note-taker to 50–67% | real, larger at scale |
| reservations remove the queue (13 ms) | true only when slots cover the peak; sized below it (24 of a 39–46 p95) the reserved tenant queued 19–26 s at p50 and lost 7% of requests after 30 s | new finding (Q6) |

What it means: the capacity story of M3/M4 (nesting, lockouts, unplaceable bursts) was the 8-seat
replica talking, and goes away with 32 seats. The quality-of-service story does not: a wall-clock
promise of 60 tokens per second cannot be kept on a prompt-heavy mix at 40–55% utilisation with a
decode-only calibration, because the step-rate curve was measured without concurrent prefill and
the engine spends most of its steps sharing the batch with 512 prefill tokens. The fixes are the
same as before and now have numbers behind them: admit against a prefill-aware curve (or reserve a
share of the batch for decode), and keep a lending reserve so that bursting does not spend the
buffer that absorbs prefill stalls.

## Q6. Do the scheduler's placement rules hold with real replicas?

**Yes: pinned reservations and an excluded shape put the right traffic on the right replica,
unpinned work spread within ±8%, no ledger was exceeded and vLLM never preempted. The new result
is that a reservation sized below the tenant's peak is a cap, because a reserved tenant queues only
on its own slots.**

- Setting (3), chat 24 + note-taker 4 slots pinned to replica A with the 60 tps shape excluded from
  A: A served 2,627–2,628 chat and all 184–188 note-taker requests and no guaranteed 60 tps request
  (3–15 agent context bursts ran on A as the 15 tps class: the exclusion applies to the class a
  request runs as, which is the intended reading); B and C took 1,143–1,601 agent steps and 676–987
  batch jobs each. Fill rate 0.61–0.62 on A against 0.42–0.49 on B/C.
- Setting (4), the same slots unpinned: the gateway placed 8 chat and 1–2 note-taker slots on each
  replica; requests and output stayed within ±8% across replicas in every unpinned run.
- Ledgers: engine streams peaked at 22–32 per replica against the 32-seat plan, KV at 29–60%,
  gateway queues at 8–107 interactive and 7–11 burst requests; vLLM preemptions 0 in all 30
  replica-runs; vLLM's own waiting queue peaked at 5–22, which is prefill backlog behind the
  512-token chunk, not admission.
- The cap: chat's what-if p95 occupancy was 39–46 streams and one replica holds 24 chat slots
  beside the note-taker's, so the reservation covered the p50 (12–14) twice but not the peak. Pinned,
  chat waited 19–26 s at p50 and 30 s at p95 and 197–202 requests per run timed out
  (`capacity:reservation_busy`); spread, 13.6–18.4 s and 56–130. The note-taker lost 7–18 the same
  way. The remedy is the existing `admission.reservation_spill` knob (overflow competes on demand),
  which was not run; the design row is in DESIGN.md §12.

## Q7. What are the economics of a sellable configuration?

**At the 7B market basket with cache hits priced as cache reads, three 4090s at $0.89 each earn
1.6× their cost with bursting off (41% utilisation) and 2.1× with it on (54%): gross margin 36% →
53%, about 63% at a 70% utilisation, break-even near 26% utilisation. Prompt tokens still carry
the revenue; the per-second price of a seat pays 0.68× its cost share. Reservations at 2.3× cost
share undercharge a well-used slot: 3.5–4.9× would match per-request revenue.**

| per run, mean over seeds | burst off | burst on | reservations pinned | reservations spread |
|---|---|---|---|---|
| per-token revenue (cache-aware) / uncached | $1.30 / $1.73 | $1.71 / $2.25 | $1.51 / $2.30 | $1.70 / $2.21 |
| shape revenue; what tenants paid | $1.34; $1.34 | $1.76; $1.76 | $1.55; $1.36 | $1.74; $1.35 |
| GPU cost (3 × $0.89/h × 1,110 s) | $0.83 | $0.80 | $0.81 | $0.81 |
| revenue ÷ GPU cost; gross margin; at 70% utilisation | 1.58×; 36%; 62% | 2.12×; 53%; 63% | 1.86×; 46%; 56% | 2.11×; 53%; 64% |
| output tokens per GPU-hour | 1.43M | 1.88M | 1.99M | 1.86M |
| list price for a 0 / 25 / 50% margin, $ per 1M output tokens, measured utilisation | 0.111 / 0.148 / 0.222 | 0.083 / 0.110 / 0.165 | 0.094 / 0.125 / 0.188 | 0.083 / 0.111 / 0.166 |
| the same at 70% utilisation | 0.066 / 0.088 / 0.131 | 0.064 / 0.086 / 0.128 | 0.077 / 0.103 / 0.155 | 0.064 / 0.085 / 0.128 |

What it means:

- A 7B AWQ model on a rented 4090 is a sellable configuration at today's open-model prices: the
  break-even list price is 0.47–0.64× the basket's $0.175 per 1M output tokens, and a 50% margin
  needs 0.94–1.27× at the measured utilisation or 0.73–0.88× at 70%. On the 2080 Ti the same
  comparison needed everything charged in full to reach 1.6× cost; here cache-read pricing is
  already in (it removes 23–34% of the per-token revenue) and the margin survives it.
- Bursting is the single largest lever on the economics: +32% output per GPU-hour and +31% revenue
  for a 3–6% discount, because the closed-loop tenants (agent, batch) do more work when their
  requests finish sooner. Its cost is the floor (Q5), so the honest version of this margin is
  "with a lending reserve", which is unbuilt.
- The seat's per-second price alone ($0.0189 per 30 tps slot-hour) is below the seat's cost share
  ($0.0278); the prompt charge at the basket's input price is what makes shapes revenue-neutral
  against per-token billing (1.0× for the interactive tenants, 1.3–1.6× for batch, whose per-token
  reference carries a 50% discount the shape does not).
- Reservations priced at 2.3× cost share ($0.0435 per slot-hour) are a good deal for a tenant whose
  slots are 73–78% used: chat paid 0.67× (pinned) and 0.44–0.51× (spread) of its per-request bill,
  and the provider earned $0.19–0.39 less per run than per-request billing of the same tenants.
  The bill is linear in the factor, so 3.5× (pinned) to 4.9× (spread), $0.065–0.092 per slot-hour,
  is revenue parity; the M4 window of 2.3–3.2× covered the cost share, not the opportunity cost.
- Cached share depends on placement: 46% with the agent's growing contexts concentrated on two
  replicas (pinned setting) against 32–35% spread. A scheduler that keeps a session's turns on one
  replica raises cache hits and lowers prefill load and, under cache-read pricing, revenue per
  prompt token at the same time.

## What was not measured, and what to do next

- Two to three seeds per setting; direction is consistent across seeds everywhere, but the
  note-taker (180–202 requests per run) and the reservation bills are two-sample numbers.
- `admission.reservation_spill`: the fix for the reservation cap (Q6) exists as a knob and has not
  run on hardware.
- A prefill-aware seat curve or a decode share of the batch (Q5): the calibration still measures
  decode alone, so admission admits more fast streams than the engine can hold at a 15 : 1 prompt
  mix. The lending reserve for bursting is likewise unbuilt.
- The `abort` revocation policy on hardware (the `grace` runs revoked 6–19 bursts; `abort` would
  cancel them within a tick and was skipped for budget), the fp16 model, a 120 tps shape and an
  A100 cross-check: the optional experiments of PHASE2.md §3, affordable within the remaining
  credit.
- Whole-run reservations straddle quiet and peak phases; phase-length reservations
  (`reservation.phases`) exist in the load generator and ran only in the smoke tests.
- One GPU model and one model size; the Q7 margins are for a 4090 at $0.89 and a 7B AWQ model at
  the 7B basket. The report recomputes them for any `gpu_usd_per_hour` and price reference.

In order of value: reservation spill (a run, not a feature); a prefill-aware curve or decode share;
a lending reserve; session affinity for cache hits; then the optional experiments.
