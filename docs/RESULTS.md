# Results: what the proof of concept measured

Answers to the four questions of DESIGN.md §1, from the GPU runs of milestones 2–4 (2026-11-06 and
2026-11-07), and to the three phase-2 questions of PHASE2.md §1 (Q5–Q7, 2026-11-08), and to the six phase-3 questions of PHASE3.md §1 (Q8–Q13, 2026-11-09 → 11-10). Setup for
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

Q8–Q13 answer the phase-3 questions of PHASE3.md §1 (the confidence campaign, 2026-11-09 → 11-10): one RunPod
pod with three RTX 4090s ($2.68 per pod-hour, $0.893 per GPU-hour) serving Qwen2.5-7B-Instruct-AWQ, one vLLM
engine per GPU, the gateway, meter and load generator on the same pod, 68 runs over 24 settings chained unattended with a fresh gateway and a KV preflight per run (59 kept in the report beside the ten phase-2 runs: seven were re-run after a reservation was not granted and two failed by the load generator's timeout, PHASE3.md §6); six seeds of the on-demand settings, three to six of the 24 + 4 reservation settings, four of the p95 reservations, two of every sweep point; the same prices as Q5–Q7. Every comparison is paired by seed
and reported with its confidence interval and its Holm-corrected p-value (PHASE3.md §4); the campaign report is
`runs/phase3/campaign/campaign.html` with its CSVs, copied for readers to `docs/reports/phase3-campaign/`.

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

The phase-3 paragraph (Q8–Q13, 68 runs on one three-GPU pod, seeds paired): the phase-2 findings held at
six seeds (bursting +31% revenue, p 0.001; floor violations under bursting not significantly worse, +9 points,
p 0.14) and the campaign measured what they cost whom. A reservation sized at the tenant's p95 gives the reserved
tenant a bill that moves 1.4% across four seeds against 9% per token, no queue and its floors on demand's
terms, and the on-demand tenants pay for it in waiting (start p95 2.0–2.1× the baseline's) while the
provider pays in idle seats: at the campaign's 2.3× price the reservation settings earn less margin than on-demand
bursting, the margin row is met from 3.2× (p95 sizing) or 5× (24 + 4), where the reserved tenant pays 1.2–1.7× (p95) or 1.1–1.6× (24 + 4) its
per-token bill, and no price table moves the on-demand tenants' waiting (Q8, Q13). Revenue over GPU cost rises
linearly with utilisation: break-even 32% with bursting off and under the lowest measured point with it on, and
the cluster saturates at 1.5× load, where every interactive queue sits at its 30 s contract, 75–89% of requests
start more than 2 s after arrival, and the slot bill is the quote for three requests in four (Q9). The contract
held there and is the wrong contract for chat. Of the levers, the gateway's prefill cap is the one that fixes the
30 tps floors (chat 45 → 12% misses, −4 margin points) and not the 60 tps shape; overcommit sells service that is
not there; the engine's chunk size trades first-token latency against decode steadiness in both directions
(256: floors worse, TTFT doubled; 2048: floors 44 → 35%, TTFT +22–43%) without removing the interference
(Q10, Q12). Output-heavy traffic holds every floor and earns 0.6× the GPU's cost under both schemes at the 7B
basket, and bursting under per-second billing is a transfer to open-loop tenants (46% of the quoted bill given
away); a flash crowd is absorbed by the queue contract inside the 30 s bound, holds every bill at or under its
quote and moves the surge's cost onto latency, which raises the provider's margin 13 points while delivering
slower service, the case for an SLA credit on the start bound that the frontier prices at 4 points of margin at
base load, 7–8 at saturation and 5 in the surge (Q11, Q13). Two operational findings: vLLM's KV budget came out
5–6% short at every cold first engine start (four of four) and needs a post-start gate when capacity is sold by
KV; and a load generator's blanket timeout can fail a saturation run the gateway served correctly (PHASE3.md §6).

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

## Q8. Does a reservation make a tenant's bill predictable, and what does it cost the others and the provider?

**Yes, when the reservation covers the tenant's peak: with chat and the note-taker reserved at their
measured p95 (44 and 7 slots) and spill on, the configuration that would ship, their bills moved 1.4%
(chat) and 0.9% (note-taker) from seed to seed over four seeds, against 9.1% and 7.7% under per-token
billing of the same runs, and 0.1% / 0.0% without spill. On demand, slot billing is no more predictable
than per-token billing (chat 9.7% vs 7.7%, note-taker 11.3% vs 11.1% over six seeds): the predictability
is bought by the reservation fee, which is 98–100% of a reserved tenant's bill. The price of a reservation
sized at p95 is paid by the on-demand tenants (on the same four seeds the agent's queue p95 goes from
5.7 s to 15.6 s and the batch tenant's from 6.1 s to 12.8 s; the agent's start p95 from 8 s to 17 s) and
by the provider (gross margin 50% → 43%, output per GPU-hour −3%); the reserved chat tenant itself pays
0.90× its per-token bill, because the fee is still priced at 2.3× cost share (Q7). A reservation below
the tenant's peak (24 slots against a p95 of 44) is a cap on every one of six seeds: 182 chat requests per
run refused as `reservation_busy`, a 30 s queue p95 against 6 s on demand; spill removes the cap and
with it most of the predictability (CV 9.7%).**

Bill CV across seeds, the same tenant over the same runs (lower = more predictable); fee share = the
reservation fee's share of the slot bill:

| setting | n | chat: slot / per-token CV (ratio) | fee share | note-taker: slot / per-token CV (ratio) | fee share |
|---|---|---|---|---|---|
| on demand (`burst`), the baseline | 6 | 9.7% / 7.7% | – | 11.3% / 11.1% | – |
| `resv-split` (24 + 4 pinned to replica A) | 6 | 1.2% / 4.1% (0.30) | 97% | 0.1% / 9.9% (0.01) | 100% |
| `resv-shared` (24 + 4, no placement rule) | 6 | 0.6% / 6.9% (0.08) | 99.7% | 0.0% / 10.2% (0.00) | 100% |
| `resv-spill` (24 + 4, spill on) | 3 | 9.7% / 11.0% (0.88) | 57% | 2.1% / 6.7% (0.32) | 77% |
| `resv-p95` (44 + 7) | 4 | 0.1% / 9.1% (0.01) | 99.9% | 0.0% / 7.7% (0.00) | 100% |
| `resv-p95-spill` (44 + 7, spill on; would ship) | 4 | 1.4% / 9.1% (0.15) | 98% | 0.9% / 7.7% (0.11) | 99% |

What the fee buys the reserved tenant, and what it costs the others (means over seeds, per run):

| | `burst` | `resv-split` | `resv-shared` | `resv-spill` | `resv-p95` | `resv-p95-spill` |
|---|---|---|---|---|---|---|
| chat queue p95 / start p95 | 5.9 s / 7.5 s | 29.9 s / 31.6 s | 25.3 s / 26.3 s | 7.4 s / 9.1 s | 0.9 s / 2.7 s | 0.5 s / 2.7 s |
| chat refused per run (`reservation_busy`) | 0.3 | 182 | 79 | 0.3 | 0.5 | 0.5 |
| chat decode-floor violations | 35% | 54% | 25% | 28% | 10% | 7% |
| chat bill: slot / per-token | $0.68 / $0.69 | $0.32 / $0.46 | $0.31 / $0.64 | $0.55 / $0.60 | $0.57 / $0.66 | $0.59 / $0.65 |
| agent queue p95; refused per run | 6.0 s; 136 | 2.4 s; 96 | 2.8 s; 118 | 8.6 s; 197 | 16.0 s; 201 | 15.6 s; 202 |
| batch queue p95 | 4.9 s | 1.4 s | 3.5 s | 8.6 s | 11.5 s | 12.8 s |
| revenue (what tenants paid) | $1.57 | $1.23 | $1.23 | $1.51 | $1.41 | $1.42 |
| gross margin, shape pricing | 48.1% | 32.9% | 33.3% | 46.9% | 42.9% | 43.1% |
| output tokens per GPU-hour | 1.97M | 2.04M | 1.91M | 1.97M | 1.87M | 1.87M |

Paired against `burst` on the same seeds (six): `resv-split` −$0.34 revenue per run (95% CI −0.38 … −0.31,
p < 0.001 after Holm), gross margin −15 points (p = 0.004 Holm), refusal rate +2.2 points (p = 0.005 Holm),
output per GPU-hour +3.7% (p = 0.008, not significant after Holm); `resv-shared` −$0.34 revenue and −15
points of margin (both p < 0.01 Holm), decode-floor violations −5.5 points (p = 0.003, 0.09 Holm).
`resv-spill` against `resv-split` on three seeds: chat refusals −179 per run and chat start p95 −22.5 s
(both p = 0.013), revenue +$0.12 (p = 0.06). `resv-p95-spill` against `burst` on four common seeds: revenue
−$0.23 per run (95% CI −0.43 … −0.02, p = 0.04, not significant after Holm), gross margin −7.2 points
(50.3% → 43.1%, p = 0.03), output per GPU-hour −3.0% (p < 0.001, 0.012 after Holm), decode-floor violations
38% → 12% (p = 0.03), the agent's queue p95 5.7 → 15.6 s (p = 0.03) and refusals +42 per run (p = 0.004),
the batch tenant's queue p95 6.1 → 12.8 s (p = 0.02); the reserved chat tenant's floor violations 37% → 7%
(p = 0.03) and start-bound misses 47% → 10% (p = 0.05). Spill on against off at p95 sizing (four seeds)
changes nothing measurable: revenue +$0.005, floor violations −1.8 points, chat queue p95 0.9 → 0.5 s, all
p > 0.2, because the overflow is 2% of the chat bill.

What it means:

- The claim holds where it was designed to: a reservation turns the tenant's bill into a fee. Across
  six seeds the reserved chat tenant's slot bill varied 0.6–1.2% while its own per-token bill over the
  same runs varied 4–7%; for the note-taker 0.0–0.1% against 10%; at p95 sizing over four seeds 0.1% and
  0.0% against 9.1% and 7.7%. The ratio is not 0 only because a few of the tenant's requests are billed
  on demand outside the reservation (fee share 97% in the pinned setting, 99.7% spread, 99.9% at p95;
  check `reserved_bills.csv` and the per-run reports for which requests).
- On-demand slot billing does not deliver predictability, and was never going to: a slot bill is the
  time the request ran, which tracks tokens generated. Over the six `burst` seeds the slot bill's CV is
  higher than per-token's for every tenant. The design's predictability lives in reservations, as the
  phase-2 brief said.
- Spill trades predictability for availability. `resv-spill` lets the reserved tenant overflow into
  shared capacity: chat's refusals go from 182 to 0 per run and its queue p95 from 30 s to 7 s, but 43%
  of its bill is now on-demand runtime and the CV is back at 9.7%, the on-demand level. With the
  reservation sized at p95 the overflow is small (fee share 98%) and the CV stays at 1.4%: size at p95
  and let it spill.
- A p95 reservation for two tenants takes 51 of 96 seats out of the shared pool. The on-demand tenants
  pay for it in service, not money: on the same four seeds the agent's queue p95 goes from 5.7 s to
  15.6 s and its refusals from 160 to 202 per run, the batch tenant's queue p95 from 6.1 s to 12.8 s,
  while their bills are unchanged. The provider loses 3% of output per GPU-hour (idle reserved slots)
  and 7 points of gross margin, and that loss comes from the fee, not from the capacity: at 2.3× cost
  share the reserved chat tenant pays 0.90× of its per-token bill (Q7 put revenue parity at 3.5–4.9×).
  From 3.2× the provider's margin is back above the on-demand setting's (Q13).
- The under-sized reservation is a cap, now on six seeds: 182 chat requests per run refused inside a
  queue of 24 slots while the cluster had seats free (the agent and batch tenants waited less than under
  `burst`). Phase 2 measured it on two seeds (Q6); the replication puts the chat refusal difference at
  +182 per run with a 95% interval of 148–217.
- Confidence: `resv-split` and `resv-shared` have six seeds (1–2 from phase 2 on three separate pods,
  3–6 from phase 3 on one pod with three GPUs, all with the full 289k KV budget on every replica after the
  stage-I re-runs); `resv-spill` three; the p95 settings four (seeds 1–2 ran before stage I with replica A
  6% short of KV, seeds 3–4 after the chunk-512 restore with every engine at 289,056 tokens; the report's
  Capacity card reads the two groups apart). Seeds 4–6 plan about half the agent load of 1–3 (seed 4 of
  the p95 settings: start p95 2.3–2.7 s, floor violations 4%, against 12–23 s and 13–22% on seeds 1–3),
  so pooled service metrics (queue, refusals) are read per seed group in the report's Capacity section;
  the paired differences above are within-seed and unaffected.

Sources: `runs/phase3/campaign/campaign.html` (Headline, Tenants, Paired comparisons, Scorecard),
`reserved_bills.csv`, `tenants.csv`, `paired.csv`, `reservations.csv`, `scorecard.csv`; the re-runs are
`resv-split-s3..s6-20261109-*` and `resv-spill-s1..s3-20261109-12*` (DECISION-stage-I.md), the stage-J
runs `resv-p95-s3..s4-20261110-*` and `resv-p95-spill-s3..s4-20261110-*` (PHASE3.md §9 B).

## Q9. How do margin and service move with load, and where is break-even?

**Measured over four load levels (0.5×, 1×, 1.5×, 2× the phase-2 arrival rates; 26–66% utilisation): shape
revenue over GPU cost rises linearly with utilisation, break-even is at 32% utilisation with bursting off
(measured: half load loses money, 0.84× cost) and below the lowest measured point with bursting on (1.42× cost
at 26%), and the shape curve sits 1–5% above per-token's at every level. The cluster saturates at 1.5× load:
utilisation stops at 64–66%, every interactive tenant's queue p95 is pinned at its 30 s contract and 7–11% of
requests are refused; 2× only adds refusals. The contract holds at saturation and is the wrong contract for
chat: 75–89% of interactive requests start more than 2 s after arrival and chat's median wait at 2× is 27 s,
inside a queue cap no interactive tenant would accept. Bursting is worth most where seats are idle (+140% output per
GPU-hour at half load, +50% at 1×, +8–10% at saturation). A slot bill never exceeds its quote, and at
saturation it is the quote: the share of interactive requests billed in full goes from 2% at half load to 75%,
and the slot bill per output token goes from 1% below per-token's to 4–6% above it.**

Per run, means over seeds (two per point, six at 1×); revenue is what tenants paid under shapes (per-token
billing of the same requests in brackets); GPU cost 3× $0.89/h × run length:

| load | bursting | utilisation | revenue (per-token) | revenue ÷ GPU cost, shape (per-token) | gross margin, shape (per-token) | output / GPU-hour | refused | start p95 | floor misses |
|---|---|---|---|---|---|---|---|---|---|
| 0.5× | off | 28% | $0.69 ($0.67) | 0.84× (0.82×) | −19% (−23%) | 0.88M | 4.1% | 0.9 s | 2% |
| 0.5× | on | 26% | $1.14 ($1.14) | 1.42× (1.41×) | 30% (29%) | 2.12M | 0.6% | 1.6 s | 5% |
| 1× | off | 44% | $1.20 ($1.16) | 1.45× (1.41×) | 30% (28%) | 1.31M | 3.5% | 4.9 s | 24% |
| 1× | on | 46% | $1.57 ($1.54) | 1.96× (1.91×) | 48% (47%) | 1.97M | 1.8% | 7.8 s | 34% |
| 1.5× | off | 64% | $1.88 ($1.79) | 2.25× (2.14×) | 56% (53%) | 1.63M | 7.7% | 31.7 s | 69% |
| 1.5× | on | 65% | $1.95 ($1.86) | 2.39× (2.27×) | 58% (56%) | 1.77M | 7.5% | 32.7 s | 58% |
| 2× | off | 65% | $1.93 ($1.83) | 2.27× (2.16×) | 56% (54%) | 1.64M | 10.8% | 32.7 s | 68% |
| 2× | on | 66% | $2.04 ($1.95) | 2.47× (2.36×) | 60% (58%) | 1.81M | 10.5% | 32.9 s | 57% |

The margin curve, a straight line through the twelve runs of each series (revenue ÷ GPU cost against
utilisation): bursting off, shape 3.91× utilisation − 0.26 (r² 0.998, break-even 32%), per-token 3.66×
utilisation − 0.20 (break-even 33%); bursting on, shape 2.58× utilisation + 0.76 (r² 0.993, break-even 9%,
extrapolated: the lowest measured point is 25%), per-token 2.32× utilisation + 0.83 (7%). The report's
Margin curve section fits the pooled line over all settings and workloads (slope 2.8, break-even 21%, r² 0.44),
a campaign-wide tendency, beside one line per sweep series; the numbers here are the load family's lines, which
is what the claim is about (`margin_curve.csv`, `family` = load).

What the tenants saw, bursting off | on:

| per run | 0.5× | 1× | 1.5× | 2× |
|---|---|---|---|---|
| chat queue p95 | 0.07 s \| 0.15 s | 3.3 s \| 5.9 s | 29.9 s \| 29.9 s | 30.0 s \| 30.0 s |
| chat refused (`queue_timeout`) | 1.5 \| 0 | 0.5 \| 0.3 | 186 \| 206 | 418 \| 447 |
| chat delivered tps (shape 30) / floor misses | 36 \| 74; 1% \| 3% | 34 \| 44; 20% \| 35% | 27 \| 33; 64% \| 49% | 27 \| 33; 64% \| 48% |
| agent queue p95; refused | 0.07 s; 149 \| 0.14 s; 38 | 3.4 s; 211 \| 6.0 s; 136 | 29.9 s; 407 \| 29.9 s; 374 | 30.0 s; 460 \| 30.0 s; 429 |
| agent delivered tps (shape 60) / floor misses | 76 \| 76; 6% \| 16% | 59 \| 50; 42% \| 56% | 38 \| 37; 84% \| 84% | 38 \| 35; 83% \| 84% |
| batch queue p95 (lazy, 3600 s contract) | 0.08 s \| 0.14 s | 5.2 s \| 4.9 s | 457 s \| 57 s | 759 s \| 11 s |
| batch delivered tps (shape 15) | 16 \| 105 | 16 \| 88 | 16 \| 83 | 16 \| 74 |
| requests served per run | 3,551 \| 6,510 | 5,636 \| 7,202 | 7,250 \| 7,452 | 7,562 \| 7,842 |
| interactive requests billed at their quote (chat) | 2% \| 4% | 27% \| 41% | 75% \| 56% | 75% \| 54% |
| chat slot bill ÷ per-token bill, per output token | 0.99 \| 0.91 | 1.01 \| 0.99 | 1.04 \| 1.04 | 1.04 \| 1.03 |
| agent, the same | 0.99 \| 0.95 | 1.02 \| 1.01 | 1.05 \| 1.05 | 1.06 \| 1.05 |

Paired by seed, bursting on minus off: at 0.5× output per GPU-hour +140%, revenue +65%, gross margin +49
points (−19% → +30%), refusals −86%; at 1.5× output +8%, revenue +4%, margin +2.6 points; at 2× output
+10%, revenue +6%, margin +3.5 points. The 1× pair over six seeds is in Q8's table (+31% revenue, +18 points
of margin, p < 0.001 after Holm). Two seeds per sweep point: the within-seed differences are consistent in
direction everywhere; the 1.5× and 2× service numbers are two-sample.

What it means:

- Break-even is measured now, not derived: a 4090 at $0.89 serving a 7B AWQ model at the 7B basket covers its
  cost from 32% utilisation with bursting off (Q7 derived 26% from one point). Half load (26–28%) is the
  case below break-even, and bursting carries it: lending the idle seats to the closed-loop tenants (the
  agent's loops, the batch queue) produces 2.4× the output and 1.65× the revenue from the same GPU-hours, and
  turns −19% into +30%. With bursting on, the fitted line crosses 1.0× at about 9% utilisation, but nothing
  was measured below 25%.
- The shape scheme's margin curve sits above per-token's at every load (1.03–1.06× bursting off, 1.01–1.05×
  on). The gap is the batch tenant: its per-token reference carries the 50% batch discount, its slot bill does
  not (1.6× per-token at every load). For the three interactive tenants the two schemes are within 1–2% of
  each other below saturation; above it the slot bill is 4–6% dearer per output token.
- That 4–6% is the predictability claim's honest edge. A slot bill is bounded by its quote (ratio p95 ≤ 1.0 for
  every tenant at every load, by construction); below the quote the tenant is paying for seconds it did not
  use at the full rate. At half load 98% of chat requests finish under their quote (early finishes, bursts);
  at 1.5× and above, 75% are billed the quote exactly because the engine delivered at or below the shape rate
  for the whole request. The bill does not rise when the cluster is busy; the discount disappears. Per-token
  bills per output token also rise with load (+12%, chat: the mix shifts toward longer contexts as short
  requests are refused), which is why the comparison is a ratio.
- Saturation is at 1.5×, not 2×. Utilisation stops at 64–66% whether bursting is on or off; what grows past
  that is refusals (7.7% → 10.8%) and the batch tenant's wait (457 s → 759 s with bursting off). The
  interactive queues are pinned at the 30 s contract from 1.5× on, so start p95 (31.7–32.9 s) is the contract
  plus a 2 s TTFT, 75–89% of interactive requests miss the 2 s start bound (chat's median queue wait at 2× is
  27 s), and the floors are missed on 57–69% of requests. Every one of those requests met its contract: the
  30 s queue cap is a batch contract on an interactive shape, and what the saturation runs show is that the
  interactive shapes need the start bound as their contract, enforced by refusing earlier, not the queue cap.
  The ceiling is the seat count against a 15 : 1 prompt mix (Q5): the engine is prefilling for someone
  whenever 60–65% of the seats are busy, and admission has no prefill term.
- Bursting at saturation changes who waits, not how much is served: the batch tenant's queue p95 falls from
  457 s to 57 s (1.5×) and 759 s to 11 s (2×) because tps bursts let its lazy requests finish at 74–83 tps on a
  15 tps shape, freeing their seats; chat and the agent refuse slightly more (206 vs 186 per run) and the
  floors are missed less often (49% vs 64% for chat) because faster finishes shorten the queue. Output per
  GPU-hour is +8–10%. At half load the same mechanism is +140%: bursting's value is idle seats, and
  saturation has none.
- The agent's refusals with bursting off (149 per run even at half load, 13% of its requests) are loops
  outgrowing their 8k context; a context burst absorbs them (38 per run with bursting on). This is Q4's
  mechanism at every load level, not a capacity effect.

Sources: `runs/phase3/campaign/campaign.html` (Sweeps: load; Margin curve; Paired comparisons), `sweep.csv`,
`margin_curve.csv`, `tenants.csv`, `predictability.csv`; runs `load0.5-*`, `load1.5-*`, `load2-*` of
2026-11-09 (`runs/phase3/pulled.runs`).

## Q10. Do the levers add margin, and what does each cost the tenants?

**Against the on-demand bursting setting on the same two seeds: tps overcommit (1.25×, 1.5×) adds 1.5 points
of margin and 3.5% revenue by selling seats the engine cannot serve (output per GPU-hour −4 to −5%, floor
misses +32–43%, start p95 6.7 s → 12.5–13.1 s): it fails on service. The prefill cap (4,096 prompt tokens
prefilling at once per replica) is the one lever that moves the floor: misses −41% overall, chat's from 45%
to 12% and the note-taker's from 46% to 9%, the 60 tps agent unchanged at 65%; it costs 4 points of
margin (utilisation 53% → 41%, revenue −8%) and 2–3 s of queue p95 as prompts wait at the gateway instead
of inside the engine. Abort-on-revoke protects the floors a little (misses −18%) for 4 s more queueing.
Pacing itself costs throughput: the unpaced control delivers 2.19M output tokens per GPU-hour, 54% more than
pacing with bursting off and 16% more than with bursting on, at 8 points more margin than bursting off, and it
is the 60 tps shape that loses: without pacing every tenant decodes at about 50 tps.**

Per run, seeds 1–2, means (the paired difference against `burst` seeds 1–2 in brackets where it is the point):

| | `burst` (seeds 1–2) | `oc1.25` | `oc1.5` | `prefillcap` | `abort` | `nopace` (control; pair: `burstoff` s1–2) |
|---|---|---|---|---|---|---|
| utilisation | 53% | 59% (+12%) | 60% (+14%) | 41% (−22%) | 52% | 43% (`burstoff` 48%) |
| concurrency p95 (streams) | 79 | 95 | 105 | 69 | 81 | 77 |
| output tokens / GPU-hour | 1.89M | 1.82M (−4%) | 1.80M (−5%) | 1.87M (−1%) | 1.93M (+2%) | 2.19M (`burstoff` 1.42M; +54%) |
| revenue, shape | $1.74 | $1.79 (+3.4%) | $1.80 (+3.5%) | $1.60 (−8%) | $1.68 (−3%) | $1.46 (`burstoff` $1.30; +12%) |
| gross margin, shape | 53.6% | 55.1% | 55.2% | 49.6% | 52.2% | 44.9% (`burstoff` 36.7%) |
| floor misses, all requests | 44% | 58% (+32%) | 63% (+43%) | 26% (−41%) | 36% (−18%) | 26% (`burstoff` 22%) |
| chat / note-taker / agent floor misses | 45 / 46 / 64% | 67 / 66 / 67% | 75 / 69 / 67% | 12 / 9 / 65% | 32 / 28 / 64% | 21 / 20 / 68% |
| chat / agent delivered tps (shapes 30 / 60) | 36 / 43 | 30 / 38 | 27 / 35 | 46 / 48 | 41 / 46 | 53 / 51 (`burstoff` 35 / 57) |
| start p95 | 6.7 s | 12.5 s | 13.1 s | 7.6 s | 10.8 s | 5.6 s (`burstoff` 3.3 s) |
| chat queue p95 / TTFT p95 | 4.7 s / 2.9 s | 10.0 s / – | 10.0 s / – | 7.1 s / 0.85 s | 8.4 s / – | 4.3 s / 2.6 s |
| refused | 2.3% | 2.1% | 2.1% | 2.7% | 1.5% | 3.2% (`burstoff` 4.1%) |
| bursts revoked | 11.5 | 19.5 | 23 | 0 | 85.5 | – |

Two seeds per lever; the differences are consistent in direction within seed, and the t-tests on two pairs are
indicative only (`oc1.5` output per GPU-hour p = 0.009, floor misses p = 0.04; `prefillcap` floor misses p = 0.08;
`nopace` output per GPU-hour p = 0.016, margin p = 0.009; nothing survives Holm at n = 2).

What it means:

- Overcommit is the vCPU lever, and it does not work for decode: selling 1.25–1.5× the calibrated seats puts
  95–105 streams on engines calibrated for 79 and every stream slows (chat 36 → 27 tps on a 30 tps shape,
  the agent 43 → 35 on 60). Output per GPU-hour falls 4–5% because the engine spends more of each step on
  prefill for more waiting requests; revenue rises 3.5% because tenants are billed the seconds, and slower
  seconds are more of them. The extra margin is paid for by the tenants in service, which is the opposite of
  the scheme's promise, and it fails the scorecard's floor and start-p95 rows (Q13).
- The prefill cap is the mitigation for Q5's mechanism, and it fixes what it can reach. Capping the prompt
  tokens in flight per replica at 4,096 keeps prefill from stalling the running decodes: the 30 tps tenants
  go from missing their floor on 45% of requests to 9–12%, delivered tps 36 → 46 on a 30 tps shape, TTFT p95
  2.9 s → 0.85 s. The waiting moves to the gateway queue (chat queue p95 4.7 → 7.1 s, start p95 +0.9 s) and the
  engine runs emptier (utilisation 53 → 41%, revenue −8%, margin −4 points). It does nothing for the 60 tps
  shape (65% misses either way, delivered 48 tps): that shape needs a seat curve that charges it for the
  prefill it displaces, not a cap. This is the result PHASE3 section 9 waits for: the floor of the 30 tps
  shapes is fixable from the gateway; the 60 tps shape's is not.
- `abort` revokes 86 bursts per run against `grace`'s 12 (a grace revocation waits up to 15 s for the burst to
  finish; most do) and buys 8 points of floor compliance for the 30 tps tenants at the cost of their queue
  (p95 +3.7 s) and 3% revenue: the revoked requests re-queue. Neither policy touches the agent's floor.
- The control answers "what does slicing cost". With admission on and pacing off, the engine delivers 2.19M
  output tokens per GPU-hour; pacing to the shapes with bursting off delivers 1.42M (65% of it) and with
  bursting on 1.89–1.97M (86–90%). Bursting is therefore most of the way back to the raw engine, and what is
  left is the price of the guarantee. Unpaced, every tenant decodes at 51–53 tps: the 30 tps tenants get 1.7×
  their shape and the 60 tps agent gets 0.85× of it (57 tps paced, bursting off), because nothing stops the
  30 tps streams from taking the step budget. Pacing is what makes a 60 tps shape exist. The control also
  earns 8 points more margin than pacing without bursting, because the closed-loop tenants complete more
  requests; against bursting on it earns 9 points less, with 1.5 s less queueing.

Sources: `campaign.html` (Paired comparisons, Sweeps: overcommit), `paired.csv`, `tenants.csv`; runs
`oc1.25-s*`, `oc1.5-s*`, `prefillcap-s*`, `abort-s*`, `nopace-s*` of 2026-11-09.

## Q11. Is the floor finding a property of prompt-heavy traffic, and what does a flash crowd do?

**Yes. On output-heavy traffic (prompt : output 1.5 : 1 instead of 15 : 1, arrival rates scaled so the seats
are as busy) the floors hold: 14% misses with bursting off against 24% on the base scenario, and with
bursting on effectively none (0.01%), with every tenant delivered at 121–143 tps, four times the 30 tps
shapes. The economics are the other way round: output-heavy traffic does not pay for the GPU at the 7B
basket under either scheme (shape revenue 0.61× cost with bursting off, per-token 0.60×), and bursting makes
the shape result worse (0.48×) while making per-token's better (0.82×): a burst is free tokens by design, the
tenants here arrive at a fixed rate and have no backlog to fill the freed seats, so utilisation falls from
61% to 29% and the provider bills 23% fewer seconds for 85% more tokens. Under a two-minute flash crowd at
twice the peak chat rate, slot pricing holds the bills fixed and moves the cost onto latency: no chat request
was refused and no bill exceeded its quote, but the interactive queue p95 reached 17 s (bursting off) to 24 s
(on) in the minutes after the surge, 39–63% of requests started more than 2 s after arrival (start p95 16–21 s
against 5–8 s without the surge), and 54–56% of chat requests were billed exactly their quote against
27–41%. The provider's margin rises (43% against 30% with bursting off) because it sells the same seats
while delivering slower service, which is the case for an SLA credit on the start bound, priced offline in Q13.**

Per run, seeds 1–2:

| | `outheavy-burstoff` | `outheavy-burst` | paired difference |
|---|---|---|---|
| requests served | 3,242 | 5,483 | +69% (closed-loop agent and batch) |
| output tokens | 1.84M | 3.41M | +85% |
| output tokens / GPU-hour | 1.96M | 3.77M | +92% |
| utilisation | 61% | 29% | −52% |
| floor misses: all / chat / agent | 14% / 25% / 8% | 0.01% / 0 / 0.04% | |
| delivered tps: chat / agent / batch (shapes 30 / 60 / 15) | 29 / 62 / 15 | 132 / 121 / 143 | |
| start p95; refused | 0.39 s; 0.1% | 0.56 s; 0.05% | |
| revenue: shape (per-token) | $0.51 ($0.50) | $0.39 ($0.66) | shape −23%, per-token +32% |
| burst discount (quote − bill) | $0.006 | $0.33 | 46% of the quoted bill |
| revenue ÷ GPU cost: shape (per-token) | 0.61× (0.60×) | 0.48× (0.82×) | |
| gross margin: shape (per-token) | −65% (−68%) | −107% (−22%) | |
| cost per 1M output tokens at this utilisation | $0.45 | $0.24 | basket output price $0.175 |

What it means:

- The floor problem is prompt-heavy traffic. The same engines, the same seat count, the same occupancy, and
  the 30 tps floors hold on 75% (chat) to 100% (note-taker) of requests with bursting off and on 100% with
  it on; on the base scenario it is 55–80% and 50–65%. Q5's mechanism was the 15 : 1 prompt mix keeping the
  engine in prefill; take the prompts away and chunked prefill has nothing to interrupt the decodes with.
- Decode alone does not pay for a GPU at open-model prices, on three 4090s as on the 2080 Ti (Q1): at 61%
  utilisation the output-heavy mix earns 0.61× its cost under shapes and 0.60× per token. The cost of an
  output token on this hardware is $0.45 per million at 1.96M per GPU-hour, 2.6× the basket's $0.175; with
  bursting the engine makes 3.77M per GPU-hour and the cost is $0.24, still above the basket. The 7B basket
  prices are set by providers with bigger batches on faster cards; a 4090 sells prompt processing, not
  decode.
- Bursting under per-second billing is a transfer from the provider to the tenant when demand is open-loop.
  By design a request's bill is bounded by its quote and the burst is the discount; here the discount is 46%
  of the quoted revenue ($0.33 of $0.72 per run), because every request runs at four times its shape rate
  and finishes in a quarter of the quoted seconds. The base scenario's closed-loop tenants refill the freed
  seats with new requests (+31% revenue, Q7); the output-heavy chat and note-taker arrive at a fixed rate
  and do not, so the freed seats sit idle (utilisation 61 → 29%) and revenue falls 23% while per-token
  revenue for the same work rises 32%. The design consequence: bursting pays under shapes only where a
  backlog exists to fill the seats it frees (a lazy tier, closed-loop agents), and a provider selling shapes
  to open-loop tenants should cap bursts at the rate the price assumes or price the burst (DESIGN section 13 named the risk: on decode-heavy traffic the discount
  would be the larger share; measured, it is 46%).
The flash crowd (`mixed-4090-flash.yaml`: the base scenario's 480 s peak split into 240 s of peak, 120 s at
twice the peak chat rate and 120 s of peak again; the agent's bursts and the note-taker's hour marks do not
scale), per run, seeds 1–2, with the base scenario's six seeds for comparison:

| | `flash-burstoff` | `flash-burst` | base: `burstoff` / `burst` |
|---|---|---|---|
| requests served | 6,326 | 7,313 | 5,637 / 7,202 |
| utilisation | 52% | 56% | 44% / 46% |
| output tokens / GPU-hour | 1.44M | 1.84M | 1.31M / 1.97M |
| revenue: shape (per-token) | $1.45 ($1.39) | $1.81 ($1.75) | $1.20 ($1.16) / $1.57 ($1.54) |
| gross margin, shape | 43% | 55% | 30% / 48% |
| refused: all; chat | 3.9%; 1.5 requests | 2.3%; 0 | 3.5%; 0.5 / 1.8%; 0.3 |
| start p95 | 16.3 s | 20.6 s | 4.9 s / 7.8 s |
| start misses (queue + TTFT > 2 s) | 39% | 63% | 21% / 48% |
| floor misses | 44% | 48% | 24% / 34% |
| chat requests billed exactly the quote | 54% | 56% | 27% / 41% |
| chat slot bill ÷ per-token bill | 1.02 | 1.02 | 1.01 / 0.99 |
| streams in flight, worst minute (96 seats) | 84 | 80 | 55 / 71 |
| interactive queue p95 / max, worst minute | 17.0 s / 20.8 s | 23.6 s / 26.5 s | 0.4 s / 4.2 s (peak) |

Minute by minute (chat queue p95, mean of two seeds; the flash is minutes 9–11, the second peak 11–13, then
quiet): bursting off 0.2 s at minute 8, then 1.5, 1.0, 9.1, 15.3, 17.0, 16.2, 10.7 and 1.8 s at minute 16;
bursting on 3.2 s at minute 8, then 4.6, 7.2, 12.9, 17.6, 21.9, 23.5, 19.0 and 10.1 s at minute 16. Streams in
flight with bursting off: 55 at minute 8, 59, 71, 84, 69, 82, 75, 67, 44.

- The queue contract held through a surge the cluster was not sized for, and that is the contract's
  weakness, not the system's strength: no chat or note-taker request was refused for capacity and the worst
  minute's maximum wait stayed 3–9 s under the 30 s bound, so the whole surge was absorbed as waiting that the
  contract permits. The waiting comes mostly after the surge: the chat sessions started in the flash keep their
  turns coming, so the queue peaks two to three minutes after the flash ends (minute 13–14) and drains by
  minute 16 with bursting off, later with it on (the bursting runs serve 15% more requests, so there is more
  to drain).
- The start bound is what a surge breaks. 39–63% of requests started more than 2 s after arrival, p95 16–21 s.
  A provider selling the start bound rather than the queue bound has to shed earlier (a shorter `max_queue_s`
  on the interactive shapes trades start misses for refusals) or hold seats back (the reservation settings, Q8).
- The quote cap holds under a spike, and the spike fills it. No bill exceeded its quote (quote ratio p95 1.00
  for every tenant), but twice the share of chat requests were billed exactly the quote (54% against 27%),
  because the seconds the engine was slow inside the quote are billed: the slot bill per chat output token
  ended 2% above per-token's where the base scenario has it 1% above (bursting off) or 1% below (on). The
  cap bounds the tenant's exposure; the surge moves the bill up to it.
- The trade-off, stated plainly: under a surge slot pricing holds each tenant's bill at or under its quote and
  moves the cost onto latency, and the provider's margin rises (43% against 30% with bursting off, utilisation
  52–56% against 44–46%) because it sells the same seats while delivering slower service. Per-token billing
  would have moved the same surge onto the tenants' bills instead (more requests, more tokens) with the same
  waiting. That asymmetry is the case for an SLA credit on the start bound: a shape that refunds the runtime
  price of requests started late returns the surge margin to the tenants who waited, and the re-pricing
  frontier of Q13 prices it offline from the meters (real tenants would also change their behaviour under a
  surge, which no offline re-pricing captures). Bursting adds what it adds on the base scenario: +28% output
  per GPU-hour, +12 points of margin, +4 points of floor misses and +4 s of start p95 (two seeds; p 0.01–0.06
  on the revenue and margin lines, not significant on the service lines).

Sources: `campaign.html` (Paired comparisons), `paired.csv`, `tenants.csv`, `predictability.csv`; runs
`outheavy-burstoff-s*`, `outheavy-burst-s*`, `flash-*` of 2026-11-09; scenarios
`config/scenarios/mixed-4090-outheavy.yaml`, `mixed-4090-flash.yaml`.

## Q12. Does the prefill chunk size move the floor?

An operational finding from the engine restarts the chunk sweep needed, before the numbers: **the KV budget an
engine reports is not deterministic across starts.** Four times over the project (the phase-2 calibration pods,
the campaign pod's bring-up, the chunk-256 restart and, as predicted in the plan before it happened, the
chunk-2048 restart) the first engine started with a cold compile cache profiled 5–6% less KV than the others
(271k against 286–290k tokens), and a restart with the cache warm gave every engine the same budget. The budget is fixed at start, so a replica is 6% smaller until it is restarted, and a reservation
sized by KV then does not fit: seven runs of the campaign lost the note-taker's 4-slot reservation that way and
were repeated. A provider selling KV-sized capacity needs either a margin of that order or a gate after every
engine start; this project's gate (the KV budgets read from `/metrics` after each restart, the matrix preflight)
caught the third occurrence before a run was lost (ASSUMPTIONS.md "Phase 3 measurements"; PHASE3.md section 6).

**No: the engine knob goes the wrong way. Halving the prefill chunk to 256 tokens, the "smaller chunks interrupt
decodes less" hypothesis, made the floor worse on the 15 : 1 mix: floor misses 44 → 53%, start p95 6.7 → 15.8 s,
interactive queue and TTFT p95 doubled, output per GPU-hour −8% and margin −3 points against the same seeds of
the bursting baseline at chunk 512. Doubling it to 2048, vLLM's default, went the other way: floor misses
44 → 35% (p 0.05), every tenant delivered 12–18% faster, output per GPU-hour +4% and margin +1.4 points,
paid for with time to first token (p95 +22–43%, chat 2.6 → 3.7 s) while the start bound and the queue did not
move. Both directions were the opposite of the plan's hypothesis. The chunk size trades first-token latency
against decode steadiness; it does not remove the interference, and at 2048 a third of requests still miss
their floor. The fix that worked is the gateway's prefill cap (Q10), which limits how much prompt is prefilling
at once.**

Per run, seeds 1–2, paired against `burst` (chunk 512) on the same seeds:

| | `chunk256` | `burst` (512) | `chunk2048` (vLLM's default) | 256 vs 512 | 2048 vs 512 |
|---|---|---|---|---|---|
| utilisation | 59% | 53% | 53% | +12%, p 0.04 | 0 |
| output tokens / GPU-hour | 1.73M | 1.89M | 1.98M | −8%, p 0.04 | +4%, n.s. |
| revenue: shape (per-token) | $1.68 ($1.59) | $1.74 ($1.69) | $1.79 ($1.74) | −3% (−6%), p 0.02 | +3% (+3%), p 0.01 |
| gross margin, shape | 52.1% | 53.6% | 55.0% | −1.5 points, p 0.04 | +1.4 points, p 0.03 |
| floor misses: all / chat / agent / note-taker | 53 / 56 / 71 / 57% | 44 / 45 / 64 / 46% | 35 / 32 / 58 / 23% | +22%, p 0.08 | −21%, p 0.05 |
| start misses (queue + TTFT > 2 s) | 67% | 59% | 60% | +13%, n.s. | +1%, n.s. |
| start p95 | 15.8 s | 6.7 s | 7.9 s | ×2.4, n.s. | +18%, n.s. |
| interactive queue p95 | 12.4 s | 4.7–5.0 s | 5.8–6.0 s | ×2.5 | +20–25%, n.s. |
| TTFT p95: chat / agent / note-taker | 5.6 / 6.3 / 5.2 s | 2.6 / 3.8 / 3.1 s | 3.7 / 4.6 / 3.8 s | ×1.7–2.2, p 0.04–0.08 | +22–43%, p 0.04–0.4 |
| delivered tps: chat / agent / note-taker (shapes 30 / 60 / 30) | 34 / 40 / 34 | 36 / 43 / 35 | 40 / 48 / 42 | −6% (p 0.02) / −7% / −4% | +12% / +12% / +18% (p 0.015) |
| burst discount; context bursts | $0.047; 570 | $0.062; 681 | $0.079; 744 | −24% (p 0.015); −16% | +27% (p 0.02); +9% |
| requests served; batch output | 7,046; 346k | 7,443; 449k | 7,648; 505k | −5%; −23% | +3%; +12% |

What it means:

- The chunk size does not control prefill interference, the amount of prompt in flight does. The measured
  facts: every time-to-first-token doubled, the queue behind the slower prefill doubled, and the floors were
  missed on 9 more points of requests. The mechanism the numbers fit, not instrumented here: a prompt at 256
  takes twice the scheduler steps it takes at 512, so prefill is present in many more of the engine's steps,
  each of them slower than a pure decode step, and the decodes are interrupted more often, not less; the
  per-step token budget is also shared with the running decodes (one token each), so with 30–40 streams
  running a quarter of the small budget is gone before any prompt token is scheduled.
- Slower prefill looks like higher utilisation: seats were busy 59% of the time against 53%, with 8% fewer
  output tokens per GPU-hour. Under per-second billing the slower seconds are billed, so revenue fell only 3%
  while the work fell 8% (per-token revenue fell 6%), the same pattern as overcommit in Q10: a change that
  slows the engine costs the provider less than it costs the tenants.
- The result settles the section-9 question the backfill experiment (C) was waiting on: neither engine knob fixes
  the 30 tps floors; the gateway's prefill cap does (chat 45 → 12%, Q10) at −4 margin points. The floor is a
  gateway admission problem (how much prompt is admitted to prefill at once), and the prefill-aware admission
  curve DESIGN section 12 names stays the fix for the 60 tps shape.
- Chunk 2048, vLLM's default and the deployment a provider gets without tuning, is the better engine setting for
  this mix and still not a fix. Fewer, larger prefill steps leave more of the engine's steps to decode alone:
  every tenant's delivered rate rose 12–18%, the floors were missed on 9 fewer points of requests (chat 45 → 32%,
  note-taker 46 → 23%, agent 64 → 58%), the bursting tenants burst more (+27% discount) and the closed-loop
  tenants got 3% more requests through, so output per GPU-hour rose 4% and margin 1.4 points. The price is the
  first token: each large prefill step holds the engine longer, so TTFT p95 rose 22–43% (chat 2.6 → 3.7 s) and the
  start bound did not improve (60% missed, p95 7.9 s). A third of requests still miss their floor, against 12%
  for the 30 tps shapes under the prefill cap. The campaign ran at 512, the phase-1 choice from a 2080 Ti; the
  three points are 256 / 512 / 2048 at two seeds each, so the shape of the curve between and beyond them is not
  measured (DESIGN section 12 records 2048 as the candidate default, with the cap).

Sources: `campaign.html` (Paired comparisons, Capacity), `paired.csv`, `tenants.csv`; runs `chunk256-s1-20261109-215552`,
`chunk256-s2-20261109-221440`, `chunk2048-*`; the engines' KV budgets at each chunk in ASSUMPTIONS.md "Engine
sweep" (289,680 / 289,056 / 285,584 tokens at 256 / 512 / 2048); PHASE3.md section 6 for the restarts.

## Q13. What price makes a reservation pay for everyone, and does any?

**No single price does. The campaign's meters were re-priced offline over reservation factors 1.0–6.0× the
shape's price per second and SLA credits of 0 / 50 / 100% of the runtime price on two bases (a request
that missed its decode floor; a request that started more than 2 s after arrival), 245 price tables over
24 settings, each held against the scorecard fixed before any price was tried (PHASE3.md §9: the reserved
tenant's bill CV at most 2%, nobody's floor or start share worse than the on-demand bursting baseline's on
the same seeds, the on-demand tenants' start p95 at most 1.5× the baseline's and refusals within a point,
the provider's margin and output per GPU-hour at least the baseline's). Two settings pass every row: the
baseline itself and, at the campaign's prices, chunk 2048 (Q12); no reservation setting passes at any
price. The provider's margin row is met from 3.2× for the p95-sized reservations (52% against the
baseline's 50% on the same four seeds) and 5× for the 24 + 4 reservations (51% against 48%), where the
reserved tenant pays 1.2–1.7× (p95) or 1.1–1.6× (24 + 4) its per-token bill for a bill that moves 1.4% or
less across seeds. No price moves the two rows the p95 reservation breaks, the on-demand tenants' start
p95 (2.0–2.1× the baseline's) and the agent's refusals (+2.7 points): those are capacity, 51 of 96 seats
held. A credit on the start bound, the contract that actually fails under load (Q9, Q11), costs the
provider 4 points of margin at base load, 7–8 at saturation and 5 in the flash crowd at full refund.**

The shipping reservation (`resv-p95-spill`, four seeds) along the price axis, no credit; the baseline is
`burst` on the same seeds (gross margin 50.3%, output 1.93M per GPU-hour); the bill ratio is the reserved
tenant's slot bill over its per-token bill for the same requests (chat – note-taker):

| factor × shape price | revenue per run | gross margin | reserved bill ratio | reserved bill CV (max) | scorecard rows passed |
|---|---|---|---|---|---|
| 1.0 (cost share) | $1.03 | 21% | 0.40 – 0.55 | 3.1% | reserved service, floor, output |
| 2.3 (the campaign) | $1.42 | 43% | 0.90 – 1.25 | 1.4% | + reserved CV |
| 3.2 | $1.69 | 52% | 1.24 – 1.73 | 1.0% | + provider margin |
| 4.0 | $1.93 | 58% | 1.55 – 2.17 | 0.8% | same |
| 5.0 | $2.23 | 64% | 1.93 – 2.70 | 0.6% | same |
| 6.0 | $2.53 | 68% | 2.31 – 3.24 | 0.5% | same (on-demand start p95 and refusals fail at every factor) |

`resv-p95` without spill is within a point of every cell. The 24 + 4 reservations (`resv-shared`, six seeds,
baseline margin 48.1%) reach the margin row at 5× (51.5%, bill ratio 1.06 – 1.48) and fail their own
reserved tenants' service at every price (79 chat refusals per run, chat start p95 26 s); `resv-spill`
reaches it at 4× and fails the CV row at every price (9.7%).

SLA credits at the campaign's factor, gross margin with no credit → full refund of the runtime price of
every request that missed the basis (half refund costs about half the points):

| setting (seeds) | no credit | floor credit | start-bound credit | share of requests credited: floor / start |
|---|---|---|---|---|
| `burst` (6) | 48.1% | 44.4% | 44.1% | 34% / 48% |
| `load1.5-burst` (2) | 58.1% | 52.4% | 50.2% | 58% / 89% |
| `load2-burst` (2) | 59.5% | 54.1% | 52.1% | 57% / 88% |
| `flash-burst` (2) | 55.4% | 50.7% | 50.4% | 48% / 63% |
| `chunk2048` (2) | 55.0% | 51.6% | 50.7% | 35% / 60% |
| `resv-p95-spill` (4) | 43.1% | 41.7% | 40.8% | 12% / 24% |

What it means:

- The price of predictability is set by the provider's margin row at about 3× the shape's per-second
  price for a p95-sized reservation: the reserved tenant pays 1.2–1.7× per token for a bill that moves
  1% where its per-token bill moves 8–9%. Q7's revenue parity of 3.5–4.9× (two seeds, 24 + 4 pinned) is
  this frontier's 5× for the 24 + 4 reservations; p95 sizing passes lower because the fee covers 51 seats
  rather than 28 and 98–100% of the two tenants' traffic runs inside it, so the factor multiplies nearly
  the whole of their bill.
- No price table moves the on-demand tenants' waiting: the agent's start p95 is 2.0–2.1× the baseline's
  and the batch tenant's 1.9× at every factor and every credit, because the seats are held, not because
  of what they cost. Fewer reserved seats (24 + 4) pass the on-demand row and cap the reserved tenant
  instead (Q6, Q8). Giving the idle reserved seats back is the backfill experiment (PHASE3.md §9 C),
  which was planned, given pass criteria and not run.
- The credit is the sellable fallback and it is cheap because of what it refunds. The runtime price is
  the per-second part of the bill; the prompt charge, 70% of a shape bill on this traffic (Q1, Q7), is
  not credited. So refunding 88–89% of requests at saturation costs 7–8 points of margin, not half the
  revenue, and the surge's extra margin (flash-burst 55% against 43% with bursting off, Q11) is cut by
  5 points, not removed. A provider selling the start bound would price the credit from this table and
  keep bursting on.
- The start-bound credit costs more than the floor credit everywhere the load is high (saturation 7.9
  against 5.7 points, the surge 5.0 against 4.7) and about the same at base load (4.0 against 3.7),
  because at saturation 88% of interactive requests start late while 57% miss the floor. Under a
  reservation it costs 2.4 points: the reserved tenants start on time and only the on-demand ones are
  credited.
- The only setting that beat the baseline on every row was the engine's default chunk size (Q12): a
  statement about the campaign's choice of 512, not about pricing. At 2048 the floor row improves (35%
  against 44%) and the margin row (55% against 54%) with the on-demand rows unchanged.
- Offline re-pricing holds behaviour fixed. At 3–5× the shape price a real tenant reserves fewer slots
  or sends less, and with credits on offer a tenant's traffic shape changes what it is refunded; the
  meters cannot show either. The frontier says what each table would have billed for the traffic that
  ran; a run at those prices is the next experiment.

Sources: `runs/phase3/campaign/campaign.html` ("Scorecard and re-pricing frontier"), `scorecard.csv`,
`frontier.csv` (one row per setting × factor × credit × basis; `campaign_table` marks the prices the runs
were billed at); the criteria and the first read over 50 runs in PHASE3.md §9 A.

## What was not measured, and what to do next

Measured since the phase-2 list: reservation spill (`resv-spill`, `resv-p95-spill`, Q8), the `abort` revocation
policy (Q10), two more load levels each way (Q9), the overcommit and prefill-cap levers (Q10), output-heavy and
flash-crowd traffic (Q11), the engine's chunk size at three points (Q12), six seeds of the on-demand settings
and four of the shipping reservation (Q8). Not measured, in order of value:

- A start-bound contract for the interactive shapes. The 30 s queue cap is met at saturation and after a flash
  crowd while chat waits 17–30 s to start (Q9, Q11); the start bound (2 s) is reported as a placeholder and is not
  enforced by admission. The next thing to build is admission that refuses or sheds when the predicted start
  exceeds the bound, with the queue cap left to the lazy tier, and the SLA credit on the start bound (Q13) sold
  with it. Offline re-pricing priced the credit; only a run shows what tenants do when they are refused instead
  of queued.
- The prefill-aware seat curve (Q5, Q10, Q12). The prefill cap is a blunt version of it and costs 4 margin points;
  the 60 tps shape's floor (58–65% missed under every lever) waits for admission that counts prompt tokens in
  flight per replica, not seats.
- Reservation backfill (PHASE3.md §9 C): lending an idle reserved seat to a preemptable lazy request. Pass
  criteria are written (the reserved tenants' start p95 and floor not worse than `resv-p95-spill`, bill CV ≤ 2%,
  output per GPU-hour up, the lazy queue down, a reclaim-latency metric); decided 2026-11-10 after stage J: dropped for this project (the floor turned out to be admission's problem, the on-demand waiting is held seats, and about $9 of budget was left; PHASE3.md §9 C) and kept as the next experiment with the criteria unchanged.
- A burst cap or a priced burst for open-loop tenants (Q11): under per-second billing a burst is revenue given
  away unless a backlog refills the seats it frees; the knob (`burst.max_tps_factor`) is not built.
- The chunk-size curve beyond three points at two seeds (Q12): 256 / 512 / 2048 bracket the trade-off; 1024 and
  4096, and the cap combined with 2048, are not measured.
- The cause of the KV shortfall at a cold first engine start (ASSUMPTIONS.md phase 3): observed four of four
  times, gated against, not explained. vLLM's memory profile charged the first engine 1.16 GiB of peak activation
  against 0.21 GiB on the others; whether the compile cache, the CUDA-graph capture or the allocator is the
  reason was not instrumented.
- Behaviour under re-pricing (Q13): the frontier holds behaviour fixed. Real tenants would change what they buy
  and what they send at 4–5× reservation prices or with credits on offer; only a run with those prices shows it.
- Seeds: the sweep points, the levers, the chunk sizes and the scenario variants have two seeds each; the
  directions are consistent and the p-values on their service lines are mostly not significant (the tables say
  which). The on-demand settings have six, the shipping reservation four.
- One GPU model, one model size, one price basket: a 4090 at $0.893 per GPU-hour and a 7B AWQ model at the 7B
  basket; the fp16 model, a 120 tps shape and an A100 cross-check (PHASE2.md §3) were skipped for budget. The
  report recomputes the margins for any `gpu_usd_per_hour` and price reference.
- Session affinity for cache hits, phase-length reservations, and the gateway fix for requests abandoned in the
  queue (served and billed to a closed socket in one saturation run; built and tested, not run on hardware).

In order of value: the start-bound contract with its credit; the prefill-aware curve; the backfill decision;
the burst cap; then the optional experiments.
