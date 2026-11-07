# Results: what the proof of concept measured

Answers to the four questions of DESIGN.md §1, from the GPU runs of milestones 2–4 (2026-11-06 and
2026-11-07). Setup for every number here: one RTX 2080 Ti serving Qwen2.5-3B-Instruct-AWQ through
vLLM, the four-tenant `mixed.yaml` scenario (chat on `small.8k.30tps`, agent on `small.8k.60tps`,
note-taker on `small.16k.30tps`, batch on `small.8k.15tps.lazy`), 18 minutes per run, seed 1 unless
stated, GPU priced at $0.30 per hour, shape and per-token prices at parity with the `basket-3b`
market reference (the median of four 3B-class OpenRouter listings: $0.0755 per 1M prompt tokens,
$0.15 per 1M output tokens). The measurements, run ids and caveats are in ASSUMPTIONS.md; the
reports in `runs/<run id>/report*`. Earlier milestones were billed at a gpt-4o-mini placeholder 4×
higher; where a number comes from them it is marked.

The one-paragraph version: time-sliced shapes with admission control turn a card that broke its
floors on 90% of fast streams into one that holds them 80–90% of the time, at the cost of queueing
(chat waits 10–13 s at peak) and 18% of throughput. At market parity the card earns 1.6–1.7× its
cost on this traffic, but only because prompt tokens are charged; the per-second price of decode
alone cannot pay for a GPU. Bursting gives back 12–15% of the throughput for free and halves
response times, at a few points of floor compliance. Reservations deliver a genuinely different
product (no queue, 13 ms) and, at parity prices, lose money; the price they need is 2.3–3.2×
parity. The 60 tps shape should not be sold on this card: it costs six of eight seats whenever it
runs, and any reservation locks it out completely.

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

## What was not measured, and what to do next

- Two seeds at most per setting (one for the M4 runs); the direction of every finding is
  consistent across the M3 and M4 runs but the small-sample tenants (note-taker, 15 requests) are
  noise at the tenant level.
- Revocation of context bursts on hardware (see Q4); the `grace` and `abort` runs are identical
  because nothing was ever revoked.
- Prefix-cache hits are charged in full (Q1); every margin here is an upper bound for prompt-heavy
  traffic until `cached_tokens` is metered.
- A single real replica; the multi-replica scheduler and the "reservations on their own replica"
  answer to Q3 are exercised by fake replicas only.
- Whole-run reservations straddle quiet and peak phases, which is most of their idle share;
  phase-length reservations (`reservation.phases`) exist in the load generator and were run only in
  the smoke test.

In order of value: meter cached tokens and price cache reads; context bursts at the slowest tps
class with a short queue; a lending reserve; a reservation price factor of 2.3–3.2 and a second
replica (fake is enough to show placement) for the 60 tps shape; then more seeds.
