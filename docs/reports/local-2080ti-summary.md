# Results summary: stages N and O (local RTX 2080 Ti, 2026-11-19 and 20)

Full text: [RESULTS.md](../RESULTS.md) Q17, Q18 and Q19. Tables: [stage-n](stage-n/stage_n_table.csv),
[stage-o](stage-o/stage_o_table.csv), [slot-hour re-pricing](stage-o/slot_hour.csv).

## In short

- **Fair share of prompt reading works.** It cut the mean wait 17–35% and the chat tenants' wait 27–46%, with
  no output lost. It is the best result of the two stages.
- **Slot-hour billing with ceilings did not beat today's design on waits.** The tenants need more seats than
  the hardware has, and the model answers that by refusing requests or by slowing streams.
- **No paid RunPod run is recommended.** The rule set before the runs was not met. Spend is unchanged at
  $113.93 of $150.

## The test bed

One RTX 2080 Ti, Qwen2.5-3B, 8 seats at 30 tokens a second. Four tenants: two chat and two document tenants,
one of each on a 15 tps tier and one on a 30 tps tier. Three seeds, eight minutes a run, 21 runs, no request
errors. These are scaled tests of a mechanism. Read the ratios, not the seconds.

## Q17. What price gives a 25% margin with idle GPU time? (extrapolated, not measured)

- At the 7B basket prices the stage M runs earn $1.65 per GPU-hour against $0.89 rent.
- A 25% margin needs the cluster at the tested load for 72% of the hours it pays for. Break-even is 54%.
- A reserved 30 tps seat costs about $0.025 an hour at a 25% margin.

## Q18. Fair share of prompt reading between tenants (stage N)

The gateway orders waiting requests so that each tenant gets a share of prompt reading in proportion to the
rate it pays for. Arms: plain queue (`n-fifo`), a cap on prompt tokens being read (`n-cap`), and the cap with
fair share (`n-fair`).

Mean wait from arrival to first token, seconds, per seed (1 / 2 / 3):

| | `n-fifo` | `n-cap` | `n-fair` | fair against cap |
|---|---|---|---|---|
| All requests | 2.43 / 4.47 / 2.08 | 2.70 / 4.29 / 2.00 | 2.17 / 2.80 / 1.66 | −20% / −35% / −17% |
| Chat tenants | 2.05 / 3.75 / 1.62 | 2.30 / 3.57 / 1.53 | 1.66 / 1.92 / 1.12 | −28% / −46% / −27% |
| Document tenants | 5.78 / 9.42 / 5.41 | 6.31 / 9.26 / 5.44 | 6.69 / 9.18 / 5.63 | +6% / −1% / +4% |

- **The gain is from the order, not the cap.** The cap alone moved the mean wait by +11%, −4% and −4%.
- **The heavy tenants lost little.** Document waits rose 3% on average, 9–14% at p95.
- **Paying more bought less than its share.** The 30 tps document tenant queued 0.83 of the 15 tps tenant's
  time under fair share, against 1.15 without it. The line asked for 0.75.
- **Streams did not get faster.** 8–18% of chat streams ran under 95% of their rate in every arm.
- **Two documents were refused** on the heavy seed, at the 30 s queue limit.

Three of six hypotheses held (N1, N2, N5) and three failed (N3, N4, N6).

## Q19. Slot-hour billing with the rates as ceilings (stage O)

The model: every tenant pays a flat price per slot per hour, used or not. The output rate is a ceiling, with no
bursting. Prompt reading is a ceiling too, at 10 times the output rate per slot. No floor is promised.

### Part A. Offline: the stage K and M runs re-priced

- **The slots do not fit.** The tenants' p95 needs add up to 129–149% of the seats, so the model only works
  oversold.
- **Tier prices at 60% of seats sold:** $0.036 (15 tps), $0.046 (30 tps) and $0.099 (60 tps) per slot-hour.
- **The provider keeps part of its cache-routing gain.** Revenue falls 13%, against 17% per token and 19% per
  second, because tenants need 13% fewer slots. It keeps all of it only if tenants do not resize.
- **The bill is flat only for a tenant that never resizes.** Resized daily, it varies more than the per-token
  bill for chat (22% against 13%) and the note-taker (33% against 8%).
- **Price against per token:** agent on the fast tier 1.5–1.6 times, chat 0.9–1.1, batch 0.9–1.2, note-taker
  0.7–0.8.

### Part B. Local runs

Arms, each adding one thing: slots only (`o-slots`), input ceiling (`o-bucket`), tier priority (`o-prio`), and
every tenant's p95 slots on a card sold twice (`o-over`). `n-fifo` is today's design.

Per seed (1 / 2 / 3):

| | `n-fifo` | `o-slots` | `o-bucket` | `o-prio` | `o-over` | `n-fair` |
|---|---|---|---|---|---|---|
| Chat wait, mean s | 2.05 / 3.75 / 1.62 | 4.13 / 3.91 / 1.56 | 3.46 / 2.52 / 1.49 | 3.30 / 2.51 / 1.53 | 2.27 / 2.57 / 1.44 | 1.66 / 1.92 / 1.12 |
| Chat wait, p95 s | 8.1 / 16.5 / 8.2 | 17.1 / 17.7 / 6.4 | 12.9 / 13.6 / 7.8 | 12.7 / 13.8 / 8.2 | 11.2 / 13.9 / 7.1 | 5.9 / 8.1 / 4.9 |
| Document wait, mean s | 5.78 / 9.42 / 5.41 | 5.49 / 8.33 / 5.65 | 5.93 / 7.66 / 9.59 | 5.87 / 7.67 / 9.63 | 5.64 / 7.72 / 5.57 | 6.69 / 9.18 / 5.63 |
| Refused | 0 / 0 / 0 | 0 / 5 / 0 | 1 / 11 / 5 | 1 / 11 / 5 | 0 / 3 / 0 | 0 / 2 / 0 |
| Slowest 5% of chat streams, share of tier rate | 0.61 / 0.56 / 0.73 | 0.76 / 0.56 / 0.97 | 0.74 / 0.69 / 0.85 | 0.74 / 0.64 / 0.82 | 0.42 / 0.34 / 0.65 | |

- **Seats sold once:** each tenant queues for its own slots. A tenant sized above its load waits less (the
  30 tps document tenant, 10.0 s to 5.8 s on the heavy seed). A tenant sized below it waits more (the 15 tps
  chat tenant, 1.68 s to 4.65 s on seed 1).
- **The input ceiling protects the others, and one tenant pays.** Chat time to first token fell 17–26%. The
  15 tps document tenant sent 13,500 prompt tokens a minute against a limit of 9,000 and had 7%, 48% and 31%
  of its documents refused.
- **Tier priority:** the faster tier's first token came 9–11% sooner, the slower tier's 7–22% later. Waits,
  refusals and output did not move.
- **Overselling:** waits closest to today's design and 3 refusals in 1,037 requests. The slowest 5% of chat
  streams ran at 34–65% of their rate.
- **Output:** 1–5% below today's design. The load was under the card's capacity, so this is not a capacity
  measurement.
- **Fair share beat every stage O arm for chat**, on every seed.

Two of seven hypotheses held (O2, O6) and five failed (O1, O3, O4, O5, O7).

### The two questions asked of this stage

- **Who would buy it?** A tenant with long prompts and steady traffic that buys more slots than it needs. It
  gets shorter, steadier waits and a lower bill than per token. A tenant with bursty short turns gets a longer
  wait, or a slow stream on an oversold card.
- **Does the provider keep its efficiency gains?** Partly, and only while tenants do not resize.

## Limits

- One GPU, a 3B model, four tenants, eight minutes a run.
- The 30 tps document tenant also held two slots against one, so the runs do not separate a faster tier from
  more slots.
- Slot counts were set from one base run, not chosen by tenants reacting to a price.
- Arms with longer waits sent 1–5% fewer requests, because a chat turn is sent after the one before it ends.
- Not measured: fair share with cache routing across replicas, and either mechanism on the 4090s.
