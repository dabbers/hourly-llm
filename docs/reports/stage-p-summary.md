# Stage P summary: slot-hour billing without an output ceiling (Q20)

Run 2026-11-20 on the local RTX 2080 Ti (Qwen2.5-3B-Instruct-AWQ, 8 seats, four tenants). 27 runs (9 arms, seeds 1–3),
9,345 requests, 0 errors, 28 refused at the 30 s queue limit. No paid GPU time. Full text: `docs/RESULTS.md`, Q20.
Table: `docs/reports/stage-p/stage_p_table.csv`. Values below are seeds 1 / 2 / 3.

## Short answer

The pacer buys the host nothing, but removing it does not fix the slot model. No paid run is recommended.

## Paced (`o-slots`) against unpaced (`p-slots`)

| Measure | Paced | Unpaced | Change |
|---|---|---|---|
| Seat time, mean (s) | 4.16 / 5.14 / 3.43 | 4.17 / 5.05 / 3.40 | 0% / −2% / −1% |
| Output tokens | 55,012 / 49,903 / 40,945 | 56,584 / 51,541 / 41,662 | +3% / +3% / +2% |
| Chat end-to-end time, median (s) | 9.0 / 8.3 / 6.7 | 3.9 / 3.3 / 1.8 | −57% / −61% / −72% |
| Chat conversation, median (s) | 85 / 88 / 63 | 67 / 64 / 47 | −22% / −27% / −26% |
| Chat wait to first token, mean (s) | 3.58 / 4.04 / 1.44 | 3.08 / 3.64 / 1.63 | −14% / −10% / +13% |
| Chat wait to first token, p95 (s) | 14.3 / 19.3 / 6.1 | 11.8 / 17.9 / 9.9 | −18% / −7% / +63% |
| Slots wanted, sum of tenants' p95 | 19 / 19 / 10 | 18 / 18 / 12 | against 8 seats |

## Findings

1. **The ceiling holds no GPU capacity.** Slots were held for the same time with and without it (within 6% per
   tenant). It only delays the customer: chat answers took 57–72% less time end to end without it.
2. **Speed is not a tier.** Unpaced, both chat tiers got 65–100 tokens a second, 2 to 6 times what they pay for.
   The pro tier was never more than 2% faster, and was 19% slower on seed 1 (not explained).
3. **Engine priority does not separate tiers.** The standard tier waited 12–28% longer for its first token after
   admission; the pro tier gained nothing measurable.
4. **Waits are still set by slot counts.** Tenants wanted 12–18 slots against 8. Against a plain unpaced pool
   with no reservations, chat's wait p95 was 25–39% higher with slots; document tenants finished 11–23% sooner.
5. **Unpaced waits on the light seed are a recurring anomaly.** On seed 3 chat's wait p95 rose 63% without the
   pacer (and 48% on the oversold arm). The open-loop pair also missed its line there (+13%), which is no bigger
   than the difference between two identical runs (−16% to +9%). Not settled.
6. **The prefill gate moved the wait, it did not shorten it.** Time from admission to first token fell from
   0.7–0.9 s to about 0.25 s, but time from arrival to first token did not improve.
7. **Today's slot price is too low without a ceiling.** A flat slot-hour bill recovers 38–47% of the per-token
   revenue for the same requests, and the standard tier gets the same speed at half the price per slot. With one
   context size, all eight slots are the same product; one price about 2.1–2.6 times today's average slot price
   would match per-token revenue on this traffic.

## Hypotheses

| # | Line | Result |
|---|---|---|
| P1 | the ceiling holds no GPU capacity | held |
| P2 | chat answers at least 30% sooner | held (−57% / −61% / −72%) |
| P3 | chat wait p95 not more than 10% worse | failed on seed 3 (+63%) |
| P4 | conversations at least 10% shorter | held (−21% pooled) |
| P5 | the two chat tiers within 10% in speed | failed on seed 1 (pro 19% slower) |
| P6 | priority partly separates tiers | failed, both halves |
| P7 | pacing does not cause the oversold slowdowns | failed on seed 3 (−11% against a 10% line; wait p95 +48%) |
| P8 | slot needs do not grow | held, but could not fail (bounded by slots held) |
| P9 | fair share with the gate helps chat | failed |
| P10 | fair share costs no output | held |
| P11 | open loop: the ceiling holds no capacity | held |
| P12 | open loop: waits do not change | failed on seed 3 (+13%) |

## Decision rules

- **Drop the output ceiling from the slot model: yes** (P1, P2 and P11 held).
- **Stop selling speed as a tier:** the condition is not met as written, because P5 failed. It failed with the
  higher tier being slower, which points the same way. The owner's call.
- **P3 and P12 both failed on seed 3.** By the rule this needs explaining first. Engine occupancy was equal and
  the gap is inside the repeat spread, but that is a reading, not a measurement.
- **`p-fair` is not proposed** (P9 failed).
- **Paid 4090 run: no.** P3 and P12 failed, and tenants wanted 18 / 18 / 12 slots against a limit of 9.6.

## Limits

One 2080 Ti, a 3B model, four tenants: ratios, not absolute numbers. Windows restarted during the chain; the
interrupted run was discarded and re-run, and seed 3 ran about eight hours after seeds 1 and 2 (its paced
baseline matched stage O's). The open-loop arms still sent 28–40% of turns late on the busy seeds. Seed 3 is
light traffic, so its p95 rests on about 13 requests.
