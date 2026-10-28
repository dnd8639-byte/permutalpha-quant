# RESULTS v5: front-running the commodity index roll (T9, "Goldman roll")

Run 2026-09-23 against `PREREGISTRATION_v5.md`. **Two runs, both reported.** Run 2 uses the calendar fix in Amendment v5-A1, which was made *after* run 1 was seen and is disclosed there.

## Verdict (preregistered rule, run 2)

**NO EDGE.** G1, G2 and G3 all fail. The net result is significantly negative.

| | run 1 (calendar bug) | **run 2 (corrected)** |
|---|---|---|
| Sharpe, net | −1.76 | **−1.06** (95% CI −1.91 … −0.48) |
| Sharpe, before costs | −0.08 | **+0.18** |
| Placebo BD 12–20 (net) | −0.21 | −0.20 |
| 2010–2017 / 2018+ | −1.81 / −1.75 | −1.35 / −0.92 |

## Per market (run 2)

| market | net | gross | placebo | avg P&L BD 1–4 | avg P&L BD 5–9 (roll) |
|---|---|---|---|---|---|
| CL (WTI) | −0.29 | **+0.56** | +0.25 | −19 bp | **+77 bp** |
| HO | −1.11 | −0.08 | −0.72 | −8 bp | +3 bp |
| RB | −0.48 | +0.21 | −1.47 | −1 bp | +13 bp |
| NG | −0.74 | −0.41 | −0.78 | +10 bp | −35 bp |

## What it means

- **There is a faint trace of the mechanism, but it is not tradable.** Before costs the roll window is slightly positive (+0.18) and beats the no-roll placebo. In CL, the most heavily indexed contract, almost all of the gain comes during the actual roll days (BD 5–9), which is exactly where the mechanism says it should. That fits the published finding that this effect **was** strong and has been largely arbitraged away as it became well known.
- **Costs make it a loss.** The preregistered cost charges each leg separately (4 × 5 bps × about 2.8× leverage ≈ 55 bp per month). Real calendar spreads trade as one instrument and are cheaper. **Even at zero cost, though, a Sharpe of +0.18 is far below the bar**, so the verdict does not depend on the cost model.
- **The CL result is a hint, not evidence.** Picking out the best market after seeing results is exactly the search this project avoids. Testing "CL only, realistic spread costs" would be a new preregistered trial (#14), with a low prior.

## Bug found (and how it was handled)

7. **Databento's UTC-day bars include Sunday-evening and holiday stubs** (0.3–2% of volume) as separate dates. Counting them as trading days put the "BD 1–9" window about two days early, so it missed most of the roll. It was found while checking run 1, fixed as a correctness issue (v5-A1), and **both runs are reported**. The harness missed it because the synthetic data used a clean weekday calendar. The lesson is to test the calendar against the real exchange calendar, not only synthetic data. (In v3/v4 the stubs only split a day's return in two; monthly results are unaffected.)

## Harness

H13 planted roll effect → Sharpe 4.0 (PASS) · H14 no effect → 0.06 (PASS) · H15 look-ahead canary → 8.9 (PASS).

## Running tally

**13 preregistered tests, 0 passes.** The DSR bar is now 0.68 Sharpe. Large negative trials widen the spread of trial Sharpes, which raises the bar further. That makes the bar conservative, and it is noted here.
