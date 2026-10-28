# RESULTS v4: OU mean reversion on economic spreads (T8)

Run 2026-09-23 against `PREREGISTRATION_v4.md` (including Amendment v4-A1, recorded before any real-data run). Full output: `v4/results_v4.txt`.

## Verdict (preregistered rule)

**NO EDGE.** G1, G2 and G3 all fail. The net result is **significantly negative**.

| | value |
|---|---|
| Sharpe, net of costs | **−0.39** (95% CI −0.67 … −0.08; NW t −2.34) |
| Sharpe, before costs | −0.07 |
| Sharpe, 2× costs | −0.68 |
| 2010–2017 / 2018+ | −0.54 / −0.28 |
| DSR (N = 12) | 0.00 |
| annual vol (in market only ~10% of days) | 2.4% |

| spread | net | gross | in market | trades | median half-life |
|---|---|---|---|---|---|
| soybean crush | −0.02 | +0.23 | 8% | 18 | 24 d |
| crack | −0.13 | −0.04 | 11% | 30 | 17 d |
| Treasury curve | −0.36 | −0.03 | 12% | 18 | 20 d |
| WTI–Brent | −0.35 | −0.15 | 9% | 18 | 25 d |
| gold–silver | −0.26 | −0.19 | 10% | 16 | 27 d |

## What it means

- **The mathematics works; the markets didn't cooperate.** On synthetic cointegrated data the Kalman + OU machinery matched an oracle that knows the true spread (Sharpe 1.46 vs 1.44). On real spreads, divergences of 2 standard deviations did **not** reliably revert. Before costs the strategy made roughly nothing.
- **Costs turn nothing into a loss.** Each trade hits 2–3 legs, and energy legs roll monthly. On a strategy that is usually flat, that is enough to make results significantly negative.
- **It is a genuine diversifier with nothing to diversify.** Its correlation with the v3 strategies is −0.05 to −0.25, but its own return is below zero.
- **Few trades.** Only 16–30 round trips per spread in 13 years, so the per-spread numbers are noisy. The combined verdict is not borderline, though.

## Also documented (harness, before real data)

- Chan's Kalman δ is unit-dependent. In log-price units it made the filter absorb the spread entirely (true half-life 10 days, estimated 0.5).
- **Hindsight fitting of hedge ratios and OU parameters manufactures about +0.5 Sharpe on unrelated random walks.** This is the classic pairs-trading backtest trap, now quantified in this project.

## Running tally (all preregistered, all reported)

Twelve strategy/universe tests. **None passes.** Several show small positive effects historically; none is large enough to separate from luck after counting every attempt, and none shows strength since 2018.
