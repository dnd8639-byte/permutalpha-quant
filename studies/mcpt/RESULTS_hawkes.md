# RESULTS: neurotrader VolatilityHawkes through the MCPT process

Source: github.com/neurotrader888/VolatilityHawkes (MIT). Ported into `strategies.py` as `HawkesVolatility`. **The port reproduces the author's script exactly** (PF 1.0687, time in market 50.7%). Settings were fixed before running (header of `run_hawkes.py`): BTCUSDT 1h, the author's 25-combination grid, 5 bps cost per unit traded.

## The funnel

| step | data | result | pass? |
|---|---|---|---|
| 1. In-sample | 2018–2020 | best κ = 0.1, lookback 96: PF 1.133, Sharpe 2.26 | ✓ |
| 2. In-sample MCPT (1000 bar permutations) | 2018–2020 | **p = 0.002** (noise best-PF median 1.033) | ✓ |
| 2b. Sign-flip test (volatility kept, direction random; 1000) | 2018–2020 | **p = 0.002** (null median 1.038) | ✓ |
| 3. Walk-forward (3y train, re-optimize every 90 days) | 2021–2022 | PF 1.051, Sharpe 1.06; parameters stable (κ = 0.1/96 in 8 of 9 refits) | ✓ |
| 4. Walk-forward MCPT (200) | 2021–2022 | **p = 0.025** | ✓ |
| **5. Post-publication** (fixed params from author's data; 1000 perms) | **2023 → Sep 2026** (Binance.US) | **PF 1.031, Sharpe 0.67, p = 0.026** | ✓ |

The first strategy in this project to pass every stage, including data from after it was made public.

## What the extra checks add

- **2b is the important one.** The standard permutation destroys volatility clustering, which is exactly what this strategy uses. The sign-flip null keeps the real volatility pattern bar for bar and randomizes only direction, and the strategy still beats it (p = 0.002). **The direction price takes after a quiet-to-volatile transition carries information.** It's not just an artifact of volatility clustering.
- **5 is the cleanest test.** The parameters are exactly what a 2023 viewer would have picked, traded unchanged on data nobody had seen. The data source was checked against the author's file (0.996 correlation on 2022).

## Why it is not "a money machine"

| post-publication detail | value |
|---|---|
| PF by year | 2023 **1.090** · 2024 1.017 · 2025 **0.991** · 2026 (to Sep) 1.044 |
| cost sensitivity (0 / 5 / 10 / 20 bps) | Sharpe 0.85 / 0.67 / 0.48 / **0.10** |
| long side / short side | PF **1.083** (24% of time) / **0.995** (26%) |
| trades | about 68 entries per year |
| buy-and-hold BTC over the same period | Sharpe 0.95; correlation with the strategy −0.05 |

- **Decaying:** in-sample PF 1.13 → walk-forward 1.05 → post-publication 1.03, with one losing year (2025).
- **Cost-sensitive:** it needs futures-level fees (≤ 10 bps). At retail spot fees (≈ 20 bps round-trip-ish) it is breakeven.
- **All post-publication profit came from longs** during a strong BTC bull market. Shorts broke even. The permutation keeps drift, so the p-value already allows for that, but it means the short side hasn't earned its place recently. *Trying a long-only version now would be a new, post-hoc idea and would need its own test.*
- **Buy-and-hold had a higher Sharpe.** The strategy's value is that it is **uncorrelated** with BTC (−0.05), not that it beats it.
- **Multiple testing:** this idea reached you because it looked good on the author's data. That selection is what test 5 guards against. It passed, but only at p = 0.026. Treat it as "real but small, and needs live confirmation," not "proven."

## Practical notes

- Binance.US spot does not allow shorting, so the full long/short strategy can't be run there as written.
- The natural next step is **paper trading**: the Pi runs the fixed κ = 0.1/96 strategy on live hourly bars with `DRY_RUN`. Compare live signals with the backtest's signals bar for bar. That tests your *engineering* (does live match backtest?) and extends the post-publication sample.

## Ledger

All runs are in `LEDGER.csv`: in-sample MCPT, walk-forward MCPT, sign-flip and post-publication for Hawkes, plus 2 entries for the ES Donchian example. Add these to the 13 preregistered tests from v2–v5 when judging any future p-value.
