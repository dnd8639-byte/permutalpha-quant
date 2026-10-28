# RESULTS: neurotrader TrendlineBreakoutMetaLabel through the MCPT process

Source: github.com/neurotrader888/TrendlineBreakoutMetaLabel (MIT). Ported in `trendline_meta.py`; test design fixed before running (docstring of `run_trendline.py`). The author's parameters are used throughout. Costs are 5 bps per unit traded.

## Port check

- The resistance line is solved **exactly** (closed form) instead of by the author's iterative search, which is **150× faster**.
- Resistance values match the author's to a median of 4e-6 (log price). 1,172 of 1,175 trades are identical; the other 3 are borderline breakouts where the iterative search stops short of the optimum.
- Author's walk-forward result (0 cost, 2020–2022): all trades PF **1.0539**, meta-labeled **1.1423**. Port: **1.0535 / 1.1371**.
- Batched model predictions give identical results to the author's one-at-a-time version.

## A. Base breakout (every trade, no filter)

| test | PF (5 bps) | PF (0 cost) | p |
|---|---|---|---|
| A1 in-sample 2018–2019 | 0.937 | 0.972 | 0.627 |
| A2 out-of-sample 2020–2022 | 1.012 | 1.054 | 0.119 |
| A3 post-publication 2023 → now | 1.012 | 1.079 | 0.077 |

**No demonstrable edge.** It loses in-sample and is marginal afterwards.

## B. Meta-label filter (random forest picks which breakouts to take)

| test | filtered PF | vs all trades | p |
|---|---|---|---|
| B1 full-pipeline walk-forward MCPT, 2020–2022 | **1.091** | 1.012 | **0.015** |
| B2 label-shuffle null, 2020–2022 | 1.091 | (random-filter median 0.992) | **0.005** |
| B3 label-shuffle null, **post-publication 2023 → now** | 1.028 | 1.012 | **0.195** |

- **2020–2022: the filter really worked.** It beat 99.5% of filters trained on scrambled labels, and the whole pipeline beat 98.5% of permuted markets. The features (trendline slope, fit error, volume, distance, ADX) did pick better breakouts.
- **After publication it stopped working.** The improvement shrank from +0.08 PF to +0.016, which is no longer distinguishable from a filter that learned nothing (p = 0.195).

## What this teaches

1. **The permutation test caught real signal in 2020–2022, and publication (or time) removed it.** This is the decay your point about published strategies predicted, now measured.
2. **Meta-labeling can add value to a weak base signal**, but only while the relationships it learned still hold. A model that must be retrained every year is only as good as the next year resembling the last.
3. **Compare with VolatilityHawkes:** Hawkes survived publication at a reduced strength (p = 0.026). This one did not (p = 0.195 for the filter, 0.077 for the base).

## Ledger

`LEDGER.csv` now holds 12 tests: 2 ES Donchian, 4 Hawkes and 6 trendline. That's in addition to the 13 preregistered tests from v2–v5.
