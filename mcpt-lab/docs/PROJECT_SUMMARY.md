# Project summary (one page)

**What it is.** mcpt-lab is an open-source Python library that tests whether a trading strategy's backtest shows a real edge or just the result of tuning. It re-runs the entire strategy, tuning included, on hundreds of shuffled copies of the market and checks where the real result falls. It is built on neurotrader888's Monte Carlo permutation method. It adds:

- trading costs in every test;
- three new null tests: sign-flip, fixed-parameter post-publication, and label-shuffle for machine-learning filters;
- a test ledger that records every experiment;
- a validation suite and unit tests;
- reproductions of two published strategies.

## By the numbers

| | |
|---|---|
| library code | about 770 lines of Python (numpy, pandas, matplotlib) |
| examples, tests, validation, figures | about 680 lines |
| unit tests | 24, all passing, including two look-ahead "canaries" |
| permutations in the research ledger | 8,000 across 12 logged tests; each one re-runs the full strategy |
| calibration runs on synthetic markets | 90 fake markets (6,800 more permutations): 1 of 40 noise markets flagged in-sample, as chance predicts; 20 of 20 planted trends found |
| strategies reproduced from their authors' code | 2: Hawkes exactly; trendline with 1,172 of 1,175 trades identical and about 150× faster |
| data | 16 years of daily CME futures (ES test); 8.7 years of hourly BTC (76,000+ bars); 100 years of US stock factor returns (Kenneth French library) |
| publication-decay study | 10 famous published strategies (1980–2013), preregistered; 1 primary test + 9 descriptive gross + 9 descriptive net of costs, all in the ledger |

## Key findings

1. **A tuned backtest can look good and still be noise.**
   - A Donchian breakout on S&P 500 futures reached profit factor 1.11 in-sample.
   - Shuffled markets matched it (p = 0.56), and it lost money out-of-sample (PF 0.91).
2. **One published strategy survived publication, weakly.**
   - Volatility Hawkes on BTC passed all four steps and a stricter sign-flip test.
   - On 2023–2026 data, created after it was published and with its parameters frozen, it still beat shuffled markets (p = 0.026).
   - But its profit factor fell at every phase (1.13 → 1.05 → 1.03).
   - It breaks even at about 20 bps of trading costs.
3. **One published machine-learning filter stopped working after publication.**
   - Before publication (2020–22), it beat 99.5% of filters trained on scrambled labels (p = 0.005).
   - After publication, it was indistinguishable from them (p = 0.195).
4. **Across ten famous published strategies, most of the edge disappeared after publication.**
   - Preregistered replication of McLean & Pontiff (2016): value, momentum, reversal, profitability, the January, Monday, turn-of-the-month and Halloween effects, size, and Faber's moving-average timing.
   - Pooled, post-publication returns were **30% of in-sample** (p ≈ 3×10⁻⁹); all nine anomaly strategies declined.
   - No post-publication remainder survives a correction for the number of tests.
   - After measured trading costs and slippage, none keeps a reliable post-publication profit. The Monday and turn-of-the-month effects were never tradable in their own era.
   - Limits: "after publication" is also "later in time"; the cost estimates are approximate.

## Engineering decisions worth mentioning

- **Every test is logged automatically,** including failures, so the number of tries is never forgotten. That is what makes a p-value interpretable.
- **The tester was tested first,** on fake markets with known answers, before it was trusted on real ones.
- **Ports were verified against the original authors' code** before any conclusions were drawn. The trendline port replaced an iterative search with an exact closed-form solution.
- **Licensed vendor data (CME futures) and API keys are excluded** from the public repository. Only openly licensed data is bundled, with download scripts for the rest.

## Figures

The `results/figures/` folder holds all the figures:

| file | shows |
|---|---|
| `scorecard.png` | the 12 strategy tests' p-values |
| `all_tests.png` | the 12 strategy permutation distributions |
| `decay_story.png` | 10 published strategies: in-sample vs after publication vs after costs |
| `decay_growth.png` | $1 invested from each publication date, before and after costs |
| `calendar_costs.png` | Monday and turn-of-the-month effects vs their trading cost, 1926–2026 |
| `decay_tests.png` | the 18 publication-decay permutation tests |
| `hawkes_story.png` | the surviving strategy, phase by phase |
| `permutation_explainer.png` | what a permutation is |
| `validation.png` | the tester tested |
