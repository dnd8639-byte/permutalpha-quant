# Does the gate's p-value mean what it says?

The gate claims: "on a market with no edge, a search passes only about 5% of the time (alpha = 0.05)". That is a
claim about the test itself, so it can be measured. `python -m oql.calibration` runs the full gate (search, then the
same search on 39 shuffled copies) hundreds of times on synthetic books whose truth is known, under each null model.

Books: 3 markets of 2,000 daily bars each, development part = first 70%. A smaller search than the default (4 rounds)
so 300 gates fit in about 20 minutes on two cores. Raw results: `calibration_results.json`.

| markets | null | gates | pass rate (p ≤ 0.05) | mean p | shuffled best (median) | real best (median) |
|---|---|---|---|---|---|---|
| no edge, volatility clustering | bars | 100 | 3% (± 2) | 0.57 | 0.64 | 0.62 |
| no edge, volatility clustering | vol_matched | 100 | 4% (± 2) | 0.56 | 0.63 | 0.62 |
| no edge, strong clustering | bars | 100 | 1% (± 1) | 0.60 | 0.69 | 0.60 |
| no edge, strong clustering | vol_matched | 100 | 1% (± 1) | 0.55 | 0.64 | 0.60 |
| real edge (persistent trends) | bars | 100 | 96% (± 2) | 0.03 | 0.80 | 2.93 |
| real edge (persistent trends) | vol_matched | 100 | 98% (± 1) | 0.03 | 0.81 | 2.93 |

What this shows:

1. **Neither null lets luck through.** On markets with no edge both pass at or below the 5% they promise.
2. **The classic bar shuffle is too strict when volatility clusters strongly.** Scrambling *when* volatility was
   high produces shuffled markets on which the best strategy looks better (median 0.69) than on the real no-edge
   market (0.60), so real results are judged against an inflated bar: p-values pile up near 1 (mean 0.60, 90th
   percentile 0.98). A test that is too strict does not produce false discoveries, but it loses real ones.
3. **The volatility-matched null is better calibrated there** (mean p 0.55, shuffled best 0.64) and at least as
   powerful on a real edge (98% vs 96%; the difference is within the noise of 100 gates).
4. Both nulls give mean p slightly above 0.5 even with mild clustering: the whole test leans a little conservative.

What it does not show: how either null behaves on real markets (unknowable without a known truth), or for intraday
strategies. Experiments choose with `nulls=[...]`; declaring both requires p < alpha under each, the stricter test.

## Power: how strong must an edge be to be caught?

Same full gate (volatility-matched null, 39 shuffles) on synthetic books of 10 markets with persistent trends of
growing strength and about 11 years of development data, the length of the CME futures sample. Raw results:
`power_results.json` (140 of 160 gates; the run was interrupted near the end).

| planted trend | gates | best Sharpe found (median) | caught (p ≤ 0.05) |
|---|---|---|---|
| very weak | 40 | 0.36 | 8% (± 4) |
| weak | 40 | 0.41 | 10% (± 5) |
| moderate | 40 | 0.40 | 15% (± 6) |
| stronger | 20 | 0.46 | 10% (± 7) |

With 11 years of data, an edge whose best strategy shows a Sharpe around 0.4 is caught about one time in ten:
barely more than the 5% a no-edge market passes by luck. The same test caught 96-98% of edges with Sharpe near 3,
and card 0004 (time-series momentum, development Sharpe 1.07 over 45 years) passed with p = 0.005. The lab's honest
specification: it reliably confirms edges of about Sharpe 1 over several decades; it cannot confirm a Sharpe of 0.4
from a decade. No test can: the uncertainty of a Sharpe ratio measured over T years is about 1/√T.

## Does the 5% promise hold in a large search?

60 full gates with the large-search space of project 11 (about 600-700 distinct strategies per search, dense
grids), on no-edge books, volatility-matched null with 39 shuffles. Raw results: `large_search_results.json`.

| search | gates | passed (p ≤ 0.05) | mean p | best Sharpe found, real (median) | shuffled best (median) |
|---|---|---|---|---|---|
| ~650 strategies, no edge | 60 | 1.7% (1 of 60) | 0.58 | 0.90 | 0.92 |

On noise, searching hundreds of strategies finds a winner with a Sharpe near 0.9, but the same search on shuffled
copies finds winners just as good, so the gate passes at no more than its promised rate.

## A third null: the block bootstrap

The block null cuts the history into blocks of 20 consecutive bars and shuffles the blocks (one order for every
market). Inside a block everything is real; patterns longer than about 20 bars are broken. Same 300-gate study as
above (raw results: `block_results.json`):

| markets | bars | vol_matched | block | shuffled best (median), block |
|---|---|---|---|---|
| no edge, volatility clustering | 3% | 4% | 3% | 0.65 |
| no edge, strong clustering | 1% | 1% | 2% | 0.65 |
| real edge (persistent trends) | 96% | 98% | **88%** | 1.89 |

What this shows:

1. **All three nulls keep false alarms within 5%.**
2. **The block null is the strictest against trend strategies.** The planted trends last about 67 bars on average,
   so a 20-bar block still carries part of each trend: the shuffled markets keep some trendiness, the best strategy on
   them scores much higher (median 1.89 vs 0.81 for the bar shuffle), and the real edge is caught less often (88%).
   That is by design: the block null only credits structure LONGER than a block.
3. **Choosing a null is choosing a question.** Bar shuffle: "is there any time structure?"; volatility-matched:
   "is there structure beyond the volatility path?"; block: "is there structure beyond the market's short-term
   behaviour?". Declaring several (`nulls=[...]`) requires the result to survive each.
4. Under strong volatility clustering the block null is about as conservative as the bar shuffle (mean p 0.59 vs
   0.60; volatility-matched 0.55): 20-bar blocks keep clustering inside each block but scramble it across blocks.
