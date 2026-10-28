# Is tomorrow's efficiency predictable?  (development data only, holdout untouched)

Data: 35 CME futures, front contract by yesterday's volume, 2010-06 to 2021-10 (the first 70%), 100,000 market-days.
Efficiency on daily bars: ER = |close - open| / (high - low).

1. **The daily-bar ER measures the same thing as the hourly-path ER.** On 7 markets with hourly data
   (BTC, EURUSD, GBPUSD, USDJPY, gold, crude, S&P), rank correlation 0.94-0.95, same tercile 83-85% of days.
   (`research/proxy_check.py`)
2. **It is not predictable one day ahead.** Pooled gradient boosting, walk-forward by year 2013-2021:
   out-of-sample R^2 = 0.0014. Simply using each market's own average ER scores 0.0023, i.e. better.
   Predicted low / mid / high tiers realize 0.441 / 0.456 / 0.468: a real but tiny spread. (`study_er.py`)
3. **Release calendars barely matter.** Next-day ER is 1.2 points higher before FOMC and payrolls days,
   0.2 points before CPI and EIA days.
4. **Longer windows don't help.** Kaufman ER over the next 5, 20 and 60 days: R^2 -0.003, -0.015, -0.034. (`study_horizon.py`)
5. **Why: daily ER looks like a random walk's.** A simulated random walk with fat tails and clustered volatility
   gives mean ER 0.474 and median 0.48; the 35 futures give 0.455 and 0.46, with lag-1 autocorrelation -0.03.
   ER is a ratio, so volatility cancels out of it, and volatility is the part of markets that IS predictable.
   (`study_random_walk.py`)

Conclusion: the 7-market failure was not caused by too little data. With 140x more rows the answer is the same.

## Building the gate up (gate v2)

Three upgrades, same walk-forward test (`study_gate.py`). Spread = realized ER on predicted-high days minus predicted-low days.

| gate | OOS R^2 | spread | t |
|---|---|---|---|
| original features | +0.0014 | +0.027 | 8.7 |
| + calendar (month-end, holidays, options expiry, roll days) | +0.0012 | +0.028 | 8.3 |
| + market-wide efficiency and range | -0.0010 | +0.026 | 7.1 |
| everything | -0.0012 | +0.024 | 6.3 |

The volatility model works where the efficiency model doesn't: OOS R^2 0.10; predicted low / mid / high volatility days
realize 0.82x / 0.92x / 1.06x their normal range. Gate v2 = efficiency tier x volatility tier (25 gate settings).

## Development run, 35 futures, 2-D gate (`daily_engine.py dev 99`)

Engine checks first (`validate_daily.py`): on a synthetic market with a real, predictable trend regime it found
a gated breakout rule (Sharpe 5.2, whole-search p = 0.05, the smallest possible with 19 permutations) and the ablation
showed the high-efficiency gate adding value. On pure noise its best was 0.79 and p = 0.10, so it was not fooled.

Real data, 2013 to Oct 2021 (holdout sealed):
* 185 strategies tried. Best: **12-month time-series momentum, both sides, no gate**, dev Sharpe 0.60.
* Long side 0.41, short side 0.34; beta to all markets -0.08, to the S&P +0.05.
* Cost/delay Monte Carlo (costs x0.5 to x3, 0-3 hour fills): median 0.57, 5th percentile 0.43.
* **Every gate setting made it worse** (0.60 -> between +0.05 and -0.68). An uninformative gate just switches the
  position on and off, which pays costs and drops days at random.
* Deflated Sharpe 0.17. **Whole-search permutation p = 0.22**: the same search on shuffled data finds a best of 0.39
  (median) and 0.76 (95th percentile). Not significant. Holdout NOT opened.

## Can the efficiency feature be made to work?  (exploratory, development data only)

* **A strategy that is paid by efficiency exists.** Opening-range breakout on BTC (take the direction of the first
  2 hours, hold to the close): if you KNEW the day's efficiency, high days earn +94 bps and low days lose 77 bps.
  (`research/orb_feasibility.py`)
* **But the gate has to be far better than ours.** With a simulated forecast, gated ORB needs a correlation of about
  0.3 with true efficiency (tier spread ~0.17) before it reaches a Sharpe near 1. Ours is 0.045 (spread 0.027):
  about seven times too weak. The futures model transferred to BTC shows no spread at all (0.410 / 0.427 / 0.425).
* **Same-day information doesn't fix it either.** A gate decided after the first 2 or 4 hours of the day (how straight,
  how wide, how heavy the volume so far) has correlation -0.001 and -0.05 with the rest of the day's efficiency.
  (`research/intraday_gate.py`)
* **The volatility forecast works but adds nothing to sizing.** In the search, 12-month momentum sized by the model's
  volatility forecast scored 0.54 vs 0.60 sized by plain trailing 20-day volatility, which already captures it.

Attempts so far on the efficiency idea: 7-market hourly model, pooled daily model, 3 feature upgrades, 3 longer
horizons, the 2-D gate search (185 strategies), the transferred BTC gate, and 2 same-day gates. None worked.
