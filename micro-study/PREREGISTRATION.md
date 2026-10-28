# Pre-registration: does the order book predict how efficiently the next hour trades?

**Author:** David DeConti · **Written:** 2026-10-13, before any real order-book feature had been computed
**Lock date:** 2026-10-20 00:00 UTC. At the lock date `python study.py freeze` records SHA-256 fingerprints of this file and `study.py`, and both are pushed to GitHub. The public commit time is the proof of when these rules were fixed.

## 1. Why this test

The edge-engine study found that **tomorrow's efficiency cannot be predicted from price history**. Efficiency here is the net move divided by the distance travelled. On 100,000 market-days of 35 futures the forecast correlation was 0.045. Three horizons, three feature upgrades and two same-day gates also failed (`../edge-engine/futures/FINDINGS.md`). A simulation showed a strategy paid by efficiency would need a forecast correlation near 0.3.

Every failed forecast used the same information: past prices. This test uses information that past prices do not contain: the **order book and the order flow**, recorded every 10 seconds by the Raspberry Pi since 2026-10-03.

## 2. Hypothesis and mechanism

**H1.** Order-book and order-flow features measured during hour *h* predict the efficiency of hour *h+1* (rank correlation > 0).

**Mechanism.**
1. **Order flow persists.** Large traders split big orders into many small ones over hours (Kyle 1985). The sign of market order flow has long memory (Lillo & Farmer 2004). When one hour's flow is strongly one-sided, the same pressure is likely to continue. One-sided pressure moves the price in a straight line.
2. **Thin books amplify persistent flow.** Each unit of flow moves the price further when there is little resting depth.
3. **Balanced flow meeting thick books gives chop.** The price wanders back and forth: low efficiency.

**Expected signs:**

| Feature | Expected sign |
|---|---|
| `flow_agree` | + |
| `book_agree` | + |
| `abs_tfi` | + |
| `abs_imb2` | + |
| `log_depth2` | − |

## 3. Data

- **Source:** the Pi recorder.
  - Venues: Binance.US BTC/USDT, Kraken BTC/USD and Coinbase BTC/USD.
  - Each snapshot holds the top 50 price levels on each side plus every trade, taken every 10 seconds.
  - The data is public; no account is used.
- **Consensus price:** the median of the venue mid-prices at each snapshot.
- **Usable hours:** an hour is used only if at least 300 of its 360 snapshots exist. A row is used only if the next clock hour also exists. **There are no other exclusions.**
- **Pilot data:** every hour before the lock date. It is used for building, debugging and exploring, and anything may change before the lock.
- **Confirmation data:** every hour from the lock date on. The test opens once **1,000** confirmation hours have a next hour. At about 24 hours a day that is around late November 2026.
- **Feature extraction:** `extract.py` on the Pi. Every feature is measured inside hour *h*, so it is known at the end of hour *h*.

## 4. Target

ER(*h+1*) = |close − open| / (high − low) of the consensus mid over hour *h+1*. This is the same definition as the futures study, which validated it against path efficiency (rank correlation 0.95).

## 5. Features (frozen at the lock date)

Primary model, 8 features:

| name | definition (hour *h*) |
|---|---|
| `flow_agree` | sign(price change over the hour) × trade-flow imbalance, where imbalance = (buy − sell)/(buy + sell) BTC across venues |
| `book_agree` | sign(price change) × depth imbalance within 2 bps of mid, averaged over the last 5 minutes |
| `abs_tfi` | absolute value of the hour's trade-flow imbalance |
| `abs_imb2` | absolute value of the depth imbalance within 2 bps, last 5 minutes |
| `log_depth2` | log of USD depth within 2 bps of mid, summed across venues, hour average |
| `spread_bps` | average quoted spread in bps |
| `er` | this hour's efficiency |
| `log_range` | log of this hour's high–low range in bps |

Secondary set: the primary 8 plus `imb10_agree`, `log_depth10`, `imb25_agree`, `log_depth25` (these four exist only from 2026-10-14, when the recorder went to 500 levels; the tree model treats earlier hours as missing), `micro_agree`, `abs_micro`, `flow_agree_15m`, `log_dep2_change`, `venue_disp_bps`, `er_15m`, `log_rv`, `log_trades_rel`, `log_volume_rel`, `imb1_agree`, `hour_sin` and `hour_cos`. The exact formulas are in `study.py`, `build()`.

## 6. Model

The primary model is ridge regression on standardized features with α = 10.
- It is refit every UTC day on all earlier hours, with the training window growing over time.
- A training row's target is complete before the first prediction it feeds.
- Hyperparameters are not tuned on confirmation data.

## 7. Primary test and decision rule

- **Statistic:** the Spearman rank correlation ρ between predicted and realized next-hour efficiency, over all confirmation hours.
- **Null:** a day-shuffle permutation with 10,000 draws. Each day's realized efficiencies are paired, hour by hour, with the predictions of a different random day. This keeps time-of-day patterns and destroys only the link being tested. p = (1 + #null ≥ real) / (N + 1).
- **Decision:** the result is a **PASS if and only if ρ ≥ 0.10 and p < 0.01**. Anything else is a FAIL. The result is reported either way.

**Power.** With n = 1,000 hours, the null standard deviation of ρ is about 0.032, so p < 0.01 needs ρ of roughly 0.075 or more. The chance of a PASS depends on the true correlation:

| True ρ | Chance of a PASS |
|---|---|
| 0.10 | ≈ 50% |
| 0.15 | ≈ 94% |
| 0.20 | > 99% |

A forecast useful for trading (ρ near 0.3, section 1) would be detected almost surely.

## 8. Secondary results (reported, not decisive)

- **All-features model:** boosted trees on the secondary set; its ρ.
- **Price-only model:** ridge on `er`, `log_range`, `log_rv` and `er_15m`; its ρ. The order book only "adds information" if the primary model beats this.
- **Tier spread:** realized next-hour efficiency in the predicted low, mid and high tiers. Tier cut-offs come from training predictions only.
- **Economic check:**
  - **Rule:** follow the direction of the hour's order flow for the next hour, either every hour or only in predicted-efficient hours.
  - **Costs:** 5 bps per side.
  - **Reported:** Sharpe ratio and bps per hour.
  - **Threshold:** the forecast correlation needed for a Sharpe of 1, taken from the pilot simulation.

## 9. Rules

1. Before the lock date, anything may change. Changes go in the change log below.
2. After the freeze, only `python study.py status` touches confirmation data until the test opens. It reports counts and gaps, never results. `pilot` mode cannot read confirmation hours, not even as targets.
3. `confirm` is **one-shot**. It writes its lock file before computing anything, and it refuses to run if this file or `study.py` changed since the freeze, or if the lock date differs.
4. **If it passes:** the forecast becomes an hourly gate in the edge-engine and in the Pi paper trader. There it faces its own out-of-sample test. **No real money.**
5. **If it fails:** the failure is written up. The features are not changed and re-tested on the same confirmation data.

## 10. Context for multiple testing

This is the first test of order-book information. It follows 11 failed attempts on price-only efficiency forecasts (see FINDINGS.md). Those are recorded so a reader can judge this result against everything tried.

## Change log (pilot period)

- 2026-10-13: first version.
- 2026-10-13 (evening): first real pilot data (240 hours, no gaps, all 3 venues every hour). Depth within 25 bps was
  barely larger than depth within 5 bps, which suggests the 50 recorded levels do not reach 25 bps. `extract.py` now
  records how far each snapshot's levels reach (`reach_bps`) and reports a depth band as missing, not as a smaller
  number, when any venue's recording does not cover it. `depthcheck.py` measures the reach on the Pi.
- 2026-10-13 (night): `depthcheck.py` on the Pi showed 50 levels reach a median 3.3-3.9 bps on Coinbase and Kraken
  (41-51 bps on Binance.US). So the primary book features now use the **2-bps band**, which the recording has covered
  since day 1 (measured consistently across pilot and confirmation). The recorder goes to 500 levels; the 10- and
  25-bps bands join the secondary set only. Also: an index on trade time so hourly extraction stays fast.
- 2026-10-14: the recorder update accidentally dropped an earlier Coinbase fix, so Coinbase trades (not its order book)
  were missing from about 00:10 UTC until the fix was restored. Trade-flow features for those hours use Binance.US and
  Kraken only. Pilot period, so no effect on the confirmation test.
- 2026-10-14: the recorder now also records ETH order books (labels `coinbase-eth`, `kraken-eth`). `extract.py` reads
  only the three BTC markets (`binanceus`, `kraken`, `coinbase`), so this study's data is unchanged.
