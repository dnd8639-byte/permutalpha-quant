# PREREGISTRATION v3: clean futures data + new strategies

**Written:** 2026-09-23 ~20:50 America/New_York, **before the Databento data was downloaded**. No v3 strategy has been run on real data.
**Builds on:** v1 and v2. Their results stand as written. The v2 verdict for mom(1,4) is unchanged.

---

## 1. Data fix: roll-gap-free futures ("CF")

- **Source:** Databento `GLBX.MDP3` daily bars for every individual contract, from mid-2010 to latest. Markets are the v1 futures list minus the ICE softs (KC, SB, CC, CT): ES NQ YM RTY NKD · ZT ZF ZN ZB UB · 6E 6J 6B 6A 6C 6S 6M · CL BZ NG HO RB · GC SI HG PL PA · ZC ZW ZS ZM ZL ZO · LE HE.
- **Construction** (`v3/build_clean.py`): on day t, hold the contract with the most volume on day t−1, excluding contracts within 5 trading days of their final bar. The return is always close/close **of the same contract**, so roll jumps never enter a return.
- **Validated before any real data** (`v3/test_build.py`, H6): on synthetic contracts with 30%/yr contango, the naive splice shows a **+29.3%/yr fake return** and the clean method shows **0.00%** (max daily error 4×10⁻¹⁶).
- A market that fails to download, or has fewer than 36 months of data, is dropped and listed. There are no substitutes.

## 2. Tests

All tests use the causal risk engine from v2: the vol estimate is lagged by one month, and every strategy is geared with `portfolio_gearing` (24-month correlation, 10% target, cap 5). **All parameters are the published defaults. There is no tuning. Nothing is chosen after seeing results.**

| id | strategy | source | signal (known at end of month t−1, held during month t) |
|---|---|---|---|
| **T0** | mom(1,4), unchanged | v1/v2 | as in `signals.py`. This measures the effect of the data fix alone |
| **T4** | Slow time-series momentum | Moskowitz, Ooi & Pedersen (2012) | sign of the 12-month return (months t−12…t−1), position = sign / vol |
| **T5** | Cross-sectional momentum | Asness, Moskowitz & Pedersen (2013) | rank markets by risk-adjusted 12−1 return (t−12…t−2, divided by vol); long the top third, short the bottom third, each at 1/vol |
| **T6** | Cross-sectional carry | Koijen, Moskowitz, Pedersen & Vrugt (2018) | carry = (P₁ − P₂)/P₂ annualized by the days between the two nearest eligible contracts' final bars, measured on the last trading day of month t−1; long the top third, short the bottom third, each at 1/vol |
| **T7** | Equal-risk blend of T4 + T5 + T6 | — | average of the three return streams, each first scaled to 10% vol using trailing 24-month realized vol (lagged) |

Cross-sectional strategies need at least 6 live markets in a month; otherwise they stay flat.

**Costs** (bps of notional traded): equity index 2 · rates 2 · FX 3 · energy 5 · metals 5 · grains 8 · livestock 10. They are charged on position changes **and** on every roll (the old and new contract are each traded once, so 2× cost on the position held). A sensitivity run with costs doubled is reported.

## 3. Validation harness (must pass before any real-data run)

- **H6:** the roll-gap test above. Already PASSED.
- **H7:** on a GARCH random walk, T4 and T5 show a mean Sharpe within ±2 SE of 0.
- **H8:** a look-ahead canary. T5 given month-t returns in its signal shows a mean Sharpe > 1.
- **H9:** a planted carry premium, where synthetic markets with higher carry earn higher returns. T6 shows a mean Sharpe > 0.5. On a zero-premium control, T6 is within ±2 SE of 0.

## 4. Statistics and gates (per strategy, on CF, net of costs)

- **G1:** the lower bound of the bootstrap 95% CI on Sharpe > 0 (stationary block bootstrap, block 12, 10,000 draws, seed 42).
- **G2:** Deflated Sharpe > 0.95 with **N = 11 trials**: the 6 from v2, plus T0, T4, T5, T6 and T7.
- **G3:** Sharpe > 0 on **2018-01 → end**. This guards against an edge that has already disappeared, as mom(1,4) did.

**Verdict:** a strategy has an edge **only if G1, G2 and G3 all pass**. The overall verdict is "edge found" if at least one of T4–T7 passes. All results are reported, whatever they show.

**Reported, not gated:** Sharpe for 2010–2017 and 2018+, Newey–West t, max drawdown, skew, correlation with AQR TSMOM, and T4 on the v2 U6 macro data (1973+) as a long-history cross-check.
**Data-fix report (T0):** per market, the correlation between Yahoo and clean monthly returns and the annualized mean difference (roll contamination), plus the T0 Sharpe on Yahoo vs CF over the same markets and months.

---

## Amendment v3-A1 (2026-09-23 ~21:50 ET, after download, **before any strategy was run on real data**)

**Data-integrity bug found in `build_clean.py`:** CME symbols reuse a one-digit year. For example, `ESZ0` is both the December 2010 and the December 2020 contract. Keying contracts by symbol merged different contracts, which showed up as ES having only 40 "contracts" instead of about 65. Databento's `instrument_id` is unique per contract: ES has 70 IDs, and no ID maps to two symbols.

**Fix:** the contract key is now `symbol + instrument_id`. Nothing else changes: not the tests, parameters, costs, gates or N. H6 was rerun and passes.

## Amendment v3-A2 (2026-09-23 ~22:00 ET, **before any strategy was run on real data**)

**Second data bug, found by checking carry values:** contracts still trading at the end of the data all have their "final bar" on the download date (2026-09-22). Using final bars as expiry dates collapsed the gap between contracts to a few days, which inflated annualized carry to impossible values (e.g. ES −149%/yr, ZN +172%/yr, all in 2026-06..08).

**Fix:** contracts are ordered and spaced by their **delivery month read from the contract code** (e.g. `ESZ6` → Dec 2026, set to the 15th). This is known in advance, so it involves no look-ahead. One-digit years are resolved to the first matching year on or after the contract's first bar.

**Result:** financial carry now lies in plausible ranges (ES 1st–99th percentile −4.7%…+2.5%; ZN −2.7%…+5.1%). The remaining extremes are real episodes: CL super-contango in Mar–Apr 2020, seasonal spreads in HE/NG/RB, and the ZM squeeze in Aug 2014. They are **left as they are**, since T6 uses ranks, not magnitudes.

**Also noted, not changed:** BZ (CME Brent) is thinly traded, and volume leadership flips often (759 rolls). The preregistered roll rule and roll costs apply as written, which makes BZ more expensive for every strategy.

Tests, parameters, costs, gates and N are unchanged. H6 was rerun and passes.
