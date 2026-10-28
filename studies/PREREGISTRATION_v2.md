# PREREGISTRATION v2: Is vol-scaled mom(1,4) truly profitable?

**Written:** 2026-09-23 19:19 America/New_York, *before* any v2 data was downloaded or any v2 backtest was run on real data.
**Author:** David DeConti (with Claude)
**Supersedes:** nothing. `PREREGISTRATION.md` (v1) and its results stand as written.

---

## 0. What is frozen

The strategy is **not** modified. Every parameter stays exactly as in `research/signals.py`:
`N_LAGS=4, SIGNAL_CAP=2.0, TARGET_VOL=0.10, HL_FAST=6, HL_SLOW=24, VOL_FLOOR_MULT=0.5, ANN=12`,
plus `CORR_WINDOW=24` and the gearing cap of 5.0 from `backtest.py`.
No universe pruning, no parameter changes, and no date trimming after results are seen.

## 1. Bug #3, found before any v2 result (code review + synthetic test)

In `backtest.py`, `vol = realized_vol(rets)` at month *t* is an EWM std that **includes the month-t return**.
That same `vol[t]` divides the signal and the position that earns `rets[t]`. The position is therefore sized
using the return it is about to earn. This is look-ahead.

It was hidden by the old `held = geared.shift(1)`. That shift fixed the vol timing but double-lagged the signal (bug #1).
The bug #1 fix removed the shift, which exposed this problem.

**Synthetic evidence** (`v2/leakcheck.py`: 60 GARCH random walks, 10 assets, 240 months, zero predictability by construction):

| code | mean Sharpe | realized vol |
|---|---|---|
| v1 (`vol[t]`, leaks) | **+0.211** | 0.081 |
| causal (`vol[t-1]`) | −0.069 | 0.104 |

On data with no edge at all, the v1 code manufactures a Sharpe of about +0.2 (≈7 standard errors) and undershoots its vol target.
This probably also explains a large part of why the v1 shuffle-test null mean was 0.339 rather than 0.

**Preregistered fix (the only code change):** `vol = realized_vol(rets).shift(1)`. The signal still uses
returns t−1…t−4, and positions still earn month-t returns. The causal code is the **primary** specification.
The v1 code is reported alongside for continuity, but the verdict does not depend on it.

## 2. Validation harness: must pass before any real-data run

| id | test | pass condition |
|---|---|---|
| H1 | Planted trend: Markov-switching drift (±0.5σ/month, stay prob 0.95), 100 sims | causal mean Sharpe > 0.5 |
| H2 | Random walk (GARCH-t), 100 sims, zero cost | causal mean Sharpe within ±2 SE of 0 |
| H3a | Canary: v1 code on H2 data | mean Sharpe > 0 by ≥ 3 SE (detects bug #3) |
| H3b | Canary: explicit leak (signal uses r_t) | mean Sharpe > 1.0 |
| H4 | Stats: bootstrap CI covers the true Sharpe of iid data in 90–98% of 200 trials; DSR reproduces the Bailey & López de Prado (2014) worked example to 2 decimals | both hold |

If any H test fails, the problem is fixed and the whole harness is rerun before any real data is touched.

## 3. New data (free, long history)

**U5: Industries.** Ken French 49 Industry Portfolios, value-weighted, monthly, 1926-07 → latest.
Returns are in excess of the risk-free rate (the RF column of the FF 3-factor file). Values of −99.99 are set to missing.

**U6: Macro multi-asset.** Monthly, as far back as each series goes:
- Equity: FF `Mkt-RF`.
- Bond: US 10-year return constructed from FRED `GS10`: `r_t = y_{t−1}/12 − 7.5·(y_t − y_{t−1})`, minus RF.
- FX (long foreign vs USD, spot only): `EXJPUS`, `EXUSUK`, `EXCAUS`, `EXSZUS`, `EXUSAL`, and `EXGEUS` through 1998-12 spliced with `EXUSEU` from 1999-01.
- Commodities (spot % change): `WTISPLC` plus the IMF series `PCOPPUSDM`, `PALUMUSDM`, `PNICKUSDM`, `PZINCUSDM`, `PWHEAMTUSDM`, `PMAIZMTUSDM`, `PSOYBUSDM`, `PCOFFOTMUSDM`, `PSUGAISAUSDM`, `PCOTTINDUSDM`, `PNGASUSUSDM`.
- A series that fails to download is dropped and listed. **No substitutes are chosen after seeing results.**
- The backtest starts at the first month in which ≥ 5 assets have ≥ 24 months of history.

**Known limitations, stated in advance:** FX ignores interest differentials (carry), and commodities use spot prices, so there is no roll yield. U6 therefore approximates, but is not identical to, a futures book.

**Benchmark (diagnostic, not a gate):** AQR "Time Series Momentum: Factors, Monthly". Expect correlation between U6 and the AQR TSMOM factor above 0.3 over the overlapping months. If it comes in below 0.3, that is flagged as a possible implementation or universe mismatch *before* interpreting anything else.

## 4. Costs (charged on turnover)

Equity 5 bps · bonds 3 bps · FX 3 bps · commodities 10 bps · industries 10 bps. **All costs are doubled before 1990.**
A 0 / 25 / 50 bps flat sweep is also reported for every universe. This closes the outstanding futures cost-sweep item from v1.

## 5. Statistics

- Annualized Sharpe of monthly net returns.
- **Bootstrap 95% CI:** stationary block bootstrap (Politis–Romano), mean block length 12 months, 10,000 resamples, seed 42, percentile interval.
- **Deflated Sharpe Ratio** (Bailey & López de Prado 2014): number of trials **N = 6** (crypto, ETF-20, ETF-39, futures, U5, U6). The variance of trial Sharpes is taken from the six causal-code Sharpes, and DSR includes each series' skew and kurtosis.
- **Newey–West t-stat** on the mean return, 12 lags. Reported alongside the Harvey–Liu–Zhu bar of t > 3.
- Reported, not gated: subperiods pre-1990 / 1990–2008 / 2009–2015 / **2016+ (post-publication)**, max drawdown, skew.

## 6. Pass/fail (per universe, causal code, net of section-4 costs)

- **G1:** lower bound of the bootstrap 95% CI on Sharpe > 0
- **G2:** DSR > 0.95

**Verdict rule:**
- G1 and G2 pass in **both** U5 and U6 → *"profitable after preregistered tests."*
- They pass in exactly one → *"mixed: evidence only in X."*
- They pass in neither → *"no demonstrable edge."*

The old universes (ETF-20, ETF-39, futures) are re-run with the causal code and reported. They count toward N but are not part of the verdict.

All results are reported whatever they show.

---

## Amendment A1 (2026-09-23 ~19:40 ET, **before any v2 data was downloaded**)

**Problem found in section 3:** FRED's monthly `EX*`, `GS10`, `WTISPLC` and all IMF `P*USDM` series are
**monthly averages**, not month-end prices. Averaging mechanically creates positive lag-1 autocorrelation
in returns (Working, 1960). That is exactly what a 1–4 month momentum signal feeds on.

**Synthetic evidence** (`v2/avgcheck.py`: 60 daily random walks, 10 assets, 240 months, zero edge, zero cost):

| monthly price used | mean Sharpe |
|---|---|
| end-of-month | +0.03 |
| monthly **average** | **+1.07** |

Averaging alone would manufacture a Sharpe above 1. It is added to the harness as **H5**.

**Changes (made before any data was seen):**
- U6 uses **end-of-month values of daily FRED series** instead:
  `DGS10` (bond, 1962+), `DEXJPUS`, `DEXUSUK`, `DEXCAUS`, `DEXSZUS`, `DEXUSAL`, `DEXGEUS`→`DEXUSEU` (FX, 1971+),
  and `DCOILWTICO` (WTI, 1986+). Bond construction, FX direction and costs are unchanged from section 3.
- The IMF commodity series exist only as monthly averages. They are **removed from U6** and run separately as **U6-avg**,
  a labelled diagnostic that shows the averaging artifact on real data. **U6-avg is excluded from the verdict and from N.**
- Consequence: U6's commodity sleeve is WTI only. This is stated as a limitation.

Gates, N = 6, costs and the verdict rule are unchanged.

---

## Addendum B: secondary analyses (2026-09-23 ~20:10 ET, written after the main verdict and before any of these were run)

**The main verdict is locked:** *NO DEMONSTRABLE EDGE.* Nothing below can change it. These analyses explain the result:
they ask *where* an effect exists, *what* it is, and *why* the headline test could not confirm it.
All use the causal code with section-4 costs, on U5 and U6 only.

| id | question | method | reading |
|---|---|---|---|
| S1 | Does return **ordering** matter? | Joint-row permutation (all assets move together, keeping cross-asset correlation). Rows are shuffled **only within blocks with the same set of available assets**, so missing-data structure is kept. 1000 permutations, seed 42. p = share of null Sharpes ≥ real. | p < 0.05 → the time-series structure carries information |
| S2 | Is it just hidden **beta**? | OLS with Newey–West (12) SEs. U5 on FF3 (`Mkt-RF`, `SMB`, `HML`); U6 on `Mkt-RF` + the U6 US 10y excess return. | alpha t > 2 → not explained by those exposures |
| S3 | Is there **crisis alpha** (the 2016 paper's claim)? | Mean strategy return in the worst 10% of `Mkt-RF` months vs all other months; Welch t on the difference. Also reported: the correlation with `Mkt-RF` in those months. | positive difference, t > 2 → crisis alpha supported |
| S4 | Is the **post-2016 decline** real? | r_t = a + b·1[t ≥ 2016-01] + e, with Newey–West (12) t on b. | b < 0 with t < −2 → decay is statistically distinguishable from noise |
| S5 | Could this sample even **detect** the effect? | Years needed for t = 2 and t = 3 at the observed Sharpe: (t / SR)²; the chance G2 would pass if the observed Sharpe were the true one (simulated, 2000 draws). | context only |
