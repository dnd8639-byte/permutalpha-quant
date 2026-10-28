# RESULTS v2: Is vol-scaled mom(1,4) truly profitable?

Run 2026-09-23 against `PREREGISTRATION_v2.md` (incl. Amendment A1). Full output: `v2/results_v2.txt`. Monthly returns: `v2/returns_v2.csv`.

## Verdict (preregistered rule)

**NO DEMONSTRABLE EDGE.** U5 (49 industries, 1928–2026) and U6 (macro, 1973–2026) both pass G1 (bootstrap CI above 0) and both fail G2 (Deflated Sharpe > 0.95).

| universe | months | Sharpe | 95% CI | NW t | DSR | G1 | G2 |
|---|---|---|---|---|---|---|---|
| U5 industries | 1177 | +0.275 | [+0.05, +0.50] | 2.53 | 0.70 | PASS | FAIL |
| U6 macro | 645 | +0.334 | [+0.06, +0.60] | 2.35 | 0.80 | PASS | FAIL |

To pass G2 after six trials, U5 would have needed a Sharpe of at least 0.40, and U6 at least 0.45. Neither reaches the t > 3 bar either.

**Interpretation:** there is a weak positive signal over long history. It is too small to survive correction for the six universes tried, and it has **disappeared since publication**:

| Sharpe by period | pre-1990 | 1990–2008 | 2009–2015 | 2016+ |
|---|---|---|---|---|
| U5 industries | +0.35 | +0.09 | +0.71 | **−0.14** |
| U6 macro | +0.67 | +0.29 | +0.46 | **−0.35** |
| AQR's own TSMOM factor (context) | +1.51 (1985–2008) | | +0.62 | **+0.14** |

The decay is not specific to this implementation. AQR's professional trend factor fell from a Sharpe of 1.5 to 0.14 over the same window.

## Effect of bug #3 (look-ahead in vol sizing) on earlier results

| universe | v1 code (leaky) | causal code |
|---|---|---|
| ETF-20 | 0.634 | **0.411** |
| ETF-39 | 0.464 | 0.213 |
| Futures | 0.339 | **0.127** |
| Crypto | 0.024 | −0.065 |

(The v1 figures differ slightly from the recorded 0.603 / 0.442 / 0.328 because v2 excludes the zero-return warm-up months. Your original `backtest.py` run on this exact data reproduces 0.603 / 0.442 / 0.328 / 0.02 exactly.)

## Effect of monthly-average data (Amendment A1)

U6-avg, built from FRED/IMF monthly averages with the same code, shows a Sharpe of **+0.90** (t = 5.8, DSR 1.00). It would have "passed" everything. That is the averaging artifact predicted before the data was downloaded (+1.07 on pure noise). It is excluded from the verdict by design.

## Checks performed

- Harness H1–H5 all pass (synthetic trend detected; noise ≈ 0; both look-ahead canaries fire; bootstrap coverage 94%; DSR reproduces Bailey & López de Prado 0.9004).
- The original code on the exported data reproduces all recorded v1 Sharpes exactly.
- Sharpe, DSR and SR0 were recomputed independently from `returns_v2.csv` and match.
- Benchmark: U6 correlates **+0.42** with AQR TSMOM (bar: > 0.3). The implementation tracks real trend-following.
- Cost sweep (0 / 25 / 50 bps): U5 +0.41 / +0.19 / −0.03 and futures +0.32 / −0.16 / −0.63. This closes the outstanding v1 futures cost-sweep item.

## Deviations and limitations (stated, not hidden)

- `DEXGEUS` (daily DEM) failed to download. Under the preregistration it was dropped, not substituted, so EUR starts in 1999.
- U6 FX excludes interest carry, and commodities are WTI spot only (no roll yield).
- The very large 1930s industry returns (e.g. Paper +300%, Aug 1932) are genuine Ken French values and were left in.

---

## Secondary analyses (Addendum B). They explain the verdict but do not change it

Full output: `v2/secondary_results.txt`. Code: `v2/secondary.py`.

| test | U5 industries | U6 macro | reading |
|---|---|---|---|
| **S1** permutation (1000 shuffles, within availability blocks) | p = **0.009** (null mean 0.04) | p = **0.006** (null mean 0.00) | The time ordering carries real information in both |
| **S2** factor alpha (Newey–West) | +3.5%/yr, t **2.76**; R² 0.01 | +4.6%/yr, t **2.59**; R² 0.02 | Not hidden equity/size/value/bond beta |
| **S3** crisis alpha (worst 10% equity months) | +0.52%/mo vs +0.23% otherwise, t 0.54 | +0.97%/mo vs +0.25% otherwise, t 1.22 | Positive in crashes, **not statistically significant** |
| **S4** post-2016 break | −5.2%/yr, t −1.49 | −9.4%/yr, t **−2.68** | The decline in macro is statistically real |
| **S5** power | 53 yrs needed for t=2 (have 98) | 36 yrs needed for t=2 (have 54) | Even if real, G2 would pass only **14% / 21%** of the time |

**What this means together:**
1. There is good statistical evidence of a **small, genuine momentum effect in the historical record**. Shuffling destroys it (S1), and market exposures don't explain it (S2).
2. It is too small to survive the strict multiple-testing bar (the main G2 failure). The power analysis (S5) shows that even a real effect of this size would usually fail that bar, so G2's failure alone does not mean the effect is fake.
3. **It has not been present since 2016.** In U6 the drop is statistically significant (S4). The crash protection that motivated the 2016 paper points the right way but never reaches significance (S3).
4. **For a decision about trading it today:** the data supports "there was a small edge historically", not "there is an edge now".

**Caveat on S1:** p-values of 0.009 and 0.006 do not account for the six universes tried. Under a strict Bonferroni cutoff for 6 tests (0.0083), U6 clears it and U5 narrowly misses.
