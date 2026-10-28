# PREREGISTRATION v4: mean reversion in economically linked spreads (T8)

**Written:** 2026-09-23 ~22:00 America/New_York, **before any spread was computed on real data**.
**Stopping rule:** none. David chose not to set one. This is recorded so that any later test is read in that light, and **every test continues to count toward N**.

## 1. Hypothesis

Prices tied together by physical or economic arbitrage should mean-revert relative to each other. This is a different return source from the trend, momentum and carry strategies in v1–v3. The spreads are chosen from economics, **not** by searching the data:

| spread | y (dependent) | x (hedges) | economic link |
|---|---|---|---|
| S1 soybean crush | ZS | ZM, ZL | beans are crushed into meal + oil |
| S2 crack | CL | HO, RB | crude is refined into heating oil + gasoline |
| S3 Treasury curve | ZF | ZN | adjacent points on the same yield curve |
| S4 crude grades | CL | BZ | WTI vs Brent, linked by arbitrage |
| S5 precious metals | GC | SI | shared monetary/industrial drivers |

## 2. Data

Clean roll-gap-free **daily** returns (`state/v3/clean_daily.csv`, v3 build with amendments A1–A2). Each market's log price index is L = cumulative sum of log(1 + r). Missing days count as zero return.

## 3. Model (fixed; no tuning)

1. **Kalman hedge ratio** (Chan 2013 random-walk coefficients): state θ = [α, β₁…βₖ], observation L_y = [1, L_x]·θ + e.
   - δ = 1e-4, Q = δ/(1 − δ)·I.
   - R = OLS residual variance from the first 252 days (burn-in), and θ₀ = the OLS fit on those days.
   - The residual at day t uses the **prior** state θ_{t|t−1}: s_t = L_y,t − [1, L_x,t]·θ_{t|t−1}.
2. **OU fit** on the rolling 252 days of s through day t: AR(1) s_t = a + b·s_{t−1} + ε.
   - Half-life HL = −ln 2 / ln b; mean μ = a/(1 − b); equilibrium sd σ = sd(ε)/√(1 − b²).
   - z = (s_t − μ)/σ.
3. **Rules**, evaluated at the close of day t:
   - The spread is tradable only if 0 < b < 1 and **1 ≤ HL ≤ 63** trading days.
   - Enter **short** the spread if z ≥ 2, **long** if z ≤ −2.
   - Exit when z crosses 0, after 3·HL days in the trade, or when the spread stops being tradable.
4. **Positions:** a long spread holds +1 of y and −β of each x, in return-index notional. Positions set at the close of t earn returns on day t+1. No same-day trading.
5. **Sizing:** each spread's daily P&L is scaled to 10% annual vol using its trailing 60-day realized vol (lagged one day). T8 is the equal-weight average of the five scaled streams. It is then scaled to 10% vol with the trailing 60-day vol of the combination (lagged). Leverage is capped at 5.

## 4. Costs

The v3 per-class bps (rates 2, energy 5, metals 5, grains 8) apply to every change in each leg's notional. Roll costs: in each month, each leg pays 2 × bps × its average |notional| × that market's roll count (from `clean_rolls.csv`). A doubled-cost run is reported.

## 5. Harness (must pass before any real data)

- **H10:** a synthetic cointegrated pair (OU spread with HL 10 days, plus a common random walk). T8 mean Sharpe > 1.
- **H11:** independent random walks. T8 mean Sharpe within ±2 SE of 0.
- **H12:** a look-ahead canary. Positions set at t earn day-t returns (a leak). Mean Sharpe must jump well above the non-leaky run on the same H11 data (> 1.0).

## 6. Statistics and gates

Daily net returns are compounded to monthly, and all statistics are monthly, as in v2/v3.

- **G1:** the lower bound of the bootstrap 95% CI > 0.
- **G2:** DSR > 0.95 with **N = 12** (the 11 from v3 plus T8).
- **G3:** Sharpe > 0 on 2018-01 → end.

**Verdict:** an edge exists only if G1, G2 and G3 all pass. **Reported, not gated:** per-spread Sharpe, the gross (no-cost) Sharpe, the share of days in the market, the average holding period, correlation with T4/T5/T6, and 2010–2017 vs 2018+.

---

## Amendment v4-A1 (2026-09-23 ~22:15 ET, from harness failures, **before any real-data run**)

The first harness run **failed**. H10 never traded, and H12 was ill-posed. Changes:

1. **Kalman state noise was in the wrong units.** Chan's δ = 1e-4 assumes dollar price levels. In log-index units, Q = δ/(1 − δ)·I is about as large as the observation noise R, so the filter absorbed the spread: the true 10-day half-life was estimated at 0.5 days. **Fix:** Q = δ/(1 − δ)·**R**·I, the scale-free form. With this, the estimated β is 0.78 (true 0.80), HL 8 days (true 10), and the residual correlates 0.90 with the true spread.
2. **Sizing.** Scaling by trailing vol of a mostly-flat P&L stream was unstable: in-market Sharpe 1.46 became 0.87 monthly. **Fix:** each spread is sized by the ex-ante vol of the unit spread (y − β·x), trailing 60 days and lagged, targeting 10% and capped at 5×. This is the same principle as `signals.py`. T8 is the plain equal-weight average of the spreads, with no second vol-scaling layer. **Check:** before sizing, the Kalman strategy's Sharpe (1.46) matched an oracle that trades the true spread (1.44).
3. **H12 redefined.** "Earn same-day returns" is anti-profitable for mean reversion: you short *after* the spread has already risen, so the look-ahead loses money and cannot serve as a canary. The canary is now the classic pairs-trading leak: **hedge ratio and OU parameters fit on the full sample (hindsight)**. It must beat the honest H11 run by 3 SE.
4. Runs that never trade count as Sharpe 0, not NaN.

**Result:** H10 +1.23 (PASS), H11 +0.02 (PASS), H12 **+0.52** on *unrelated random walks* (PASS). Hindsight fitting alone manufactures about 0.5 Sharpe.

Spreads, rules, entry/exit thresholds, costs, gates and N = 12 are unchanged.
