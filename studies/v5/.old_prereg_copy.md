# PREREGISTRATION v5: front-running the commodity index roll (T9, "Goldman roll")

**Written:** 2026-09-23 ~22:10 America/New_York, **before any calendar-spread return was computed on real data**.
**Stopping rule:** none (David's choice, recorded in v4). This test counts as trial **13**.

## 1. Mechanism

Commodity index funds tracking the S&P GSCI must roll their futures on a **published schedule**: the 5th–9th business day of each month, about 20% of the position per day. They sell the contract they hold and buy the next one. This pushes the nearby contract **down relative to the next contract** in a predictable window (Mou, 2010, "Limits to arbitrage and commodity index investment: front-running the Goldman roll"). The Bloomberg Commodity Index rolls on business days 6–10, overlapping the same window.

**Predicted trade:** long the next contract and short the nearby one, before and through the roll window.

## 2. Scope (fixed)

- **Markets:** CL (WTI), HO (heating oil), RB (RBOB gasoline), NG (natural gas). These roll **every month** under a simple rule: in calendar month M, the index holds the contract delivering in **M+1** and rolls into **M+2**. Energy is also the largest GSCI sector. Grains, metals and livestock are **excluded**: their index roll months differ by commodity, and I would have to guess the exact schedule.
- **Data:** per-contract daily closes from Databento (`state/v3/contracts`), keyed by `symbol + instrument_id` (v3-A1). The delivery month is read from the contract code (v3-A2).

## 3. Trade (fixed; no tuning)

- **Business days (BD)** = the trading dates in the CL data. BD k = the k-th trading date of the calendar month.
- **Entry:** at the close of the last trading day of month M−1. **Exit:** at the close of BD 9 of month M. The trade earns daily returns on BD 1…9.
- **Legs** for month M: N = the contract delivering in M+1 (the nearby) and D = the contract delivering in M+2 (the next).
  - Daily spread return = r_D − r_N, each computed close-to-close **on the same contract**.
  - A leg missing a close on a day gets zero return that day.
- **Sizing:** each market's spread is scaled to 10% annualized vol. The vol comes from the standard deviation of that market's daily (D − N) spread returns over all in-window days in the prior 12 months, lagged, with a minimum of 60 observations. Leverage is capped at 5. T9 is the equal-weight average across the markets that are live that month.
- **Costs:** 5 bps per leg per trade (energy rate from v3), so each month pays 2 legs × 2 trades × 5 bps × notional. This is conservative, since calendar spreads can trade as one instrument.

## 4. Mechanism check (reported, not gated)

**Placebo:** the identical trade on **BD 12–20**, when no index is rolling. The mechanism predicts roll-window Sharpe > placebo Sharpe. A roll window that beats zero but not the placebo points to a generic spread drift, not the roll.

## 5. Harness (must pass before any real data)

- **H13:** synthetic contracts with a planted roll effect: the nearby loses 15 bps/day against the next on BD 5–9, plus noise. T9 mean Sharpe > 1.
- **H14:** the same without the planted effect. Mean Sharpe within ±2 SE of 0.
- **H15:** a look-ahead canary. The spread direction is chosen after seeing that month's realized window return, on H14 data. Mean Sharpe > 1.

## 6. Statistics and gates

Monthly net returns (P&L from each month's window).

- **G1:** the lower bound of the bootstrap 95% CI on Sharpe > 0.
- **G2:** DSR > 0.95 with **N = 13**.
- **G3:** Sharpe > 0 on 2018-01 → end.

**Verdict:** an edge exists only if all three pass. **Reported:** per-market Sharpe, gross vs net, 2× cost, placebo, 2010–2017 vs 2018+, and the average P&L contribution of BD 1–4 (pre-roll) vs BD 5–9 (roll).

---

## Amendment v5-A1 (2026-09-23 ~22:35 ET), made **AFTER run 1 was seen**; disclosed as such

**Run 1** (as preregistered; output kept in `v5/results_v5_run1_calendar_bug.txt`): net Sharpe −1.76, gross −0.08, placebo −0.21. NO EDGE.

**Bug found while checking run 1:** Databento daily bars are grouped by UTC calendar day. The Sunday-evening session opening (about 0.3% of volume) and exchange holidays (1–2% of normal volume) appear as separate dated bars. Counting these as "trading dates" inflated every business-day count. Run 1's "BD 1–9" was really about BD 1–7, which **missed most of the actual roll window**. The harness did not catch this because the synthetic data used a clean weekday calendar.

**Fix (correctness, not tuning):** a business day is a Monday–Friday date on which CL total volume is at least 20% of that month's median daily volume. This proxies the exchange's published holiday calendar, which is known in advance. Sunday and holiday stubs are still used for price continuity: returns are close-to-close across them, but they are not counted as business days.

**Integrity note:** the fix changes only which dates are counted. Markets, legs, window (BD 1–9), placebo (BD 12–20), sizing, costs, gates and N = 13 are unchanged. Because the fix came after a result was seen, **both runs are reported**. The verdict comes from run 2, and the reader should weigh it knowing that a post-result correction was made. Run 1 was already a clear failure, so the fix was not made to rescue a result.
