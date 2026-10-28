# PREREGISTRATION v6: do famous published strategies lose their edge after publication? (T14, "publication decay")

**Written:** 2026-09-24, **before any anomaly return was downloaded or computed.** Only the data source's availability was checked (an HTTP HEAD request).
**Stopping rule:** none (David's choice, recorded in v4).
**Trial count:** 25 before this round (13 preregistered + 12 MCPT). The primary pooled test (H1) is trial **26**. The 9 per-strategy post-publication sign-flip tests (strategies 1–9) are also logged in `mcpt-lab/results/LEDGER.csv` and count as trials 27–35, although they are descriptive.

## 1. Question and mechanism

Two forces predict that a published strategy earns less after publication:

1. **Statistical bias (data mining).** Authors publish the effects that looked strongest in their own sample. Some of that strength is luck, so it shrinks on any new data, even before anyone reads the paper.
2. **Publication-informed trading (arbitrage).** Once the effect is public, investors trade on it and bid it away.

The design follows McLean & Pontiff (2016). Each strategy's history is split into three windows, which separate the two forces:

| window | dates | what a decline here means |
|---|---|---|
| **IS** | the authors' own sample period | (baseline) |
| **OOS** | after the sample ended, before publication | statistical bias only (nobody could trade on it yet) |
| **POST** | from the month after journal publication to the end of the data | statistical bias + publication-informed trading |

**This is a replication, not a discovery.** The literature already reports decay for several of these strategies, and I (and the AI assisting me) know some of those results. The contribution is: (a) running them through one transparent, reproducible pipeline, (b) fixed published rules with no tuning, and (c) extending the post-publication sample to 2026. **Prior expectation (stated in advance):** McLean & Pontiff found about −26% in OOS and about −58% in POST, relative to IS.

**Selection bias disclosure:** these strategies were chosen because they are famous. Fame is partly caused by strong in-sample results, which is exactly the statistical-bias channel, so this sample could overstate decay compared with a random draw of published anomalies.

## 2. Data (fixed)

**Source:** Kenneth R. French Data Library (Dartmouth), downloaded by `mcpt-lab/data/download_french.py`. It is free but not redistributed in the repo (git-ignored). The files:

- `F-F_Research_Data_Factors` (monthly and daily): Mkt-RF, SMB, HML, RF, from 1926-07.
- `F-F_Momentum_Factor` (monthly): Mom, from 1927-01.
- `F-F_ST_Reversal_Factor` (monthly): ST_Rev, from 1926-02.
- `F-F_Research_Data_5_Factors_2x3` (monthly): RMW, from 1963-07.

**Data end:** the last month common to all files at download time. Values are percent and are converted to decimals. −99.99 / −999 are treated as missing.

## 3. The strategies (fixed list; nothing added or removed after results)

All returns are **gross of trading costs** (as in McLean & Pontiff), except strategy 10. The published rule is used as is: **no parameters are tuned.**

| # | strategy | paper (journal, issue) | authors' sample → IS | publication → POST starts | monthly return series |
|---|---|---|---|---|---|
| 1 | Size | Banz (1981), *JFE* 9(1), Mar 1981 | 1936-01 – 1975-12 | 1981-03 → 1981-04 | SMB |
| 2 | Value | Fama & French (1992), *JF* 47(2), Jun 1992 | 1963-07 – 1990-12 | 1992-06 → 1992-07 | HML |
| 3 | Momentum | Jegadeesh & Titman (1993), *JF* 48(1), Mar 1993 | 1965-01 – 1989-12 | 1993-03 → 1993-04 | Mom (UMD) |
| 4 | Short-term reversal | Jegadeesh (1990), *JF* 45(3), Jul 1990 | 1934-01 – 1987-12 | 1990-07 → 1990-08 | ST_Rev |
| 5 | Profitability | Novy-Marx (2013), *JFE* 108(1), Apr 2013 | 1963-07 – 2010-12 | 2013-04 → 2013-05 | RMW (proxy, see note a) |
| 6 | Small-firm January effect | Keim (1983), *JFE* 12(1), Jun 1983 | 1963-01 – 1979-12 | 1983-06 → 1983-07 | SMB in January, 0 in other months |
| 7 | Weekend (Monday) effect | French (1980), *JFE* 8(1), Mar 1980 | 1953-01 – 1977-12 | 1980-03 → 1980-04 | see 3.1 |
| 8 | Turn-of-the-month | Lakonishok & Smidt (1988), *RFS* 1(4), 1988 | 1897 – 1986-12 → data start (1926-07) – 1986-12 | 1988-12 → 1989-01 (note b) | see 3.1 |
| 9 | Halloween / "Sell in May" | Bouman & Jacobsen (2002), *AER* 92(5), Dec 2002 | 1970-01 – 1998-08 | 2002-12 → 2003-01 | see 3.1 |
| 10 | 10-month moving-average timing | Faber (2007), *J. Wealth Mgmt* 9(4), Spring 2007 | 1973-01 – 2005-12 | 2007-03 → 2007-04 | see 3.2 (**not pooled**) |

**Note a:** Novy-Marx used *gross* profitability. French's RMW uses *operating* profitability (Fama & French 2015). RMW is a close relative, not an exact replication, and this is disclosed.
**Note b:** the RFS 1(4) issue month is uncertain. Allowed **before the run only:** David may correct any date in this table from the journal's own page, and each correction is recorded in section 8. No dates may change after results are seen.

### 3.1 Calendar strategies (built from daily Mkt-RF, excess of the T-bill)

Each is measured **against normal days in the same month**, so the stock-market premium does not count as part of the anomaly:

- **7, Monday:** for month m, d_m = −( Σ Monday excess returns − n_Monday × mean excess return of the other days in m ). Positive means Mondays were worse than other days, as French found.
- **8, Turn-of-the-month (TOM):** TOM days for month m = the last trading day of month m−1 plus the first 3 trading days of month m. d_m = Σ TOM excess returns − 4 × mean excess return of month m's non-TOM days (excluding m's own last trading day, which belongs to month m+1's TOM).
- **9, Halloween:** the monthly market excess return, **+1×** in Nov–Apr and **−1×** in May–Oct (long winter, short summer). The mean equals half the winter-minus-summer difference.

### 3.2 Faber timing (reported separately, not in the pooled test)

- **Total-return index:** TR from monthly Mkt = Mkt-RF + RF.
- **Signal:** at each month end t, hold the market in month t+1 if TR_t > the average of TR over months t−9…t (10 values). Otherwise hold T-bills (RF).
- **Costs:** 10 bps per switch.
- **Why separate:** Faber's claim is *similar return with lower risk*, not extra return, so its IS mean excess over buy-and-hold is not a meaningful scale for the pooled test.
- **Reported per window:** Sharpe (timed) − Sharpe (buy-and-hold), max drawdown of each, and the timed-minus-buy-and-hold mean.

## 4. Replication check (before pooling)

A strategy enters the pooled test only if our data reproduce it in-sample: **IS mean > 0 and IS t-stat ≥ 1.5.** Any that fail are reported as "did not replicate in-sample" and left out. This rule is fixed now.

## 5. Statistics

### Primary test (H1), trial 26

For each included strategy i, scale its monthly returns by its own IS mean: y_it = r_it / μ_IS,i. Months before the authors' sample start are dropped. Pool all strategies and regress:

  y_it = a + b_OOS · OOS_it + b_POST · POST_it + e_it

- **Standard errors:** clustered by calendar month (the strategies are correlated within a month).
- **Reading the coefficients:** b_POST = −0.40 means POST returns average 60% of IS.
- **H1:** b_POST < 0. One-sided normal p-value; **pass if p < 0.05.**

### Secondary (reported, not gated)

- **H2:** b_POST < b_OOS, the publication effect beyond statistical bias. The difference is tested with the clustered covariance, one-sided.
- **Per strategy:** mean, t-stat and annualized Sharpe in IS / OOS / POST, and the decay ratio μ_POST / μ_IS.
- **Per strategy, on "the system":** a sign-flip permutation test on POST returns (`mcptlab.permutation_test`, 10,000 draws; each month's return keeps its size and gets a random sign). Question: *is anything left after publication?* Logged to the ledger.
- **Robustness:** H1 re-run (a) dropping the calendar strategies 6–9, and (b) with POST starting 2 years earlier (a rough proxy for working-paper circulation). Reported only; the verdict comes from the primary spec.

## 6. Harness (must pass before real data is loaded)

- **H16:** 9 synthetic strategies with planted decay: IS mean μ, OOS 0.75μ, POST 0.40μ, plus noise. The regression must recover b_POST ≈ −0.60 (within 2 SE) with p < 0.05.
- **H17:** the same with no decay: b_POST within ±2 SE of 0.
- **H18:** calendar builders on synthetic daily data. A planted TOM effect (+10 bps/day on TOM days) and a planted Monday effect are recovered with the right sign and size. On data with no planted effect, their means are within ±2 SE of 0.
- **H19:** look-ahead canary for Faber. Changing all returns after month t must not change any signal at or before t.

## 7. What would change my mind

- **Decay confirmed:** H1 passes. Expected, per the literature.
- **Decay not found:** H1 fails. That is still a publishable replication result and would be reported just as prominently.
- **No trading conclusions will be drawn.** Nothing in this study will be traded. It measures how edges decay; it is not a search for one.

## 8. Amendments

**v6-A1 (2026-09-24, before any real-data run).** Header corrected from "10 per-strategy tests, trials 27–36" to "9, trials 27–35". Section 5 already limits the sign-flip test to the pooled strategies; Faber is judged by its Sharpe difference (3.2). This fixes a counting error only.

**Harness record (2026-09-24, synthetic data only).** H16–H19 pass (`tests/test_decay.py`).
- **H16:** b_POST −0.572 (planted −0.60), p 0.021.
- **H17:** seed 0 gives z 0.34.
- **Extra calibration:** 200 no-decay worlds give p < 0.05 in 3.5% of them (target 5%). 200 planted-60%-decay worlds: power 0.64, mean b_POST −0.569.
- **Power caveat:** the primary test can miss a real decay of this size about a third of the time.

**v6-A2 (2026-09-24, before any real-data run).** A dry run of the whole script on *scrambled* data (every return given a random sign; outputs written to a scratch folder, not the real ledger) found two crashes and one design flaw:
1. **Design flaw:** in the "POST starts 2 years earlier" robustness check, Fama-French 1992's shifted POST start (1990-07) fell *inside* its own sample (which ends 1990-12). Fix: POST can never start before the month after the IS window ends.
2. **Crash:** the pooled regression failed when no OOS months exist (the case above).
3. **Crash:** the pooled regression failed when the "drop calendar strategies" robustness set is empty.

These are robustness-only and crash fixes. The primary specification (H1), the strategy list, the dates and the gates are unchanged. The harness still passes.
