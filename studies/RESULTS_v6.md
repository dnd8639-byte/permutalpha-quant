# RESULTS v6: publication decay (T14; trials 26–35)

**Run:** 2026-09-24, as preregistered in `PREREGISTRATION_v6.md` (amendments A1 and A2 were both made before the run).
**Data:** Kenneth R. French Data Library, 1926-07 → 2026-07.
**Code:** `mcpt-lab/examples/06_publication_decay.py`.
**Outputs:** `mcpt-lab/results/decay/`; ledger rows 13–22; nulls 13–21.
**Reproducibility:** a second run with a scratch ledger reproduced every number and every null file byte for byte.

## Verdict

**H1 PASS: published strategies lost most of their edge after publication.**
- Across the 8 strategies that replicated in-sample, post-publication returns average **30% of their in-sample level** (b_POST = −0.70, clustered SE 0.12, one-sided p ≈ 3×10⁻⁹).
- For comparison, McLean & Pontiff (2016) found about 42% remaining across 97 anomalies. Ours is somewhat steeper, which fits the preregistered warning that famous strategies may overstate decay.

## Per strategy (gross monthly returns, %)

| strategy | paper | IS | OOS | POST | POST / IS | POST sign-flip p |
|---|---|---|---|---|---|---|
| Size (SMB) | Banz 1981 | 0.157 (t 1.2) | 1.223 | 0.015 | 9% | 0.457 |
| Value (HML) | Fama-French 1992 | 0.425 (t 3.0) | 0.682 | 0.181 | 43% | 0.133 |
| Momentum (UMD) | Jegadeesh-Titman 1993 | 0.821 (t 4.0) | 1.167 | 0.389 | 47% | 0.048 |
| Short-term reversal | Jegadeesh 1990 | 0.963 (t 8.4) | 0.358 | 0.139 | 14% | 0.210 |
| Profitability (RMW) | Novy-Marx 2013 | 0.276 (t 2.9) | 0.221 | 0.196 | 71% | 0.157 |
| Small-firm January | Keim 1983 | 0.341 (t 3.2) | 0.145 | 0.034 | 10% | 0.214 |
| Weekend / Monday | French 1980 | 0.934 (t 9.4) | 0.617 | 0.225 | 24% | 0.026 |
| Turn-of-the-month | Lakonishok-Smidt 1988 | 0.598 (t 7.3) | 1.343 | 0.211 | 35% | 0.031 |
| Halloween | Bouman-Jacobsen 2002 | 0.516 (t 2.1) | 0.785 | 0.153 | 30% | 0.275 |

- **Every strategy is weaker after publication than in its authors' sample. 9 of 9 declined.**
- **Size failed the replication rule** (IS t = 1.2 < 1.5), so under the preregistered rule it is **excluded** from the pooled test. French's SMB is a broader size portfolio than Banz's smallest-firm portfolios, and that likely explains the weak in-sample result. Even so, its post-publication return is essentially zero.
- **Is anything left?** Three strategies have POST sign-flip p < 0.05: momentum 0.048, Monday 0.026, turn-of-the-month 0.031. But this round ran 9 such tests. With a Bonferroni threshold of 0.05 / 9 = 0.0056, **none survive.** Across all 35 trials in the project, the bar is higher still.

## Secondary results

- **H2** (a publication effect beyond statistical bias): diff −0.80, SE 0.38, p = 0.017. **But the OOS estimate is very noisy** (b_OOS = +0.10, SE 0.38): pre-publication windows are only 1–6 years long. Read H2 as *consistent with* arbitrage after publication, not as proof of it.
- **Robustness:**
  - Dropping the calendar strategies: b_POST −0.62, p = 0.0002.
  - Starting POST 2 years earlier: b_POST −0.65, p < 0.0001.
  - **The verdict does not depend on these choices.**
- **Faber 10-month moving-average timing** (not pooled):

  | window | Sharpe timed vs hold | max drawdown timed vs hold | return vs hold |
  |---|---|---|---|
  | IS 1973–2005 | 0.39 vs 0.36 | −24% vs −45% | −0.08%/mo |
  | POST 2007–2026 | 0.74 vs 0.65 | −18% vs −50% | −0.20%/mo |

  Faber claimed lower risk, not higher return, and **after publication that claim held.** The drawdown advantage, however, comes mostly from one event (2008), so it rests on very few independent observations.

## Limits (read before quoting any number)

1. **"After publication" is also "later in time".** This design cannot separate publication from a general rise in market efficiency: cheaper trading, more quant capital, faster information. McLean & Pontiff handle this with time-trend controls. This study does not.
2. **Gross of trading costs.** Short-term reversal, Monday and turn-of-the-month trade very often. After realistic costs, their small post-publication remainders would likely vanish. The Monday and turn-of-the-month series are also *relative* measures (anomaly days vs normal days), not directly tradable portfolios.
3. **Famous-strategy selection.** These were chosen because they are well known, which may overstate decay (preregistered disclosure).
4. **Proxies.** RMW stands in for Novy-Marx's gross profitability, and French's SMB for Banz's size portfolios.
5. **Publication dates** come from the journal issues and have not been independently verified. Lakonishok & Smidt's month is the least certain. A few months' error barely moves windows that span decades.
6. **Replication, not discovery.** Decay in several of these strategies is already documented. The contribution here is the reproducible pipeline, fixed published rules, and data extended to 2026.

## What this means for the project

- It fits the earlier results: **13 preregistered futures tests with 0 passes**, and the trendline filter that died after publication (p = 0.195).
- It gives the Hawkes result context: its post-publication PF of 1.031 is weak but alive, the same pattern seen here.
- **No trading conclusions.** Nothing here passes the multiple-testing bar as a tradable edge.

## Trial count

- **35 in total:** 25 before this round, plus 1 primary test and 9 descriptive tests.
- The pooled row in the ledger shows p as 0.0 because of 4-decimal rounding. The exact value is 2.7×10⁻⁹ (`results/decay/pooled_primary.csv`).
