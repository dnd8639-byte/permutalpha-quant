# RESULTS v7: publication decay after trading costs and slippage (trials 36–44)

**Run:** 2026-09-24, as preregistered in `PREREGISTRATION_v7.md`. The cost assumptions were fixed before any net number was computed. This is a disclosed post-hoc follow-up to v6.
**Code:** `mcpt-lab/examples/07_decay_after_costs.py`.
**Outputs:** `results/decay/net_of_costs.csv`, `faber_net.csv`, `run_output_net.txt`; ledger rows 23–31; nulls 22–30.
**Harness:** H20–H22 pass. The whole script was also dry-run on scrambled data first.

## Verdict

**After realistic costs, no published strategy leaves a statistically reliable profit after publication.**
- The best is turn-of-the-month: +0.15%/mo net, p = 0.087. That is far from the Bonferroni bar of 0.0056.
- Five of the nine strategies lose money net after publication.

## Net returns (%/month; costs include slippage for a small trader)

| strategy | IS gross | IS net | POST gross | POST cost | **POST net** | t | POST net at 0.5× / 2× / era-scaled | break-even one-way cost |
|---|---|---|---|---|---|---|---|---|
| Size | 0.157 | 0.117 | 0.015 | 0.040 | **−0.025** | −0.2 | −0.005 / −0.065 / −0.014 | 0.61% |
| Value | 0.425 | 0.375 | 0.181 | 0.050 | **+0.131** | 0.8 | 0.156 / 0.081 / 0.149 | 3.10% |
| Momentum | 0.821 | 0.171 | 0.389 | 0.650 | **−0.261** | −1.1 | 0.064 / −0.911 / −0.012 | 0.56% |
| Short-term reversal | 0.963 | −0.687 | 0.139 | 1.650 | **−1.511** | −8.8 | −0.686 / −3.161 / −0.925 | 0.08% |
| Profitability | 0.276 | 0.246 | 0.196 | 0.030 | **+0.166** | 0.9 | 0.181 / 0.136 / 0.181 | 5.00% |
| Small-firm January | 0.341 | −0.201 | 0.034 | 0.541 | **−0.507** | −6.0 | −0.237 / −1.048 / −0.344 | 0.10% |
| Monday | 0.934 | −6.799 | 0.225 | 0.432 | **−0.207** | −1.7 | 0.009 / −0.638 / −0.207 | 0.03% |
| Turn-of-the-month | 0.598 | −1.139 | 0.211 | 0.059 | **+0.152** | 1.4 | 0.182 / 0.093 / 0.152 | 0.11% |
| Halloween | 0.516 | 0.407 | 0.153 | 0.007 | **+0.146** | 0.6 | 0.149 / 0.139 / 0.146 | 0.46% |

**Break-even** is the one-way cost per 100% of notional traded at which the post-publication return reaches zero. It needs no cost assumption.

**POST net sign-flip p** (10,000 draws):

| strategy | p |
|---|---|
| Size | 0.586 |
| Value | 0.209 |
| Momentum | 0.860 |
| Reversal | 1.000 |
| Profitability | 0.197 |
| January | 1.000 |
| Monday | 0.959 |
| TOM | 0.087 |
| Halloween | 0.284 |

**None pass**, even before correcting for 9 tests.

**Faber moving-average timing, net of era costs:** it switches only about 1.5 times a year, so costs are negligible (0.003%/mo after publication).
- Post-publication Sharpe: 0.75 vs 0.65 for buy-and-hold.
- Max drawdown: −18% vs −50%.
- The v6 conclusion is unchanged.

## What the costs reveal

1. **Some famous anomalies were never tradable when they were discovered.**
   - The Monday effect and the turn-of-the-month effect were large in-sample (t = 9.4 and 7.3). But trading the whole market before 1975 cost about 1% one-way.
   - Net of that, Monday lost 6.8%/mo in-sample and TOM lost 1.1%/mo.
   - These were real price patterns that no one could profit from. That is one reason they could persist.
2. **By the time trading got cheap, the effects had shrunk.**
   - After S&P futures (1982) and decimalization (2001), TOM became cheap to trade. Its post-publication net is +0.15%/mo, but not significant (p = 0.087).
   - Monday's break-even is about 3 bps one-way, close to today's cost of roughly 2 bps. It sits right at the edge.
3. **High-turnover stock strategies are consumed by costs.**
   - Short-term reversal loses money net even in its authors' sample. This matches Novy-Marx & Velikov's finding.
   - Momentum's post-publication gross return of 0.39%/mo is below its measured cost of 0.65%/mo. With era-scaled (cheaper modern) costs it is about zero.
4. **Low-turnover strategies barely notice costs, but have little left anyway.**
   - Value, profitability and Halloween keep +0.13 to +0.17%/mo after publication.
   - None is statistically distinguishable from zero.

## Limits

- **Costs are estimates:**
  - The Novy-Marx & Velikov figures are 1963–2013 averages for decile portfolios, not the exact French factors.
  - The market-cost eras are a four-step approximation built from published anchors, not a measured series.
  - The 0.5× / 2× / era runs show which conclusions depend on this. Momentum's sign does; the others don't.
- **Market impact** for large trades is represented only by the 2× run.
- **Taxes, short-borrow fees and financing costs are ignored.** All of them would make the net results worse.
- **Post-hoc disclosure:** this analysis was designed after the gross results were known. Every sensitivity run is reported.

## Trial count

**44 in total:** 35 before this round, plus 9 descriptive net-of-cost tests. The ledger has 31 rows. The other 13 trials are the preregistered futures tests in `research/` and `v2`–`v5`.

## Sources for the cost assumptions

- Novy-Marx, R. and M. Velikov (2016), "A Taxonomy of Anomalies and Their Trading Costs", *Review of Financial Studies* 29(1), 104–147, Table 3. NBER w20721.
- Jones, C. M. (2002), "A Century of Stock Market Liquidity and Trading Costs", Columbia working paper. Fixed NYSE commissions of 0.27% (1925) to 0.90% (1974) one-way; deregulated May 1975.
