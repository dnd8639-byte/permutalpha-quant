# PREREGISTRATION v7: v6 publication-decay returns after trading costs and slippage

**Written:** 2026-09-24, **after the v6 gross results were seen and before any net-of-cost number was computed.**

**Post-hoc disclosure:** this is a follow-up to v6. Because the gross returns are already known, the cost assumptions below come from published sources, not from our own choice. Every sensitivity run is reported, and no specification is chosen after results.

**Trial count:** 35 before this round. The 9 per-strategy post-publication net-of-cost sign-flip tests are trials **36–44** (descriptive).

## 1. Question

How much of each v6 strategy survives realistic trading costs, in each window (IS / OOS / POST), and in particular **today** (POST)?

## 2. Cost model (fixed)

Cost = (units traded per month) × (one-way cost per unit traded). One unit = buying or selling 100% of the strategy's notional.

**Slippage:** the stock costs below are *effective spreads* (Hasbrouck 2009), the price actually paid compared with the quote midpoint. That already includes the slippage a small trader pays with market orders. It does **not** include the market impact of large orders, so a **2× cost** run stands in for impact and for a larger trader.

### 2a. Factor strategies (long-short stock portfolios): published measured costs

The costs come from Novy-Marx & Velikov (2016, *RFS* 29(1)), Table 3. They are value-weighted decile long-short strategies, averaged over 1963–2013, and applied as a **constant monthly cost:**

| strategy | NMV analog | turnover per side, %/mo | cost, %/mo |
|---|---|---|---|
| Size | Size | 1.23 | 0.04 |
| Value | Value | 2.91 | 0.05 |
| Profitability | Gross profitability | 1.96 | 0.03 |
| Momentum | Momentum | 34.52 | 0.65 |
| Short-term reversal | Short-run reversals | 90.87 | 1.65 |

**Limitation:** a constant average overstates costs in recent decades and understates them in early ones. NMV's Figure 1 shows costs falling over time. The era-scaled run (2d) addresses this.

### 2b. Small-firm January (SMB held in January only)

- **Trades:** enter both legs at the December close and exit both at the January close, so 4 units a year, charged in January.
- **Cost per unit:** the NMV cost-to-turnover ratio of the Size strategy, 0.04 / (2 × 1.23%) = **1.63% one-way.** This is the measured cost of trading the size decile legs.

### 2c. Market-timing strategies (Monday, turn-of-the-month, Halloween, Faber)

**Instrument:**
- a basket of stocks before S&P 500 futures (listed April 1982);
- then S&P 500 futures / an index ETF.

**One-way cost per unit c_m by era** (includes commission + half-spread + slippage):

| era | c_m | basis |
|---|---|---|
| to 1975-04 | 1.00% | fixed NYSE commissions of 0.27–0.90% one-way (Jones 2002) plus half-spread |
| 1975-05 – 1982-04 | 0.50% | after May Day 1975; commissions deregulated and falling |
| 1982-05 – 2000-12 | 0.05% | S&P 500 futures |
| 2001-01 → | 0.02% | futures / SPY after decimalization (2 bps) |

**Units traded:**
- **Monday:** sell at Friday's close, buy back at Monday's close, so 2 units per Monday.
- **Turn-of-the-month:** in before the last trading day, out after day 3, so 2 units per month.
- **Halloween:** long ↔ short at the end of April and the end of October, so 2 units per switch and 4 a year.
- **Faber:** 1 unit per switch between the market and T-bills (replaces v6's flat 10 bps).

**Caveat:** the Monday and TOM series are *relative* measures (anomaly days vs normal days), so their net series tell us whether the effect is larger than what trading it costs.

### 2d. Sensitivity (all reported)

1. **2× costs** (impact / larger trader) and **0.5× costs** (a patient trader using limit orders; also because the size-decile rate in 2b is likely high for SMB, which holds the whole bottom half of NYSE size, not just the smallest decile).
2. **Era-scaled factor costs (strategies 1–6):** the NMV averages scaled ×1.5 before 1982-05, ×1.0 for 1982-05 – 2000-12, ×0.5 from 2001. The multipliers are fixed here, a rough stand-in for NMV's Figure 1 trend.
3. **Break-even one-way cost:** the POST gross mean divided by the units traded per month, i.e. the cost at which the net return is zero. It needs no cost assumption.

## 3. Statistics

- **Per strategy × window:** gross, cost and net mean (%/mo), net t-stat, and net annualized Sharpe.
- **POST net sign-flip test** (10,000 draws) for strategies 1–9, logged to the ledger (trials 36–44).
- **Multiple-testing bar:** Bonferroni over this round, 0.05 / 9 = 0.0056.
- **No pooled decay regression on net returns:** net IS means can be ≤ 0, so scaling by them is undefined. The v6 gross result stands as the decay test.
- **Faber:** net Sharpe and max drawdown, timed vs buy-and-hold, per window.

## 4. Harness

- **H20:** a strategy with zero gross return and known turnover has net mean = −(units × cost) exactly.
- **H21:** the unit counts on synthetic calendars:
  - Monday: 2 × the number of Mondays;
  - TOM: 2 per month;
  - Halloween: 4 per year;
  - January: 4 per year.
- **H22:** break-even cost × units equals the gross mean.

## 5. Amendments

(none yet)
