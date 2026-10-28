# Quant mini-projects

Small, self-contained Python projects. Each one is a single file with the idea, the method and a built-in check at the top. They need only numpy, pandas and matplotlib.

```bash
cd ~/Desktop/ahl-node && source .venv/bin/activate && cd quant-projects
python p01_tail_index.py
python p02_intraday_seasonality.py
python p03_overfitting_detector.py
```
Figures are saved in `figures/`. Data comes from `../mcpt-lab/data/` (BTC hourly) and, if present, `../state/v3/clean_daily.csv` (CME futures; licensed, so don't publish that file).

To get the public data for projects 8-10 (no account needed):

```bash
python download_data.py
```

## Projects

| # | file | project | result |
|---|---|---|---|
| 1 | `p01_tail_index.py` | Tail index (Hill estimator) | Tail exponents of about 3 for BTC, the S&P 500, crude oil and gold. A Gaussian 99% VaR was breached 1.4-2.3% of days instead of 1%. |
| 2 | `p02_intraday_seasonality.py` | Intraday seasonality map | *When* BTC moves is predictable (p = 0.002; the 2018-21 map correlates +0.63 with 2022+). *Which way* is not (out-of-sample correlation -0.02). |
| 3 | `p03_overfitting_detector.py` | Backtest overfitting detector | Best of 1,000 random rules: Sharpe 1.06; the noise formula predicted 1.12. Its Deflated Sharpe falls from 0.999 to 0.43 once the search is counted. |
| 4 | `p04_momentum_regimes.py` | Regime-dependent momentum | Momentum earns +10.4%/yr in normal markets and -9.3%/yr in bear markets, where its market beta falls to -0.90 on rebounds. Volatility scaling lifts the Sharpe from 0.43 to 0.70, but only 0.10 to 0.26 since 2016. |
| 5 | `p05_factor_neutrality.py` | Factor neutrality check | Self-test: 5.0% false alarms on 500 fake strategies. Momentum keeps a 9.0%/yr alpha against five factors (t = 4.3). My Hawkes strategy has no exposure to BTC's direction; its alpha t-stat falls from 4.4 in-sample to 1.3 after publication. |
| 6 | `p06_american_put.py` | American put: Crank-Nicolson + projected SOR | Price 6.0896 vs 6.0904 from a binomial tree; error shrinks about 3.8x per grid halving (close to second order). Exercise boundary starts at S = 81. |
| 7 | `p07_autodiff_greeks.py` | Monte Carlo Greeks by automatic differentiation | Hand-written dual numbers. Delta is unbiased with 40x less noise than fresh-random bumping. Naive second-order autodiff returns a gamma of exactly 0; a likelihood-ratio fix recovers it. |
| 8 | `p08_implied_distribution.py` | Implied risk-neutral distribution | Checks pass (recovers the lognormal density; rounding to cents breaks raw differencing but not the smoothed method). SPY options, 30 Oct 2026 expiry: skew -1.7, and a 0.9% priced chance of a 15%+ fall where a bell curve says about zero. |
| 9 | `p09_hawkes_trades.py` | Hawkes process on trade arrivals | Checks pass (recovers branching ratios 0.00 / 0.29 / 0.68 / 0.89 for true 0 / 0.3 / 0.7 / 0.9). Real BTC trades: branching ratio 0.47, half-life 0.7 s, but the residual test fails, so one exponential kernel is too simple. |
| 10 | `p10_news_diffusion.py` | News-to-price information diffusion | Checks pass (recovers a 15-minute half-life; the placebo finds no effect). Around 32 Fed and CPI releases, BTC activity jumps about 45x at the release minute and fades with a 5-minute half-life. Later hours of raised activity may be time-of-day seasonality. |

`common.py` holds the shared helpers (data loading, regression with Newey-West errors, a small optimizer) and `hawkes.py` the Hawkes-process code used by projects 9 and 10.
