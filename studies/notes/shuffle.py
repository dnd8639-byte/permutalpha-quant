import numpy as np, pandas as pd
from data_etf import load_prices
from backtest import run_backtest

px = load_prices().resample("ME").last().dropna(axis=0, how="all")
rets = px.pct_change()
real = run_backtest(px, cost_bps=10.0)["metrics"]["sharpe"]

rng = np.random.default_rng(42)
null = []
for i in range(1000):
    perm = rng.permutation(len(rets))
    shuf = rets.iloc[perm]              # whole rows move together
    shuf.index = rets.index             # keep NaN-free ordering intact
    fake_px = (1 + shuf).cumprod() * 100
    try:
        s = run_backtest(fake_px, cost_bps=10.0)["metrics"]["sharpe"]
        if np.isfinite(s):
            null.append(s)
    except Exception:
        pass
    if (i + 1) % 200 == 0:
        print(f"  {i+1}/1000")

null = np.array(null)
print(f"\nreal sharpe:      {real: .3f}")
print(f"null mean:        {null.mean(): .3f}")
print(f"null std:         {null.std(): .3f}")
print(f"null 95th pctile: {np.percentile(null, 95): .3f}")
print(f"percentile rank:  {(null < real).mean() * 100: .1f}")
print(f"p-value:          {(null >= real).mean(): .4f}")
