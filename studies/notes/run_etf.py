from data_etf import load_prices
from backtest import run_backtest
import numpy as np

px = load_prices().resample("ME").last()
px = px.dropna(axis=0, how="all")
print(f"{px.shape[0]} monthly bars, {px.shape[1]} assets")
print(f"range: {px.index[0].date()} to {px.index[-1].date()}\n")

res = run_backtest(px, cost_bps=10.0)
for k, v in res["metrics"].items():
    print(f"{k:12s} {v: .4f}")

r = px.pct_change().corr()
n = r.shape[0]
avg = (r.values.sum() - n) / (n * (n - 1))
print(f"\navg pairwise corr: {avg: .3f}")
print(f"effective bets:    {n / (1 + (n - 1) * avg): .1f}")
