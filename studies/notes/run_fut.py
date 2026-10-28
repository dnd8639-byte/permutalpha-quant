import numpy as np
from data_fut import load_prices
from backtest import run_backtest

px = load_prices().resample("ME").last().dropna(axis=0, how="all")
print(f"{px.shape[0]} monthly bars, {px.shape[1]} contracts")
print(f"range: {px.index[0].date()} to {px.index[-1].date()}\n")

res = run_backtest(px, cost_bps=10.0)
for k, v in res["metrics"].items():
    print(f"{k:12s} {v: .4f}")

c = px.pct_change().corr()
n = c.shape[0]
avg = (np.nansum(c.values) - n) / (n * (n - 1))
ev = np.linalg.eigvalsh(c.fillna(0).values)
ev = ev[ev > 0]
print(f"\navg pairwise corr: {avg: .3f}")
print(f"effective bets:    {n / (1 + (n - 1) * avg): .1f}")
print(f"eigenvalue ENB:    {(ev.sum() ** 2) / (ev ** 2).sum(): .1f}")
