import numpy as np, pandas as pd
from data_etf import load_prices
from backtest import run_backtest

px = load_prices().resample("ME").last().dropna(axis=0, how="all")
r  = run_backtest(px, cost_bps=10.0)["returns"]
spy = px["SPY"].pct_change().reindex(r.index)

d = pd.concat([r, spy], axis=1).dropna()
d.columns = ["strat", "spy"]
beta, alpha = np.polyfit(d["spy"], d["strat"], 1)
resid = d["strat"] - (alpha + beta * d["spy"])
t = alpha / (resid.std() / np.sqrt(len(d))) 

print(f"n months:        {len(d)}")
print(f"beta to SPY:     {beta: .3f}")
print(f"alpha (ann):     {alpha * 12: .4f}")
print(f"t-stat on alpha: {t: .2f}")
print(f"resid sharpe:    {(resid.mean()*12)/(resid.std()*np.sqrt(12)): .3f}")
