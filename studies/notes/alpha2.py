import numpy as np, pandas as pd
from data_etf import load_prices
from backtest import run_backtest

px = load_prices().resample("ME").last().dropna(axis=0, how="all")
r  = run_backtest(px, cost_bps=10.0)["returns"]
F  = px[["SPY","TLT","GLD","UUP"]].pct_change().reindex(r.index)

d = pd.concat([r.rename("strat"), F], axis=1).dropna()
y = d["strat"].values
X = np.column_stack([np.ones(len(d)), d[["SPY","TLT","GLD","UUP"]].values])
b, *_ = np.linalg.lstsq(X, y, rcond=None)
resid = y - X @ b
s2 = (resid @ resid) / (len(y) - X.shape[1])
se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))

names = ["alpha","SPY","TLT","GLD","UUP"]
for n, coef, e in zip(names, b, se):
    print(f"{n:6s} {coef: .4f}  t={coef/e: .2f}")
print(f"\nalpha annualized: {b[0]*12: .4f}")
print(f"R-squared:        {1 - resid.var()/y.var(): .3f}")
