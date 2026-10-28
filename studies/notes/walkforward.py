import numpy as np, pandas as pd
from data_etf import load_prices
from backtest import run_backtest

px = load_prices().resample("ME").last().dropna(axis=0, how="all")
res = run_backtest(px, cost_bps=10.0)
r = res["returns"]

splits = {
    "2008-2012": ("2008-01-01", "2012-12-31"),
    "2013-2017": ("2013-01-01", "2017-12-31"),
    "2018-2021": ("2018-01-01", "2021-12-31"),
    "2022-2026": ("2022-01-01", "2026-12-31"),
}
print(f"{'period':12s} {'n':>4s} {'sharpe':>8s} {'cagr':>8s} {'maxdd':>8s} {'skew':>7s}")
for name, (a, b) in splits.items():
    s = r.loc[a:b]
    if len(s) < 12:
        continue
    eq = (1 + s).cumprod()
    dd = (eq / eq.cummax() - 1).min()
    vol = s.std() * np.sqrt(12)
    sh = (s.mean() * 12) / vol if vol else np.nan
    cagr = eq.iloc[-1] ** (12 / len(s)) - 1
    print(f"{name:12s} {len(s):4d} {sh:8.3f} {cagr:8.3f} {dd:8.3f} {s.skew():7.2f}")

half = len(r) // 2
for label, s in [("first half", r.iloc[:half]), ("second half", r.iloc[half:])]:
    sh = (s.mean() * 12) / (s.std() * np.sqrt(12))
    print(f"\n{label:12s} n={len(s):3d}  sharpe {sh: .3f}")
