"""v2 backtest. Strategy logic is imported UNCHANGED from research/signals.py.
Only differences from research/backtest.py:
  * causal=True  -> vol = realized_vol(rets).shift(1)   (bug #3 fix, preregistered)
  * cost can be a scalar or a per-asset Series of bps; optional doubling before 1990
  * output is trimmed to start at the first month with a live position
    (v1 counted the ~24 warm-up months as 0% returns, which deflates Sharpe slightly)
"""
import os, sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "research"))
from signals import realized_vol, momentum_signal, raw_positions, portfolio_gearing

CORR_WINDOW = 24


def run_backtest(prices=None, cost_bps=10.0, causal=True, double_pre1990=False, rets=None):
    if rets is None:
        rets = prices.pct_change(fill_method=None)
    vol = realized_vol(rets)
    if causal:
        vol = vol.shift(1)
    sig = momentum_signal(rets, vol)
    pos = raw_positions(sig, vol)

    geared = pd.DataFrame(np.nan, index=pos.index, columns=pos.columns)
    for i, t in enumerate(pos.index):
        if i < CORR_WINDOW:
            continue
        corr = rets.iloc[i - CORR_WINDOW:i].corr().fillna(0.0)
        c = corr.to_numpy(copy=True)
        np.fill_diagonal(c, 1.0)
        corr = pd.DataFrame(c, index=corr.index, columns=corr.columns)
        geared.loc[t] = pos.loc[t] * portfolio_gearing(pos.loc[t], corr, vol.loc[t])

    held = geared
    gross = (held * rets).sum(axis=1)
    dturn = (held.fillna(0) - held.shift(1).fillna(0)).abs()
    if np.isscalar(cost_bps):
        cost = dturn.sum(axis=1) * cost_bps / 1e4
    else:
        cost = (dturn * cost_bps.reindex(held.columns).fillna(10.0) / 1e4).sum(axis=1)
    if double_pre1990:
        cost = cost * np.where(cost.index < "1990-01-01", 2.0, 1.0)
    net = gross - cost

    live = held.abs().sum(axis=1) > 0
    start = live.idxmax() if live.any() else net.index[-1]
    net = net.loc[start:]
    return {"returns": net, "positions": held.loc[start:],
            "turnover": dturn.sum(axis=1).loc[start:]}
