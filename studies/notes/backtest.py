import numpy as np
import pandas as pd
from signals import (realized_vol, momentum_signal, raw_positions,
                     portfolio_gearing, ANN, TARGET_VOL)

CORR_WINDOW = 24


def run_backtest(prices, cost_bps=10.0):
    rets = prices.pct_change()
    vol  = realized_vol(rets)
    sig  = momentum_signal(rets, vol)
    pos  = raw_positions(sig, vol)

    geared = pd.DataFrame(index=pos.index, columns=pos.columns, dtype=float)
    for i, t in enumerate(pos.index):
        if i < CORR_WINDOW:
            continue
        window = rets.iloc[i - CORR_WINDOW:i]
        corr   = window.corr().fillna(0.0).copy()
        np.fill_diagonal(corr.values.copy() if not corr.values.flags.writeable else corr.values, 1.0)
        g = portfolio_gearing(pos.loc[t], corr, vol.loc[t])
        geared.loc[t] = pos.loc[t] * g

    held  = geared  # signal already uses only t-1..t-4 returns
    gross = (held * rets).sum(axis=1)
    turn  = (held - held.shift(1)).abs().sum(axis=1)
    net   = gross - turn * (cost_bps / 1e4)

    return {"returns": net.dropna(), "positions": held,
            "turnover": turn, "metrics": tearsheet(net.dropna())}


def tearsheet(r):
    eq   = (1 + r).cumprod()
    dd   = (eq / eq.cummax() - 1).min()
    vol  = r.std() * np.sqrt(ANN)
    cagr = eq.iloc[-1] ** (ANN / len(r)) - 1
    return {
        "cagr": cagr, "vol": vol,
        "sharpe": (r.mean() * ANN) / vol if vol else np.nan,
        "max_dd": dd,
        "calmar": cagr / abs(dd) if dd else np.nan,
        "skew": r.skew(),
        "hit_rate": (r > 0).mean(),
    }
