"""v3 strategies on monthly returns. One risk engine for all of them (same as v2 causal engine).
Timing: row t of R is the return DURING month t. Every signal for row t uses data through t-1 only."""
import os, sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "research"))
from signals import realized_vol, momentum_signal, raw_positions, portfolio_gearing, ANN

CORR_WINDOW = 24
MIN_XS = 6


def causal_vol(R):
    return realized_vol(R).shift(1)


def engine(R, raw_pos, cost_bps=0.0, rolls=None):
    """Gear raw positions to 10% ex-ante vol, apply trading + roll costs. Returns net monthly returns."""
    vol = causal_vol(R)
    geared = pd.DataFrame(np.nan, index=R.index, columns=R.columns)
    for i, t in enumerate(R.index):
        if i < CORR_WINDOW:
            continue
        w = raw_pos.loc[t]
        if w.fillna(0).abs().sum() == 0:
            continue
        c = R.iloc[i - CORR_WINDOW:i].corr().fillna(0.0).to_numpy(copy=True)
        np.fill_diagonal(c, 1.0)
        corr = pd.DataFrame(c, index=R.columns, columns=R.columns)
        geared.loc[t] = w * portfolio_gearing(w, corr, vol.loc[t])
    held = geared.fillna(0.0)
    gross = (held * R.fillna(0.0)).sum(axis=1)
    bps = (pd.Series(cost_bps, index=R.columns) if np.isscalar(cost_bps)
           else cost_bps.reindex(R.columns).fillna(10.0)) / 1e4
    trade = (held - held.shift(1).fillna(0.0)).abs()
    cost = (trade * bps).sum(axis=1)
    if rolls is not None:
        cost += (held.abs() * rolls.reindex_like(held).fillna(0.0) * 2 * bps).sum(axis=1)
    net = gross - cost
    live = held.abs().sum(axis=1) > 0
    return net.loc[live.idxmax():] if live.any() else net.iloc[0:0]


# ---------------- raw position builders (all return DataFrame aligned to R)
def pos_T0(R):
    vol = causal_vol(R)
    return raw_positions(momentum_signal(R, vol), vol)


def pos_T4(R):
    vol = causal_vol(R)
    past12 = np.log1p(R).shift(1).rolling(12, min_periods=12).sum()
    return np.sign(past12) / vol


def _xs(score, vol):
    ok = score.notna() & vol.notna() & (vol > 0)
    rank = score.where(ok).rank(axis=1, pct=True)
    n = ok.sum(axis=1)
    side = pd.DataFrame(0.0, index=score.index, columns=score.columns)
    side[rank > 2 / 3] = 1.0
    side[rank <= 1 / 3] = -1.0
    side[n < MIN_XS] = 0.0
    return side / vol


def pos_T5(R, leak=False):
    vol = causal_vol(R)
    lag = 0 if leak else 2                    # leak=True is the H8 canary (uses month t)
    ret = np.expm1(np.log1p(R).shift(lag).rolling(11, min_periods=11).sum())
    return _xs(ret / vol, vol)


def pos_T6(R, carry_monthend):
    """carry_monthend: carry measured on the last trading day of each month (index = month end)."""
    vol = causal_vol(R)
    return _xs(carry_monthend.reindex(R.index).shift(1), vol)


def blend(streams, target=0.10):
    scaled = []
    for s in streams:
        sd = s.rolling(24, min_periods=24).std().shift(1) * np.sqrt(ANN)
        scaled.append(s * (target / sd))
    df = pd.concat(scaled, axis=1).dropna()
    return df.mean(axis=1)
