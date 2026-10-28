"""Price permutations for Monte Carlo permutation tests (MCPT).

Adapted from neurotrader888/mcpt `bar_permute.py` (MIT License, (c) 2025 neurotrader888).
A permutation keeps the *distribution* of price changes (mean, volatility, fat tails) and,
for several markets, their cross-correlation, but destroys the *order* of those changes.
Any edge that depends on order (trends, reversals, patterns) should vanish in a permutation.
"""
import numpy as np
import pandas as pd


def permute_ohlc(ohlc, start_index=0, seed=None):
    """neurotrader bar permutation for OHLC data (one DataFrame or a list of aligned DataFrames).
    Bars before start_index are left untouched (used for walk-forward permutation tests)."""
    rng = np.random.default_rng(seed)
    single = not isinstance(ohlc, list)
    mkts = [ohlc] if single else ohlc
    idx0 = mkts[0].index
    for m in mkts:
        assert (m.index == idx0).all(), "indexes do not match"
    n = len(idx0); p0 = start_index + 1; pn = n - p0
    rel = []
    for m in mkts:
        lb = np.log(m[["open", "high", "low", "close"]])
        rel.append(dict(o=(lb["open"] - lb["close"].shift()).to_numpy()[p0:],
                        h=(lb["high"] - lb["open"]).to_numpy()[p0:],
                        l=(lb["low"] - lb["open"]).to_numpy()[p0:],
                        c=(lb["close"] - lb["open"]).to_numpy()[p0:],
                        start=lb.iloc[start_index].to_numpy(), real=lb.to_numpy()))
    perm1, perm2 = rng.permutation(pn), rng.permutation(pn)     # intrabar moves, gaps: shuffled separately
    out = []
    for r in rel:
        o, h, l, c = r["o"][perm2], r["h"][perm1], r["l"][perm1], r["c"][perm1]
        # vectorized form of: open_i = close_{i-1} + gap ; close_i = open_i + c ; high/low = open_i + h/l
        close_path = r["start"][3] + np.cumsum(o + c)
        open_path = np.concatenate([[r["start"][3]], close_path[:-1]]) + o
        b = np.zeros((n, 4)); b[:start_index] = r["real"][:start_index]; b[start_index] = r["start"]
        b[p0:, 0], b[p0:, 1], b[p0:, 2], b[p0:, 3] = open_path, open_path + h, open_path + l, close_path
        out.append(pd.DataFrame(np.exp(b), index=idx0, columns=["open", "high", "low", "close"]))
    return out[0] if single else out


def permute_close(close, start_index=0, seed=None):
    """Close-only version (for data that has no open/high/low, like our roll-adjusted futures).
    Shuffles log returns after start_index; the same shuffle is applied to every column so
    cross-market correlation is preserved. Accepts a Series or a DataFrame of closes."""
    rng = np.random.default_rng(seed)
    df = close.to_frame() if isinstance(close, pd.Series) else close
    lr = np.log(df).diff().to_numpy()
    n = len(df); p0 = start_index + 1
    perm = rng.permutation(np.arange(p0, n))
    lr_new = lr.copy(); lr_new[p0:] = lr[perm]
    logp = np.log(df.to_numpy()).copy()
    if p0 < n:
        logp[p0:] = logp[p0 - 1] + np.cumsum(lr_new[p0:], axis=0)
    res = pd.DataFrame(np.exp(logp), index=df.index, columns=df.columns)
    return res.iloc[:, 0] if isinstance(close, pd.Series) else res


def signflip_ohlc(ohlc, start_index=0, seed=None):
    """Stricter null for VOLATILITY strategies: randomly mirror each bar around its open (flip the
    direction of the gap and of the bar) with probability 1/2. Bar sizes -- and therefore volatility
    clustering -- are preserved exactly, in their real order; only direction is randomized.
    Note: it also removes drift, so it is a test of direction timing, not of drift capture."""
    rng = np.random.default_rng(seed)
    lb = np.log(ohlc[["open", "high", "low", "close"]])
    n = len(lb); p0 = start_index + 1
    o = (lb["open"] - lb["close"].shift()).to_numpy()
    h = (lb["high"] - lb["open"]).to_numpy(); l = (lb["low"] - lb["open"]).to_numpy()
    c = (lb["close"] - lb["open"]).to_numpy()
    flip = rng.random(n) < 0.5; flip[:p0] = False
    o2, c2 = np.where(flip, -o, o), np.where(flip, -c, c)
    h2, l2 = np.where(flip, -l, h), np.where(flip, -h, l)
    real = lb.to_numpy(); b = real.copy()
    close_path = real[start_index, 3] + np.cumsum(o2[p0:] + c2[p0:])
    open_path = np.concatenate([[real[start_index, 3]], close_path[:-1]]) + o2[p0:]
    b[p0:, 0], b[p0:, 1], b[p0:, 2], b[p0:, 3] = open_path, open_path + h2[p0:], open_path + l2[p0:], close_path
    return pd.DataFrame(np.exp(b), index=ohlc.index, columns=["open", "high", "low", "close"])
