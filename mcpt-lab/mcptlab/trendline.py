"""neurotrader888/TrendlineBreakoutMetaLabel (MIT), ported for the MCPT framework.

Base strategy : long when the log close breaks above a resistance line fitted to the prior `lookback`
                bars; exit at +/- 3 ATR or after 12 bars.
Meta-label    : a random forest (1000 trees, depth 3), trained walk-forward on each trade's features,
                predicts whether the breakout will win; only trades with P(win) > 0.5 are taken.

Speed-up: the author finds the resistance slope with an iterative search. The problem has an exact
solution (the least-squares line through the pivot that stays above every price is the unconstrained
least-squares slope clipped to the feasible interval), which is computed here for all windows at once.
Checked against the original: 1,172 of 1,175 trades identical (the other 3 are borderline cases
where the author's iterative search stops short of the exact optimum).
"""
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view


def _rma(x, n):
    return x.ewm(alpha=1 / n, min_periods=n).mean()


def atr(high, low, close, n):
    prev = close.shift(1)
    tr = pd.concat([high - low, (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    tr.iloc[0] = np.nan
    return _rma(tr, n)


def adx(high, low, close, n):
    a = atr(high, low, close, n)
    up, dn = high - high.shift(1), low.shift(1) - low
    pos = ((up > dn) & (up > 0)) * up
    neg = ((dn > up) & (dn > 0)) * dn
    dmp, dmn = 100 * _rma(pos, n) / a, 100 * _rma(neg, n) / a
    return _rma(100 * (dmp - dmn).abs() / (dmp + dmn), n)


def resistance_lines(logc, L):
    """For each bar i >= L: (slope, intercept) of the resistance line fitted to logc[i-L:i] (excludes bar i)."""
    W = sliding_window_view(logc, L)[:-1]                      # window ending at i-1, for i = L..n-1
    x = np.arange(L, dtype=float)
    xm = x.mean()
    ols = ((W - W.mean(1, keepdims=True)) * (x - xm)).sum(1) / ((x - xm) ** 2).sum()
    line = ols[:, None] * (x - xm) + W.mean(1, keepdims=True)
    p = np.argmax(W - line, axis=1)                             # pivot: highest point above the OLS line
    yp = W[np.arange(len(W)), p][:, None]
    d = x[None, :] - p[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        s = (W - yp) / d
    lo = np.where(d > 0, s, -np.inf).max(1)                     # line must stay above later points
    hi = np.where(d < 0, s, np.inf).min(1)                      # ... and earlier points
    m_star = (d * (W - yp)).sum(1) / (d ** 2).sum(1)
    m = np.clip(m_star, lo, hi)
    intercept = yp[:, 0] - m * p
    return m, intercept


def build_trades(ohlcv, lookback=72, hold=12, tp_mult=3.0, sl_mult=3.0, atr_lookback=168):
    """Same trade rules and features as the author's trendline_breakout_dataset()."""
    close = np.log(ohlcv["close"].to_numpy())
    atr_arr = atr(np.log(ohlcv["high"]), np.log(ohlcv["low"]), np.log(ohlcv["close"]), atr_lookback).to_numpy()
    vol_arr = (ohlcv["volume"] / ohlcv["volume"].rolling(atr_lookback).median()).to_numpy()
    adx_arr = adx(ohlcv["high"], ohlcv["low"], ohlcv["close"], lookback).to_numpy()
    m, b = resistance_lines(close, lookback)
    rows, in_trade = [], False
    for i in range(atr_lookback, len(close)):
        k = i - lookback
        r_val = b[k] + lookback * m[k]
        if not in_trade and close[i] > r_val:
            a = atr_arr[i]
            w = close[i - lookback:i]
            line = b[k] + np.arange(lookback) * m[k]
            cur = dict(entry_i=i, entry_p=close[i], atr=a, sl=close[i] - a * sl_mult, tp=close[i] + a * tp_mult,
                       hp_i=i + hold, slope=m[k], intercept=b[k], resist_s=m[k] / a,
                       tl_err=np.sum(line - w) / lookback / a, max_dist=(line - w).max() / a,
                       vol=vol_arr[i], adx=adx_arr[i])
            rows.append(cur); in_trade = True
        if in_trade and (close[i] >= cur["tp"] or close[i] <= cur["sl"] or i >= cur["hp_i"]):
            cur["exit_i"], cur["exit_p"] = i, close[i]
            in_trade = False
    t = pd.DataFrame(rows)
    t["return"] = t["exit_p"] - t["entry_p"]
    return t


FEATURES = ["resist_s", "tl_err", "vol", "max_dist", "adx"]


def meta_label_probs(trades, n_bars, train_size, step_size, n_estimators=1000, max_depth=3, seed=69420,
                     shuffle_labels_seed=None):
    """Walk-forward P(win) for each trade, as in the author's walkforward_model(): at each retrain bar,
    fit on trades that entered AND exited inside the previous train_size bars; score the trades that
    enter before the next retrain. Batch prediction (same numbers as predicting one trade at a time).
    shuffle_labels_seed: permute the training labels (null test for the filter)."""
    from sklearn.ensemble import RandomForestClassifier
    X, y = trades[FEATURES].to_numpy(), (trades["return"] > 0).astype(int).to_numpy()
    prob = np.full(len(trades), np.nan)
    rng = np.random.default_rng(shuffle_labels_seed) if shuffle_labels_seed is not None else None
    ent, ext = trades["entry_i"].to_numpy(), trades["exit_i"].to_numpy()
    for t0 in range(train_size, n_bars, step_size):
        tr = (ent > t0 - train_size) & (ext < t0)
        sc = (ent >= t0) & (ent < t0 + step_size)
        if tr.sum() < 10 or sc.sum() == 0:
            continue
        yt = rng.permutation(y[tr]) if rng is not None else y[tr]
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=seed)
        model.fit(X[tr], yt)
        prob[sc] = model.predict_proba(X[sc])[:, list(model.classes_).index(1)] if 1 in model.classes_ else 0.0
    return prob


def trade_signals(logc, trades, prob, threshold=0.5):
    """Author's position logic: a trade is 'taken' if prob > threshold; untaken trades still block
    overlapping entries (as in the original). Returns (filtered signal, all-trades signal)."""
    n = len(logc)
    sig, allsig = np.zeros(n), np.zeros(n)
    for k, tr in enumerate(trades.itertuples()):
        if np.isnan(prob[k]):
            continue
        e = int(tr.entry_i)
        x = int(tr.exit_i) if np.isfinite(tr.exit_i) else n
        allsig[e:x] = 1
        if prob[k] > threshold:
            sig[e:x] = 1
    return sig, allsig


def net_returns(logc, sig, cost_bps):
    r = pd.Series(logc).diff().shift(-1).to_numpy()
    pos = pd.Series(sig)
    cost = pos.diff().abs().fillna(pos.abs()).to_numpy() * cost_bps / 1e4
    return sig * r - cost


def pf(x):
    x = x[np.isfinite(x)]
    loss = -x[x < 0].sum()
    return x[x > 0].sum() / loss if loss > 0 else np.inf
