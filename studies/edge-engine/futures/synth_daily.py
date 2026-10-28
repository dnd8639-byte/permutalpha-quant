"""Synthetic daily books with the same calendar as the real futures, for validating the daily engine.

noise:  random walk with clustered volatility and fat tails -> there is nothing to find.
regime: each market switches between a TREND state (persistent direction, efficient bars) and a CHOP state
        (mean-reverting, inefficient bars); the state is sticky, so tomorrow's efficiency is predictable and a
        trend rule gated on high predicted efficiency should earn money. The engine must find it.
Each day is simulated as 23 hourly steps to produce open/high/low/close."""
import numpy as np, pandas as pd
from pooled import CLASS


def _day_paths(rng, n, sig, drift_h, revert):
    """n days x 23 hourly log steps; drift_h per hour, revert = pull toward the day's open."""
    steps = rng.standard_t(5, (n, 23)) * sig[:, None] / np.sqrt(23) * 0.8
    x = np.zeros((n, 24))
    for k in range(23):
        x[:, k + 1] = x[:, k] + drift_h + steps[:, k] - revert * x[:, k]
    return x


def make_book(kind, seed=0, start="2010-06-08", end="2021-11-01"):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range(start, end, inclusive="left")
    n = len(days)
    book = {}
    for i, r in enumerate(sorted(CLASS)):
        lv = np.zeros(n); h = np.log(0.01)
        for t in range(1, n):                                                   # clustered volatility
            h = 0.97 * h + 0.03 * np.log(0.01) + rng.normal(0, 0.12)
            lv[t] = h
        sig = np.exp(lv)
        if kind == "noise":
            drift = np.zeros(n); revert = np.zeros(n)
        else:
            state = np.zeros(n, int); d = np.ones(n)
            for t in range(1, n):
                state[t] = state[t - 1] if rng.random() < 0.95 else 1 - state[t - 1]
                d[t] = d[t - 1] if (state[t] == state[t - 1] and rng.random() > 0.02) else rng.choice([-1, 1])
            drift = np.where(state == 1, 0.25 * d * sig / 23, 0.0)              # trend: a quarter sigma a day
            revert = np.where(state == 1, 0.0, 0.15)                             # chop: pulled back to the open
        x = np.zeros((n, 24))
        for t0 in range(0, n, 500):                                              # vectorise by chunks
            sl = slice(t0, min(n, t0 + 500))
            m = sl.stop - sl.start
            steps = rng.standard_t(5, (m, 23)) * sig[sl, None] / np.sqrt(23) * 0.8
            xx = np.zeros((m, 24))
            for k in range(23):
                xx[:, k + 1] = xx[:, k] + drift[sl] + steps[:, k] - revert[sl] * xx[:, k]
            x[sl] = xx
        close = np.cumsum(x[:, -1]) + np.log(100)
        open_ = close - x[:, -1]
        A = pd.DataFrame({"open": np.exp(open_), "high": np.exp(open_ + x.max(1)), "low": np.exp(open_ + x.min(1)),
                          "close": np.exp(close), "volume": rng.integers(1000, 5000, n).astype(float),
                          "contract": r + "Z0"}, index=days)
        A["ret"] = np.log(A["close"]).diff()
        book[r] = A
    return book
