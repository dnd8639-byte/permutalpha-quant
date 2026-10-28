"""Synthetic hourly markets with known answers, for testing the engine itself."""
import numpy as np
import pandas as pd


def regime_market(days=2400, seed=0, drift=0.0012, stay=0.93):
    """Days switch between TREND and CHOP regimes (a hidden two-state Markov chain).
    TREND: price drifts steadily one way, and the way persists through the spell -> high efficiency, trend rules win.
    CHOP:  hourly moves partly reverse, and each day partly undoes the previous one -> low efficiency, trend rules lose.
    Because regimes persist, tomorrow's efficiency is predictable from today's. The right answer: a trend rule,
    switched on only on days predicted to be efficient."""
    rng = np.random.default_rng(seed)
    state, d, rets, last_day = 1, 1, [], 0.0
    for _ in range(days):
        if rng.random() > stay:
            state = 1 - state
            if state == 1:
                d = rng.choice([-1, 1])
        prev, tot = 0.0, 0.0
        for _ in range(24):
            if state == 1:
                r = d * drift + 0.006 * rng.standard_normal()
            else:
                r = -0.5 * prev - 0.6 * last_day / 24 + 0.006 * rng.standard_normal()
            rets.append(r); prev = r; tot += r
        last_day = tot
    return _bars(np.array(rets), rng)


def noise_market(days=2400, seed=0):
    """GARCH-t hourly returns: fat tails and volatility clustering, but nothing to predict. The right answer: no edge."""
    rng = np.random.default_rng(seed)
    n = days * 24; r = np.empty(n); s = 0.006
    for t in range(n):
        r[t] = s * rng.standard_t(5) / np.sqrt(5 / 3)
        s = np.sqrt(3e-6 + 0.06 * r[t] ** 2 + 0.88 * s ** 2)
    return _bars(r, rng)


def _bars(r, rng):
    c = 100 * np.exp(np.cumsum(r))
    o = np.r_[100, c[:-1]]
    w = np.abs(rng.normal(0, 0.002, len(c)))
    idx = pd.date_range("2018-01-01", periods=len(c), freq="h")
    return pd.DataFrame(dict(open=o, high=np.maximum(o, c) * np.exp(w), low=np.minimum(o, c) * np.exp(-w), close=c,
                             volume=rng.lognormal(0, 0.3, len(c))), index=idx)
