"""Shared helpers for the mini projects: data loading and plot style. Needs numpy, pandas, matplotlib."""
import os
from statistics import NormalDist

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
N = NormalDist()
INK, MUTED, GRID, A, B, C = "#17212C", "#5A6777", "#D8DEE6", "#2F5D8C", "#B3BDCA", "#A33A3A"


def _find(*parts):
    for base in (os.path.join(HERE, ".."), os.path.join(HERE, "..", "..")):
        p = os.path.join(base, *parts)
        if os.path.exists(p):
            return p
    return None


def btc_hourly():
    """BTC/USDT hourly bars, 2018 -> latest (bundled 2018-2022 file + Binance.US download if present).
    Timestamps are UTC."""
    def load(p):
        d = pd.read_csv(p).dropna()
        d["date"] = pd.to_datetime(d["date"])
        return d.set_index("date").sort_index()[["open", "high", "low", "close", "volume"]]
    a = load(_find("mcpt-lab", "data", "BTCUSDT_1h_2018_2022.csv"))
    f = _find("mcpt-lab", "data", "btc_1h_binanceus.csv")
    if f:
        fresh = load(f).loc["2023-01-01":]
        fresh[["open", "high", "low", "close"]] *= a["close"].iloc[-1] / load(f)["close"].asof(a.index[-1])
        a = pd.concat([a, fresh])
    return a


def futures_daily():
    """Daily returns of CME futures (roll-adjusted), if the licensed file is on this machine; else None."""
    p = _find("state", "v3", "clean_daily.csv")
    if not p:
        return None
    r = pd.read_csv(p, index_col=0, parse_dates=True)
    return r[r.index.dayofweek < 5]


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(labelsize=8, colors=MUTED)


def plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as p
    os.makedirs(FIG, exist_ok=True)
    return p


def french(name):
    """Monthly table from the Kenneth French Data Library (percent -> decimals), index = month end.
    name: 'F-F_Research_Data_Factors', 'F-F_Momentum_Factor', 'F-F_Research_Data_5_Factors_2x3'.
    Files come from  python ../mcpt-lab/data/download_french.py"""
    import io, zipfile
    p = _find("mcpt-lab", "data", "french", name + "_CSV.zip")
    if not p:
        raise FileNotFoundError(f"{name}: run  python ../mcpt-lab/data/download_french.py  first")
    z = zipfile.ZipFile(p)
    lines = z.read(z.namelist()[0]).decode("latin-1").splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith(","))
    rows = []
    for l in lines[start + 1:]:
        parts = [x.strip() for x in l.split(",")]
        if len(parts[0]) != 6 or not parts[0].isdigit():       # monthly block ends at the annual table
            break
        rows.append(parts)
    cols = [c.strip() for c in lines[start].split(",")[1:]]
    d = pd.DataFrame(rows, columns=["m"] + cols)
    d.index = pd.to_datetime(d.pop("m"), format="%Y%m") + pd.offsets.MonthEnd(0)
    d = d.astype(float)
    return d[(d > -99).all(axis=1)] / 100


def ols(y, X, lags=0):
    """OLS with an intercept. Returns (coefficients, t-stats, R^2); coefficient 0 is the intercept.
    lags > 0 uses Newey-West standard errors (robust to autocorrelation and changing volatility)."""
    y = np.asarray(y, float); X = np.column_stack([np.ones(len(y)), np.asarray(X, float)])
    XtXi = np.linalg.inv(X.T @ X)
    b = XtXi @ X.T @ y
    e = y - X @ b
    Xe = X * e[:, None]
    S = Xe.T @ Xe
    for L in range(1, lags + 1):
        G = Xe[L:].T @ Xe[:-L]
        S += (1 - L / (lags + 1)) * (G + G.T)
    se = np.sqrt(np.diag(XtXi @ S @ XtXi) * len(y) / (len(y) - X.shape[1]))
    return b, b / se, 1 - e.var() / y.var()


def nelder_mead(f, x0, iters=2000, tol=1e-10, step=0.25):
    """Small derivative-free minimizer (so the projects need only numpy)."""
    x0 = np.asarray(x0, float); n = len(x0)
    sim = np.vstack([x0] + [x0 + step * np.abs(x0[i] if x0[i] else 1) * np.eye(n)[i] for i in range(n)])
    fs = np.array([f(x) for x in sim])
    for _ in range(iters):
        o = np.argsort(fs); sim, fs = sim[o], fs[o]
        if abs(fs[-1] - fs[0]) < tol * (abs(fs[0]) + tol):
            break
        c = sim[:-1].mean(0)
        xr = c + (c - sim[-1]); fr = f(xr)
        if fr < fs[0]:
            xe = c + 2 * (c - sim[-1]); fe = f(xe)
            sim[-1], fs[-1] = (xe, fe) if fe < fr else (xr, fr)
        elif fr < fs[-2]:
            sim[-1], fs[-1] = xr, fr
        else:
            xc = c + 0.5 * (sim[-1] - c); fc = f(xc)
            if fc < fs[-1]:
                sim[-1], fs[-1] = xc, fc
            else:
                sim[1:] = sim[0] + 0.5 * (sim[1:] - sim[0]); fs[1:] = [f(x) for x in sim[1:]]
    return sim[np.argmin(fs)], fs.min()


def bs_call(S, K, r, T, sigma):
    """Black-Scholes call price (no dividends). Works on arrays."""
    S, K, sigma = np.asarray(S, float), np.asarray(K, float), np.asarray(sigma, float)
    d1 = (np.log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
    cdf = np.vectorize(N.cdf)
    return S * cdf(d1) - K * np.exp(-r * T) * cdf(d1 - sigma * np.sqrt(T))
