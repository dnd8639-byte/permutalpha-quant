"""Publication decay: do published strategies earn less after their paper appears?

Design after McLean & Pontiff (2016, J. Finance): each strategy's monthly returns are split into
    IS    the authors' own sample
    OOS   after the sample ended, before publication   (decline here = statistical bias only)
    POST  after journal publication                     (decline here = bias + arbitrage)
and a pooled regression of IS-scaled returns on OOS/POST dummies measures the average decay.
Preregistration: PREREGISTRATION_v6.md.  Data: Kenneth R. French Data Library (data/download_french.py).
"""
from __future__ import annotations

import math
import os
import zipfile
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import DATA_DIR

FRENCH_DIR = os.path.join(DATA_DIR, "french")
FRENCH_FILES = {
    "ff3_monthly": "F-F_Research_Data_Factors_CSV.zip",
    "ff3_daily": "F-F_Research_Data_Factors_daily_CSV.zip",
    "mom_monthly": "F-F_Momentum_Factor_CSV.zip",
    "strev_monthly": "F-F_ST_Reversal_Factor_CSV.zip",
    "ff5_monthly": "F-F_Research_Data_5_Factors_2x3_CSV.zip",
}


# ============================================================================ French data
def parse_french(raw: str) -> pd.DataFrame:
    """First table of a French-library CSV (monthly YYYYMM or daily YYYYMMDD), in decimals.
    Later tables (annual averages etc.) are ignored."""
    lines = raw.splitlines()
    start = next(i for i, l in enumerate(lines)
                 if l.strip().startswith(",") and any(c.isalpha() for c in l))
    header = [c.strip() for c in lines[start].split(",")]
    rows = []
    for l in lines[start + 1:]:
        parts = [p.strip() for p in l.split(",")]
        if not parts[0].isdigit() or len(parts[0]) not in (6, 8):
            break
        rows.append(parts)
    df = pd.DataFrame(rows, columns=["date"] + header[1:])
    fmt = "%Y%m" if len(df["date"].iloc[0]) == 6 else "%Y%m%d"
    df["date"] = pd.to_datetime(df["date"], format=fmt)
    if fmt == "%Y%m":
        df["date"] = df["date"].dt.to_period("M")
    df = df.set_index("date").apply(pd.to_numeric, errors="coerce")
    df = df.mask(df <= -99.99)
    return df / 100.0


def load_french(key: str, folder: str = FRENCH_DIR) -> pd.DataFrame:
    path = os.path.join(folder, FRENCH_FILES[key])
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} missing -- run: python data/download_french.py")
    with zipfile.ZipFile(path) as z:
        raw = z.read(z.namelist()[0]).decode("latin-1")
    return parse_french(raw)


# ============================================================================ calendar strategies
def monday_effect(daily_excess: pd.Series) -> pd.Series:
    """French (1980). Per month: -(sum of Monday excess returns - n_Monday * mean of other days).
    Positive = Mondays did worse than the month's other days."""
    d = daily_excess.dropna()
    df = pd.DataFrame({"r": d.values, "mon": d.index.dayofweek == 0}, index=d.index)
    out = {}
    for m, g in df.groupby(d.index.to_period("M")):
        mon, other = g.loc[g["mon"], "r"], g.loc[~g["mon"], "r"]
        if len(mon) == 0 or len(other) == 0:
            continue
        out[m] = -(mon.sum() - len(mon) * other.mean())
    return pd.Series(out, name="monday").sort_index()


def turn_of_month(daily_excess: pd.Series, first_days: int = 3) -> pd.Series:
    """Lakonishok & Smidt (1988). TOM(m) = last trading day of m-1 + first 3 trading days of m.
    Per month: sum of TOM excess returns - 4 * mean of m's non-TOM days
    (m's own last trading day belongs to m+1's TOM, so it is excluded from both)."""
    d = daily_excess.dropna()
    months = d.index.to_period("M")
    groups = {m: g for m, g in d.groupby(months)}
    keys = sorted(groups)
    out = {}
    for prev, m in zip(keys[:-1], keys[1:]):
        g = groups[m]
        tom = pd.concat([groups[prev].iloc[-1:], g.iloc[:first_days]])
        normal = g.iloc[first_days:-1]
        if len(normal) < 5:
            continue
        out[m] = tom.sum() - len(tom) * normal.mean()
    return pd.Series(out, name="tom").sort_index()


def halloween(monthly_excess: pd.Series) -> pd.Series:
    """Bouman & Jacobsen (2002): +market excess in Nov-Apr, -market excess in May-Oct."""
    m = monthly_excess.dropna()
    winter = m.index.month.isin([11, 12, 1, 2, 3, 4])
    return pd.Series(np.where(winter, m.values, -m.values), index=m.index, name="halloween")


def january_only(monthly: pd.Series) -> pd.Series:
    """Keim (1983): hold the series in January only, 0 otherwise."""
    m = monthly.dropna()
    return m.where(m.index.month == 1, 0.0).rename(f"{m.name}_jan")


# ============================================================================ Faber timing
def faber_timing(mkt_total: pd.Series, rf: pd.Series, months: int = 10, cost: float = 0.001):
    """Faber (2007). At month end t: hold the market in t+1 if TR_t > mean(TR_{t-9..t}), else T-bills.
    Returns a DataFrame with signal (decided at t, applied at t+1), timed and buy-and-hold returns."""
    tr = (1 + mkt_total).cumprod()
    sig = (tr > tr.rolling(months).mean()).astype(float)
    sig[tr.rolling(months).mean().isna()] = np.nan
    pos = sig.shift(1)                                      # decided at t, earned in t+1
    switch = pos.diff().abs().fillna(0.0)
    timed = pos * mkt_total + (1 - pos) * rf - switch * cost
    return pd.DataFrame({"signal": sig, "pos": pos, "timed": timed, "bh": mkt_total, "rf": rf}).dropna()


def max_drawdown(rets: pd.Series) -> float:
    w = (1 + rets).cumprod()
    return float((w / w.cummax() - 1).min())


# ============================================================================ windows + regression
@dataclass
class Published:
    key: str
    label: str
    paper: str
    is_start: str       # "YYYY-MM"
    is_end: str
    pub: str            # journal publication month; POST starts the month after
    pooled: bool = True


def windows(r: pd.Series, spec: Published, post_shift_months: int = 0) -> pd.Series:
    """Label each month IS / OOS / POST (pre-sample months -> dropped)."""
    idx = r.index
    is0, is1 = pd.Period(spec.is_start, "M"), pd.Period(spec.is_end, "M")
    post0 = max(pd.Period(spec.pub, "M") + 1 - post_shift_months, is1 + 1)   # POST never overlaps IS
    lab = pd.Series(np.nan, index=idx, dtype=object)
    lab[(idx >= is0) & (idx <= is1)] = "IS"
    lab[(idx > is1) & (idx < post0)] = "OOS"
    lab[idx >= post0] = "POST"
    return lab


def window_stats(r: pd.Series, lab: pd.Series, periods_per_year=12) -> pd.DataFrame:
    rows = {}
    for w in ("IS", "OOS", "POST"):
        x = r[lab == w].dropna()
        n = len(x)
        mu, sd = (x.mean(), x.std()) if n > 1 else (np.nan, np.nan)
        rows[w] = dict(start=str(x.index[0]) if n else "", end=str(x.index[-1]) if n else "", n=n,
                       mean_pct=100 * mu, t=mu / sd * math.sqrt(n) if n > 1 and sd > 0 else np.nan,
                       sharpe=mu / sd * math.sqrt(periods_per_year) if n > 1 and sd > 0 else np.nan)
    return pd.DataFrame(rows).T


def norm_sf(z):
    """P(Z > z) for a standard normal (scipy-free)."""
    return 0.5 * math.erfc(z / math.sqrt(2))


def pooled_decay(series: dict, labels: dict):
    """McLean-Pontiff regression: y_it = r_it / mu_IS,i = a + b_OOS*OOS + b_POST*POST + e,
    standard errors clustered by calendar month. Returns coefficients, clustered SEs, one-sided p's."""
    ys, Xs, cl = [], [], []
    for k, r in series.items():
        lab = labels[k].reindex(r.index)
        ok = lab.notna() & r.notna()
        r, lab = r[ok], lab[ok]
        mu = r[lab == "IS"].mean()
        ys.append((r / mu).to_numpy())
        Xs.append(np.column_stack([np.ones(len(r)), (lab == "OOS").to_numpy(float), (lab == "POST").to_numpy(float)]))
        cl.append(np.array([str(p) for p in r.index]))
    y, X, g = np.concatenate(ys), np.vstack(Xs), np.concatenate(cl)
    if not X[:, 1].any():                     # no OOS months anywhere (possible when POST is shifted)
        X = np.column_stack([X[:, 0], X[:, 2]])
        return {**_fit(y, X, g, ["a", "b_post"]), "b_oos": np.nan, "se_oos": np.nan, "p_oos": np.nan,
                "b_post_minus_oos": np.nan, "se_diff": np.nan, "p_diff": np.nan, "n_strategies": len(series)}
    XtX_inv = np.linalg.inv(X.T @ X)
    b = XtX_inv @ X.T @ y
    u = y - X @ b
    meat = np.zeros((3, 3))
    uniq, inv = np.unique(g, return_inverse=True)
    S = np.zeros((len(uniq), 3))
    np.add.at(S, inv, X * u[:, None])
    meat = S.T @ S
    G, n, k = len(uniq), len(y), 3
    V = XtX_inv @ meat @ XtX_inv * (G / (G - 1)) * ((n - 1) / (n - k))
    se = np.sqrt(np.diag(V))
    diff = b[2] - b[1]
    se_diff = math.sqrt(V[2, 2] + V[1, 1] - 2 * V[1, 2])
    return dict(a=b[0], b_oos=b[1], b_post=b[2], se_oos=se[1], se_post=se[2],
                p_post=norm_sf(-b[2] / se[2]), p_oos=norm_sf(-b[1] / se[1]),
                b_post_minus_oos=diff, se_diff=se_diff, p_diff=norm_sf(-diff / se_diff),
                n_obs=n, n_months=G, n_strategies=len(series))


def _fit(y, X, g, names):
    """OLS with month-clustered SEs, for the no-OOS case."""
    XtX_inv = np.linalg.inv(X.T @ X)
    b = XtX_inv @ X.T @ y
    u = y - X @ b
    uniq, inv = np.unique(g, return_inverse=True)
    S = np.zeros((len(uniq), X.shape[1]))
    np.add.at(S, inv, X * u[:, None])
    G, n, k = len(uniq), len(y), X.shape[1]
    V = XtX_inv @ (S.T @ S) @ XtX_inv * (G / (G - 1)) * ((n - 1) / (n - k))
    se = np.sqrt(np.diag(V))
    out = dict(zip(names, b))
    out.update(se_post=se[-1], p_post=norm_sf(-b[-1] / se[-1]), n_obs=n, n_months=G)
    return out


# ============================================================================ sign-flip null for return series
def signflip_series(r, start_index=0, seed=None):
    """Null for 'is the mean > 0?': each return keeps its size, gets a random sign."""
    rng = np.random.default_rng(seed)
    s = np.where(rng.random(len(r)) < 0.5, -1.0, 1.0)
    s[:start_index] = 1.0
    return r * s


def mean_stat(r):
    return float(np.mean(r))


# ============================================================================ trading costs (PREREGISTRATION_v7.md)
# Novy-Marx & Velikov (2016, RFS) Table 3: turnover per side (%/mo) and measured cost (%/mo), 1963-2013.
NMV = {"size": (1.23, 0.04), "value": (2.91, 0.05), "profitability": (1.96, 0.03),
       "momentum": (34.52, 0.65), "reversal": (90.87, 1.65)}
SIZE_LEG_COST = NMV["size"][1] / (2 * NMV["size"][0])        # 1.63% one-way per unit of size-decile trading
MARKET_ERAS = [("1975-04", 0.0100), ("1982-04", 0.0050), ("2000-12", 0.0005), ("9999-12", 0.0002)]
FACTOR_ERA_SCALE = [("1982-04", 1.5), ("2000-12", 1.0), ("9999-12", 0.5)]


def _by_era(idx, eras):
    out = np.empty(len(idx))
    for i, p in enumerate(idx):
        out[i] = next(v for end, v in eras if p <= pd.Period(end, "M"))
    return pd.Series(out, index=idx)


def market_cost(idx):
    """One-way cost per unit of trading the whole market, by era (basket -> futures -> ETF)."""
    return _by_era(idx, MARKET_ERAS)


def factor_era_scale(idx):
    return _by_era(idx, FACTOR_ERA_SCALE)


def monthly_units(key, idx, daily_index=None, faber_pos=None):
    """Units of one-way trading per month for each strategy (1 unit = 100% of notional)."""
    if key in NMV:
        return pd.Series(2 * NMV[key][0] / 100, index=idx)
    if key == "january":
        return pd.Series(np.where(idx.month == 1, 4.0, 0.0), index=idx)
    if key == "monday":
        d = pd.Series(daily_index.dayofweek == 0, index=daily_index)
        return (2.0 * d.groupby(daily_index.to_period("M")).sum()).reindex(idx).fillna(0.0)
    if key == "tom":
        return pd.Series(2.0, index=idx)
    if key == "halloween":
        return pd.Series(np.where(np.isin(idx.month, [5, 11]), 2.0, 0.0), index=idx)
    if key == "faber":
        return faber_pos.diff().abs().fillna(0.0).reindex(idx)
    raise KeyError(key)


def monthly_cost(key, idx, mult=1.0, era_scaled=False, **kw):
    """Cost in return units per month. Factors: NMV measured cost; january: size-leg rate;
    market-timing strategies: market_cost by era."""
    units = monthly_units(key, idx, **kw)
    if key in NMV:
        c = pd.Series(NMV[key][1] / 100, index=idx)
    elif key == "january":
        c = units * SIZE_LEG_COST
    else:
        c = units * market_cost(idx)
    if era_scaled and (key in NMV or key == "january"):
        c = c * factor_era_scale(idx)
    return mult * c
