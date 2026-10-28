"""edge-engine core: daily efficiency ratio, features, the ER model, strategies, and execution with costs.

Time convention (no look-ahead):
  * Each UTC day d has hourly bars. Its efficiency ratio is ER_d = |net move| / (sum of |hourly moves|), in log prices.
  * Features for day d use only data up to the END of day d. The model predicts ER_{d+1}.
  * A strategy's position for day d+1 is decided at the end of day d and filled `delay_h` hours into day d+1,
    at that hour's close, plus slippage. P&L is computed hour by hour from the fill onward.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MIN_BARS = 18  # a day needs at least this many hourly bars to have a meaningful ER


# ---------------------------------------------------------------------------------------------- data
def load_hourly(path):
    """CSV with a date/time column (UTC) and open, high, low, close[, volume]."""
    d = pd.read_csv(path)
    d.columns = [c.strip().lower() for c in d.columns]
    tcol = next(c for c in d.columns if c in ("date", "datetime", "time", "timestamp", "ts"))
    if pd.api.types.is_numeric_dtype(d[tcol]):
        d[tcol] = pd.to_datetime(d[tcol], unit="ms" if d[tcol].iloc[0] > 1e11 else "s")
    else:
        d[tcol] = pd.to_datetime(d[tcol], utc=True).dt.tz_localize(None)
    cols = [c for c in ("open", "high", "low", "close", "volume") if c in d.columns]
    d = d.set_index(tcol)[cols].sort_index()
    d = d[~d.index.duplicated()].dropna(subset=["close"])
    d.index.name = "date"
    return d[d["close"] > 0]


def daily_table(h):
    """One row per UTC day: open/close, ER, realized variance, range, bar count, volume."""
    lc = np.log(h["close"])
    day = h.index.floor("D")
    step = lc.diff()
    step[day != pd.Series(day, index=h.index).shift(1).values] = np.nan   # first bar of each day: no in-day step
    g = pd.DataFrame({"lc": lc, "abs": step.abs(), "sq": step ** 2, "hi": np.log(h["high"]), "lo": np.log(h["low"]),
                      "vol": h["volume"] if "volume" in h else 0.0}, index=h.index).groupby(day)
    D = pd.DataFrame({"first": g["lc"].first(), "last": g["lc"].last(), "path": g["abs"].sum(), "rv": g["sq"].sum(),
                      "hi": g["hi"].max(), "lo": g["lo"].min(), "n": g["lc"].count(), "volume": g["vol"].sum()})
    D = D[D["n"] >= MIN_BARS]
    D["net"] = D["last"] - D["first"]
    D["er"] = (D["net"].abs() / D["path"]).where(D["path"] > 0, 0.0).clip(0, 1)
    D["ret"] = D["last"].diff()                                            # close-to-close daily log return
    return D


# ---------------------------------------------------------------------------------------------- features
FEATURES = ["er", "er_5", "er_20", "absret", "rv_ratio", "range_ratio", "vol_ratio", "dow_next", "trend_20", "event_next"]


def features(D, event_days=()):
    """Everything here is known at the end of day d. Target column `er_next` = ER of day d+1."""
    F = pd.DataFrame(index=D.index)
    F["er"] = D["er"]
    F["er_5"] = D["er"].rolling(5).mean()
    F["er_20"] = D["er"].rolling(20).mean()
    F["absret"] = D["net"].abs() / np.sqrt(D["rv"].rolling(20).mean())
    F["rv_ratio"] = np.log(D["rv"] / D["rv"].rolling(20).mean())
    F["range_ratio"] = (D["hi"] - D["lo"]) / (D["hi"] - D["lo"]).rolling(20).mean()
    F["vol_ratio"] = np.log((D["volume"] + 1) / (D["volume"].rolling(20).mean() + 1))
    nxt = D.index.to_series().shift(-1).fillna(D.index[-1] + pd.Timedelta(days=1))
    F["dow_next"] = nxt.dt.dayofweek.values
    F["trend_20"] = (D["last"] - D["last"].shift(20)) / np.sqrt(D["rv"].rolling(20).sum())
    ev = set(pd.to_datetime(list(event_days)).normalize()) if len(event_days) else set()
    F["event_next"] = [1.0 if t in ev else 0.0 for t in nxt.dt.normalize()]   # scheduled release tomorrow (known in advance)
    F["er_next"] = D["er"].shift(-1)
    return F.replace([np.inf, -np.inf], np.nan)


# ---------------------------------------------------------------------------------------------- ER model
class ERModel:
    """Predicts tomorrow's efficiency ratio. Gradient-boosted trees if scikit-learn is installed, else ridge regression.
    Always reported next to two baselines: 'persistence' (tomorrow = today's 20-day mean ER) and the training mean."""

    def __init__(self, seed=0):
        self.seed = seed

    def fit(self, F):
        d = F.dropna()
        X, y = d[FEATURES].to_numpy(float), d["er_next"].to_numpy(float)
        try:
            from sklearn.ensemble import HistGradientBoostingRegressor
            self.m = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=200, min_samples_leaf=40,
                                                   l2_regularization=1.0, random_state=self.seed).fit(X, y)
            self.kind = "gbm"
        except ImportError:
            self.mu, self.sd = X.mean(0), X.std(0) + 1e-12
            Z = np.column_stack([np.ones(len(X)), (X - self.mu) / self.sd])
            self.w = np.linalg.solve(Z.T @ Z + 5.0 * np.eye(Z.shape[1]), Z.T @ y)
            self.kind = "ridge"
        fitted = self.predict(d)
        self.cuts = np.quantile(fitted, [1 / 3, 2 / 3])             # tercile cut-offs from TRAINING predictions only
        return self

    def predict(self, F):
        X = F[FEATURES].to_numpy(float)
        ok = ~np.isnan(X).any(1)
        out = np.full(len(X), np.nan)
        if ok.any():
            if self.kind == "gbm":
                out[ok] = self.m.predict(X[ok])
            else:
                out[ok] = np.column_stack([np.ones(ok.sum()), (X[ok] - self.mu) / self.sd]) @ self.w
        return out

    def tier(self, pred):
        """0 = low, 1 = medium, 2 = high predicted efficiency; -1 where there is no prediction."""
        t = np.where(pred < self.cuts[0], 0, np.where(pred < self.cuts[1], 1, 2))
        return np.where(np.isnan(pred), -1, t)


# ---------------------------------------------------------------------------------------------- strategies
RULES = {
    "breakout": dict(lookback=[5, 10, 20, 40, 60]),
    "ma_cross": dict(fast=[3, 5, 10, 20], slow=[20, 40, 60, 100]),
    "momentum": dict(lookback=[1, 3, 5, 10, 20, 60]),
    "reversal": dict(lookback=[1, 2, 3, 5]),
}
GATES = {"all": (0, 1, 2), "mid+high": (1, 2), "high": (2,), "low": (0,), "low+mid": (0, 1)}
SIDES = ("both", "long", "short")


def raw_signal(D, rule, p):
    """Direction for day d+1 decided from closes up to day d: +1, -1 or 0 (array aligned with D)."""
    c = D["last"]
    if rule == "breakout":
        hi, lo = c.rolling(p["lookback"]).max().shift(1), c.rolling(p["lookback"]).min().shift(1)
        s = pd.Series(np.where(c > hi, 1.0, np.where(c < lo, -1.0, np.nan)), index=D.index).ffill().fillna(0.0)
    elif rule == "ma_cross":
        s = np.sign(c.rolling(p["fast"]).mean() - c.rolling(p["slow"]).mean()).fillna(0.0)
    elif rule == "momentum":
        s = np.sign(c - c.shift(p["lookback"])).fillna(0.0)
    elif rule == "reversal":
        s = -np.sign(c - c.shift(p["lookback"])).fillna(0.0)
    else:
        raise ValueError(rule)
    return s.to_numpy(float)


def positions(D, genome, tiers):
    """Daily target positions (decided at the end of each day, for the next day)."""
    s = raw_signal(D, genome["rule"], genome["params"])
    if genome["side"] == "long":
        s = np.maximum(s, 0)
    elif genome["side"] == "short":
        s = np.minimum(s, 0)
    allowed = np.isin(tiers, GATES[genome["gate"]])
    return np.where(allowed, s, 0.0)


# ---------------------------------------------------------------------------------------------- execution
class Market:
    """Hourly + daily data for one market, with execution assumptions."""

    def __init__(self, name, hourly, cost_bps=5.0, delay_h=1, event_days=()):
        self.name, self.h = name, hourly
        self.D = daily_table(hourly)
        self.F = features(self.D, event_days)
        self.cost_bps, self.delay_h = cost_bps, delay_h
        self.lc = np.log(hourly["close"].to_numpy())
        self.hday = hourly.index.floor("D")
        self.day_pos_index = self.D.index                            # days with a decision


def execute(mkt, day_pos, days, cost_bps=None, delay_h=None, slip_draw=None):
    """Hour-by-hour P&L of holding `day_pos[i]` from `delay_h` hours into the day after days[i].
    Returns a daily Series of net log returns (indexed by the day the return accrues in)."""
    cost_bps = mkt.cost_bps if cost_bps is None else cost_bps
    delay_h = mkt.delay_h if delay_h is None else delay_h
    h = mkt.h.index
    start = days + pd.Timedelta(days=1) + pd.Timedelta(hours=delay_h)          # fill time for each decision
    idx = h.searchsorted(start)                                                  # first hourly bar at/after fill time
    pos_h = np.zeros(len(h))
    marks = np.full(len(h), np.nan)
    ok = idx < len(h)
    marks[idx[ok]] = np.asarray(day_pos)[ok]
    pos_h = pd.Series(marks).ffill().fillna(0.0).to_numpy()
    r = np.diff(mkt.lc, append=mkt.lc[-1])                                       # return from bar t to t+1
    dpos = np.abs(np.diff(pos_h, prepend=0.0))
    per_unit = cost_bps / 1e4 if slip_draw is None else slip_draw(len(h)) / 1e4
    net = pos_h * r - dpos * per_unit
    out = pd.Series(net, index=mkt.hday).groupby(level=0).sum()
    lo, hi = days[0] + pd.Timedelta(days=1), days[-1] + pd.Timedelta(days=1)
    return out.loc[lo:hi]


def sharpe(x, per_year=365):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return float(x.mean() / x.std() * np.sqrt(per_year)) if len(x) > 2 and x.std() > 0 else 0.0
