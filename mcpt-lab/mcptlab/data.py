"""Loading price data.

Any CSV works if it has a date/time column plus `close` (and ideally open, high, low, volume).
"""
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(HERE), "data")
SAMPLE_BTC = os.path.join(DATA_DIR, "BTCUSDT_1h_2018_2022.csv")


def load_ohlc(path, date_col=None, start=None, end=None):
    """Read a CSV into a DataFrame indexed by time with lowercase open/high/low/close[/volume]."""
    d = pd.read_csv(path)
    d.columns = [c.strip().lower() for c in d.columns]
    date_col = date_col or next(c for c in d.columns if c in ("date", "datetime", "time", "timestamp", "ts"))
    if pd.api.types.is_numeric_dtype(d[date_col]):          # epoch milliseconds or seconds
        unit = "ms" if d[date_col].iloc[0] > 1e11 else "s"
        d[date_col] = pd.to_datetime(d[date_col], unit=unit)
    else:
        d[date_col] = pd.to_datetime(d[date_col])
    cols = [c for c in ("open", "high", "low", "close", "volume") if c in d.columns]
    d = d.set_index(date_col)[cols].sort_index().dropna()
    d = d[~d.index.duplicated()]
    d.index.name = "date"
    return d.loc[start:end]


def load_sample_btc(start=None, end=None):
    """Bundled BTC/USDT 1-hour bars, 2018-01-01 .. 2022-12-31 (from neurotrader888's repos, MIT)."""
    return load_ohlc(SAMPLE_BTC, start=start, end=end)


def synthetic(kind="noise", n=2500, seed=0, strength=0.25, start="2010-01-04"):
    """Test data with a known answer.
    kind="noise" : GARCH-t returns (fat tails, volatility clustering) with NO exploitable order.
    kind="trend" : persistent up/down regimes (average 50 bars) -- a trend strategy SHOULD work."""
    rng = np.random.default_rng(seed)
    r = np.empty(n)
    if kind == "noise":
        s = 0.01
        for t in range(n):
            r[t] = s * rng.standard_t(5) / np.sqrt(5 / 3)
            s = np.sqrt(1e-5 + 0.08 * r[t] ** 2 + 0.9 * s ** 2)
    elif kind == "trend":
        state = 1
        for t in range(n):
            if rng.random() < 1 / 50:
                state = -state
            r[t] = 0.01 * (strength * state + rng.standard_normal())
    else:
        raise ValueError(kind)
    return pd.Series(100 * np.exp(np.cumsum(r)), index=pd.bdate_range(start, periods=n), name="close")
