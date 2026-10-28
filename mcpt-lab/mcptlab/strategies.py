"""Strategy definitions.

A strategy is: a parameter grid + a function that turns prices into a position (-1, 0, +1 or
fractional) at each bar, using ONLY data up to and including that bar. The framework applies
the position to the NEXT bar's return, so the signal function never needs to shift anything.

To add your own strategy: subclass Strategy, set `name` and `grid`, write `positions`. That's it.
Optional: `bars_per_year` (default 252) so Sharpe ratios are annualized correctly.
"""
import numpy as np
import pandas as pd


class Strategy:
    name = "base"
    grid = []                                     # list of parameter values (any hashable)

    bars_per_year = 252

    def positions(self, data, param) -> pd.Series:
        """data: a close-price Series or an OHLC DataFrame. Return a position per bar
        (+1 long, -1 short, 0 flat, or fractional) using only data up to that bar."""
        raise NotImplementedError

    def __repr__(self):
        return f"{type(self).__name__}(grid of {len(self.grid)})"


class DonchianBreakout(Strategy):
    """neurotrader's example: long when close breaks the prior N-bar high, short below the N-bar low."""
    name = "donchian"

    def __init__(self, lookbacks=range(10, 201, 5)):
        self.grid = list(lookbacks)

    def positions(self, data, lookback):
        close = data["close"] if isinstance(data, pd.DataFrame) else data
        upper = close.rolling(lookback - 1).max().shift(1)
        lower = close.rolling(lookback - 1).min().shift(1)
        sig = pd.Series(np.nan, index=close.index)
        sig[close > upper] = 1.0
        sig[close < lower] = -1.0
        return sig.ffill().fillna(0.0)


class MovingAverageCross(Strategy):
    """Long when the fast average is above the slow one, short otherwise."""
    name = "ma_cross"

    def __init__(self, fast=(5, 10, 20, 50), slow=(50, 100, 150, 200)):
        self.grid = [(f, s) for f in fast for s in slow if f < s]

    def positions(self, data, param):
        close = data["close"] if isinstance(data, pd.DataFrame) else data
        f, s = param
        return np.sign(close.rolling(f).mean() - close.rolling(s).mean()).fillna(0.0)


# ----------------------------------------------------------------------------- neurotrader VolatilityHawkes
def wilder_atr(high, low, close, length):
    """ATR as in pandas_ta (true range, Wilder RMA = ewm(alpha=1/length, min_periods=length))."""
    prev = close.shift(1)
    tr = pd.concat([high - low, (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    tr.iloc[0] = np.nan
    return tr.ewm(alpha=1 / length, min_periods=length).mean()


def hawkes(x, kappa):
    """y_t = y_{t-1} * exp(-kappa) + x_t, started at the first valid value; output * kappa."""
    a = np.exp(-kappa)
    arr = np.asarray(x, float); out = np.full(len(arr), np.nan)
    valid = np.where(~np.isnan(arr))[0]
    if len(valid):
        i0 = valid[0]
        try:
            from scipy.signal import lfilter
            out[i0:] = lfilter([1.0], [1.0, -a], arr[i0:])
        except ImportError:                                  # no scipy (e.g. on the Pi): plain loop
            acc = 0.0
            for i in range(i0, len(arr)):
                acc = acc * a + arr[i]; out[i] = acc
    return out * kappa


class HawkesVolatility(Strategy):
    """neurotrader888/VolatilityHawkes (MIT). Volatility 'excitement' (Hawkes process on ATR-normalized
    bar ranges). When it jumps from below its rolling 5th percentile to above its 95th, trade in the
    direction price moved since the quiet point; go flat when it falls back below the 5th percentile.
    Grid = the author's own heatmap: kappa x threshold lookback."""
    name = "hawkes_vol"
    bars_per_year = 24 * 365
    NORM_LOOKBACK = 336                                      # author's fixed value (not optimized)

    def __init__(self, kappas=(0.5, 0.25, 0.1, 0.05, 0.01), lookbacks=(24, 48, 96, 168, 336)):
        self.grid = [(k, lb) for k in kappas for lb in lookbacks]
        self._cache_key, self._norm = None, None

    def _norm_range(self, ohlc):
        key = (id(ohlc), len(ohlc), float(ohlc["close"].iloc[-1]))
        if key != self._cache_key:
            lh, ll, lc = np.log(ohlc["high"]), np.log(ohlc["low"]), np.log(ohlc["close"])
            self._norm = ((lh - ll) / wilder_atr(lh, ll, lc, self.NORM_LOOKBACK)).to_numpy()
            self._cache_key = key
        return self._norm

    def positions(self, ohlc, param):
        kappa, lookback = param
        vh = hawkes(self._norm_range(ohlc), kappa)
        s = pd.Series(vh)
        q05 = s.rolling(lookback).quantile(0.05).to_numpy()
        q95 = s.rolling(lookback).quantile(0.95).to_numpy()
        close = ohlc["close"].to_numpy()
        sig = np.zeros(len(close)); last_below, cur = -1, 0.0
        for i in range(len(close)):
            if vh[i] < q05[i]:
                last_below, cur = i, 0.0
            if vh[i] > q95[i] and vh[i - 1] <= q95[i - 1] and last_below > 0:
                cur = 1.0 if close[i] - close[last_below] > 0 else -1.0
            sig[i] = cur
        return pd.Series(sig, index=ohlc.index)
