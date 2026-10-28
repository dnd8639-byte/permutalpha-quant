import numpy as np
import pandas as pd

N_LAGS         = 4
SIGNAL_CAP     = 2.0
TARGET_VOL     = 0.10
HL_FAST        = 6
HL_SLOW        = 24
VOL_FLOOR_MULT = 0.5
ANN            = 12


def realized_vol(returns):
    fast = returns.ewm(halflife=HL_FAST, min_periods=HL_FAST).std()
    slow = returns.ewm(halflife=HL_SLOW, min_periods=HL_SLOW).std()
    return np.maximum(fast, VOL_FLOOR_MULT * slow)


def momentum_signal(returns, vol):
    w = np.full(N_LAGS, 1.0 / N_LAGS)
    norm = np.sqrt((w ** 2).sum())
    stacked = sum(w[i] * returns.shift(i + 1) for i in range(N_LAGS))
    sig = stacked / (vol * norm)
    return sig.clip(-SIGNAL_CAP, SIGNAL_CAP)


def raw_positions(signal, vol):
    return signal / vol.replace(0, np.nan)


def portfolio_gearing(weights, corr, vol, target=TARGET_VOL):
    w = weights.fillna(0.0).values
    s = vol.fillna(vol.median()).values
    cov = corr.values * np.outer(s, s)
    ex_ante = np.sqrt(max(w @ cov @ w, 1e-12)) * np.sqrt(ANN)
    return float(np.clip(target / ex_ante, 0.0, 5.0))
