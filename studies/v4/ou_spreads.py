"""T8: Kalman hedge ratio + Ornstein-Uhlenbeck mean reversion on economic spreads (PREREGISTRATION_v4).
All signals at close t use data through t; positions set at close t earn day t+1 returns."""
import numpy as np
import pandas as pd

DELTA, BURN, OU_WIN = 1e-4, 252, 252
Z_IN, HL_MIN, HL_MAX, HOLD_MULT = 2.0, 1.0, 63.0, 3.0
VOL_WIN, TARGET, LEV_CAP = 60, 0.10, 5.0

SPREADS = {"S1_crush": ("ZS", ["ZM", "ZL"]), "S2_crack": ("CL", ["HO", "RB"]),
           "S3_curve": ("ZF", ["ZN"]), "S4_wti_brent": ("CL", ["BZ"]), "S5_gold_silver": ("GC", ["SI"])}


def kalman_residual(Ly, Lx):
    """Prior-state residual s_t and prior coefficients theta_{t|t-1} (rows aligned to dates)."""
    T, k = len(Ly), Lx.shape[1]
    H = np.column_stack([np.ones(T), Lx])
    X0, y0 = H[:BURN], Ly[:BURN]
    th = np.linalg.lstsq(X0, y0, rcond=None)[0]
    R = np.var(y0 - X0 @ th, ddof=k + 1)
    P = np.eye(k + 1) * R          # modest initial uncertainty, scaled to the residual variance
    Q = DELTA / (1 - DELTA) * R * np.eye(k + 1)   # scale-free: state drift relative to obs noise (v4-A1)
    s = np.full(T, np.nan); coef = np.full((T, k + 1), np.nan)
    for t in range(BURN, T):
        P = P + Q                                    # predict
        coef[t] = th
        e = Ly[t] - H[t] @ th                        # residual with PRIOR state
        s[t] = e
        S = H[t] @ P @ H[t] + R
        K = P @ H[t] / S
        th = th + K * e                              # update with day t (used from t+1 on)
        P = P - np.outer(K, H[t]) @ P
    return s, coef


def ou_signals(s):
    """Rolling AR(1) on the spread -> z, half-life, tradable flag (all at t, using data through t)."""
    ss = pd.Series(s)
    x, y = ss.shift(1), ss
    n = OU_WIN
    mx, my = x.rolling(n).mean(), y.rolling(n).mean()
    cov = (x * y).rolling(n).mean() - mx * my
    varx = (x * x).rolling(n).mean() - mx ** 2
    b = cov / varx
    a = my - b * mx
    resid_var = ((y - (a + b * x)) ** 2).rolling(n).mean()     # approx residual variance in window
    ok = (b > 0) & (b < 1)
    hl = (-np.log(2) / np.log(b)).where(ok)
    mu = (a / (1 - b)).where(ok)
    sig = (np.sqrt(resid_var) / np.sqrt(1 - b ** 2)).where(ok)
    z = (ss - mu) / sig
    trad = ok & (hl >= HL_MIN) & (hl <= HL_MAX)
    return z.values, hl.values, trad.fillna(False).values


def spread_positions(z, hl, trad):
    """State machine: +1 long spread, -1 short, 0 flat. Decided at close t."""
    T = len(z); pos = np.zeros(T); cur, age, maxage = 0, 0, 0
    for t in range(T):
        if not trad[t] or not np.isfinite(z[t]):
            cur = 0
        elif cur == 0:
            if z[t] >= Z_IN: cur, age, maxage = -1, 0, HOLD_MULT * hl[t]
            elif z[t] <= -Z_IN: cur, age, maxage = 1, 0, HOLD_MULT * hl[t]
        else:
            age += 1
            if (cur == 1 and z[t] >= 0) or (cur == -1 and z[t] <= 0) or age >= maxage:
                cur = 0
        pos[t] = cur
    return pos


def hindsight_residual(Ly, Lx):
    """H12 canary only: full-sample OLS hedge ratio (uses the future)."""
    H = np.column_stack([np.ones(len(Ly)), Lx])
    th = np.linalg.lstsq(H, Ly, rcond=None)[0]
    coef = np.tile(th, (len(Ly), 1)).astype(float)
    return Ly - H @ th, coef


def hindsight_ou(s):
    """H12 canary only: OU parameters fit on the FULL sample (uses the future)."""
    x, y = s[:-1], s[1:]
    b, a = np.polyfit(x, y, 1)
    eps = y - (a + b * x)
    ok = 0 < b < 1
    hl = -np.log(2) / np.log(b) if ok else np.nan
    mu = a / (1 - b) if ok else np.nan
    sig = eps.std() / np.sqrt(1 - b ** 2) if ok else np.nan
    z = (s - mu) / sig
    T = len(s)
    return z, np.full(T, hl), np.full(T, ok and HL_MIN <= hl <= HL_MAX * 100)


def run_spread(Rd, y, xs, bps, leak=False):
    """Returns daily gross P&L, leg notionals (DataFrame), for one spread."""
    cols = [y] + xs
    r = Rd[cols].fillna(0.0)
    L = np.log1p(r).cumsum()
    if leak:
        s, coef = hindsight_residual(L[y].values, L[xs].values)
        z, hl, trad = hindsight_ou(s)
    else:
        s, coef = kalman_residual(L[y].values, L[xs].values)
        z, hl, trad = ou_signals(s)
    p = spread_positions(z, hl, trad)
    # size by ex-ante vol of the UNIT spread (y - beta.x), lagged; same idea as signals.py (v4-A1)
    beta = pd.DataFrame(np.nan_to_num(coef[:, 1:]), index=Rd.index, columns=xs)
    unit = r[y] - (beta.shift(1) * r[xs]).sum(axis=1)
    uvol = unit.rolling(VOL_WIN).std().shift(1) * np.sqrt(252)
    size = (TARGET / uvol).clip(upper=LEV_CAP).fillna(0.0).values
    w = pd.DataFrame(0.0, index=Rd.index, columns=cols)
    w[y] = p * size
    for j, xname in enumerate(xs):
        w[xname] = -p * size * np.nan_to_num(coef[:, j + 1])
    held = w.shift(1).fillna(0.0)
    gross = (held * r).sum(axis=1)
    return gross, held, p, hl


def build_T8(Rd, bps_map, cost_mult=1.0, rolls_m=None, leak=False, spreads=SPREADS):
    streams, legs, info = {}, {}, {}
    for name, (y, xs) in spreads.items():
        if not all(c in Rd.columns for c in [y] + xs):
            continue
        gross, held, p, hl = run_spread(Rd, y, xs, bps_map, leak=leak)
        bps = pd.Series({c: bps_map.get(c, 10.0) * cost_mult for c in held.columns}) / 1e4
        cost = (held.diff().abs().fillna(held.abs()) * bps).sum(axis=1)
        if rolls_m is not None:
            avg_abs = held.abs().resample("ME").mean()
            rc = (avg_abs * rolls_m.reindex(avg_abs.index)[held.columns].fillna(0) * 2 * bps).sum(axis=1)
            last_day = held.index.to_series().groupby(held.index.to_period("M")).max()
            daily_rc = pd.Series(0.0, index=held.index)
            for per, d in last_day.items():
                m = per.to_timestamp("M")
                if m in rc.index:
                    daily_rc.loc[d] = rc.loc[m]
            cost = cost + daily_rc
        streams[name] = gross - cost
        info[name] = dict(in_market=float((p != 0).mean()), trades=int(((np.diff(p) != 0) & (p[1:] != 0)).sum()),
                          med_hl=float(np.nanmedian(hl)))
    S = pd.DataFrame(streams)
    t8 = S.mean(axis=1, skipna=True)          # equal-weight average of risk-sized spreads (v4-A1)
    return t8.dropna(), S, info


def to_monthly(d):
    d = d.dropna()
    return (1 + d).resample("ME").prod() - 1
