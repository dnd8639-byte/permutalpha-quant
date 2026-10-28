"""T9: front-run the commodity index roll with calendar spreads (PREREGISTRATION_v5).
Input per market: DataFrame with columns date, key, close, dmonth (delivery month as Period 'M')."""
import numpy as np
import pandas as pd

TARGET, LEV_CAP, BPS, MIN_OBS = 0.10, 5.0, 5.0, 60
MAIN, PLACEBO = (1, 9), (12, 20)


def bd_table(dates):
    """Map each trading date -> (month period, business-day number within month)."""
    d = pd.Series(pd.DatetimeIndex(sorted(set(dates))))
    per = d.dt.to_period("M")
    return pd.DataFrame({"date": d, "per": per, "bd": d.groupby(per).cumcount() + 1})


def window_spread(close, keyN, keyD, entry_day, days, look_ahead=False):
    """Daily spread returns r_D - r_N on `days`, same-contract close-to-close from entry_day's close."""
    # include any stub dates between entry and window end for price continuity (returns are close-to-close)
    span = close.loc[entry_day:days[-1], [keyN, keyD]].ffill()
    sub = span.loc[[entry_day] + list(days)]
    if sub.iloc[0].isna().any():
        return None
    r = sub.pct_change().iloc[1:].fillna(0.0)
    return (r[keyD] - r[keyN])


def run_T9(markets, calendar, window=MAIN, bps=BPS, direction=None, choose_by_outcome=False):
    """markets: {name: df(date,key,close,dmonth)}. Returns monthly T9, per-market dict, diagnostics."""
    bt = bd_table(calendar)
    per_mkt, split = {}, {}
    for name, df in markets.items():
        close = df.pivot_table(index="date", columns="key", values="close", aggfunc="last").sort_index()
        close = close.reindex(sorted(set(close.index) | set(bt["date"])))
        dmon = df.groupby("key")["dmonth"].first()
        hist, rows, parts = [], {}, {}
        for per, g in bt.groupby("per"):
            wd = g[(g.bd >= window[0]) & (g.bd <= window[1])]["date"].tolist()
            if len(wd) < window[1] - window[0] + 1:
                continue
            if window[0] == 1:
                prev = bt[bt.per == per - 1]["date"]
                if prev.empty:
                    continue
                entry = prev.iloc[-1]
            else:
                entry = g[g.bd == window[0] - 1]["date"].iloc[0]
            kN = dmon[dmon == per + 1].index; kD = dmon[dmon == per + 2].index
            if len(kN) == 0 or len(kD) == 0:
                continue
            # if a delivery month maps to several keys, take the one with most closes in the window
            pick = lambda ks: max(ks, key=lambda k: close.loc[wd, k].notna().sum())
            sp = window_spread(close, pick(kN), pick(kD), entry, wd)
            if sp is None or sp.abs().sum() == 0:
                continue
            # sizing from PRIOR 12 months of in-window spread returns
            past = [x for (p, x) in hist if per - 12 <= p < per]
            obs = np.concatenate(past) if past else np.array([])
            hist.append((per, sp.values))
            if len(obs) < MIN_OBS or obs.std() == 0:
                continue
            s = min(TARGET / (obs.std(ddof=1) * np.sqrt(252)), LEV_CAP)
            sign = 1.0
            if choose_by_outcome:                     # H15 canary: direction from the realized outcome
                sign = 1.0 if sp.sum() >= 0 else -1.0
            gross = np.prod(1 + sign * s * sp.values) - 1
            cost = 2 * 2 * bps / 1e4 * s
            rows[per.to_timestamp("M")] = gross - cost
            if window == MAIN:
                parts[per.to_timestamp("M")] = (s * sp.iloc[:4].sum(), s * sp.iloc[4:].sum())
        per_mkt[name] = pd.Series(rows, dtype=float)
        split[name] = pd.DataFrame(parts, index=["bd1_4", "bd5_9"]).T if parts else pd.DataFrame()
    M = pd.DataFrame(per_mkt)
    return M.mean(axis=1, skipna=True).dropna(), M, split
