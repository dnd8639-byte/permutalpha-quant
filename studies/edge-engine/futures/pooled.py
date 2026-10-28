"""Pooled daily-bar efficiency data set for the 35 futures + the pooled ER model study.

Bar efficiency  ER_d = |close - open| / (high - low)   (validated against the hourly-path ER: rank corr ~0.95)
Everything in the features for day d is known at the close of day d; the target is ER_{d+1}.
The 70/30 split is by calendar date across ALL markets; nothing here ever reads a date after the split
unless phase='holdout'.
"""
import glob, os
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CLASS = {**{k: "fx" for k in ["6A", "6B", "6C", "6E", "6J", "6M", "6S"]},
         **{k: "energy" for k in ["BZ", "CL", "HO", "NG", "RB"]},
         **{k: "metal" for k in ["GC", "SI", "HG", "PA", "PL"]},
         **{k: "equity" for k in ["ES", "NQ", "YM", "RTY", "NKD"]},
         **{k: "rates" for k in ["ZT", "ZF", "ZN", "ZB", "UB"]},
         **{k: "ags" for k in ["ZC", "ZS", "ZW", "ZL", "ZM", "ZO"]},
         **{k: "meat" for k in ["HE", "LE"]}}
CLASSES = sorted(set(CLASS.values()))
EVENTS = ["fomc", "cpi", "nfp", "eia"]
FEATURES = (["er", "er_5", "er_20", "move_ratio", "range_ratio", "range_ratio_5", "nr7", "clv", "trend_20", "vol_ratio",
             "dow_next"] + [f"{e}_next" for e in EVENTS] + [f"is_{c}" for c in CLASSES])


def load_daily(root):
    D = pd.read_csv(os.path.join(HERE, "daily", f"{root}.csv"), index_col=0, parse_dates=True)
    return D


def features(D, root, events):
    rng = (D["high"] - D["low"])
    F = pd.DataFrame(index=D.index)
    F["er"] = ((D["close"] - D["open"]).abs() / rng).where(rng > 0)
    F["er_5"], F["er_20"] = F["er"].rolling(5).mean(), F["er"].rolling(20).mean()
    m20 = rng.rolling(20).mean()
    F["move_ratio"] = (D["close"] - D["open"]).abs() / m20
    F["range_ratio"] = np.log(rng / m20)
    F["range_ratio_5"] = np.log(rng.rolling(5).mean() / m20)
    F["nr7"] = (rng <= rng.rolling(7).min()).astype(float)
    F["clv"] = ((D["close"] - D["low"]) / rng).where(rng > 0)                       # where today closed in its range
    F["trend_20"] = (D["close"] - D["close"].shift(20)) / (m20 * np.sqrt(20))
    F["vol_ratio"] = np.log((D["volume"] + 1) / (D["volume"].rolling(20).mean() + 1))
    nxt = D.index.to_series().shift(-1)
    # next trading day; for the last row assume the next weekday
    last = D.index[-1] + pd.offsets.BDay(1)
    nxt = nxt.fillna(last)
    F["dow_next"] = nxt.dt.dayofweek.values
    for e in EVENTS:
        days = set(events.loc[events["event"] == e, "date"])
        F[f"{e}_next"] = nxt.isin(days).astype(float).values
    for c in CLASSES:
        F[f"is_{c}"] = float(CLASS[root] == c)
    F["er_next"] = F["er"].shift(-1)
    F["ret_next"] = D["ret"].shift(-1)                                              # for strategy work, never a feature
    F["market"] = root
    return F.replace([np.inf, -np.inf], np.nan)


def panel():
    ev = pd.read_csv(os.path.join(HERE, "events_full.csv"), parse_dates=["date"])
    parts = [features(load_daily(r), r, ev) for r in sorted(CLASS)]
    return pd.concat(parts).sort_index()


def split_date(P, frac=0.7):
    days = P.index.unique().sort_values()
    return days[int(len(days) * frac)]


# ---------------------------------------------------------------------------------------------- gate v2 features
CROSS = ["mx_er", "mx_er_5", "cls_er", "cls_er_5", "mx_range", "mx_trend_abs", "mkt_er_exp"]
CALENDAR = ["dom_next", "month_end_next", "pre_holiday_next", "post_holiday_next", "opex_next", "roll_next", "days_since_roll"]


def _calendar(D, idx_next):
    from pandas.tseries.holiday import USFederalHolidayCalendar
    hol = USFederalHolidayCalendar().holidays("2009-01-01", "2027-12-31")
    nxt = pd.DatetimeIndex(idx_next)
    C = pd.DataFrame(index=D.index)
    C["dom_next"] = nxt.day
    C["month_end_next"] = (nxt == (nxt + pd.offsets.BMonthEnd(0))).astype(float)          # last business day of month
    C["pre_holiday_next"] = (nxt + pd.offsets.BDay(1)).isin(hol).astype(float)          # day after tomorrow is a US holiday
    C["post_holiday_next"] = (nxt - pd.offsets.BDay(1)).isin(hol).astype(float)
    third_fri = (nxt.dayofweek == 4) & (nxt.day >= 15) & (nxt.day <= 21)
    C["opex_next"] = third_fri.astype(float)
    con = D["contract"]
    C["roll_next"] = (con.shift(-1) != con).astype(float).fillna(0.0)                   # tomorrow's contract chosen with today's volume
    C.iloc[-1, C.columns.get_loc("roll_next")] = 0.0
    grp = (con != con.shift(1)).cumsum()
    C["days_since_roll"] = con.groupby(grp).cumcount().astype(float)
    return C


def panel_v2():
    """Base features + calendar + cross-market features. Cross-market values for date d use only day-d closes."""
    ev = pd.read_csv(os.path.join(HERE, "events_full.csv"), parse_dates=["date"])
    parts = []
    for r in sorted(CLASS):
        D = load_daily(r)
        F = features(D, r, ev)
        nxt = D.index.to_series().shift(-1).fillna(D.index[-1] + pd.offsets.BDay(1))
        F = F.join(_calendar(D, nxt.values))
        F["mkt_er_exp"] = F["er"].expanding(60).mean()
        F["cls"] = CLASS[r]
        parts.append(F)
    P = pd.concat(parts).sort_index()
    P.index.name = "date"
    P["abs_trend"] = P["trend_20"].abs()
    by_day = P.groupby(level=0)
    P["mx_er"] = by_day["er"].transform("mean")
    P["mx_range"] = by_day["range_ratio"].transform("mean")
    P["mx_trend_abs"] = by_day["abs_trend"].transform("mean")
    P["cls_er"] = P.groupby([P.index, "cls"])["er"].transform("mean")
    # 5-day smooths of the daily cross-sectional means (per-date series, then mapped back)
    mx = by_day["er"].mean().rolling(5).mean()
    P["mx_er_5"] = P.index.map(mx)
    cl = P.groupby([P.index, "cls"])["er"].mean().unstack().rolling(5).mean()
    P["cls_er_5"] = [cl.at[d, c] for d, c in zip(P.index, P["cls"])]
    P["range_next"] = P.groupby("market")["range_ratio"].shift(-1)
    return P.replace([np.inf, -np.inf], np.nan)
