"""Front-contract daily OHLC for the 35 Databento CME futures (2010-06 onward).

Same no-look-ahead roll rule as v3/build_clean.py: on day t hold the contract with the most volume on day t-1,
excluding contracts in their last 5 trading days. Every number for day t comes from that one contract,
so a roll never splices two contracts into one bar or one return.

Bars are UTC calendar days. Sunday's short evening session is merged into Monday (open = Sunday open,
close = Monday close), so every row is one weekday session.

Output (local only; licensed data, never published):  futures/daily/<ROOT>.csv
  date, open, high, low, close, volume, ret   (ret = log close/prev close of the SAME contract)
"""
import glob, os, sys
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "../../state/v3/contracts")
DST = os.path.join(HERE, "daily")
BUFFER = 5


def build(path):
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["ts_event"].astype(str).str[:10])
    df["key"] = df["symbol"].astype(str) + "_" + df["instrument_id"].astype(str)   # CME reuses 1-digit years
    # weekend sessions belong to the next Monday
    wd = df["date"].dt.dayofweek
    df.loc[wd == 6, "date"] += pd.Timedelta(days=1)
    df.loc[wd == 5, "date"] += pd.Timedelta(days=2)
    df = df.sort_values("ts_event")
    g = df.groupby(["date", "key"])
    bars = pd.DataFrame({"open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(),
                         "close": g["close"].last(), "volume": g["volume"].sum()}).reset_index()
    P = {c: bars.pivot(index="date", columns="key", values=c).sort_index() for c in ("open", "high", "low", "close", "volume")}
    close = P["close"]
    pos = pd.Series(np.arange(len(close)), index=close.index)
    last_pos = close.apply(lambda c: pos[c.last_valid_index()])
    elig = pd.DataFrame(pos.values[:, None] <= (last_pos.values[None, :] - BUFFER), index=close.index, columns=close.columns)
    v = P["volume"].fillna(0).where(elig & close.notna(), -1.0)
    held = v.idxmax(axis=1).where(v.max(axis=1) > 0).shift(1)                      # chosen with yesterday's volume
    ok = held.notna()
    dates, keys = held.index[ok], held[ok].to_numpy()
    take = lambda M: M.reindex(columns=pd.Index(keys).unique()).to_numpy()[
        M.index.get_indexer(dates), pd.Index(pd.Index(keys).unique()).get_indexer(keys)]
    pc = take(close.shift(1))
    cl = take(close)
    out = pd.DataFrame({"open": take(P["open"]), "high": take(P["high"]), "low": take(P["low"]), "close": cl,
                        "volume": take(P["volume"]), "ret": np.log(cl / pc),
                        "contract": [k.split("_")[0] for k in keys]}, index=dates)
    out.index.name = "date"
    out = out.dropna(subset=["close"])
    return out[(out[["open", "high", "low", "close"]] > 0).all(axis=1)]


if __name__ == "__main__":
    os.makedirs(DST, exist_ok=True)
    for p in sorted(glob.glob(os.path.join(SRC, "*.csv"))):
        root = os.path.basename(p)[:-4]
        D = build(p)
        D.to_csv(os.path.join(DST, f"{root}.csv"))
        print(f"{root:4s} {D.index[0].date()} -> {D.index[-1].date()}  days {len(D):5d}  "
              f"ret sd {D['ret'].std():.4f}  max|ret| {D['ret'].abs().max():.3f}")
