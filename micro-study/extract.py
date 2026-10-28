"""Order-book recordings -> one row per hour. Runs on the Pi, next to the recorder's data.

    python extract.py              process every complete hour recorded so far (finished days are cached)
    python extract.py --rebuild    ignore the cache and redo everything

Reads  ~/recorder/data/book_YYYY-MM.db   (read-only; safe while the recorder is writing)
Writes ./data/hourly.csv                  one row per UTC hour, features measured INSIDE that hour
       ./data/days/YYYY-MM-DD.csv         cache of finished days (and .partial.csv for today, so hourly runs are quick)

Every number in a row is known at the END of its hour. The study script pairs it with the NEXT hour's
efficiency; nothing here looks ahead.

Per 10-second snapshot and venue:  mid, spread, USD depth within 5 and 25 bps of mid, depth imbalance
(bid - ask) / (bid + ask) within 5 and 25 bps and at the top level, and the microprice offset
(top-of-book size-weighted price minus mid). The three venues are combined into one 'consensus' mid
(the median), which is also the price used for efficiency.
"""
import glob, json, os, sqlite3, sys, time, zlib

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.environ.get("RECORDER_DATA", os.path.expanduser("~/recorder/data"))
OUT = os.environ.get("MICRO_DATA", os.path.join(HERE, "data"))
CACHE = os.path.join(OUT, "days")
HOUR = 3_600_000
DAY = 24 * HOUR
MIN_SNAPS = 300            # an hour needs at least 300 of its 360 consensus snapshots
VENUES = tuple(os.environ.get("MICRO_VENUES", "binanceus,kraken,coinbase").split(","))   # the BTC markets only
BANDS = (2, 10, 25)        # basis points from mid. 2 bps is inside the recording since day 1; 10 and 25 need the deeper recorder


def connect(path):
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


def snapshot_rows(db, venue, t0, t1):
    out = []
    for ts, bid, ask, blob in db.execute("SELECT ts, bid, ask, book FROM book WHERE venue=? AND ts>=? AND ts<? ORDER BY ts",
                                         (venue, t0, t1)):
        b = json.loads(zlib.decompress(blob))
        bids, asks = np.asarray(b["b"], float), np.asarray(b["a"], float)
        if bids.ndim != 2 or asks.ndim != 2 or not len(bids) or not len(asks):
            continue
        mid = (bid + ask) / 2
        reach = min(mid - bids[-1, 0], asks[-1, 0] - mid) / mid * 1e4      # how far the recorded levels go
        r = [ts, mid, (ask - bid) / mid * 1e4, reach]
        for B in BANDS:
            if reach < B:                                                    # recording too shallow: unknown, not zero
                r += [np.nan, np.nan]; continue
            bd = bids[bids[:, 0] >= mid * (1 - B / 1e4), 1].sum()
            ad = asks[asks[:, 0] <= mid * (1 + B / 1e4), 1].sum()
            r += [(bd + ad) * mid, (bd - ad) / (bd + ad) if bd + ad > 0 else 0.0]
        b1, a1 = bids[0, 1], asks[0, 1]
        r.append((b1 - a1) / (b1 + a1) if b1 + a1 > 0 else 0.0)
        micro = (bid * a1 + ask * b1) / (a1 + b1) if a1 + b1 > 0 else mid
        r.append((micro - mid) / mid * 1e4)
        out.append(r)
    cols = ["ts", "mid", "spread_bps", "reach_bps"] + [f"{k}{B}" for B in BANDS for k in ("dep", "imb")] + ["imb1", "micro_bps"]
    return pd.DataFrame(out, columns=cols).assign(venue=venue)


def load_range(t0, t1):
    """All snapshots and trades with time in [t0, t1), from whichever monthly files hold them."""
    snaps, trades = [], []
    for f in sorted(glob.glob(os.path.join(SRC, "book_*.db"))):
        db = connect(f)
        try:
            venues = [v for (v,) in db.execute("SELECT DISTINCT venue FROM health") if v in VENUES]
            for v in venues:
                s = snapshot_rows(db, v, t0, t1)
                if len(s):
                    snaps.append(s)
            q = ",".join("?" * len(VENUES))
            tr = pd.read_sql_query(f"SELECT venue, ts, price, amount, side FROM trades WHERE ts>=? AND ts<? AND venue IN ({q})",
                                   db, params=(t0, t1, *VENUES))
            if len(tr):
                trades.append(tr)
        finally:
            db.close()
    S = pd.concat(snaps) if snaps else pd.DataFrame()
    T = pd.concat(trades).drop_duplicates() if trades else pd.DataFrame(columns=["venue", "ts", "price", "amount", "side"])
    return S, T


def hours_from(S, T):
    """One row per complete hour."""
    if S.empty:
        return pd.DataFrame()
    mids = S.pivot_table(index="ts", columns="venue", values="mid")
    cons = mids.median(axis=1)
    disp = (mids.sub(cons, axis=0).abs().div(cons, axis=0) * 1e4).median(axis=1)
    rows = []
    for h, m in cons.groupby(cons.index // HOUR):
        if len(m) < MIN_SNAPS:
            continue
        t0, t1 = h * HOUR, (h + 1) * HOUR
        m = m.sort_index()
        o, c, hi, lo = m.iloc[0], m.iloc[-1], m.max(), m.min()
        last15 = m[m.index >= t1 - 15 * 60_000]
        lr = np.diff(np.log(m.to_numpy()))
        row = dict(hour=pd.Timestamp(t0, unit="ms"), n_snaps=len(m), venues=int(mids.loc[m.index].notna().any().sum()),
                   open=o, close=c, high=hi, low=lo,
                   er=abs(c - o) / (hi - lo) if hi > lo else 0.0, ret_bps=(c - o) / o * 1e4, range_bps=(hi - lo) / o * 1e4,
                   rv_bps2=float((lr ** 2).sum() * 1e8),
                   er_15m=abs(last15.iloc[-1] - last15.iloc[0]) / (last15.max() - last15.min()) if last15.max() > last15.min() else 0.0,
                   ret_15m_bps=(last15.iloc[-1] - last15.iloc[0]) / last15.iloc[0] * 1e4,
                   venue_disp_bps=float(disp.loc[m.index].mean()))
        sh = S[(S["ts"] >= t0) & (S["ts"] < t1)]
        l5 = sh[sh["ts"] >= t1 - 5 * 60_000]
        per_v = sh.groupby("venue")
        n_v = sh["venue"].nunique()

        def vmean(frame, k):
            """Per-venue average of k; NaN for a venue whose recording covered k in < 90% of its snapshots."""
            g = frame.groupby("venue")[k]
            return g.mean().where(g.count() >= 0.9 * g.size())

        def all_venues(x, how):                      # a band feature exists only if EVERY venue's book reached it
            return float(getattr(x, how)()) if x.notna().sum() == n_v else np.nan

        row["spread_bps"] = float(per_v["spread_bps"].mean().mean())
        row["reach_bps"] = float(per_v["reach_bps"].median().min())
        for B in BANDS:
            row[f"dep{B}_usd"] = all_venues(vmean(sh, f"dep{B}"), "sum")
        row["dep2_last5_ratio"] = all_venues(vmean(l5, "dep2") / vmean(sh, "dep2"), "mean") if len(l5) else np.nan
        for k in [f"imb{B}" for B in BANDS] + ["imb1", "micro_bps"]:
            row[f"{k}_mean"] = all_venues(vmean(sh, k), "mean")
            row[f"{k}_last5"] = all_venues(vmean(l5, k), "mean") if len(l5) else np.nan
        th = T[(T["ts"] >= t0) & (T["ts"] < t1)]
        t15 = th[th["ts"] >= t1 - 15 * 60_000]
        for tag, x in (("1h", th), ("15m", t15)):
            buy = x.loc[x["side"] == "buy", "amount"].sum(); sell = x.loc[x["side"] == "sell", "amount"].sum()
            row[f"tfi_{tag}"] = (buy - sell) / (buy + sell) if buy + sell > 0 else 0.0
        row["n_trades"] = len(th)
        row["btc_traded"] = float(th["amount"].sum())
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    rebuild = "--rebuild" in sys.argv
    os.makedirs(CACHE, exist_ok=True)
    files = sorted(glob.glob(os.path.join(SRC, "book_*.db")))
    if not files:
        sys.exit(f"No recorder files in {SRC}")
    first = min(connect(f).execute("SELECT MIN(ts) FROM book").fetchone()[0] or 1e18 for f in files)
    now = int(time.time() * 1000)
    day0 = int(first // DAY * DAY)
    parts = []
    for d in range(day0, now, DAY):
        name = time.strftime("%Y-%m-%d", time.gmtime(d / 1000))
        path = os.path.join(CACHE, f"{name}.csv")
        part = os.path.join(CACHE, f"{name}.partial.csv")
        finished = d + DAY + 10 * 60_000 <= now                      # 10 minutes of slack for late trades
        if finished and os.path.exists(path) and not rebuild:
            C = pd.read_csv(path, parse_dates=["hour"])
            if len(C):
                parts.append(C)
            continue
        # hours of this day already done on an earlier run (the hourly timer) are reused, not recomputed
        P = pd.read_csv(part, parse_dates=["hour"]) if os.path.exists(part) and not rebuild else pd.DataFrame()
        start = int(P["hour"].max().value // 1_000_000) + HOUR if len(P) else d
        t_end = d + DAY if finished else now // HOUR * HOUR          # only complete hours
        H = P
        if start < t_end:
            S, T = load_range(start, t_end)
            new = hours_from(S, T)
            H = pd.concat([P, new]) if len(P) else new
        if finished:
            (H if len(H) else pd.DataFrame(columns=["hour"])).to_csv(path, index=False)   # an outage day is cached as empty
            if os.path.exists(part):
                os.remove(part)
        elif len(H):
            H.to_csv(part, index=False)
        if len(H):
            parts.append(H)
        print(f"{name}: {len(H):2d} hours" + ("" if finished else "  (today, partial)"), flush=True)
    A = pd.concat(parts).sort_values("hour").drop_duplicates("hour") if parts else pd.DataFrame()
    A.to_csv(os.path.join(OUT, "hourly.csv"), index=False)
    print(f"\n{len(A)} hours -> {os.path.join(OUT, 'hourly.csv')}"
          + (f"  ({A['hour'].min()} to {A['hour'].max()} UTC)" if len(A) else ""))


if __name__ == "__main__":
    main()
