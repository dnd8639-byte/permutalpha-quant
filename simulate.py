"""Fake order-book recordings in exactly the recorder's format, for testing the pipeline before trusting it.

    python simulate.py planted OUTDIR [days]   informed order flow that persists from hour to hour: while it lasts,
                                               trades and book depth lean one way and the price walks that way
                                               (efficient hours); otherwise the price chops around (inefficient).
    python simulate.py noise OUTDIR [days]     random walk, random flow: there is nothing to find.
"""
import json, os, sqlite3, sys, zlib

import numpy as np

SCHEMA = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS book(venue TEXT, ts INTEGER, bid REAL, ask REAL, book BLOB, PRIMARY KEY(venue, ts));
CREATE TABLE IF NOT EXISTS trades(venue TEXT, id TEXT, ts INTEGER, price REAL, amount REAL, side TEXT, PRIMARY KEY(venue, id));
CREATE TABLE IF NOT EXISTS health(ts INTEGER, venue TEXT, ok INTEGER, ms INTEGER, error TEXT);
"""
VENUES = ["binanceus", "kraken", "coinbase"]
T0 = 1_782_864_000_000                                          # 2026-07-01 00:00 UTC (fake data ends before today)


def main(kind, out, days=12, seed=1):
    rng = np.random.default_rng(seed)
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "book_2026-09.db")
    if os.path.exists(path):
        os.remove(path)
    db = sqlite3.connect(path); db.executescript(SCHEMA)
    n = days * 8640
    ts = T0 + np.arange(n) * 10_000
    hours = n // 360
    # hidden hourly state: +1 / -1 = informed buying / selling, 0 = no informed flow
    state = np.zeros(hours, int)
    if kind == "planted":
        for h in range(1, hours):
            if state[h - 1] != 0 and rng.random() < 0.75:
                state[h] = state[h - 1]
            else:
                state[h] = rng.choice([-1, 0, 0, 1])
    d = np.repeat(state, 360)
    p = np.empty(n); p[0] = np.log(60000.0)
    anchor = p[0]
    eps = rng.standard_normal(n) * 1.5e-4
    for t in range(1, n):
        if t % 360 == 0:
            anchor = p[t - 1]
        if d[t] != 0:
            p[t] = p[t - 1] + d[t] * 0.9e-5 + eps[t]                 # informed: steady walk one way
        elif kind == "planted":
            p[t] = p[t - 1] - 0.03 * (p[t - 1] - anchor) + eps[t]    # chop: pulled back toward the hour's open
        else:
            p[t] = p[t - 1] + eps[t]
    lean = 0.35 * d                                                     # how far depth and trades lean with the flow
    book_rows, trade_rows, health = [], [], []
    tid = 0
    for v_i, v in enumerate(VENUES):
        mid = np.exp(p + rng.standard_normal(n) * 2e-5)
        half = mid * (0.5 + 0.5 * v_i) / 1e4
        for t in range(n):
            bid, ask = mid[t] - half[t], mid[t] + half[t]
            sz_b = rng.exponential(0.5, 10) * (1 + lean[t] + rng.normal(0, 0.15))
            sz_a = rng.exponential(0.5, 10) * (1 - lean[t] + rng.normal(0, 0.15))
            step = mid[t] * 1e-4
            bk = {"b": [[round(bid - k * step, 2), round(max(s, 0.001), 4)] for k, s in enumerate(sz_b)],
                  "a": [[round(ask + k * step, 2), round(max(s, 0.001), 4)] for k, s in enumerate(sz_a)]}
            book_rows.append((v, int(ts[t]), bid, ask, zlib.compress(json.dumps(bk, separators=(",", ":")).encode())))
            for _ in range(rng.poisson(2)):
                buy = rng.random() < 0.5 + 0.5 * lean[t]
                trade_rows.append((v, str(tid), int(ts[t] - rng.integers(0, 10_000)), ask if buy else bid,
                                   float(rng.exponential(0.05)), "buy" if buy else "sell"))
                tid += 1
            if v_i == 0:
                health.extend((int(ts[t]), x, 1, 100, "") for x in VENUES)
        db.executemany("INSERT INTO book VALUES(?,?,?,?,?)", book_rows); book_rows.clear()
    db.executemany("INSERT INTO trades VALUES(?,?,?,?,?,?)", trade_rows)
    db.executemany("INSERT INTO health VALUES(?,?,?,?,?)", health)
    db.commit(); db.close()
    print(f"{kind}: {days} days, {n * 3:,} snapshots, {len(trade_rows):,} trades -> {path}")
    np.save(os.path.join(out, "true_state.npy"), state)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 12)
