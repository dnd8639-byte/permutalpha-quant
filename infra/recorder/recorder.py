"""Order-book and trade recorder.

Every 10 seconds, for each market in config.VENUES: save the top bid and ask levels of the order book, and every
trade since the last look. All markets are fetched at the same time (one thread each), then written together, so
adding markets does not stretch the 10-second cycle. One SQLite file per month in data/. Read-only public data;
it cannot place orders.

    python recorder.py test     one cycle, printing what each venue returned (run this first)
    python recorder.py          record forever; Control + C stops it
    python check.py             summary of what has been collected

Tables
  book(venue, ts, bid, ask, book)   ts = this computer's clock in ms; bid/ask = best prices;
                                    book = zlib-compressed JSON {"b": [[price, size], ...], "a": [...]}
  trades(venue, id, ts, price, amount, side)        ts = the exchange's trade time in ms
  health(ts, venue, ok, ms, error)                  one row per venue per cycle: did it work, how long it took
"""
import json, os, sqlite3, sys, time, zlib
from concurrent.futures import ThreadPoolExecutor

import config as C

SCHEMA = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS book(venue TEXT, ts INTEGER, bid REAL, ask REAL, book BLOB, PRIMARY KEY(venue, ts));
CREATE TABLE IF NOT EXISTS trades(venue TEXT, id TEXT, ts INTEGER, price REAL, amount REAL, side TEXT, PRIMARY KEY(venue, id));
CREATE TABLE IF NOT EXISTS health(ts INTEGER, venue TEXT, ok INTEGER, ms INTEGER, error TEXT);
CREATE INDEX IF NOT EXISTS trades_ts ON trades(ts);
"""


def db_for(ts_ms, cache={}):
    """Open (or reuse) this month's database file."""
    name = time.strftime("book_%Y-%m.db", time.gmtime(ts_ms / 1000))
    if cache.get("name") != name:
        if cache.get("db"):
            cache["db"].close()
        os.makedirs(C.DATA_DIR, exist_ok=True)
        cache["db"] = sqlite3.connect(os.path.join(C.DATA_DIR, name)); cache["db"].executescript(SCHEMA); cache["name"] = name
    return cache["db"]


def pack(book, depth):
    return zlib.compress(json.dumps({"b": [l[:2] for l in book["bids"][:depth]], "a": [l[:2] for l in book["asks"][:depth]]},
                                    separators=(",", ":")).encode())


def unpack(blob):
    return json.loads(zlib.decompress(blob))


def venues():
    """config.VENUES entries are (exchange, symbol), labelled by the exchange name (the original BTC markets), or
    (label, exchange, symbol, depth) for any further market. Labels are what the database stores; they must be unique."""
    out = []
    for v in C.VENUES:
        if len(v) == 2:
            out.append((v[0], v[0], v[1], C.DEPTH))
        else:
            out.append((v[0], v[1], v[2], v[3] if len(v) > 3 else C.DEPTH))
    labels = [x[0] for x in out]
    if len(set(labels)) != len(labels):
        raise ValueError(f"venue labels must be unique: {labels}")
    return out


def fetch_venue(ex, symbol, depth, now_ms, last_trade_ts):
    """Network only (runs in a worker thread). Returns (book, trades, trade_error)."""
    book = ex.fetch_order_book(symbol, limit=depth)
    if not book["bids"] or not book["asks"] or book["bids"][0][0] >= book["asks"][0][0]:
        raise ValueError("empty or crossed book")
    since = last_trade_ts + 1 if last_trade_ts else None
    try:
        try:
            trades = ex.fetch_trades(symbol, since=since, limit=1000)
        except Exception as e:                                   # Coinbase insists on an end time whenever a start time is given
            if since is None or "until" not in str(e):
                raise
            trades = ex.fetch_trades(symbol, since=since, limit=1000, params={"until": now_ms})
        return book, trades, ""
    except Exception as e:                                       # keep the book even if trades failed
        return book, [], f"trades: {type(e).__name__}: {str(e)[:140]}"


def write_venue(db, label, depth, now_ms, book, trades, last_trade_ts):
    """Database only (main thread). Returns (number of new trades, newest trade time)."""
    db.execute("INSERT OR REPLACE INTO book VALUES(?,?,?,?,?)",
               (label, now_ms, book["bids"][0][0], book["asks"][0][0], pack(book, depth)))
    new = 0
    for t in trades:
        tid = str(t.get("id") or f"{t['timestamp']}-{t['price']}-{t['amount']}")
        new += db.execute("INSERT OR IGNORE INTO trades VALUES(?,?,?,?,?,?)",
                          (label, tid, t["timestamp"], t["price"], t["amount"], t.get("side") or "")).rowcount
        last_trade_ts = max(last_trade_ts or 0, t["timestamp"])
    return new, last_trade_ts


def cycle(exchanges, state, now_ms, verbose=False, pool=None):
    db = db_for(now_ms)
    vs = venues()

    def job(v):
        label, _, symbol, depth = v
        t0 = time.time()
        try:
            book, trades, terr = fetch_venue(exchanges[label], symbol, depth, now_ms, state.get(label))
            return label, depth, book, trades, terr, int((time.time() - t0) * 1000)
        except Exception as e:                                   # one market failing must not stop the others
            return label, depth, None, [], f"{type(e).__name__}: {str(e)[:150]}", int((time.time() - t0) * 1000)

    own = pool is None
    pool = pool or ThreadPoolExecutor(max_workers=len(vs))
    try:
        results = list(pool.map(job, vs))
    finally:
        if own:
            pool.shutdown()
    for label, depth, book, trades, err, ms in results:
        new = 0
        if book is not None:
            try:
                new, state[label] = write_venue(db, label, depth, now_ms, book, trades, state.get(label))
            except Exception as e:
                err = f"write: {type(e).__name__}: {str(e)[:140]}"
        ok = int(book is not None and not err)
        db.execute("INSERT INTO health VALUES(?,?,?,?,?)", (now_ms, label, ok, ms, err))
        if verbose:
            row = db.execute("SELECT bid, ask, length(book) FROM book WHERE venue=? AND ts=?", (label, now_ms)).fetchone()
            print(f"  {label:14s} " + (f"OK   bid {row[0]:,.2f}  ask {row[1]:,.2f}  {new} trades  {row[2]} bytes  {ms} ms" if ok
                                       else f"FAILED  {err}"))
    db.commit()


def make_exchanges():
    """One exchange object per market, so markets on the same exchange can be fetched in parallel safely."""
    import ccxt
    return {label: getattr(ccxt, ex)({"enableRateLimit": True, "timeout": C.TIMEOUT_MS}) for label, ex, _, _ in venues()}


if __name__ == "__main__":
    exchanges, state = make_exchanges(), {}
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        print("One test cycle:"); cycle(exchanges, state, int(time.time() * 1000), verbose=True); sys.exit()
    print(f"Recording {len(venues())} markets every {C.INTERVAL_S}s into {C.DATA_DIR}. Control + C to stop.", flush=True)
    pool = ThreadPoolExecutor(max_workers=len(venues()))
    while True:
        now = time.time()
        nxt = (now // C.INTERVAL_S + 1) * C.INTERVAL_S           # stay on a fixed 10-second grid
        time.sleep(nxt - now)
        try:
            cycle(exchanges, state, int(nxt * 1000), pool=pool)
        except Exception as e:                                   # e.g. disk problem: report and keep trying
            print("cycle failed:", type(e).__name__, e, flush=True)
