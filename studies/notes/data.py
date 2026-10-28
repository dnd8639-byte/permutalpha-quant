import ccxt, duckdb, pandas as pd, datetime as dt, time

UNIVERSE = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "LINK/USDT"]
DB = "../state/ahl.duckdb"


def to_dt(ms):
    return dt.datetime.fromtimestamp(ms / 1000, dt.timezone.utc).replace(tzinfo=None)


def fetch_all(symbol, exchange, timeframe="1d", since_year=2019):
    since = exchange.parse8601(f"{since_year}-01-01T00:00:00Z")
    out = []
    while True:
        batch = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
        if not batch:
            break
        out += batch
        since = batch[-1][0] + 1
        time.sleep(exchange.rateLimit / 1000)
        if len(batch) < 1000:
            break
    return out


def build():
    ex = ccxt.binanceus()
    ex.load_markets()
    con = duckdb.connect(DB)
    con.execute("""CREATE TABLE IF NOT EXISTS bars(
        ts TIMESTAMP, symbol VARCHAR, open DOUBLE, high DOUBLE,
        low DOUBLE, close DOUBLE, volume DOUBLE, PRIMARY KEY(ts, symbol))""")
    for sym in UNIVERSE:
        try:
            rows = [(to_dt(c[0]), sym, *c[1:]) for c in fetch_all(sym, ex)]
            con.executemany("INSERT OR REPLACE INTO bars VALUES (?,?,?,?,?,?,?)", rows)
            print(f"{sym}: {len(rows)} bars")
        except Exception as e:
            print(f"{sym}: FAILED -> {e}")
    con.close()


def load_prices(db=DB):
    con = duckdb.connect(db)
    df = con.execute("SELECT ts, symbol, close FROM bars ORDER BY ts").df()
    con.close()
    return df.pivot(index="ts", columns="symbol", values="close")


if __name__ == "__main__":
    build()
