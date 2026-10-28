"""Download BTC/USDT 1-hour bars from Binance.US (via ccxt, already in your .venv) for 2022-01-01 -> now.
2022 overlaps the author's file (used to check the two sources agree); 2023+ is the post-publication test.
Run from the mcpt folder:  python data_btc_hourly.py      (a few minutes)"""
import time, datetime as dt
import ccxt
import pandas as pd

ex = ccxt.binanceus()
since = ex.parse8601("2022-01-01T00:00:00Z")
rows = []
while True:
    batch = ex.fetch_ohlcv("BTC/USDT", "1h", since=since, limit=1000)
    if not batch:
        break
    rows += batch
    since = batch[-1][0] + 3600_000
    print(f"  {len(rows):6d} bars, up to {dt.datetime.fromtimestamp(batch[-1][0] / 1000, dt.timezone.utc):%Y-%m-%d}", flush=True)
    time.sleep(ex.rateLimit / 1000)
    if len(batch) < 1000:
        break
df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"]).drop_duplicates("ts")
df["date"] = pd.to_datetime(df["ts"], unit="ms").dt.strftime("%Y-%m-%d %H:%M:%S")
df[["open", "high", "low", "close", "volume", "date"]].to_csv("btc_1h_binanceus.csv", index=False)
print(f"Saved {len(df)} bars to btc_1h_binanceus.csv. Tell Claude it finished.")
