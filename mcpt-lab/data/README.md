# Data

| file | contents | source / license | in repo? |
|---|---|---|---|
| `BTCUSDT_1h_2018_2022.csv` | BTC/USDT 1-hour OHLCV, 2018-01-01 → 2022-12-31 (43,823 bars) | shipped with neurotrader888's MIT-licensed repos (Binance data) | yes |
| `btc_1h_binanceus.csv` | BTC/USDT 1-hour OHLCV, 2022-01-01 → today | Binance.US public API: `python data/download_binanceus.py` | no (download it) |

The two sources agree closely where they overlap: hourly returns in 2022 have a correlation of 0.996.

**Using your own data.** `mcptlab.load_ohlc("file.csv")` reads any CSV that has:
- a date/time column (`date`, `datetime`, `time`, `timestamp` or `ts`);
- a `close` column;
- optionally `open`, `high`, `low` and `volume`.

With OHLC data the bar permutation is used. With close-only data the returns are shuffled instead.

**Not included.** The ES futures test in `results/` used licensed CME data from Databento. That data can't be redistributed, so only the results (the ledger row and null distribution) are published here.
