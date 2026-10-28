"""Download the public data for projects 8, 9 and 10 into quant-projects/data/.

    python download_data.py            all three (about 3-5 minutes)
    python download_data.py options    SPY option chain (Yahoo Finance, via yfinance)
    python download_data.py trades     last 24 hours of BTC/USDT trades (Binance.US, via ccxt)
    python download_data.py news       BTC/USDT one-minute bars around FOMC and CPI releases (Binance.US)
No account or API key is needed. If one part fails, the others still run.
"""
import os, sys, time, datetime as dt
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data"); os.makedirs(OUT, exist_ok=True)


def options():
    import yfinance as yf
    tk = yf.Ticker("SPY")
    today = dt.date.today()
    cands = [e for e in tk.options if 25 <= (dt.date.fromisoformat(e) - today).days <= 70]
    best, chain = None, None
    for e in cands:                                            # the expiry with the most strikes (the monthly one)
        ch = tk.option_chain(e)
        if best is None or len(ch.calls) > len(chain.calls):
            best, chain = e, ch
    spot = float(tk.history(period="5d")["Close"].iloc[-1])
    rows = pd.concat([chain.calls.assign(type="call"), chain.puts.assign(type="put")])[["type", "strike", "bid", "ask", "lastPrice", "volume", "openInterest"]]
    rows["expiry"], rows["spot"] = best, spot
    rows["T_years"] = (dt.date.fromisoformat(best) - today).days / 365
    rows["downloaded"] = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    rows.to_csv(os.path.join(OUT, "spy_chain.csv"), index=False)
    print(f"  options: SPY expiry {best}, {len(chain.calls)} calls + {len(chain.puts)} puts, spot {spot:.2f}")


def trades(hours=24):
    import ccxt
    ex = ccxt.binanceus()
    now = ex.milliseconds(); since = now - hours * 3600_000
    rows = []
    while since < now:
        batch = ex.fetch_trades("BTC/USDT", since=since, limit=1000)
        if not batch:
            since += 3600_000                                  # quiet hour: move on
            continue
        rows += [(t["timestamp"], t["price"], t["amount"], t["side"]) for t in batch]
        since = batch[-1]["timestamp"] + 1
        time.sleep(ex.rateLimit / 1000)
        if len(rows) % 10000 < 1000:
            print(f"    {len(rows):7d} trades so far", flush=True)
    df = pd.DataFrame(rows, columns=["timestamp", "price", "amount", "side"]).drop_duplicates()
    df.to_csv(os.path.join(OUT, "btc_trades.csv"), index=False)
    print(f"  trades: {len(df):,} BTC/USDT trades over the last {hours} hours")


def news():
    import ccxt
    from p10_news_diffusion import events_utc, HALF
    ex = ccxt.binanceus()
    rows = []
    for ts, kind in events_utc():
        start = int((ts.timestamp() - HALF - 600) * 1000); end = int((ts.timestamp() + HALF + 600) * 1000)
        if end > ex.milliseconds():
            continue
        since, n0 = start, len(rows)
        while since < end:
            batch = ex.fetch_ohlcv("BTC/USDT", "1m", since=since, limit=1000)
            if not batch:
                break
            rows += [b for b in batch if b[0] <= end]
            since = batch[-1][0] + 60_000
            time.sleep(ex.rateLimit / 1000)
        print(f"    {kind} {ts:%Y-%m-%d %H:%M} UTC: {len(rows) - n0} bars", flush=True)
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"]).drop_duplicates("ts")
    df.to_csv(os.path.join(OUT, "btc_1m_news_windows.csv"), index=False)
    print(f"  news: {len(df):,} one-minute bars")


if __name__ == "__main__":
    want = sys.argv[1:] or ["options", "trades", "news"]
    for name in want:
        print(f"{name} ...", flush=True)
        try:
            {"options": options, "trades": trades, "news": news}[name]()
        except Exception as e:                                 # keep going; report at the end
            print(f"  {name} FAILED: {type(e).__name__}: {e}")
    print("Done. Tell Claude it finished (and paste any FAILED lines).")
