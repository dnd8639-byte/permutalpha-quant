import yfinance as yf
import pandas as pd

UNIVERSE = {
    "SPY": "US equity",     "EFA": "Intl equity",  "EEM": "EM equity",
    "IWM": "US smallcap",   "TLT": "US 30y bond",  "IEF": "US 10y bond",
    "LQD": "IG credit",     "HYG": "HY credit",    "TIP": "Inflation",
    "GLD": "Gold",          "SLV": "Silver",       "USO": "Oil",
    "DBA": "Agriculture",   "DBB": "Base metals",  "UUP": "USD index",
    "FXE": "Euro",          "FXY": "Yen",          "VNQ": "US REIT",
    "XLE": "Energy eq",     "XLU": "Utilities",
}

def build(start="2006-01-01"):
    df = yf.download(list(UNIVERSE), start=start, auto_adjust=True,
                     progress=False)["Close"]
    df.to_parquet("../state/etf_prices.parquet")
    print(f"{df.shape[0]} daily bars, {df.shape[1]} assets")
    print(f"range: {df.index[0].date()} to {df.index[-1].date()}")
    print("\nfirst valid date per asset:")
    print(df.apply(lambda c: c.first_valid_index().date()).sort_values())
    return df

def load_prices():
    return pd.read_parquet("../state/etf_prices.parquet")

if __name__ == "__main__":
    build()
