import yfinance as yf
import pandas as pd

# Selected for asset-class orthogonality, not performance.
UNIVERSE = {
    # Equity regions (1 bet, kept small deliberately)
    "SPY": "US equity", "EFA": "Intl dev equity", "EEM": "EM equity",
    # Rates / duration
    "SHY": "US 1-3y", "IEF": "US 7-10y", "TLT": "US 20y+",
    "BWX": "Intl sovereign", "EMB": "EM sovereign",
    # Credit spread
    "LQD": "IG credit", "HYG": "HY credit", "TIP": "TIPS",
    # Precious metals
    "GLD": "Gold", "SLV": "Silver", "PPLT": "Platinum",
    # Energy
    "USO": "WTI crude", "UNG": "Natural gas", "BNO": "Brent",
    # Agriculture
    "DBA": "Ag basket", "CORN": "Corn", "WEAT": "Wheat", "SOYB": "Soybeans",
    # Industrial metals
    "DBB": "Base metals", "COPX": "Copper miners", "JJC": "Copper",
    # Currencies
    "UUP": "USD index", "FXE": "Euro", "FXY": "Yen", "FXB": "Sterling",
    "FXA": "Aussie", "FXF": "Swiss franc", "FXC": "CAD",
    # Real assets / sectors with distinct drivers
    "VNQ": "US REIT", "RWX": "Intl REIT", "XLE": "Energy eq",
    "XLU": "Utilities", "XLP": "Staples", "GDX": "Gold miners",
    "IYR": "Real estate", "XME": "Metals & mining", "MOO": "Agribusiness",
}

def build(start="2006-01-01"):
    df = yf.download(list(UNIVERSE), start=start, auto_adjust=True,
                     progress=False)["Close"]
    df = df.dropna(axis=1, how="all")
    df.to_parquet("../state/wide_prices.parquet")
    print(f"{df.shape[0]} daily bars, {df.shape[1]} assets")
    print(f"range: {df.index[0].date()} to {df.index[-1].date()}\n")
    firsts = df.apply(lambda c: c.first_valid_index())
    print("inception dates:")
    for t, d in firsts.sort_values().items():
        print(f"  {t:6s} {d.date()}  {UNIVERSE.get(t,'')}")
    return df

def load_prices():
    return pd.read_parquet("../state/wide_prices.parquet")

if __name__ == "__main__":
    build()
