import yfinance as yf
import pandas as pd

# Continuous front-month futures. Chosen for sector orthogonality.
UNIVERSE = {
    # Equity index
    "ES=F": "S&P 500", "NQ=F": "Nasdaq 100", "YM=F": "Dow",
    "RTY=F": "Russell 2000", "NKD=F": "Nikkei 225",
    # Rates & bonds
    "ZT=F": "2y Note", "ZF=F": "5y Note", "ZN=F": "10y Note",
    "ZB=F": "30y Bond", "UB=F": "Ultra Bond",
    # FX
    "6E=F": "Euro", "6J=F": "Yen", "6B=F": "Sterling",
    "6A=F": "Aussie", "6C=F": "CAD", "6S=F": "Swiss", "6M=F": "Peso",
    # Energy
    "CL=F": "WTI crude", "BZ=F": "Brent", "NG=F": "Nat gas",
    "HO=F": "Heating oil", "RB=F": "RBOB gasoline",
    # Metals
    "GC=F": "Gold", "SI=F": "Silver", "HG=F": "Copper",
    "PL=F": "Platinum", "PA=F": "Palladium",
    # Grains
    "ZC=F": "Corn", "ZW=F": "Wheat", "ZS=F": "Soybeans",
    "ZM=F": "Soymeal", "ZL=F": "Soyoil", "ZO=F": "Oats",
    # Softs
    "KC=F": "Coffee", "SB=F": "Sugar", "CC=F": "Cocoa", "CT=F": "Cotton",
    # Livestock
    "LE=F": "Live cattle", "HE=F": "Lean hogs",
}

def build(start="2000-01-01"):
    df = yf.download(list(UNIVERSE), start=start, auto_adjust=True,
                     progress=False)["Close"]
    df = df.dropna(axis=1, how="all")
    # drop series with fewer than 5 years of data
    keep = df.columns[df.notna().sum() >= 1250]
    dropped = sorted(set(df.columns) - set(keep))
    df = df[keep]
    df.to_parquet("../state/fut_prices.parquet")
    print(f"{df.shape[0]} daily bars, {df.shape[1]} contracts kept")
    print(f"range: {df.index[0].date()} to {df.index[-1].date()}")
    if dropped:
        print(f"dropped (<5y history): {dropped}")
    print("\ninception dates:")
    for t, d in df.apply(lambda c: c.first_valid_index()).sort_values().items():
        print(f"  {t:7s} {d.date()}  {UNIVERSE.get(t,'')}")
    return df

def load_prices():
    return pd.read_parquet("../state/fut_prices.parquet")

if __name__ == "__main__":
    build()
