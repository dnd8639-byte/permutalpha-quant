"""Download daily prices for EVERY individual CME futures contract (not Yahoo's spliced
front month), so returns can be computed without roll gaps.

Source: Databento historical API, dataset GLBX.MDP3 (CME, CBOT, NYMEX, COMEX), from mid-2010.
Run from the v3 folder:   python data_databento.py
- Asks for your API key once (hidden) and stores it in ../state/.databento_key
- Checks the PRICE first and stops unless you type 'yes'. Hard stop above $20.
- Resumable: roots already downloaded are skipped.
"""
import os, sys, io, json, getpass, time
import pandas as pd

try:
    from curl_cffi import requests as http
    KW = {"impersonate": "chrome"}
except Exception:
    import requests as http
    KW = {}

BASE = "https://hist.databento.com/v0"
DATASET = "GLBX.MDP3"
OUT = "../state/v3/contracts"
KEYFILE = "../state/.databento_key"
HARD_CAP_USD = 55.0

# Same markets as data_fut.py, minus the ICE softs (KC, SB, CC, CT), which CME does not list.
ROOTS = ["ES", "NQ", "YM", "RTY", "NKD",
         "ZT", "ZF", "ZN", "ZB", "UB",
         "6E", "6J", "6B", "6A", "6C", "6S", "6M",
         "CL", "BZ", "NG", "HO", "RB",
         "GC", "SI", "HG", "PL", "PA",
         "ZC", "ZW", "ZS", "ZM", "ZL", "ZO",
         "LE", "HE"]

os.makedirs(OUT, exist_ok=True)

if os.path.exists(KEYFILE):
    KEY = open(KEYFILE).read().strip()
else:
    KEY = getpass.getpass("Paste your Databento API key (it won't show), then press Return: ").strip()
    with open(KEYFILE, "w") as f:
        f.write(KEY)
    os.chmod(KEYFILE, 0o600)
AUTH = (KEY, "")


def call(path, params, timeout=300):
    r = http.get(f"{BASE}/{path}", params=params, auth=AUTH, timeout=timeout, **KW)
    if r.status_code not in (200, 206):   # 206 = success with partially resolved symbols (Databento docs)
        raise RuntimeError(f"{path} -> HTTP {r.status_code}: {r.text[:300]}")
    return r


rng = call("metadata.get_dataset_range", {"dataset": DATASET}).json()
START, END = rng["start"][:10], rng["end"][:10]
print(f"Dataset {DATASET}: {START} -> {END}")

todo = [r for r in ROOTS if not os.path.exists(f"{OUT}/{r}.csv") and not os.path.exists(f"{OUT}/../raw/{r}.csv")]
if not todo:
    print("All roots already downloaded.")
else:
    base = {"dataset": DATASET, "stype_in": "parent", "schema": "ohlcv-1d", "start": START, "end": END}
    total = 0.0
    for r in todo:
        c = float(call("metadata.get_cost", {**base, "symbols": f"{r}.FUT"}).text)
        total += c
    print(f"\nEstimated cost for {len(todo)} markets: ${total:.2f}  (free credit on new accounts: $125)")
    if total > HARD_CAP_USD:
        sys.exit(f"STOP: above the ${HARD_CAP_USD:.0f} safety cap. Nothing was downloaded. Tell Claude.")
    if input("Type yes to download: ").strip().lower() != "yes":
        sys.exit("Cancelled. Nothing was downloaded.")

    for n, r in enumerate(todo, 1):
        print(f"[{n}/{len(todo)}] {r} ...", flush=True)
        try:
            resp = call("timeseries.get_range", {**base, "symbols": f"{r}.FUT", "encoding": "csv",
                                                  "compression": "none", "pretty_px": "true",
                                                  "pretty_ts": "true", "map_symbols": "true"})
            os.makedirs(f"{OUT}/../raw", exist_ok=True)
            with open(f"{OUT}/../raw/{r}.csv", "w") as f:     # keep what we paid for, before parsing
                f.write(resp.text)
            df = pd.read_csv(io.StringIO(resp.text))
            if "symbol" in df.columns:
                df = df[~df["symbol"].astype(str).str.contains(r"[-: ]", regex=True)]   # outrights only
            else:
                print(f"      note: no 'symbol' column; columns = {list(df.columns)}", flush=True)
            df.to_csv(f"{OUT}/{r}.csv", index=False)
            print(f"      OK  {len(df):6d} rows, {df['symbol'].nunique():3d} contracts", flush=True)
        except Exception as e:
            print(f"      FAIL {r}: {type(e).__name__}: {str(e)[:300]}", flush=True)
        time.sleep(0.3)

print("\nDone. Tell Claude it finished.")
