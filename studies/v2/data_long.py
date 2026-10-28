"""Download the v2 long-history data (PREREGISTRATION_v2 section 3) and export the
existing universes to CSV. Run from the v2 folder:   python data_long.py
Needs only pandas + requests (already in your .venv). Writes to ../state/v2/."""
import io, os, zipfile, time
import pandas as pd
import requests

OUT = "../state/v2"
os.makedirs(OUT, exist_ok=True)
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/128.0 Safari/537.36", "Accept": "text/csv,*/*"}
log = []


def note(msg):
    print(msg); log.append(msg)


try:
    # yfinance ships curl_cffi; it presents a real Chrome TLS fingerprint, which
    # sites with bot protection (Yahoo, FRED) accept when python-requests is stalled.
    from curl_cffi import requests as cffi_requests
except Exception:
    cffi_requests = None


def get(url, tries=2, timeout=(10, 90)):
    for k in range(tries):
        try:
            if cffi_requests is not None:
                r = cffi_requests.get(url, impersonate="chrome", timeout=timeout[1])
            else:
                r = requests.get(url, headers=UA, timeout=timeout)
            r.raise_for_status()
            return r.content
        except Exception as e:
            if k == tries - 1:
                raise
            time.sleep(2)


# ---------- Ken French ----------
KF = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
for name, fname in [("F-F_Research_Data_Factors_CSV.zip", "ff_factors_raw.csv"),
                    ("49_Industry_Portfolios_CSV.zip", "ff49_raw.csv")]:
    if os.path.exists(f"{OUT}/{fname}"):
        note(f"skip Ken French {name} (already downloaded)"); continue
    try:
        z = zipfile.ZipFile(io.BytesIO(get(KF + name)))
        with open(f"{OUT}/{fname}", "wb") as f:
            f.write(z.read(z.namelist()[0]))
        note(f"OK   Ken French {name}")
    except Exception as e:
        note(f"FAIL Ken French {name}: {e}")

# ---------- FRED ----------
FRED_DAILY = ["DGS10", "DEXJPUS", "DEXUSUK", "DEXCAUS", "DEXSZUS", "DEXUSAL", "DEXGEUS",
              "DEXUSEU", "DCOILWTICO"]   # end-of-month taken later (Amendment A1)
FRED = FRED_DAILY + ["GS10", "TB3MS", "EXJPUS", "EXUSUK", "EXCAUS", "EXSZUS", "EXUSAL", "EXGEUS", "EXUSEU",
        "WTISPLC", "PCOPPUSDM", "PALUMUSDM", "PNICKUSDM", "PZINCUSDM", "PWHEAMTUSDM",
        "PMAIZMTUSDM", "PSOYBUSDM", "PCOFFOTMUSDM", "PSUGAISAUSDM", "PCOTTINDUSDM", "PNGASUSUSDM"]
os.makedirs(f"{OUT}/fred", exist_ok=True)


def fred_series(sid):
    """Try the CSV endpoint, then the plain-text endpoint."""
    try:
        df = pd.read_csv(io.BytesIO(get(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}")),
                         na_values=".")
        return pd.Series(df.iloc[:, 1].values, index=pd.to_datetime(df.iloc[:, 0]), name=sid)
    except Exception as e1:
        txt = get(f"https://fred.stlouisfed.org/data/{sid}.txt").decode("latin-1").splitlines()
        i = next(k for k, l in enumerate(txt) if l.strip().upper().startswith("DATE"))
        rows = [l.split() for l in txt[i + 1:] if l.strip()]
        return pd.Series(pd.to_numeric([r[1] for r in rows], errors="coerce"),
                         index=pd.to_datetime([r[0] for r in rows]), name=sid)


for n, sid in enumerate(FRED, 1):
    path = f"{OUT}/fred/{sid}.csv"
    if os.path.exists(path):
        note(f"skip FRED {sid} (already downloaded)"); continue
    print(f"[{n}/{len(FRED)}] FRED {sid} ...", flush=True)
    if n == 1:
        print("      (using " + ("curl_cffi / Chrome fingerprint" if cffi_requests else "plain requests") + ")", flush=True)
    try:
        s = fred_series(sid).dropna()
        s.to_csv(path)
        note(f"OK   FRED {sid:13s} {s.index[0].date()} -> {s.index[-1].date()}  ({len(s)} obs)")
    except Exception as e:
        note(f"FAIL FRED {sid}: {type(e).__name__}")
    time.sleep(0.5)

# ---------- AQR benchmark (raw xlsx; parsed later) ----------
AQR = ("https://www.aqr.com/-/media/AQR/Documents/Insights/Data-Sets/"
       "Time-Series-Momentum-Factors-Monthly.xlsx")
if os.path.exists(f"{OUT}/aqr_tsmom.xlsx"):
    note("skip AQR (already downloaded)")
else:
  try:
    with open(f"{OUT}/aqr_tsmom.xlsx", "wb") as f:
        f.write(get(AQR))
    note("OK   AQR TSMOM factors")
  except Exception as e:
    if os.path.exists(f"{OUT}/aqr_tsmom.xlsx"): os.remove(f"{OUT}/aqr_tsmom.xlsx")
    note(f"FAIL AQR TSMOM: {e}  -> download it by hand from aqr.com (Data Sets > "
         f"Time Series Momentum: Factors, Monthly) and save as state/v2/aqr_tsmom.xlsx")

# ---------- export existing universes to CSV (unchanged data) ----------
for p in ["etf_prices", "wide_prices", "fut_prices"]:
    try:
        pd.read_parquet(f"../state/{p}.parquet").to_csv(f"{OUT}/{p}.csv")
        note(f"OK   exported {p}")
    except Exception as e:
        note(f"FAIL export {p}: {e}")
try:
    import duckdb
    con = duckdb.connect("../state/ahl.duckdb", read_only=True)
    df = con.execute("SELECT ts, symbol, close FROM bars ORDER BY ts").df()
    con.close()
    df.pivot(index="ts", columns="symbol", values="close").to_csv(f"{OUT}/crypto_prices.csv")
    note("OK   exported crypto")
except Exception as e:
    note(f"FAIL export crypto: {e}")

with open(f"{OUT}/download_log.txt", "w") as f:
    f.write(pd.Timestamp.now().isoformat() + "\n" + "\n".join(log) + "\n")
print("\nDone. Tell Claude it finished.")
