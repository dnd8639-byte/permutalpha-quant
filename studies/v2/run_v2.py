"""v2 main run (PREREGISTRATION_v2 + Amendment A1). Reads ../state/v2/, writes results_v2.txt
and returns_v2.csv next to this file. Needs numpy, pandas, scipy (+ openpyxl for the AQR file)."""
import io, os, sys, warnings
import numpy as np
import pandas as pd
import backtest_v2 as bt
import stats

warnings.filterwarnings("ignore")
D = "../state/v2"
out_lines = []


def p(s=""):
    print(s); out_lines.append(s)


# ------------------------------------------------------------------ parsers
def parse_french(path):
    """First block of a Ken French CSV (monthly), percent -> decimal, -99.99 -> NaN."""
    rows, header = [], None
    for line in open(path, encoding="latin-1"):
        parts = [x.strip() for x in line.rstrip("\n").split(",")]
        if header is None:
            if line.startswith(",") and len(parts) > 2:
                header = parts[1:]
            continue
        if len(parts[0]) == 6 and parts[0].isdigit():
            rows.append(parts[: len(header) + 1])
        elif rows:
            break
    df = pd.DataFrame(rows).set_index(0).astype(float)
    df.columns = header
    df.index = pd.to_datetime(df.index, format="%Y%m") + pd.offsets.MonthEnd(0)
    return df.replace([-99.99, -999], np.nan) / 100.0


def load_fred():
    import glob
    cols = {}
    for path in sorted(glob.glob(f"{D}/fred/*.csv")):
        s = pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]
        cols[os.path.basename(path)[:-4]] = s
    return pd.DataFrame(cols)


def eom(s):
    return s.dropna().resample("ME").last()


def prices_csv(name):
    df = pd.read_csv(f"{D}/{name}.csv", index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index).tz_localize(None) if getattr(df.index, "tz", None) else df.index
    return df.resample("ME").last().dropna(how="all")


# ------------------------------------------------------------------ universes
ff = parse_french(f"{D}/ff_factors_raw.csv")
RF = ff["RF"]
fred = load_fred()

U = {}      # name -> dict(rets, cost, double, start)

# old universes (prices, total returns, 10 bps flat as in v1)
for name, f in [("crypto", "crypto_prices"), ("etf20", "etf_prices"),
                ("etf39", "wide_prices"), ("futures", "fut_prices")]:
    if os.path.exists(f"{D}/{f}.csv"):
        U[name] = dict(rets=prices_csv(f).pct_change(fill_method=None), cost=10.0, double=False)

# U5: 49 industries, excess of RF
ind = parse_french(f"{D}/ff49_raw.csv")
U["U5_industries"] = dict(rets=ind.sub(RF.reindex(ind.index), axis=0), cost=10.0, double=True)

# U6: macro, end-of-month from daily FRED (Amendment A1)
m = {}
m["US_equity"] = ff["Mkt-RF"]
y = eom(fred["DGS10"]) / 100.0
m["US_10y"] = (y.shift(1) / 12 - 7.5 * y.diff()) - RF.reindex(y.index)
fx_inv = {"JPY": "DEXJPUS", "CAD": "DEXCAUS", "CHF": "DEXSZUS"}
fx_dir = {"GBP": "DEXUSUK", "AUD": "DEXUSAL"}
for k, sid in fx_inv.items():
    m[k] = (1.0 / eom(fred[sid])).pct_change(fill_method=None)
for k, sid in fx_dir.items():
    m[k] = eom(fred[sid]).pct_change(fill_method=None)
eur = eom(fred["DEXUSEU"]).loc["1999-01-01":]
if "DEXGEUS" in fred:
    dem_as_eur = 1.95583 / eom(fred["DEXGEUS"]).loc[:"1998-12-31"]
    m["EUR_DEM"] = pd.concat([dem_as_eur, eur]).pct_change(fill_method=None)
else:   # DEXGEUS failed to download -> dropped per prereg s.3; EUR runs from 1999 only
    m["EUR_DEM"] = eur.pct_change(fill_method=None)
wti = eom(fred["DCOILWTICO"])
m["WTI"] = wti.where(wti > 0).pct_change(fill_method=None)
u6 = pd.DataFrame(m).sort_index()
cost6 = pd.Series({"US_equity": 5, "US_10y": 3, "JPY": 3, "CAD": 3, "CHF": 3, "GBP": 3,
                   "AUD": 3, "EUR_DEM": 3, "WTI": 10}, dtype=float)
hist = u6.notna().cumsum()
start6 = ((hist >= 24).sum(axis=1) >= 5).idxmax()
U["U6_macro"] = dict(rets=u6, cost=cost6, double=True, start=start6)

# U6-avg: monthly-AVERAGE series (diagnostic only, Amendment A1)
avg_ids = ["EXJPUS", "EXUSUK", "EXCAUS", "EXSZUS", "EXUSAL", "WTISPLC", "PCOPPUSDM", "PALUMUSDM",
           "PNICKUSDM", "PZINCUSDM", "PWHEAMTUSDM", "PMAIZMTUSDM", "PSOYBUSDM", "PCOFFOTMUSDM",
           "PSUGAISAUSDM", "PCOTTINDUSDM", "PNGASUSUSDM"]
avg = {}
for sid in avg_ids:
    if sid in fred:
        s = fred[sid].dropna().resample("ME").last()
        if sid in ("EXJPUS", "EXCAUS", "EXSZUS"):
            s = 1.0 / s
        avg[sid] = s.pct_change(fill_method=None)
if avg:
    U["U6avg_DIAGNOSTIC"] = dict(rets=pd.DataFrame(avg), cost=10.0, double=False)

# ------------------------------------------------------------------ run
res = {}
for name, u in U.items():
    kw = dict(rets=u["rets"], cost_bps=u["cost"], double_pre1990=u["double"])
    r_c = bt.run_backtest(causal=True, **kw)["returns"]
    r_v1 = bt.run_backtest(causal=False, **kw)["returns"]
    if "start" in u:
        r_c, r_v1 = r_c.loc[u["start"]:], r_v1.loc[u["start"]:]
    sweep = {}
    for c in (0, 25, 50):
        rs = bt.run_backtest(rets=u["rets"], cost_bps=float(c), causal=True)["returns"]
        sweep[c] = stats.sharpe(rs.loc[u.get("start", rs.index[0]):])
    res[name] = dict(r=r_c, v1=stats.sharpe(r_v1), sweep=sweep, n_assets=u["rets"].shape[1])

TRIALS = ["crypto", "etf20", "etf39", "futures", "U5_industries", "U6_macro"]
trial_sr = [stats.sharpe(res[k]["r"]) for k in TRIALS if k in res]

p("=" * 78)
p("v2 RESULTS  (causal code = primary; costs per PREREGISTRATION_v2 s.4)")
p(f"DSR trials N = {len(trial_sr)}: " + ", ".join(f"{k}={s:+.3f}" for k, s in zip(TRIALS, trial_sr)))
p("=" * 78)
hdr = f"{'universe':18s} {'n':>3s} {'months':>6s} {'range':17s} {'SR':>6s} {'CI95':>15s} {'NW t':>6s} {'DSR':>5s} {'v1 SR':>6s}  G1 G2"
p(hdr); p("-" * len(hdr))
reports = {}
for name, d in res.items():
    rep = stats.full_report(d["r"].values, trial_sr)
    reports[name] = rep
    rng_ = f"{d['r'].index[0]:%Y-%m}..{d['r'].index[-1]:%Y-%m}"
    g = "" if name in ("U5_industries", "U6_macro") else "  (not gated)"
    p(f"{name:18s} {d['n_assets']:3d} {rep['months']:6d} {rng_:17s} {rep['sharpe']:+6.3f} "
      f"[{rep['ci_lo']:+.2f},{rep['ci_hi']:+.2f}] {rep['nw_t']:6.2f} {rep['dsr']:5.2f} {d['v1']:+6.3f}  "
      f"{'P' if rep['G1'] else 'F'}  {'P' if rep['G2'] else 'F'}{g}")
p(f"\nSR0 (expected max Sharpe of {len(trial_sr)} null trials, annualized): {reports['U5_industries']['sr0_ann']:.3f}")

p("\nDetail: ann return / ann vol / max DD / skew")
for name, rep in reports.items():
    p(f"  {name:18s} {rep['ann_ret']:+.3f} / {rep['ann_vol']:.3f} / {rep['max_dd']:+.3f} / {rep['skew']:+.2f}")

p("\nCost sweep (flat bps, causal): Sharpe at 0 / 25 / 50 bps")
for name, d in res.items():
    s = d["sweep"]; p(f"  {name:18s} {s[0]:+.3f} / {s[25]:+.3f} / {s[50]:+.3f}")

p("\nSubperiods (causal, net): Sharpe [months]")
cuts = [("pre-1990", None, "1989-12-31"), ("1990-2008", "1990-01-01", "2008-12-31"),
        ("2009-2015", "2009-01-01", "2015-12-31"), ("2016+ post-pub", "2016-01-01", None)]
p(f"  {'':18s}" + "".join(f"{c[0]:>18s}" for c in cuts))
for name, d in res.items():
    row = ""
    for _, a, b in cuts:
        s = d["r"].loc[a:b]
        row += f"{stats.sharpe(s):+8.2f} [{len(s):4d}]  " if len(s) >= 24 else f"{'-':>18s}"
    p(f"  {name:18s}{row}")

# ------------------------------------------------------------------ AQR benchmark
p("\nAQR TSMOM benchmark (diagnostic; expect corr > 0.3 for U6)")
try:
    raw = None
    for sh, df in pd.read_excel(f"{D}/aqr_tsmom.xlsx", sheet_name=None, header=None).items():
        hits = [i for i in range(len(df)) if "TSMOM" in [str(x).strip() for x in df.iloc[i]]]
        if hits:
            raw, hr = df, hits[0]; break
    if raw is None:
        raise ValueError("no sheet with a TSMOM header row")
    cols = ["Date"] + [str(c).strip() for c in raw.iloc[hr, 1:]]
    aq = raw.iloc[hr + 1:].copy(); aq.columns = cols
    aq.index = pd.to_datetime(aq["Date"], errors="coerce") + pd.offsets.MonthEnd(0)
    aq = aq[aq.index.notna()].drop(columns="Date").apply(pd.to_numeric, errors="coerce")
    aq = aq.loc[:, [c for c in aq.columns if c.startswith("TSMOM")]]
    for name in ("U6_macro", "U5_industries", "futures", "etf20"):
        if name not in res:
            continue
        j = pd.concat([res[name]["r"].rename("x"), aq], axis=1).dropna(subset=["x"])
        cs = {c: j["x"].corr(j[c]) for c in aq.columns if j[c].notna().sum() > 24}
        n = int(j.dropna().shape[0])
        p(f"  {name:18s} " + "  ".join(f"{c}={v:+.2f}" for c, v in cs.items()) + f"   (n={n})")
except Exception as e:
    p(f"  AQR file unavailable or unparsed: {e}")

# ------------------------------------------------------------------ verdict
g5 = reports["U5_industries"]["G1"] and reports["U5_industries"]["G2"]
g6 = reports["U6_macro"]["G1"] and reports["U6_macro"]["G2"]
p("\n" + "=" * 78)
p(f"GATES  U5_industries: G1 {'PASS' if reports['U5_industries']['G1'] else 'FAIL'}, "
  f"G2 {'PASS' if reports['U5_industries']['G2'] else 'FAIL'}   |   "
  f"U6_macro: G1 {'PASS' if reports['U6_macro']['G1'] else 'FAIL'}, G2 {'PASS' if reports['U6_macro']['G2'] else 'FAIL'}")
if g5 and g6:
    v = "PROFITABLE AFTER PREREGISTERED TESTS"
elif g5 or g6:
    v = f"MIXED: evidence only in {'U5_industries' if g5 else 'U6_macro'}"
else:
    v = "NO DEMONSTRABLE EDGE"
p(f"VERDICT (preregistered rule): {v}")
p("=" * 78)

open("results_v2.txt", "w").write("\n".join(out_lines) + "\n")
pd.DataFrame({k: d["r"] for k, d in res.items()}).to_csv("returns_v2.csv")
