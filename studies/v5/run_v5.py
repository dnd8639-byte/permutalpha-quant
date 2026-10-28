"""v5 main run (PREREGISTRATION_v5). Writes results_v5.txt and returns_v5.csv."""
import os, sys, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "v2")); sys.path.insert(0, os.path.join(HERE, "..", "v3"))
import stats
from build_clean import load_contracts, nominal_expiry
from roll_strategy import run_T9

SRC = os.path.join(HERE, "..", "state", "v3", "contracts")
out = []


def p(s=""):
    print(s, flush=True); out.append(s)


markets = {}
for m in ["CL", "HO", "RB", "NG"]:
    df = load_contracts(f"{SRC}/{m}.csv")
    first = df.groupby("symbol")["date"].min()
    exp = {k: nominal_expiry(k, first[k]) for k in first.index}
    df["dmonth"] = df["symbol"].map(lambda k: exp[k].to_period("M") if pd.notna(exp[k]) else pd.NaT)
    df = df.dropna(subset=["dmonth"]).rename(columns={"symbol": "key"})
    markets[m] = df[["date", "key", "close", "dmonth"]]
# v5-A1: business days = Mon-Fri with CL volume >= 20% of that month's median (drops Sunday/holiday stubs)
_cl = pd.read_csv(f"{SRC}/CL.csv", usecols=["ts_event", "volume"])
_cl["date"] = pd.to_datetime(_cl["ts_event"].astype(str).str[:10])
_v = _cl.groupby("date")["volume"].sum()
_med = _v.groupby(_v.index.to_period("M")).transform("median")
calendar = _v.index[(_v.index.dayofweek < 5) & (_v >= 0.2 * _med)]

t9, per, split = run_T9(markets, calendar)
t9g, perg, _ = run_T9(markets, calendar, bps=0)
t92, _, _ = run_T9(markets, calendar, bps=10)
plc, plcm, _ = run_T9(markets, calendar, window=(12, 20))

v2 = pd.read_csv(os.path.join(HERE, "..", "v2", "returns_v2.csv"), index_col=0, parse_dates=True)
v3 = pd.read_csv(os.path.join(HERE, "..", "v3", "returns_v3.csv"), index_col=0, parse_dates=True)
v4 = pd.read_csv(os.path.join(HERE, "..", "v4", "returns_v4.csv"), index_col=0, parse_dates=True)
trial = [stats.sharpe(v2[c].dropna()) for c in ["crypto", "etf20", "etf39", "futures", "U5_industries", "U6_macro"]]
trial += [stats.sharpe(v3[c].dropna()) for c in v3.columns] + [stats.sharpe(v4["T8"].dropna()), stats.sharpe(t9)]
rep = stats.full_report(t9.values, trial)
late = stats.sharpe(t9.loc["2018-01-01":])
G1, G2, G3 = rep["G1"], rep["G2"], late > 0

p("=" * 80)
p(f"v5 RESULTS  T9 index-roll front-running (CL HO RB NG), {len(t9)} months {t9.index[0]:%Y-%m}..{t9.index[-1]:%Y-%m}, N = {len(trial)}")
p("=" * 80)
p(f"Sharpe (net)  {rep['sharpe']:+.3f}   95% CI [{rep['ci_lo']:+.2f}, {rep['ci_hi']:+.2f}]   NW t {rep['nw_t']:.2f}   DSR {rep['dsr']:.2f}  (SR0 {rep['sr0_ann']:.3f})")
p(f"Sharpe gross  {stats.sharpe(t9g):+.3f}    2x cost {stats.sharpe(t92):+.3f}")
p(f"PLACEBO BD12-20 (net) {stats.sharpe(plc):+.3f}   -> mechanism predicts roll window > placebo")
p(f"2010-2017 {stats.sharpe(t9.loc[:'2017-12-31']):+.2f}   2018+ {late:+.2f}")
p(f"ann ret {rep['ann_ret']:+.4f}  vol {rep['ann_vol']:.4f}  maxDD {rep['max_dd']:+.3f}  skew {rep['skew']:+.2f}  hit rate {(t9 > 0).mean():.0%}")
p("\nPer market: net | gross | placebo  ; mean monthly P&L (units of 10%-vol risk) BD1-4 vs BD5-9")
for m in per.columns:
    s = split[m]
    p(f"  {m}  {stats.sharpe(per[m].dropna()):+.3f} | {stats.sharpe(perg[m].dropna()):+.3f} | {stats.sharpe(plcm[m].dropna()):+.3f}   "
      f"BD1-4 {s['bd1_4'].mean()*1e4:+.1f} bp   BD5-9 {s['bd5_9'].mean()*1e4:+.1f} bp   (n={len(s)})")
p("\n" + "=" * 80)
p(f"GATES  G1 {'PASS' if G1 else 'FAIL'}   G2 {'PASS' if G2 else 'FAIL'}   G3 {'PASS' if G3 else 'FAIL'}")
p(f"VERDICT (preregistered): {'EDGE FOUND' if (G1 and G2 and G3) else 'NO EDGE'}")
p("=" * 80)
open(os.path.join(HERE, "results_v5.txt"), "w").write("\n".join(out) + "\n")
pd.DataFrame({"T9": t9, "T9_gross": t9g, "placebo": plc}).to_csv(os.path.join(HERE, "returns_v5.csv"))
