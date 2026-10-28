"""v3 main run (PREREGISTRATION_v3). Reads ../state/v3/clean_*.csv (+ v2 files for context).
Writes results_v3.txt and returns_v3.csv."""
import os, sys, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "v2"))
import stats
from strategies import engine, pos_T0, pos_T4, pos_T5, pos_T6, blend

S3, S2 = "../state/v3", "../state/v2"
out = []


def p(s=""):
    print(s, flush=True); out.append(s)


R = pd.read_csv(f"{S3}/clean_monthly.csv", index_col=0, parse_dates=True)
C = pd.read_csv(f"{S3}/clean_carry.csv", index_col=0, parse_dates=True)
ROLL = pd.read_csv(f"{S3}/clean_rolls.csv", index_col=0, parse_dates=True)
dropped = [c for c in R.columns if R[c].notna().sum() < 36]
R = R.drop(columns=dropped)
C = C.reindex(columns=R.columns)
ROLL = ROLL.reindex(index=R.index, columns=R.columns).fillna(0)

CLS = {**{k: "equity" for k in ["ES", "NQ", "YM", "RTY", "NKD"]},
       **{k: "rates" for k in ["ZT", "ZF", "ZN", "ZB", "UB"]},
       **{k: "fx" for k in ["6E", "6J", "6B", "6A", "6C", "6S", "6M"]},
       **{k: "energy" for k in ["CL", "BZ", "NG", "HO", "RB"]},
       **{k: "metals" for k in ["GC", "SI", "HG", "PL", "PA"]},
       **{k: "grains" for k in ["ZC", "ZW", "ZS", "ZM", "ZL", "ZO"]},
       **{k: "livestock" for k in ["LE", "HE"]}}
BPS = {"equity": 2, "rates": 2, "fx": 3, "energy": 5, "metals": 5, "grains": 8, "livestock": 10}
cost = pd.Series({c: BPS[CLS.get(c, "grains")] for c in R.columns}, dtype=float)

p("=" * 84)
p(f"v3 RESULTS  clean futures: {R.shape[1]} markets, {R.index[0]:%Y-%m} -> {R.index[-1]:%Y-%m}")
if dropped:
    p(f"dropped (<36 months): {dropped}")
p("=" * 84)


def run_all(mult=1.0):
    k = dict(cost_bps=cost * mult, rolls=ROLL)
    s = {"T0_mom14": engine(R, pos_T0(R), **k),
         "T4_tsmom12": engine(R, pos_T4(R), **k),
         "T5_xsmom": engine(R, pos_T5(R), **k),
         "T6_carry": engine(R, pos_T6(R, C), **k)}
    s["T7_blend"] = blend([s["T4_tsmom12"], s["T5_xsmom"], s["T6_carry"]])
    return s


S = run_all(1.0)
S2x = run_all(2.0)

prev = pd.read_csv(os.path.join(HERE, "..", "v2", "returns_v2.csv"), index_col=0, parse_dates=True)
trial_sr = [stats.sharpe(prev[c].dropna()) for c in ["crypto", "etf20", "etf39", "futures", "U5_industries", "U6_macro"]]
trial_sr += [stats.sharpe(S[k]) for k in S]
p(f"DSR trials N = {len(trial_sr)}")

hdr = f"{'strategy':12s} {'months':>6s} {'range':17s} {'SR':>6s} {'CI95':>15s} {'NW t':>6s} {'DSR':>5s} {'2018+':>6s} {'2x cost':>7s}  G1 G2 G3"
p(hdr); p("-" * len(hdr))
verdicts = {}
for k, r in S.items():
    rep = stats.full_report(r.values, trial_sr)
    late = stats.sharpe(r.loc["2018-01-01":])
    g3 = late > 0
    verdicts[k] = rep["G1"] and rep["G2"] and g3
    gated = "" if k != "T0_mom14" else "  (data-fix measure; not in verdict)"
    p(f"{k:12s} {rep['months']:6d} {r.index[0]:%Y-%m}..{r.index[-1]:%Y-%m} {rep['sharpe']:+6.3f} "
      f"[{rep['ci_lo']:+.2f},{rep['ci_hi']:+.2f}] {rep['nw_t']:6.2f} {rep['dsr']:5.2f} {late:+6.2f} {stats.sharpe(S2x[k]):+7.3f}  "
      f"{'P' if rep['G1'] else 'F'}  {'P' if rep['G2'] else 'F'}  {'P' if g3 else 'F'}{gated}")
p(f"SR0 (expected max of {len(trial_sr)} null trials, annualized): {rep['sr0_ann']:.3f}")

p("\nSubperiods: Sharpe 2010-2017 | 2018+   ann ret / vol / maxDD / skew")
for k, r in S.items():
    rep = stats.full_report(r.values, trial_sr)
    p(f"  {k:12s} {stats.sharpe(r.loc[:'2017-12-31']):+.2f} | {stats.sharpe(r.loc['2018-01-01':]):+.2f}    "
      f"{rep['ann_ret']:+.3f} / {rep['ann_vol']:.3f} / {rep['max_dd']:+.3f} / {rep['skew']:+.2f}")

p("\nCorrelations between strategies")
p(pd.DataFrame(S).corr().round(2).to_string())

# ---------------- data-fix report: Yahoo vs clean
p("\nDATA FIX: Yahoo =F vs clean, monthly, overlapping months")
try:
    Y = pd.read_csv(f"{S2}/fut_prices.csv", index_col=0, parse_dates=True).resample("ME").last().pct_change(fill_method=None)
    rows = []
    for c in R.columns:
        yc = f"{c}=F"
        if yc in Y:
            j = pd.concat([R[c], Y[yc]], axis=1, keys=["clean", "yahoo"]).dropna()
            if len(j) > 24:
                rows.append((c, CLS.get(c, "?"), j["clean"].corr(j["yahoo"]), (j["yahoo"].mean() - j["clean"].mean()) * 12, len(j)))
    df = pd.DataFrame(rows, columns=["mkt", "class", "corr", "yahoo_minus_clean_ann", "n"]).sort_values("yahoo_minus_clean_ann")
    p(df.to_string(index=False, float_format=lambda v: f"{v:+.3f}"))
    p(f"  median |annual contamination|: {df['yahoo_minus_clean_ann'].abs().median():.2%}   "
      f"worst: {df.iloc[df['yahoo_minus_clean_ann'].abs().argmax()]['mkt']} {df['yahoo_minus_clean_ann'].abs().max():.1%}/yr")
    common = [c for c in R.columns if f"{c}=F" in Y]
    Yc = Y[[f"{c}=F" for c in common]].loc[R.index[0]:R.index[-1]]; Yc.columns = common
    yahoo_t0 = engine(Yc, pos_T0(Yc), cost_bps=cost.reindex(common))
    p(f"  T0 mom(1,4), same markets & months:  Yahoo SR {stats.sharpe(yahoo_t0):+.3f}   clean SR {stats.sharpe(S['T0_mom14']):+.3f}")
except Exception as e:
    p(f"  unavailable: {e}")

# ---------------- AQR + U6 cross-check
p("\nContext")
try:
    a = pd.read_excel(f"{S2}/aqr_tsmom.xlsx", header=None).iloc[18:, :2].dropna()
    a.index = pd.to_datetime(a.iloc[:, 0]) + pd.offsets.MonthEnd(0); aq = a.iloc[:, 1].astype(float)
    p("  corr with AQR TSMOM: " + "  ".join(f"{k}={S[k].corr(aq):+.2f}" for k in S))
except Exception as e:
    p(f"  AQR unavailable: {e}")
try:
    sys.path.insert(0, os.path.join(HERE, "..", "v2"))
    cwd = os.getcwd(); os.chdir(os.path.join(HERE, "..", "v2"))
    src = open("run_v2.py").read().split("# ------------------------------------------------------------------ run")[0]
    g = {}; exec(src, g); os.chdir(cwd)
    u6r = g["U"]["U6_macro"]["rets"]
    t4u6 = engine(u6r, pos_T4(u6r), cost_bps=g["cost6"]).loc[g["start6"]:]
    p(f"  T4 on U6 macro 1973+: SR {stats.sharpe(t4u6):+.3f}  (pre-2016 {stats.sharpe(t4u6.loc[:'2015-12-31']):+.2f}, "
      f"2016+ {stats.sharpe(t4u6.loc['2016-01-01':]):+.2f})")
except Exception as e:
    p(f"  U6 cross-check unavailable: {e}")

p("\n" + "=" * 84)
new = {k: v for k, v in verdicts.items() if k != "T0_mom14"}
for k, v in new.items():
    p(f"  {k:12s} {'EDGE (G1+G2+G3 pass)' if v else 'no edge'}")
p(f"VERDICT (preregistered): {'EDGE FOUND in ' + ', '.join(k for k, v in new.items() if v) if any(new.values()) else 'NO EDGE FOUND'}")
p("=" * 84)
open("results_v3.txt", "w").write("\n".join(out) + "\n")
pd.DataFrame(S).to_csv("returns_v3.csv")
