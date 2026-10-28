"""v4 main run (PREREGISTRATION_v4 + A1). Writes results_v4.txt and returns_v4.csv."""
import os, sys, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore"); np.seterr(all="ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "v2"))
import stats
from ou_spreads import build_T8, to_monthly, SPREADS

S3 = os.path.join(HERE, "..", "state", "v3")
out = []


def p(s=""):
    print(s, flush=True); out.append(s)


Rd = pd.read_csv(f"{S3}/clean_daily.csv", index_col=0, parse_dates=True)
ROLL = pd.read_csv(f"{S3}/clean_rolls.csv", index_col=0, parse_dates=True)
BPS = {"ZS": 8, "ZM": 8, "ZL": 8, "CL": 5, "HO": 5, "RB": 5, "BZ": 5, "ZF": 2, "ZN": 2, "GC": 5, "SI": 5}

t8, per, info = build_T8(Rd, BPS, rolls_m=ROLL)
t8_2x, per2x, _ = build_T8(Rd, BPS, cost_mult=2.0, rolls_m=ROLL)
t8_0, per0, _ = build_T8(Rd, {k: 0 for k in BPS})
first = per.abs().sum(axis=1).gt(0).idxmax()                    # first day any spread is live
M = to_monthly(t8.loc[first:]); M2 = to_monthly(t8_2x.loc[first:]); M0 = to_monthly(t8_0.loc[first:])

v2 = pd.read_csv(os.path.join(HERE, "..", "v2", "returns_v2.csv"), index_col=0, parse_dates=True)
v3 = pd.read_csv(os.path.join(HERE, "..", "v3", "returns_v3.csv"), index_col=0, parse_dates=True)
trial = [stats.sharpe(v2[c].dropna()) for c in ["crypto", "etf20", "etf39", "futures", "U5_industries", "U6_macro"]]
trial += [stats.sharpe(v3[c].dropna()) for c in v3.columns] + [stats.sharpe(M)]
rep = stats.full_report(M.values, trial)
late = stats.sharpe(M.loc["2018-01-01":])
G1, G2, G3 = rep["G1"], rep["G2"], late > 0

p("=" * 80)
p(f"v4 RESULTS  T8 OU mean reversion, {len(M)} months {M.index[0]:%Y-%m}..{M.index[-1]:%Y-%m}, DSR N = {len(trial)}")
p("=" * 80)
p(f"Sharpe (net)     {rep['sharpe']:+.3f}   95% CI [{rep['ci_lo']:+.2f}, {rep['ci_hi']:+.2f}]   NW t {rep['nw_t']:.2f}   DSR {rep['dsr']:.2f}  (SR0 {rep['sr0_ann']:.3f})")
p(f"Sharpe gross     {stats.sharpe(M0):+.3f}     2x cost {stats.sharpe(M2):+.3f}")
p(f"2010-2017 {stats.sharpe(M.loc[:'2017-12-31']):+.2f}   2018+ {late:+.2f}")
p(f"ann ret {rep['ann_ret']:+.3f}  vol {rep['ann_vol']:.3f}  maxDD {rep['max_dd']:+.3f}  skew {rep['skew']:+.2f}")
p("\nPer spread (monthly, net | gross):  in-market share, trades, median half-life (days)")
for k in per.columns:
    a = to_monthly(per[k].loc[first:]); b = to_monthly(per0[k].loc[first:])
    i = info[k]
    p(f"  {k:15s} {stats.sharpe(a):+.3f} | {stats.sharpe(b):+.3f}   in-mkt {i['in_market']:.0%}  trades {i['trades']:4d}  HL {i['med_hl']:.1f}")
p("\nCorrelation of T8 with v3 strategies (monthly)")
j = pd.concat([M.rename("T8"), v3], axis=1).dropna()
p("  " + "  ".join(f"{c}={j['T8'].corr(j[c]):+.2f}" for c in v3.columns))
p("\n" + "=" * 80)
p(f"GATES  G1 {'PASS' if G1 else 'FAIL'}   G2 {'PASS' if G2 else 'FAIL'}   G3 {'PASS' if G3 else 'FAIL'}")
p(f"VERDICT (preregistered): {'EDGE FOUND' if (G1 and G2 and G3) else 'NO EDGE'}")
p("=" * 80)
open(os.path.join(HERE, "results_v4.txt"), "w").write("\n".join(out) + "\n")
pd.DataFrame({"T8": M, "T8_gross": M0}).to_csv(os.path.join(HERE, "returns_v4.csv"))
