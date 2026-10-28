"""Addendum B secondary analyses S1-S5 (PREREGISTRATION_v2). Verdict is locked; these explain it.
Usage: python secondary.py [n_perm]      (default 1000; S1 is the slow part)"""
import sys, os, warnings
import numpy as np
import pandas as pd
from multiprocessing import Pool
from scipy.stats import norm, skew, kurtosis, ttest_ind

warnings.filterwarnings("ignore")
_src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_v2.py")).read()
exec(_src.split("# ------------------------------------------------------------------ run")[0])
import backtest_v2 as bt
import stats

N_PERM = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
out = []


def p(s=""):
    print(s, flush=True); out.append(s)


def strat(name, rets=None):
    u = U[name]
    r = bt.run_backtest(rets=u["rets"] if rets is None else rets, cost_bps=u["cost"],
                        double_pre1990=u["double"])["returns"]
    return r.loc[u["start"]:] if "start" in u else r


def nw_ols(y, X, lags=12):
    X = np.column_stack([np.ones(len(y)), X]); b = np.linalg.lstsq(X, y, rcond=None)[0]
    e = y - X @ b; T = len(y); XtX_inv = np.linalg.inv(X.T @ X)
    S = (X * e[:, None]).T @ (X * e[:, None]) / T
    for L in range(1, lags + 1):
        w = 1 - L / (lags + 1); G = (X[L:] * e[L:, None]).T @ (X[:-L] * e[:-L, None]) / T
        S += w * (G + G.T)
    se = np.sqrt(np.diag(T * XtX_inv @ S @ XtX_inv))
    return b, b / se, 1 - e.var() / y.var()


# ---------------- S1 permutation (within blocks of identical asset availability)
def _perm_job(args):
    name, seed = args
    rets = U[name]["rets"]
    pat = rets.notna().apply(tuple, axis=1)
    block = (pat != pat.shift()).cumsum().values
    rng = np.random.default_rng(seed)
    order = np.arange(len(rets))
    for b_ in np.unique(block):
        idx = np.where(block == b_)[0]
        order[idx] = rng.permutation(idx)
    shuf = pd.DataFrame(rets.values[order], index=rets.index, columns=rets.columns)
    return stats.sharpe(strat(name, shuf))


R = {n: strat(n) for n in ("U5_industries", "U6_macro")}
p(f"SECONDARY ANALYSES (Addendum B). Verdict locked: NO DEMONSTRABLE EDGE.  n_perm={N_PERM}")
p("=" * 78)

p("\nS1  Permutation test (does ordering matter?)")
with Pool(2) as pool:
    for n, r in R.items():
        real = stats.sharpe(r)
        null = np.array(pool.map(_perm_job, [(n, 42 + k) for k in range(N_PERM)]))
        null = null[np.isfinite(null)]
        pv = (np.sum(null >= real) + 1) / (len(null) + 1)
        p(f"  {n:14s} real SR {real:+.3f} | null mean {null.mean():+.3f} sd {null.std():.3f} "
          f"95th {np.percentile(null, 95):+.3f} | p = {pv:.4f}  -> {'ordering matters' if pv < 0.05 else 'not distinguishable from shuffled'}")

# ---------------- S2 factor regressions
p("\nS2  Factor regressions (Newey-West 12)")
F = ff[["Mkt-RF", "SMB", "HML"]]
d = pd.concat([R["U5_industries"].rename("y"), F], axis=1).dropna()
b, t, r2 = nw_ols(d["y"].values, d[["Mkt-RF", "SMB", "HML"]].values)
p(f"  U5 ~ FF3:   alpha {b[0]*12:+.3%}/yr (t {t[0]:+.2f})  Mkt {b[1]:+.3f} (t {t[1]:+.2f})  SMB {b[2]:+.3f} (t {t[2]:+.2f})  HML {b[3]:+.3f} (t {t[3]:+.2f})  R2 {r2:.3f}  n={len(d)}")
d = pd.concat([R["U6_macro"].rename("y"), F["Mkt-RF"], u6["US_10y"].rename("Bond")], axis=1).dropna()
b, t, r2 = nw_ols(d["y"].values, d[["Mkt-RF", "Bond"]].values)
p(f"  U6 ~ Mkt+Bond: alpha {b[0]*12:+.3%}/yr (t {t[0]:+.2f})  Mkt {b[1]:+.3f} (t {t[1]:+.2f})  Bond {b[2]:+.3f} (t {t[2]:+.2f})  R2 {r2:.3f}  n={len(d)}")

# ---------------- S3 crisis alpha
p("\nS3  Crisis alpha: worst 10% of US equity months")
for n, r in R.items():
    d = pd.concat([r.rename("y"), ff["Mkt-RF"].rename("m")], axis=1).dropna()
    cut = d["m"].quantile(0.10); cr = d[d["m"] <= cut]; ok = d[d["m"] > cut]
    tt = ttest_ind(cr["y"], ok["y"], equal_var=False)
    p(f"  {n:14s} crisis months n={len(cr):3d} (equity avg {cr['m'].mean():+.2%}): strategy {cr['y'].mean():+.2%}/mo "
      f"vs other {ok['y'].mean():+.2%}/mo  diff t {tt.statistic:+.2f}  | corr w/ equity in crises {cr['y'].corr(cr['m']):+.2f}")
    for lab, a, b_ in [("pre-2016", None, "2015-12-31"), ("2016+", "2016-01-01", None)]:
        c2 = cr.loc[a:b_]
        if len(c2) >= 5:
            p(f"      {lab:9s} crisis months n={len(c2):3d}: strategy {c2['y'].mean():+.2%}/mo")

# ---------------- S4 structural break
p("\nS4  Post-2016 break: r = a + b*1[t>=2016]")
for n, r in R.items():
    dummy = (r.index >= "2016-01-01").astype(float)
    b, t, _ = nw_ols(r.values, dummy[:, None])
    p(f"  {n:14s} pre mean {b[0]*12:+.2%}/yr  change b {b[1]*12:+.2%}/yr  NW t {t[1]:+.2f}  -> "
      f"{'decay statistically real' if t[1] < -2 else 'decline not distinguishable from noise'}")

# ---------------- S5 power
p("\nS5  Power / detectability")
trial = pd.read_csv("returns_v2.csv", index_col=0, parse_dates=True)
tr = [stats.sharpe(trial[c].dropna()) for c in ["crypto", "etf20", "etf39", "futures", "U5_industries", "U6_macro"]]
rng = np.random.default_rng(7)
for n, r in R.items():
    sr = stats.sharpe(r); T = len(r); x = r.values
    y2, y3 = (2 / sr) ** 2, (3 / sr) ** 2
    # chance G2 passes if the observed SR were the TRUE SR: iid resamples of the actual monthly returns
    base = x
    passes = 0
    for k in range(2000):
        s = rng.choice(base, T, replace=True)
        dsr, _ = stats.deflated_sharpe(s, tr)
        passes += dsr > 0.95
    p(f"  {n:14s} SR {sr:.3f}: years needed for t=2: {y2:5.0f}   t=3: {y3:5.0f}   | sample has {T/12:.0f} yrs"
      f" | P(G2 pass | true SR = observed) = {passes/2000:.0%}")

open("secondary_results.txt", "w").write("\n".join(out) + "\n")
