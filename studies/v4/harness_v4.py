"""H10-H12 (PREREGISTRATION_v4 s.5)."""
import sys, os
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "v2"))
import stats
from ou_spreads import build_T8, to_monthly

rng = np.random.default_rng(404)
N_SIMS, T = 40, 252 * 12
idx = pd.bdate_range("2010-06-07", periods=T)
SP = {"P": ("Y", ["X"])}
BPS = {"Y": 0.0, "X": 0.0}


def world(coint):
    common = np.cumsum(0.01 * rng.standard_normal(T))
    if coint:                                   # OU spread, half-life 10 days
        phi = 0.5 ** (1 / 10); u = np.zeros(T)
        for t in range(1, T):
            u[t] = phi * u[t - 1] + 0.004 * rng.standard_normal()
        Lx, Ly = common, 0.8 * common + u
    else:
        Lx, Ly = common, np.cumsum(0.01 * rng.standard_normal(T))
    L = pd.DataFrame({"Y": Ly, "X": Lx}, index=idx)
    return np.expm1(L.diff()).fillna(0.0)


def sr(Rd, leak=False):
    t8, _, _ = build_T8(Rd, BPS, spreads=SP, leak=leak)
    v = stats.sharpe(to_monthly(t8))
    return 0.0 if not np.isfinite(v) else v      # never trading = zero Sharpe


def summ(name, xs, need, ok):
    x = np.array(xs); se = x.std(ddof=1) / np.sqrt(len(x)); p = ok(x.mean(), se)
    print(f"{name:44s} mean SR {x.mean():+.3f} (SE {se:.3f})  need {need:14s} {'PASS' if p else 'FAIL'}")
    return p


res = []
res.append(summ("H10 cointegrated pair (OU, HL 10d)", [sr(world(True)) for _ in range(N_SIMS)], "> 1.0", lambda m, se: m > 1.0))
rw = [world(False) for _ in range(N_SIMS)]
h11 = [sr(R) for R in rw]
res.append(summ("H11 independent random walks", h11, "|mean|<2SE", lambda m, se: abs(m) < 2 * se))
res.append(summ("H12 canary: hindsight hedge ratio + OU fit", [sr(R, leak=True) for R in rw], "> H11 + 3SE", lambda m, se: m > np.mean(h11) + 3 * se))
print("\nHARNESS v4:", "ALL PASS" if all(res) else "FAILED - do not run real data")
