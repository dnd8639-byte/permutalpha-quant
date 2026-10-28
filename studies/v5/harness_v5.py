"""H13-H15 (PREREGISTRATION_v5 s.5)."""
import sys, os
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "v2"))
import stats
from roll_strategy import run_T9, bd_table

rng = np.random.default_rng(505)
days = pd.bdate_range("2010-06-01", "2026-08-31")
BT = bd_table(days)
N_SIMS = 20


def synth_market(pressure_bps):
    spot = 60 * np.exp(np.cumsum(0.02 * rng.standard_normal(len(days))))
    rows = []
    months = pd.period_range("2010-07", "2026-11", freq="M")
    roll_day = {d: (p, b) for d, p, b in zip(BT.date, BT.per, BT.bd)}
    for k, dm in enumerate(months):
        exp = dm.to_timestamp("M")
        live = (days >= (exp - pd.Timedelta(days=150))) & (days <= exp - pd.Timedelta(days=10))
        adj, idio = 0.0, 0.0
        for i in np.where(live)[0]:
            d = days[i]
            p, b = roll_day[d]
            if dm == p + 1 and 5 <= b <= 9:            # this contract is being rolled out of
                adj -= pressure_bps / 1e4
            idio += 0.003 * rng.standard_normal()
            ttm = (exp - d).days / 365
            rows.append((d, f"K{k}", spot[i] * np.exp(0.05 * ttm + adj + idio), dm))
    return pd.DataFrame(rows, columns=["date", "key", "close", "dmonth"])


def sr(markets, **kw):
    t9, _, _ = run_T9(markets, days, **kw)
    v = stats.sharpe(t9)
    return 0.0 if not np.isfinite(v) else v


def summ(name, xs, need, ok):
    x = np.array(xs); se = x.std(ddof=1) / np.sqrt(len(x)); p = ok(x.mean(), se)
    print(f"{name:44s} mean SR {x.mean():+.3f} (SE {se:.3f})  need {need:12s} {'PASS' if p else 'FAIL'}")
    return p


res = []
planted = [{m: synth_market(15) for m in "ABCD"} for _ in range(N_SIMS)]
res.append(summ("H13 planted roll pressure (15 bps/day)", [sr(w, bps=0) for w in planted], "> 1.0", lambda m, se: m > 1.0))
null = [{m: synth_market(0) for m in "ABCD"} for _ in range(N_SIMS)]
res.append(summ("H14 no roll pressure", [sr(w, bps=0) for w in null], "|mean|<2SE", lambda m, se: abs(m) < 2 * se))
res.append(summ("H15 canary: direction from realized outcome", [sr(w, bps=0, choose_by_outcome=True) for w in null], "> 1.0", lambda m, se: m > 1.0))
print("\nHARNESS v5:", "ALL PASS" if all(res) else "FAILED - do not run real data")
