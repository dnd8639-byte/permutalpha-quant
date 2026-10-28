"""H7-H9 (PREREGISTRATION_v3 s.3). Must pass before any real-data run."""
import sys, os
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "v2"))
import stats
from strategies import engine, pos_T4, pos_T5, pos_T6

rng = np.random.default_rng(2027)
N_SIMS, T, N = 60, 192, 20
idx = pd.date_range("2010-07-31", periods=T, freq="ME")


def garch(T, N):
    r = np.empty((T, N)); s = np.full(N, 0.05)
    for t in range(T):
        r[t] = s * rng.standard_t(5, N) / np.sqrt(5 / 3)
        s = np.sqrt(0.0005 + 0.1 * r[t] ** 2 + 0.85 * s ** 2)
    return pd.DataFrame(r, index=idx)


def carry_world(k):
    """Persistent carry per market; next-month return = k * carry + noise."""
    c = np.zeros((T, N)); c[0] = rng.normal(0, 0.05, N)
    for t in range(1, T):
        c[t] = 0.9 * c[t - 1] + rng.normal(0, 0.02, N)
    r = np.zeros((T, N))
    r[1:] = k * c[:-1] / 12 * 6 + 0.04 * rng.standard_normal((T - 1, N))
    return pd.DataFrame(r, index=idx), pd.DataFrame(c, index=idx)


def summ(name, xs, need, ok):
    x = np.array(xs); se = x.std(ddof=1) / np.sqrt(len(x)); p = ok(x.mean(), se)
    print(f"{name:40s} mean SR {x.mean():+.3f} (SE {se:.3f})  need {need:16s} {'PASS' if p else 'FAIL'}")
    return p


res = []
rws = [garch(T, N) for _ in range(N_SIMS)]
res.append(summ("H7a T4 slow TSMOM on random walk", [stats.sharpe(engine(R, pos_T4(R))) for R in rws], "|mean|<2SE", lambda m, se: abs(m) < 2 * se))
res.append(summ("H7b T5 XS momentum on random walk", [stats.sharpe(engine(R, pos_T5(R))) for R in rws], "|mean|<2SE", lambda m, se: abs(m) < 2 * se))
res.append(summ("H8  T5 canary: signal uses month t", [stats.sharpe(engine(R, pos_T5(R, leak=True))) for R in rws], "> 1.0", lambda m, se: m > 1.0))
planted, control = [], []
for _ in range(N_SIMS):
    R, C = carry_world(1.0); planted.append(stats.sharpe(engine(R, pos_T6(R, C))))
    R, C = carry_world(0.0); control.append(stats.sharpe(engine(R, pos_T6(R, C))))
res.append(summ("H9a T6 carry, planted premium", planted, "> 0.5", lambda m, se: m > 0.5))
res.append(summ("H9b T6 carry, zero-premium control", control, "|mean|<2SE", lambda m, se: abs(m) < 2 * se))
print("\nHARNESS v3:", "ALL PASS" if all(res) else "FAILED - do not run real data")
