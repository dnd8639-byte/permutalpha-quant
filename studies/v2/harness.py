"""Validation harness (PREREGISTRATION_v2 section 2). Must pass before any real-data run.
Usage: python harness.py"""
import numpy as np
import pandas as pd
import backtest_v2 as bt
import signals  # resolved to ../research/signals.py by backtest_v2
import stats

N_SIMS, T, N = 100, 240, 10
rng = np.random.default_rng(2026)
idx = pd.date_range("2000-01-31", periods=T, freq="ME")


def garch_t(T, N, rng):
    r = np.empty((T, N)); s = np.full(N, 0.04)
    for t in range(T):
        r[t] = s * rng.standard_t(5, N) / np.sqrt(5 / 3)
        s = np.sqrt(0.0004 + 0.1 * r[t] ** 2 + 0.85 * s ** 2)
    return r


def planted_trend(T, N, rng, sigma=0.04, stay=0.95):
    state = rng.choice([-1, 1], N); r = np.empty((T, N))
    for t in range(T):
        flip = rng.random(N) > stay
        state = np.where(flip, -state, state)
        r[t] = 0.5 * sigma * state + sigma * rng.standard_normal(N)
    return r


def leaky_signal(returns, vol):
    # same as signals.momentum_signal but uses lags 0..3 -> includes r_t (deliberate leak)
    w = np.full(signals.N_LAGS, 1.0 / signals.N_LAGS)
    stacked = sum(w[i] * returns.shift(i) for i in range(signals.N_LAGS))
    return (stacked / (vol * np.sqrt((w ** 2).sum()))).clip(-signals.SIGNAL_CAP, signals.SIGNAL_CAP)


def run(r, **kw):
    return stats.sharpe(bt.run_backtest(rets=pd.DataFrame(r, index=idx), cost_bps=0.0, **kw)["returns"])


def summary(name, x, cond, ok):
    x = np.asarray(x); se = x.std(ddof=1) / np.sqrt(len(x))
    print(f"{name:34s} mean SR {x.mean():+.3f}  (SE {se:.3f})  need {cond:22s} {'PASS' if ok(x.mean(), se) else 'FAIL'}")
    return ok(x.mean(), se)


results = []
h1 = [run(planted_trend(T, N, rng)) for _ in range(N_SIMS)]
results.append(summary("H1  planted trend, causal", h1, "> 0.5", lambda m, se: m > 0.5))

rw = [garch_t(T, N, rng) for _ in range(N_SIMS)]
h2 = [run(r) for r in rw]
results.append(summary("H2  random walk, causal", h2, "|mean| < 2 SE", lambda m, se: abs(m) < 2 * se))
h3a = [run(r, causal=False) for r in rw]
results.append(summary("H3a canary: v1 code on random walk", h3a, "> 3 SE (leak detected)", lambda m, se: m > 3 * se))

orig = bt.momentum_signal
bt.momentum_signal = leaky_signal
h3b = [run(r) for r in rw]
bt.momentum_signal = orig
results.append(summary("H3b canary: signal uses r_t", h3b, "> 1.0", lambda m, se: m > 1.0))

# H4a: bootstrap CI coverage on iid data with known Sharpe
true_sr = 0.5; cover = 0
for k in range(200):
    r = 0.10 / np.sqrt(12) * (rng.standard_normal(240) + true_sr / np.sqrt(12))
    lo, hi = stats.bootstrap_sharpe_ci(r, n_boot=2000, seed=k)
    cover += lo <= true_sr <= hi
ok4a = 0.90 <= cover / 200 <= 0.98
print(f"{'H4a bootstrap CI coverage':34s} {cover/200:.1%} of 200    need 90%-98%            {'PASS' if ok4a else 'FAIL'}")

# H4b: Bailey & Lopez de Prado (2014) worked example: SR_ann 2.5, T=1250 daily, N=100, V[SR_ann]=0.5,
#      skew -3, kurtosis 10  ->  SR0 = 0.1132 (daily), DSR = 0.9004
sr0 = stats.expected_max_sr(0.5 / 250, 100)
dsr = stats.psr(2.5 / np.sqrt(250), sr0, 1250, -3, 10)
ok4b = round(sr0, 4) == 0.1132 and round(dsr, 2) == 0.90
print(f"{'H4b DSR worked example':34s} SR0 {sr0:.4f} DSR {dsr:.4f}   need 0.1132 / 0.90        {'PASS' if ok4b else 'FAIL'}")

results += [ok4a, ok4b]
print("\nHARNESS:", "ALL PASS" if all(results) else "FAILED - do not run real data")
