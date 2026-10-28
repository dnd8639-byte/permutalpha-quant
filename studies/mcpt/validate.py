"""Validate the permutation test itself before trusting it.
  V1 calibration : on pure noise, in-sample MCPT p-values should be ~uniform (about 5% below 0.05).
  V2 power       : on data with a planted trend regime, p should usually be < 0.05.
  V3 walk-forward: same two checks for walk-forward MCPT (fewer runs; it's slower).
Run:  python validate.py        (takes several minutes)"""
import numpy as np
import pandas as pd
from mcpt import in_sample_mcpt, walk_forward_mcpt
from strategies import DonchianBreakout

rng = np.random.default_rng(7)
STRAT = DonchianBreakout(range(10, 201, 10))
idx = pd.bdate_range("2010-01-04", periods=2500)


def noise():
    r = np.empty(len(idx)); s = 0.01
    for t in range(len(idx)):                       # GARCH-t: fat tails + volatility clustering
        r[t] = s * rng.standard_t(5) / np.sqrt(5 / 3)
        s = np.sqrt(1e-5 + 0.08 * r[t] ** 2 + 0.9 * s ** 2)
    return pd.Series(100 * np.exp(np.cumsum(r)), index=idx)


def trending(strength=0.25):
    state, r = 1, np.empty(len(idx))
    for t in range(len(idx)):                       # persistent up/down regimes, avg length 50 bars
        if rng.random() < 1 / 50:
            state = -state
        r[t] = 0.01 * (strength * state + rng.standard_normal())
    return pd.Series(100 * np.exp(np.cumsum(r)), index=idx)


if __name__ == "__main__":
    K, NP = 40, 100
    p_noise = [in_sample_mcpt(STRAT, noise(), n_perm=NP, cost_bps=0, log=False)["p"] for _ in range(K)]
    frac = np.mean(np.array(p_noise) < 0.05)
    ok1 = np.sum(np.array(p_noise) < 0.05) <= 5 and 0.35 < np.mean(p_noise) < 0.65
    print(f"V1 IS-MCPT on noise:    share p<0.05 = {frac:.0%} (expect ~5%, allow <=5/40)   mean p = {np.mean(p_noise):.2f}   {'PASS' if ok1 else 'FAIL'}")

    p_trend = [in_sample_mcpt(STRAT, trending(), n_perm=NP, cost_bps=0, log=False)["p"] for _ in range(20)]
    ok2 = np.mean(np.array(p_trend) < 0.05) >= 0.8
    print(f"V2 IS-MCPT on planted trend: share p<0.05 = {np.mean(np.array(p_trend) < 0.05):.0%} (need >=80%)   {'PASS' if ok2 else 'FAIL'}")

    wf_noise = [walk_forward_mcpt(STRAT, noise(), 1000, 125, n_perm=40, log=False)["p"] for _ in range(10)]
    wf_trend = [walk_forward_mcpt(STRAT, trending(), 1000, 125, n_perm=40, log=False)["p"] for _ in range(10)]
    ok3 = np.mean(np.array(wf_noise) < 0.05) <= 0.2 and np.mean(np.array(wf_trend) < 0.05) >= 0.7
    print(f"V3 WF-MCPT: noise share p<0.05 = {np.mean(np.array(wf_noise) < 0.05):.0%} (mean p {np.mean(wf_noise):.2f});  "
          f"trend share p<0.05 = {np.mean(np.array(wf_trend) < 0.05):.0%}   {'PASS' if ok3 else 'FAIL'}")
    print("\nFRAMEWORK:", "VALIDATED" if (ok1 and ok2 and ok3) else "NOT VALIDATED")
