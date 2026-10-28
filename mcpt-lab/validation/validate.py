"""Validate the permutation test itself before trusting it (about 5-10 minutes).

  V1 calibration : on pure noise, in-sample MCPT p-values should be roughly uniform
                   (about 5% below 0.05, mean about 0.5).
  V2 power       : on data with a planted trend, p should almost always be < 0.05.
  V3 walk-forward: the same two checks for walk-forward MCPT.
Writes validation/validation_pvalues.csv and results/figures/validation.png.

Run:  python validation/validate.py
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import numpy as np
import pandas as pd
from mcptlab import DonchianBreakout, in_sample_mcpt, synthetic, walk_forward_mcpt

STRAT = DonchianBreakout(range(10, 201, 10))

if __name__ == "__main__":
    K, NP = 40, 100
    rows = []
    for i in range(K):
        rows.append(("IS", "noise", in_sample_mcpt(STRAT, synthetic("noise", seed=100 + i), n_perm=NP, log=False)["p"]))
    for i in range(20):
        rows.append(("IS", "trend", in_sample_mcpt(STRAT, synthetic("trend", seed=200 + i), n_perm=NP, log=False)["p"]))
    for i in range(10):
        rows.append(("WF", "noise", walk_forward_mcpt(STRAT, synthetic("noise", seed=300 + i), 1000, 125, n_perm=40, log=False)["p"]))
        rows.append(("WF", "trend", walk_forward_mcpt(STRAT, synthetic("trend", seed=400 + i), 1000, 125, n_perm=40, log=False)["p"]))
    df = pd.DataFrame(rows, columns=["test", "data", "p"])
    df.to_csv(os.path.join(HERE, "validation_pvalues.csv"), index=False)

    g = lambda t, d: df[(df.test == t) & (df.data == d)].p.to_numpy()
    ok1 = np.sum(g("IS", "noise") < 0.05) <= 5 and 0.35 < g("IS", "noise").mean() < 0.65
    ok2 = np.mean(g("IS", "trend") < 0.05) >= 0.8
    ok3 = np.mean(g("WF", "noise") < 0.05) <= 0.2 and np.mean(g("WF", "trend") < 0.05) >= 0.7
    lines = [
        f"V1 in-sample MCPT on noise ({K} markets): {np.mean(g('IS', 'noise') < 0.05):.0%} below 0.05, mean p {g('IS', 'noise').mean():.2f}  {'PASS' if ok1 else 'FAIL'}",
        f"V2 in-sample MCPT on planted trend (20): {np.mean(g('IS', 'trend') < 0.05):.0%} below 0.05  {'PASS' if ok2 else 'FAIL'}",
        f"V3 walk-forward MCPT: noise {np.mean(g('WF', 'noise') < 0.05):.0%} below 0.05 (mean p {g('WF', 'noise').mean():.2f}), "
        f"trend {np.mean(g('WF', 'trend') < 0.05):.0%}  {'PASS' if ok3 else 'FAIL'}",
        "FRAMEWORK: " + ("VALIDATED" if ok1 and ok2 and ok3 else "NOT VALIDATED"),
    ]
    print("\n".join(lines))
    open(os.path.join(HERE, "validation_output.txt"), "w").write("\n".join(lines) + "\n")

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    bins = np.linspace(0, 1, 21)
    for ax, t, title in zip(axes, ("IS", "WF"), ("In-sample permutation test", "Walk-forward permutation test")):
        ax.hist(g(t, "noise"), bins=bins, color="#9aa7b8", edgecolor="white", label="pure noise (should spread out)")
        ax.hist(g(t, "trend"), bins=bins, color="#1f2d3d", edgecolor="white", alpha=0.85, label="planted trend (should pile up near 0)")
        ax.axvline(0.05, color="#c0392b", lw=1, ls="--"); ax.text(0.06, ax.get_ylim()[1] * 0.9, "p = 0.05", color="#c0392b", fontsize=8)
        ax.set_title(title, loc="left", fontsize=10); ax.set_xlabel("p-value", fontsize=8)
        for s in ("top", "right"): ax.spines[s].set_visible(False)
        ax.legend(fontsize=7, frameon=False)
    fig.suptitle("Testing the tester: p-values on fake markets where the right answer is known", x=0.01, ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = os.path.join(os.path.dirname(HERE), "results", "figures", "validation.png")
    os.makedirs(os.path.dirname(out), exist_ok=True); fig.savefig(out, dpi=130)
