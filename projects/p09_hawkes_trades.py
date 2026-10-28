"""Project 9: do trades trigger more trades? A Hawkes process fitted to trade arrival times.

Idea     Same maths as earthquake aftershocks. If trades arrived independently they would be a
         Poisson process. A Hawkes process lets each trade raise the chance of the next one.
         The key output is the branching ratio n (see hawkes.py): the share of trades that are
         reactions to earlier trades rather than responses to outside information.
Method   Exponential kernel, maximum likelihood with the O(N) recursion (no double loop).
Checks   (1) Simulate Hawkes processes with known parameters; the fit must recover them.
         (2) Fit a plain Poisson process simulated with no self-excitation: n must come out ~0.
         (3) Ogata's residual test on every fit: rescale time by the fitted intensity; the gaps
             should then be Exponential(1). A small KS p-value means the model is too simple.
Real     BTC/USDT trades from Binance.US, if you have run  python download_data.py trades
Limits   Binance.US is a small venue (about 2,000 trades a day), so this is one quiet day on one
         exchange. On real trades the residual test FAILS: one exponential kernel is too simple
         (real reactions happen on several time scales). The branching ratio is still a useful
         summary, but it is an estimate from a model the data rejects.
Run:  python p09_hawkes_trades.py        (about a minute)
"""
import os
import numpy as np
import pandas as pd
from common import A, B, C, FIG, HERE, INK, MUTED, plt, style
from hawkes import fit, ks_exponential, residuals, simulate


def report(label, times, horizon, truth=None):
    f = fit(times, horizon)
    d, p = ks_exponential(residuals(times, f["mu"], f["alpha"], f["beta"]))
    tr = f"   (true n {truth:.2f})" if truth is not None else ""
    print(f"  {label:30s} {len(times):6d} events   n = {f['n']:.3f}{tr}   half-life {np.log(2) / f['beta']:7.2f}s   "
          f"gain over Poisson {f['loglik'] - f['loglik_poisson']:9.0f} log-lik   residual KS {d:.3f} (p = {p:.2f})")
    return f


if __name__ == "__main__":
    rng = np.random.default_rng(3)
    print("CHECKS on simulated data")
    for mu, alpha, beta in [(0.5, 0.0, 1.0), (0.5, 0.6, 2.0), (0.2, 1.4, 2.0), (0.05, 4.5, 5.0)]:
        H = 4000 / (mu / (1 - alpha / beta))                   # horizon giving about 4,000 events
        t = simulate(mu, max(alpha, 1e-12), beta, H, rng)
        report(f"mu {mu}, alpha {alpha}, beta {beta}", t, H, truth=alpha / beta)

    path = os.path.join(HERE, "data", "btc_trades.csv")
    real = None
    if os.path.exists(path):
        tr = pd.read_csv(path)
        ts = np.unique(tr["timestamp"].to_numpy()) / 1000.0    # fills of one order share a timestamp: count once
        ts = ts + np.random.default_rng(0).uniform(0, 0.001, len(ts)); ts.sort()   # spread within the millisecond
        t0 = ts[0]; hours = (ts[-1] - t0) / 3600
        print(f"\nREAL: Binance.US BTC/USDT, {len(tr):,} fills -> {len(ts):,} distinct trade times over {hours:.1f} hours "
              f"({pd.to_datetime(tr.timestamp.iloc[0], unit='ms'):%Y-%m-%d %H:%M} UTC start)")
        rows = [(0, len(ts), report("whole sample", ts - t0, ts[-1] - t0))]
        BLOCK = 6                                              # hours per block (Binance.US is a quiet venue)
        for h in range(0, int(round(hours)), BLOCK):
            w = ts[(ts >= t0 + 3600 * h) & (ts < t0 + 3600 * (h + BLOCK))] - (t0 + 3600 * h)
            if len(w) >= 200:
                rows.append((h, len(w), report(f"hours {h:2d}-{h + BLOCK:2d}", w, 3600.0 * BLOCK)))
        real = rows
    else:
        print("\nREAL data not found: run  python download_data.py trades")

    p = plt()
    fig, axes = p.subplots(1, 3 if real else 2, figsize=(14 if real else 10, 3.6))
    t = simulate(0.2, 1.4, 2.0, 300, rng)
    grid = np.linspace(0, 300, 6000)
    lam = 0.2 + 1.4 * np.array([np.exp(-2.0 * (g - t[t < g])).sum() for g in grid])
    axes[0].plot(grid, lam, color=A, lw=0.9); axes[0].plot(t, np.zeros_like(t) - 0.2, "|", color=INK, ms=6)
    axes[0].set_xlim(0, 120); axes[0].set_xlabel("seconds", fontsize=8, color=MUTED)
    axes[0].set_title("Simulated Hawkes process (n = 0.7): events arrive in bursts", loc="left", fontsize=10, color=INK); style(axes[0])
    f = fit(t := simulate(0.2, 1.4, 2.0, 8000, rng), 8000)
    res = np.sort(residuals(t, f["mu"], f["alpha"], f["beta"])); pois = np.sort(np.diff(t) * len(t) / 8000)
    q = -np.log(1 - (np.arange(1, len(res) + 1) - 0.5) / len(res))
    axes[1].plot(q, pois[:len(q)], color=B, lw=1.5, label="if you assume Poisson"); axes[1].plot(q, res, color=A, lw=1.5, label="after the Hawkes fit")
    axes[1].plot([0, 8], [0, 8], color=C, ls="--", lw=1); axes[1].set_xlim(0, 8); axes[1].set_ylim(0, 12)
    axes[1].set_xlabel("expected gap (Exponential(1) quantiles)", fontsize=8, color=MUTED); axes[1].set_ylabel("observed gap in model time", fontsize=8, color=MUTED)
    axes[1].set_title("Residual check: on the dashed line = good fit", loc="left", fontsize=10, color=INK); axes[1].legend(frameon=False, fontsize=8); style(axes[1])
    if real:
        axes[2].bar(["all 24h"] + [f"h {r[0]}-{r[0] + 6}" for r in real[1:]], [r[2]["n"] for r in real], color=[INK] + [A] * (len(real) - 1))
        axes[2].set_ylim(0, 1); axes[2].set_ylabel("branching ratio n", fontsize=8, color=MUTED)
        axes[2].set_title("Real BTC trades: share of trades triggered by other trades", loc="left", fontsize=10, color=INK); style(axes[2])
    fig.tight_layout(); fig.savefig(f"{FIG}/p09_hawkes_trades.png", dpi=130)
