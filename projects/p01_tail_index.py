"""Project 1: how fat are the tails? Hill estimator of the tail index, and what it does to Gaussian VaR.

Idea     Far-out losses follow a power law: P(loss > x) ~ c * x^(-alpha). Smaller alpha = fatter tail.
         alpha < 4 means kurtosis is infinite; alpha < 2 would mean variance is infinite.
Hill     Sort losses largest first. With the k largest:  alpha_hat = 1 / mean(log(x_i / x_(k+1))).
The work Choosing k. Too small = noisy; too large = you're no longer in the tail. Plot alpha against k
         and read it where the curve is flat. Here: the median over k = 2%..5% of the sample.
Checks   (a) the estimator on simulated data with a known alpha. It is exact for a pure power law
             (Pareto) and biased LOW for Student-t tails, more so for thinner tails -- so treat
             real-data estimates as "roughly 3" rather than "2.96";
         (b) a 99% Gaussian VaR (trailing 250-day volatility, no look-ahead) should be breached 1% of
             days. Count the real breaches, and how often a breach follows a breach (clustering).
Run:  python p01_tail_index.py
"""
import numpy as np
import pandas as pd
from common import A, B, C, FIG, INK, MUTED, N, btc_hourly, futures_daily, plt, style


def hill_curve(losses, kmax=None):
    """alpha_hat for every k = 1..kmax (losses = positive numbers)."""
    x = np.sort(losses[losses > 0])[::-1]
    kmax = kmax or len(x) // 5
    lx = np.log(x)
    k = np.arange(1, kmax + 1)
    return k, 1.0 / (np.cumsum(lx)[:kmax] / k - lx[k])


def hill(losses, lo=0.02, hi=0.05):
    """Point estimate: median of the Hill curve for k between lo and hi of the sample."""
    n = int((losses > 0).sum())
    k, a = hill_curve(losses, int(n * hi) + 1)
    return float(np.median(a[int(n * lo):int(n * hi)]))


def var_breaches(r, level=0.99, window=250):
    """Share of days with loss beyond the Gaussian VaR, and P(breach tomorrow | breach today)."""
    vol = r.rolling(window).std().shift(1)                    # yesterday's estimate: no look-ahead
    b = (r < -N.inv_cdf(level) * vol)[vol.notna()]
    return b.mean(), (b & b.shift(1, fill_value=False)).sum() / max(b.sum(), 1), int(b.sum())


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    print("CHECK on simulated data (20,000 draws each; true alpha known)")
    for name, true, x in [("Student-t, 3 dof", 3, rng.standard_t(3, 20000)), ("Student-t, 5 dof", 5, rng.standard_t(5, 20000)),
                          ("Pareto alpha=2.5", 2.5, -(rng.pareto(2.5, 20000) + 1)), ("Normal", np.inf, rng.standard_normal(20000))]:
        print(f"  {name:18s} true alpha {true:>4}   Hill estimate {hill(-x):5.2f}")

    assets = {"BTC": np.log(btc_hourly()["close"].resample("D").last()).diff().dropna()}
    fut = futures_daily()
    if fut is not None:
        for m, label in [("ES", "S&P 500"), ("CL", "Crude oil"), ("GC", "Gold"), ("ZN", "10y Treasury"), ("6E", "Euro")]:
            assets[label] = np.log1p(fut[m].dropna())
    print("\nREAL daily returns: left tail (losses)")
    print(f"  {'asset':13s} {'days':>5s} {'alpha':>6s} {'kurtosis':>9s} {'VaR breaches':>13s} {'expected':>9s} {'P(breach|breach)':>17s}")
    rows = {}
    for name, r in assets.items():
        a = hill(-r.to_numpy())
        rate, clus, nb = var_breaches(r)
        rows[name] = (a, rate)
        print(f"  {name:13s} {len(r):5d} {a:6.2f} {r.kurt() + 3:9.1f} {rate:12.2%} {'1.00%':>9s} {clus:16.1%}")

    p = plt()
    fig, (ax1, ax2) = p.subplots(1, 2, figsize=(11, 4))
    for (name, r), col in zip(assets.items(), [INK, A, C, "#1F7A4D", "#9A6414", "#6F7F94"]):
        k, a = hill_curve(-r.to_numpy())
        ax1.plot(k / (r < 0).sum() * 100, a, lw=1.2, color=col, label=name)
    ax1.axvspan(2, 5, color=B, alpha=0.25); ax1.set_ylim(1, 7); ax1.set_xlim(0, 20)
    ax1.set_xlabel("k, as % of losses used", fontsize=8, color=MUTED); ax1.set_ylabel("tail index alpha", fontsize=8, color=MUTED)
    ax1.set_title("Hill plot: read alpha where the curve is flat (shaded = 2-5%)", loc="left", fontsize=10, color=INK)
    ax1.legend(frameon=False, fontsize=8, ncol=2); style(ax1)
    names = list(rows)
    ax2.bar(names, [rows[n][1] * 100 for n in names], color=A)
    ax2.axhline(1, color=C, ls="--", lw=1); ax2.text(len(names) - 0.5, 1.03, "Gaussian promise: 1%", color=C, fontsize=8, ha="right")
    ax2.set_ylabel("% of days beyond 99% VaR", fontsize=8, color=MUTED); ax2.tick_params(axis="x", rotation=30)
    ax2.set_title("Gaussian 99% VaR is breached more often than promised", loc="left", fontsize=10, color=INK); style(ax2)
    fig.tight_layout(); fig.savefig(f"{FIG}/p01_tail_index.png", dpi=130)
