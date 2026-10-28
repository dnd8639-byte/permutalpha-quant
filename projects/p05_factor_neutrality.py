"""Project 5: factor neutrality check. Is a strategy's return really its own, or a known factor in disguise?

Idea     Regress the strategy's returns on the returns of known "factors". The slopes (betas) show
         what it is secretly exposed to; the intercept (alpha) is what is left after removing them.
         A strategy is "factor neutral" if every beta is near zero. If alpha vanishes once the
         factors are included, the strategy was just a repackaged factor.
Checks   (0) Self-test: 500 fake strategies = 0.5 x market + noise with zero alpha. The test must
             find beta 0.5, and flag a false alpha only about 5% of the time.
         (1) The US momentum factor vs the five Fama-French factors (1963 -> 2026).
         (2) My own VolatilityHawkes BTC strategy vs Bitcoin itself: is it a hidden bet on BTC going
             up? And does it depend on BTC being volatile (|return|)? In-sample and after publication.
Stats    t-statistics use Newey-West standard errors (robust to autocorrelation). |t| > 2 ~ real.
Run:  python p05_factor_neutrality.py
"""
import os, sys
import numpy as np
import pandas as pd
from common import A, B, C, FIG, HERE, INK, MUTED, btc_hourly, french, ols, plt, style


def neutrality(strategy, factors, periods_per_year, lags=6, label=""):
    """Print and return alpha, betas and t-stats of `strategy` regressed on the `factors` DataFrame."""
    d = pd.concat([strategy.rename("y"), factors], axis=1).dropna()
    b, t, r2 = ols(d["y"], d[factors.columns], lags)
    raw_t = d["y"].mean() / d["y"].std() * np.sqrt(len(d))
    print(f"{label}  ({len(d)} periods)")
    print(f"  raw mean return {d['y'].mean() * periods_per_year:7.1%}/yr (t = {raw_t:5.2f})   ->   "
          f"alpha after factors {b[0] * periods_per_year:7.1%}/yr (t = {t[0]:5.2f})   R^2 = {r2:.2f}")
    print("  betas: " + "   ".join(f"{c} {bb:+.2f} (t {tt:+.1f})" for c, bb, tt in zip(factors.columns, b[1:], t[1:])))
    return dict(alpha=b[0] * periods_per_year, t_alpha=t[0], betas=dict(zip(factors.columns, b[1:])),
                t=dict(zip(factors.columns, t[1:])), r2=r2, raw=d["y"].mean() * periods_per_year)


if __name__ == "__main__":
    ff5 = french("F-F_Research_Data_5_Factors_2x3").drop(columns="RF")
    rng = np.random.default_rng(1)
    bs, ts = [], []
    for _ in range(500):                                       # 500 fake strategies with NO alpha
        fake = 0.5 * ff5["Mkt-RF"].to_numpy() + rng.normal(0, 0.02, len(ff5))
        b, t, _ = ols(fake, ff5, 6)
        bs.append(b[1]); ts.append(t[0])
    print(f"(0) SELF-TEST on 500 fake strategies (0.5 x market + noise, zero alpha):\n"
          f"  average market beta found {np.mean(bs):.3f} (true 0.5);  false alarms (|t alpha| > 2): "
          f"{np.mean(np.abs(ts) > 2):.1%} (should be about 5%)")

    mom = french("F-F_Momentum_Factor")["Mom"]
    print(); res_m = neutrality(mom, ff5, 12, label="(1) US MOMENTUM vs Fama-French five factors, monthly")

    lab = os.path.join(HERE, "..", "mcpt-lab")
    res_h = {}
    if os.path.isdir(lab):
        sys.path.insert(0, lab)
        from mcptlab import HawkesVolatility, strategy_returns
        btc = btc_hourly()[["open", "high", "low", "close"]]
        hourly = strategy_returns(btc, HawkesVolatility().positions(btc, (0.1, 96)), 5.0)
        strat = hourly.resample("D").sum()
        r = np.log(btc["close"].resample("D").last()).diff()
        print()
        for name, sl in [("in-sample 2018-2020", slice("2018-02", "2020-12")), ("after publication 2023 ->", slice("2023-01", None))]:
            fac = pd.DataFrame({"BTC": r.loc[sl], "|BTC|": r.loc[sl].abs() - r.loc[sl].abs().mean()})
            res_h[name] = neutrality(strat.loc[sl], fac[["BTC"]], 365, lags=5, label=f"(2) HAWKES on BTC, daily, {name}: direction only")
            v = neutrality(strat.loc[sl], fac, 365, lags=5, label="    adding the size of BTC's move (not tradable, so its alpha is not a profit you could keep)")
            res_h[name]["vol_beta"], res_h[name]["vol_t"] = v["betas"]["|BTC|"], v["t"]["|BTC|"]

    p = plt()
    fig, axes = p.subplots(1, 2, figsize=(12, 3.8))
    ax = axes[0]
    ax.bar(list(res_m["betas"]), list(res_m["betas"].values()), color=[A if abs(res_m["t"][k]) > 2 else B for k in res_m["betas"]])
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_title(f"Momentum's factor betas (dark = |t| > 2). Alpha {res_m['alpha']:.1%}/yr, t = {res_m['t_alpha']:.1f}",
                 loc="left", fontsize=10, color=INK); style(ax)
    ax = axes[1]
    if res_h:
        names = list(res_h); x = np.arange(len(names))
        ax.bar(x - 0.2, [res_h[n]["raw"] * 100 for n in names], 0.4, color=B, label="raw return, %/yr")
        ax.bar(x + 0.2, [res_h[n]["alpha"] * 100 for n in names], 0.4, color=A, label="alpha after removing exposure to BTC's direction")
        for i, n in enumerate(names):
            ax.text(i + 0.2, res_h[n]["alpha"] * 100 + 2, f"t = {res_h[n]['t_alpha']:.1f}", ha="center", fontsize=8, color=INK)
        ax.set_xticks(x); ax.set_xticklabels(names); ax.legend(frameon=False, fontsize=8)
        ax.set_title("Hawkes strategy: how much return is left after removing factor exposure", loc="left", fontsize=10, color=INK)
    style(ax)
    fig.tight_layout(); fig.savefig(f"{FIG}/p05_factor_neutrality.png", dpi=130)
