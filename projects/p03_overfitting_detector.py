"""Project 3: backtest overfitting detector (Deflated Sharpe Ratio, Bailey & Lopez de Prado 2014).

Problem  Try enough strategies on any data and the best one looks excellent, by luck alone.
Part A   1,000 random trading rules on real BTC daily prices (random long/short, random holding
         periods; they know nothing). Backtest all, keep the best. Compare its Sharpe ratio with
         the formula for the best-of-N you should EXPECT from pure noise:
             E[max SR] ~ sd(SR) * [ (1 - g) * Z(1 - 1/N) + g * Z(1 - 1/(N*e)) ],  g = 0.5772...
Part B   The Deflated Sharpe Ratio: the probability that a strategy's true Sharpe is above that
         noise benchmark, given the track length T and the skew and kurtosis of its returns.
             DSR = Phi( (SR - SR0) * sqrt(T - 1) / sqrt(1 - skew*SR + (kurt - 1)/4 * SR^2) )
         (SR per period, not annualized.)  Rule of thumb: want DSR > 0.95.
Part C   Run it on a strategy I actually believed in: VolatilityHawkes on BTC (from mcpt-lab).
Run:  python p03_overfitting_detector.py
"""
import os, sys
import numpy as np
import pandas as pd
from common import A, B, C, FIG, HERE, INK, MUTED, N, btc_hourly, plt, style

EULER = 0.5772156649


def expected_max_sharpe(n_trials, sd_sharpe):
    """Sharpe of the best of n_trials strategies that all have zero true skill."""
    if n_trials < 2:
        return 0.0
    return sd_sharpe * ((1 - EULER) * N.inv_cdf(1 - 1 / n_trials) + EULER * N.inv_cdf(1 - 1 / (n_trials * np.e)))


def deflated_sharpe(returns, n_trials, sd_sharpe):
    """P(true Sharpe > best-of-noise benchmark). sd_sharpe = spread of the trials' per-period Sharpes."""
    r = np.asarray(returns, float); r = r[np.isfinite(r)]
    sr = r.mean() / r.std(ddof=1)
    z = (r - r.mean()) / r.std()
    skew, kurt = (z ** 3).mean(), (z ** 4).mean()
    sr0 = expected_max_sharpe(n_trials, sd_sharpe)
    return N.cdf((sr - sr0) * np.sqrt(len(r) - 1) / np.sqrt(1 - skew * sr + (kurt - 1) / 4 * sr ** 2))


def random_rules(n_rules, n_days, rng):
    """Each rule: start long or short at random, flip with probability 1/h each day (h = 1..40 days)."""
    pos = np.empty((n_rules, n_days))
    for i in range(n_rules):
        flips = rng.random(n_days) < 1 / rng.integers(1, 41)
        pos[i] = rng.choice([-1, 1]) * np.where(np.cumsum(flips) % 2 == 0, 1, -1)
    return pos


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    btc = btc_hourly()
    daily = np.log(btc["close"].resample("D").last()).diff().dropna().to_numpy()
    T, NR = len(daily), 1000

    rets = random_rules(NR, T, rng) * daily                      # no costs: generous to the rules
    sr = rets.mean(1) / rets.std(1, ddof=1)                      # per-day Sharpe of each rule
    ann = sr * np.sqrt(365)
    best = ann.argmax()
    print(f"PART A  {NR} random rules on {T} days of BTC")
    print(f"  best rule: annualized Sharpe {ann[best]:.2f}   (looks like a strategy; it is a coin flip)")
    print(f"  formula's expected best-of-{NR} from noise: {expected_max_sharpe(NR, sr.std(ddof=1)) * np.sqrt(365):.2f}")
    print(f"  rules with Sharpe > 1: {(ann > 1).sum()}")
    print(f"\nPART B  Deflated Sharpe of that best rule")
    print(f"  ignoring the search (N = 1):   {deflated_sharpe(rets[best], 1, sr.std(ddof=1)):.3f}   <- 'significant' if you hide the other 999")
    print(f"  counting the search (N = {NR}): {deflated_sharpe(rets[best], NR, sr.std(ddof=1)):.3f}")

    lab = os.path.join(HERE, "..", "mcpt-lab")
    hawkes = None
    if os.path.isdir(lab):
        sys.path.insert(0, lab)
        from mcptlab import HawkesVolatility, strategy_returns
        strat, COST = HawkesVolatility(), 5.0
        ohlc = btc[["open", "high", "low", "close"]]
        train, post = ohlc.loc[:"2020-12-31"], ohlc.loc["2022-10-01":]
        trials = {p: strategy_returns(train, strat.positions(train, p), COST).dropna() for p in strat.grid}
        srs = pd.Series({p: v.mean() / v.std() for p, v in trials.items()})
        bestp = max(trials, key=lambda p: srs[p])
        pp = strategy_returns(post, strat.positions(post, bestp), COST).loc["2023-01-01":].dropna()
        hawkes = (bestp, srs, trials[bestp], pp)
        print(f"\nPART C  VolatilityHawkes on BTC hourly, 5 bps costs, {len(strat.grid)} parameter sets tried")
        print(f"  in-sample 2018-2020, best {bestp}: annualized Sharpe {srs.max() * np.sqrt(8760):.2f}, "
              f"DSR (N = 25) = {deflated_sharpe(trials[bestp], 25, srs.std(ddof=1)):.3f}")
        print(f"  post-publication 2023+, same parameters frozen: annualized Sharpe {pp.mean() / pp.std() * np.sqrt(8760):.2f}")
        print(f"     no search on this data (N = 1): {deflated_sharpe(pp, 1, 0):.3f}")
        print(f"     if I charge it for all 25 tests in my research ledger: {deflated_sharpe(pp, 25, srs.std(ddof=1)):.3f}")

    p = plt()
    fig, ax = p.subplots(figsize=(8, 4))
    ax.hist(ann, bins=50, color=B, edgecolor="white", linewidth=0.4)
    em = expected_max_sharpe(NR, sr.std(ddof=1)) * np.sqrt(365)
    ax.axvline(em, color=C, ls="--", lw=1.2, label=f"formula: expected best from noise = {em:.2f}")
    ax.axvline(ann[best], color=INK, lw=2, label=f"best of {NR} random rules = {ann[best]:.2f}")
    ax.set_xlabel("annualized Sharpe ratio", fontsize=8, color=MUTED); ax.set_ylabel("random rules", fontsize=8, color=MUTED)
    ax.set_title(f"{NR} random trading rules on real BTC prices: the winner is luck, and the formula predicts it",
                 loc="left", fontsize=10, color=INK)
    ax.legend(frameon=False, fontsize=8); style(ax)
    fig.tight_layout(); fig.savefig(f"{FIG}/p03_overfitting_detector.png", dpi=130)
