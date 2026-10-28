"""Statistics the gate reports beside the permutation p-value."""
from __future__ import annotations

from statistics import NormalDist

import numpy as np
import pandas as pd

from .search import Scorer, sharpe

ND = NormalDist()


def deflated_sharpe(r, n_trials: int) -> float:
    """Probability that the true per-bar Sharpe beats the best one expected from n_trials no-skill strategies
    (Bailey & Lopez de Prado 2014). The no-skill spread is taken from the track length (sd ~ 1/sqrt(T)),
    adjusted for skew and fat tails."""
    r = np.asarray(r, float); r = r[np.isfinite(r)]
    if len(r) < 10 or r.std() == 0:
        return float("nan")
    sr = r.mean() / r.std(ddof=1)
    z = (r - r.mean()) / r.std()
    skew, kurt = (z ** 3).mean(), (z ** 4).mean()
    sd = 1 / np.sqrt(len(r))
    g = 0.5772156649
    sr0 = 0.0 if n_trials < 2 else sd * ((1 - g) * ND.inv_cdf(1 - 1 / n_trials) + g * ND.inv_cdf(1 - 1 / (n_trials * np.e)))
    return ND.cdf((sr - sr0) * np.sqrt(len(r) - 1) / np.sqrt(max(1 - skew * sr + (kurt - 1) / 4 * sr ** 2, 1e-12)))


def cost_monte_carlo(scorer: Scorer, g: dict, n: int = 200, seed: int = 0, max_delay: float = 3 / 23) -> dict:
    """Rerun with random costs (lognormal, median x1.3, often x2-3) and random fill delays."""
    rng = np.random.default_rng(seed)
    s = []
    for _ in range(n):
        port, _, _ = scorer.returns(g, cost_mult=float(np.exp(rng.normal(0.25, 0.45))), delay=float(rng.uniform(0, max_delay)))
        s.append(sharpe(port.to_numpy(), scorer.prep.ppy))
    s = np.array(s)
    return dict(median=float(np.median(s)), p5=float(np.percentile(s, 5)), share_positive=float((s > 0).mean()))


def side_and_beta(scorer: Scorer, g: dict, delay: float) -> dict:
    """Does it make money on both sides, and how much of it is just being long the market?"""
    port, _, _ = scorer.returns(g, delay=delay)
    long_, _, _ = scorer.returns({**g, "side": "long"}, delay=delay)
    short, _, _ = scorer.returns({**g, "side": "short"}, delay=delay)
    always_long = {n: np.ones(len(scorer.prep.index[n])) for n in scorer.prep.names}
    from .engine import portfolio
    mkt, _, _ = portfolio(scorer.prep, always_long, cost_mult=0.0, delay=delay)
    j = pd.concat([port.rename("s"), mkt.rename("m")], axis=1).dropna()
    beta = float(np.cov(j.s, j.m)[0, 1] / j.m.var()) if j.m.var() > 0 else float("nan")
    ppy = scorer.prep.ppy
    return dict(sharpe=sharpe(port, ppy), sharpe_long=sharpe(long_, ppy), sharpe_short=sharpe(short, ppy),
                beta_to_market=beta, corr_to_market=float(j.s.corr(j.m)), market_sharpe=sharpe(mkt, ppy),
                ann_vol=float(port.std() * np.sqrt(ppy)))


def sharpe_se(sr: float, years: float) -> float:
    """Standard error of an annualized Sharpe ratio measured over `years` (Lo 2002, independent returns)."""
    return float(np.sqrt((1 + 0.5 * sr ** 2) / max(years, 1e-9)))


def holdout_power(real: float, null, years: float, dev_sharpe: float | None = None, dev_years: float | None = None,
                  alpha: float = 0.05) -> dict:
    """What could this holdout have detected? A FAIL means little if the holdout was too short to see the edge.

    critical    the Sharpe the holdout needed to beat (the shuffled holdouts' 1 - alpha quantile)
    mde80       the smallest true Sharpe the holdout would detect 80% of the time
    power_at_dev  the chance of passing if the true Sharpe equalled the development Sharpe (which is the best of a
                search, so it overstates the true Sharpe a little: this power is an upper bound)
    decay_p     one-sided test that the holdout Sharpe is below the development Sharpe (small = the edge shrank)
    """
    null = np.asarray(null, float)
    crit = float(np.quantile(null, 1 - alpha))
    se = sharpe_se(real, years)
    se_c = sharpe_se(crit, years)
    out = dict(holdout_years=float(years), critical=crit, se=se, ci95=[real - 1.96 * se, real + 1.96 * se],
               mde80=float(crit + ND.inv_cdf(0.8) * se_c))
    if dev_sharpe is not None and dev_years:
        se_h = sharpe_se(dev_sharpe, years)
        se_d = sharpe_se(dev_sharpe, dev_years)
        out.update(dev_sharpe=float(dev_sharpe), dev_years=float(dev_years),
                   power_at_dev=float(1 - ND.cdf((crit - dev_sharpe) / se_h)),
                   decay_p=float(1 - ND.cdf((dev_sharpe - real) / np.sqrt(se ** 2 + se_d ** 2))))
    return out


def holdout_power_text(pw: dict) -> str:
    lines = [f"- Holdout length: {pw['holdout_years']:.1f} years. Sharpe 95% interval: "
             f"{pw['ci95'][0]:+.2f} to {pw['ci95'][1]:+.2f}.",
             f"- To pass, the Sharpe had to beat {pw['critical']:.2f} (the shuffled holdouts' 95th percentile).",
             f"- Smallest true Sharpe this holdout detects 80% of the time: **{pw['mde80']:.2f}**."]
    if "power_at_dev" in pw:
        lines += [f"- If the edge were as strong as in development ({pw['dev_sharpe']:.2f} over {pw['dev_years']:.0f} years), "
                  f"this holdout would have passed with probability {pw['power_at_dev']:.0%} (an upper bound: the development "
                  f"Sharpe is the best of a search).",
                  f"- Did the edge shrink? One-sided test of holdout below development: p = {pw['decay_p']:.4f}."]
        verdict = ("The holdout was long enough to see the development edge, and it did not: the edge shrank or vanished."
                   if pw["power_at_dev"] >= 0.8 and pw["decay_p"] < 0.05 else
                   "The holdout was too short to tell: a FAIL here is not evidence that the edge is gone."
                   if pw["power_at_dev"] < 0.5 else
                   "Partly informative: the holdout had a fair but not high chance of seeing the development edge.")
        lines.append(f"- **Reading:** {verdict}")
    return "\n".join(lines)
