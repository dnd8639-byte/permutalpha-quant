"""Project 7: Monte Carlo Greeks by automatic differentiation, written from scratch.

Problem  A Monte Carlo pricer gives a price. Traders need its sensitivities ("Greeks"):
         delta = d price / d spot, vega = d price / d volatility, gamma = d delta / d spot.
Old way  "Bump and reprice": price at S and at S + h, subtract, divide by h. Small h -> the random
         noise of the two runs is divided by a tiny number. Large h -> the answer is biased.
Autodiff Carry the derivative through every arithmetic step of the simulation (forward-mode
         automatic differentiation with "dual numbers": each value travels with its derivative,
         and +, *, exp, ... apply the chain rule). One run, exact derivative of what was simulated.
         The Dual class below is about 30 lines; libraries like JAX do the same thing at scale.
Checks   European call, where Black-Scholes gives the exact delta, vega and gamma.
         (1) delta and vega: autodiff vs bumping with fresh random numbers vs bumping with the
             same random numbers, over 200 repeats.
         (2) gamma: differentiate twice. This FAILS (gives exactly 0): the payoff max(S - K, 0)
             has a kink, and its second derivative is zero everywhere except one point that no
             simulated path ever hits. Autodiff differentiates the code, not the maths.
         (3) the fix: get one of the two derivatives from the probability density instead
             (likelihood-ratio method), which has no kink.
Run:  python p07_autodiff_greeks.py
"""
import numpy as np
from common import A, B, C, FIG, INK, MUTED, N, plt, style

S0, K, R, SIGMA, T = 100.0, 100.0, 0.05, 0.20, 1.0


class Dual:
    """A value and its derivative. Works on numbers, numpy arrays, and other Duals (for 2nd derivatives)."""
    def __init__(self, v, d=0.0):
        self.v, self.d = v, d

    def _w(self, o):
        return o if isinstance(o, Dual) else Dual(o, 0.0)

    def __add__(self, o): o = self._w(o); return Dual(self.v + o.v, self.d + o.d)
    def __sub__(self, o): o = self._w(o); return Dual(self.v - o.v, self.d - o.d)
    def __mul__(self, o): o = self._w(o); return Dual(self.v * o.v, self.d * o.v + self.v * o.d)      # product rule
    def __truediv__(self, o): o = self._w(o); return Dual(self.v / o.v, (self.d * o.v - self.v * o.d) / (o.v * o.v))
    def __neg__(self): return Dual(-self.v, -self.d)
    __radd__ = __add__; __rmul__ = __mul__
    def __rsub__(self, o): return self._w(o) - self


def exp(x):
    if isinstance(x, Dual):
        e = exp(x.v); return Dual(e, e * x.d)                  # chain rule
    return np.exp(x)


def sqrt(x):
    if isinstance(x, Dual):
        s = sqrt(x.v); return Dual(s, x.d / (2 * s))
    return np.sqrt(x)


def relu(x):
    """max(x, 0). Derivative: 1 where x > 0, else 0."""
    if isinstance(x, Dual):
        pos = (val(x.v) > 0) * 1.0
        return Dual(relu(x.v), x.d * pos)
    return np.maximum(x, 0.0)


def mean(x):
    return Dual(mean(x.v), mean(x.d)) if isinstance(x, Dual) else np.mean(x)


def val(x):
    return val(x.v) if isinstance(x, Dual) else x


def price(s0, sigma, z):
    """Monte Carlo price of a European call. Written once; works with floats or Duals."""
    st = s0 * exp((R - sigma * sigma / 2) * T + sigma * sqrt(T) * z)
    return float(np.exp(-R * T)) * mean(relu(st - K))


def black_scholes():
    d1 = (np.log(S0 / K) + (R + SIGMA**2 / 2) * T) / (SIGMA * np.sqrt(T))
    pdf = np.exp(-d1**2 / 2) / np.sqrt(2 * np.pi)
    return N.cdf(d1), S0 * pdf * np.sqrt(T), pdf / (S0 * SIGMA * np.sqrt(T))


if __name__ == "__main__":
    delta_bs, vega_bs, gamma_bs = black_scholes()
    rng = np.random.default_rng(0)
    NP, REP = 20000, 200
    print(f"European call, S = K = 100, {NP:,} paths, {REP} repeats.  Exact: delta {delta_bs:.4f}  vega {vega_bs:.3f}  gamma {gamma_bs:.5f}\n")

    res = {k: [] for k in ["autodiff", "bump 1.0, same randoms", "bump 1.0, fresh randoms", "bump 0.1, fresh randoms"]}
    vega_ad, gamma_ad, gamma_lr, gamma_bump = [], [], [], []
    for _ in range(REP):
        z, z2 = rng.standard_normal(NP), rng.standard_normal(NP)
        res["autodiff"].append(price(Dual(S0, 1.0), SIGMA, z).d)
        res["bump 1.0, same randoms"].append((price(S0 + 1, SIGMA, z) - price(S0, SIGMA, z)) / 1.0)
        res["bump 1.0, fresh randoms"].append((price(S0 + 1, SIGMA, z2) - price(S0, SIGMA, z)) / 1.0)
        res["bump 0.1, fresh randoms"].append((price(S0 + 0.1, SIGMA, z2) - price(S0, SIGMA, z)) / 0.1)
        vega_ad.append(price(S0, Dual(SIGMA, 1.0), z).d)
        gamma_ad.append(price(Dual(Dual(S0, 1.0), Dual(1.0, 0.0)), SIGMA, z).d.d)        # derivative of the derivative
        st = S0 * np.exp((R - SIGMA**2 / 2) * T + SIGMA * np.sqrt(T) * z)
        h = np.where(st > K, st, 0.0)                                                    # pathwise delta * S0
        gamma_lr.append(np.exp(-R * T) * (np.mean(h * z) / (S0**2 * SIGMA * np.sqrt(T)) - np.mean(h) / S0**2))
        gamma_bump.append((price(S0 + 1, SIGMA, z) - 2 * price(S0, SIGMA, z) + price(S0 - 1, SIGMA, z)) / 1.0)

    print("(1) DELTA                         mean      bias    noise (std)")
    for k, v in res.items():
        print(f"    {k:26s} {np.mean(v):8.4f} {np.mean(v) - delta_bs:+9.4f} {np.std(v):10.4f}")
    print(f"    VEGA by autodiff           {np.mean(vega_ad):8.3f} {np.mean(vega_ad) - vega_bs:+9.3f} {np.std(vega_ad):10.3f}")
    print("\n(2)-(3) GAMMA                     mean      bias    noise (std)")
    for k, v in [("autodiff twice (naive)", gamma_ad), ("bump 1.0, same randoms", gamma_bump), ("likelihood-ratio fix", gamma_lr)]:
        print(f"    {k:26s} {np.mean(v):8.5f} {np.mean(v) - gamma_bs:+9.5f} {np.std(v):10.5f}")

    p = plt()
    fig, (ax1, ax2) = p.subplots(1, 2, figsize=(12, 3.8))
    bins = np.linspace(delta_bs - 0.12, delta_bs + 0.12, 61)
    for (k, v), col in zip(res.items(), [A, "#1F7A4D", B, "#E3AE55"]):
        if "0.1" in k:
            continue
        ax1.hist(v, bins=bins, color=col, alpha=0.75, label=f"{k} (std {np.std(v):.4f})")
    ax1.axvline(delta_bs, color=C, lw=1.2); ax1.set_xlabel("delta estimate", fontsize=8, color=MUTED)
    ax1.set_title("Delta: autodiff is exact on average and far less noisy than fresh-random bumping", loc="left", fontsize=9, color=INK)
    ax1.legend(frameon=False, fontsize=7); style(ax1)
    names = ["autodiff twice\n(naive)", "bump and reprice", "likelihood-ratio\nfix"]
    means = [np.mean(gamma_ad), np.mean(gamma_bump), np.mean(gamma_lr)]; sds = [np.std(gamma_ad), np.std(gamma_bump), np.std(gamma_lr)]
    ax2.bar(names, means, yerr=sds, color=[C, B, A], capsize=4)
    ax2.axhline(gamma_bs, color=INK, ls="--", lw=1); ax2.text(2.45, gamma_bs * 1.03, "exact", fontsize=8, ha="right")
    ax2.set_title("Gamma: naive second-order autodiff returns exactly zero", loc="left", fontsize=10, color=INK); style(ax2)
    fig.tight_layout(); fig.savefig(f"{FIG}/p07_autodiff_greeks.png", dpi=130)
