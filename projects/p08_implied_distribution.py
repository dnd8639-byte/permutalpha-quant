"""Project 8: the market's implied probability distribution (Breeden-Litzenberger 1978).

Idea     Option prices contain the market's probabilities for where the price will be at expiry.
         The (risk-neutral) density is the second derivative of the call price in the strike:
             q(K) = e^(rT) * d2C/dK2
         Intuition: buy calls at K-d and K+d, sell two at K (a "butterfly"). It pays only if the
         price lands near K, so its cost is the market's price for that outcome.
Catch    A second derivative amplifies noise. Real quotes are rounded to cents and have bid-ask
         spreads, so differentiating raw prices gives nonsense (even negative probabilities).
Method   (1) convert each price to an implied volatility, (2) fit a smooth curve through the
         volatilities, (3) turn the curve back into smooth call prices, (4) differentiate those.
Checks   (a) fake chain from Black-Scholes with one volatility: must recover the lognormal curve;
         (b) same chain with prices rounded to cents: raw differencing breaks, the smooth method doesn't;
         (c) fake chain with a volatility skew: must recover the fat left tail that was put in.
Real     SPY options, if you have run  python download_data.py options
"""
import os
import numpy as np
import pandas as pd
from common import A, B, FIG, HERE, INK, MUTED, N, plt, style
from common import C as RED

cdf = np.vectorize(N.cdf)


def black_call(F, K, T, sigma, df=1.0):
    """Call price from the forward price F (Black 1976). df = discount factor e^(-rT)."""
    sd = sigma * np.sqrt(T)
    d1 = np.log(F / K) / sd + sd / 2
    return df * (F * cdf(d1) - K * cdf(d1 - sd))


def implied_vol(price, F, K, T, df=1.0):
    """Invert black_call by bisection (works on arrays). NaN where the price is outside no-arbitrage bounds."""
    price, K = np.asarray(price, float), np.asarray(K, float)
    lo, hi = np.full(K.shape, 1e-4), np.full(K.shape, 5.0)
    bad = (price <= df * np.maximum(F - K, 0) + 1e-10) | (price >= df * F)
    for _ in range(60):
        mid = (lo + hi) / 2
        too_low = black_call(F, K, T, mid, df) < price
        lo, hi = np.where(too_low, mid, lo), np.where(too_low, hi, mid)
    return np.where(bad, np.nan, (lo + hi) / 2)


def density_raw(K, C, df=1.0):
    """Second difference of the quoted prices (uneven strikes allowed). The naive way."""
    h1, h2 = K[1:-1] - K[:-2], K[2:] - K[1:-1]
    return K[1:-1], 2 * (h1 * C[2:] - (h1 + h2) * C[1:-1] + h2 * C[:-2]) / (h1 * h2 * (h1 + h2)) / df


def density_smooth(K, C, F, T, df=1.0, degree=4, n_grid=400, min_tv=0.05):
    """Smooth in implied-volatility space, then differentiate. Returns (strike grid, density, vol curve)."""
    iv = implied_vol(C, F, K, T, df)
    time_value = C - df * np.maximum(F - K, 0)
    ok = np.isfinite(iv) & (time_value >= min_tv)             # a 1-cent option says nothing reliable about vol
    x = np.log(K[ok] / F)
    sd = iv[ok] * np.sqrt(T)
    vega = np.exp(-(x / sd - sd / 2) ** 2 / 2)                # weight: how much the price tells us about vol
    coef = np.polyfit(x, iv[ok], degree, w=vega)
    grid = np.linspace(K[ok].min(), K[ok].max(), n_grid)
    vol = np.polyval(coef, np.log(grid / F))
    cg = black_call(F, grid, T, vol, df)
    q = np.gradient(np.gradient(cg, grid), grid) / df
    return grid[2:-2], q[2:-2], vol[2:-2]                     # the 2 edge points use one-sided differences


def summarize(k, q, F, label):
    w = np.trapezoid(q, k) if hasattr(np, "trapezoid") else np.trapz(q, k)
    tz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    mean = tz(q * k, k) / w; sd = np.sqrt(tz(q * (k - mean) ** 2, k) / w)
    skew = tz(q * ((k - mean) / sd) ** 3, k) / w
    p10 = tz(q[k < 0.85 * F], k[k < 0.85 * F])
    print(f"  {label:34s} strikes {k[0]:.0f}-{k[-1]:.0f} hold {w:.1%} of probability  mean/forward {mean / F:.4f}  skew {skew:+.2f}  P(15%+ drop, within quoted strikes) {p10:.2%}  negative points {(q < -1e-6).sum()}")
    return p10


if __name__ == "__main__":
    F, T, SIG = 100.0, 0.25, 0.20
    K = np.arange(60.0, 141.0, 1.0)
    k_fine = np.linspace(60, 140, 400)
    lognormal = np.exp(-(np.log(k_fine / F) + SIG**2 * T / 2) ** 2 / (2 * SIG**2 * T)) / (k_fine * SIG * np.sqrt(2 * np.pi * T))
    print("CHECKS on fake option chains (forward 100, 3 months)")
    C = black_call(F, K, T, SIG)
    g, q, _ = density_smooth(K, C, F, T)
    print(f"  (a) flat 20% vol: max error vs the exact lognormal density = {np.abs(q - np.interp(g, k_fine, lognormal)).max():.5f} (peak height {lognormal.max():.3f})")
    Cr = np.round(C, 2)
    kr, qr = density_raw(K, Cr)
    g2, q2, _ = density_smooth(K, Cr, F, T)
    print(f"  (b) prices rounded to cents: raw differencing max error {np.abs(qr - np.interp(kr, k_fine, lognormal)).max():.4f}, "
          f"negative points {(qr < 0).sum()};  smooth method max error {np.abs(q2 - np.interp(g2, k_fine, lognormal)).max():.5f}")
    skew_vol = 0.20 - 0.35 * np.log(K / F) + 0.4 * np.log(K / F) ** 2
    gs, qs, _ = density_smooth(K, black_call(F, K, T, skew_vol), F, T)
    summarize(g, q, F, "(c) flat vol (lognormal)"); summarize(gs, qs, F, "    skewed vol (as in real markets)")

    path = os.path.join(HERE, "data", "spy_chain.csv")
    real = None
    if os.path.exists(path):
        ch = pd.read_csv(path)
        meta = ch.iloc[0]
        T_r, spot, r = float(meta["T_years"]), float(meta["spot"]), 0.04
        df = np.exp(-r * T_r)
        ch["mid"] = np.where((ch.bid > 0) & (ch.ask > 0), (ch.bid + ch.ask) / 2, ch.lastPrice)
        c = ch[ch.type == "call"].set_index("strike")["mid"]; p_ = ch[ch.type == "put"].set_index("strike")["mid"]
        both = pd.concat([c, p_], axis=1, keys=["c", "p"]).dropna()
        both = both[(both.index > 0.75 * spot) & (both.index < 1.2 * spot) & (both.c > 0.05) & (both.p > 0.05)]
        k_atm = (both.c - both.p).abs().idxmin()
        F_r = k_atm + (both.c[k_atm] - both.p[k_atm]) / df                     # put-call parity gives the forward
        Kr = both.index.to_numpy(float)
        call = np.where(Kr < F_r, both.p + df * (F_r - Kr), both.c)            # out-of-the-money side only
        gr, qr_, vol_r = density_smooth(Kr, call, F_r, T_r, df)
        ln = np.exp(-(np.log(gr / F_r) + np.interp(F_r, gr, vol_r)**2 * T_r / 2) ** 2 / (2 * np.interp(F_r, gr, vol_r)**2 * T_r)) / (gr * np.interp(F_r, gr, vol_r) * np.sqrt(2 * np.pi * T_r))
        print(f"\nREAL: SPY, expiry {meta['expiry']}, downloaded {meta['downloaded']}, spot {spot:.2f}, forward {F_r:.2f}, {len(Kr)} strikes, at-the-money vol {np.interp(F_r, gr, vol_r):.1%}")
        a = summarize(gr, qr_, F_r, "implied by option prices"); b = summarize(gr, ln, F_r, "lognormal with the same ATM vol")
        real = (gr, qr_, ln, F_r, meta["expiry"])
    else:
        print("\nREAL data not found: run  python download_data.py options")

    p = plt()
    fig, axes = p.subplots(1, 3 if real else 2, figsize=(14 if real else 11, 3.8))
    ax = axes[0]
    ax.plot(k_fine, lognormal, color=INK, lw=1.2, label="exact")
    ax.plot(kr, qr, "o-", color=RED, lw=0.8, ms=3, label="raw 2nd difference of rounded prices")
    ax.plot(g2, q2, color=A, lw=1.5, ls="--", label="smoothed in implied-vol space")
    ax.axhline(0, color=B, lw=0.8); ax.set_title("Check: rounding to cents breaks the naive method", loc="left", fontsize=10, color=INK)
    ax.legend(frameon=False, fontsize=7); style(ax)
    ax = axes[1]
    ax.plot(g, q, color=MUTED, lw=1.2, ls="--", label="flat vol (lognormal bell)"); ax.plot(gs, qs, color=A, lw=1.5, label="with a volatility skew")
    ax.fill_between(gs, 0, qs, where=gs < 85, color=RED, alpha=0.2)
    ax.set_title("Check: a skew shows up as a fat left tail", loc="left", fontsize=10, color=INK); ax.legend(frameon=False, fontsize=7); style(ax)
    if real:
        ax = axes[2]; gr, qr_, ln, F_r, ex = real
        ax.plot(gr, ln, color=MUTED, lw=1.2, ls="--", label="lognormal, same ATM vol"); ax.plot(gr, qr_, color=A, lw=1.5, label="implied by SPY options")
        ax.axvline(F_r, color=B, lw=0.8); ax.set_xlabel("SPY price at expiry", fontsize=8, color=MUTED)
        ax.set_title(f"Real: SPY, expiry {ex}", loc="left", fontsize=10, color=INK); ax.legend(frameon=False, fontsize=7); style(ax)
    fig.tight_layout(); fig.savefig(f"{FIG}/p08_implied_distribution.png", dpi=130)
