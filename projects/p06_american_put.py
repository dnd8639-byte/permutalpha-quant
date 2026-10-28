"""Project 6: the American put and its free boundary (Crank-Nicolson + projected SOR).

Problem  A European put can only be exercised at expiry, so its price has a formula. An American
         put can be exercised any time. At every moment you must ask "is waiting worth more than
         exercising now?", so there is no formula, and the price where you should exercise (the
         "free boundary") has to be found together with the option's value.
Method   Black-Scholes PDE on a grid of prices (S) and times, stepped backwards from expiry with
         Crank-Nicolson (average of an explicit and an implicit step). Each step is a small linear
         system. For the American put the solution must also stay above the payoff max(K - S, 0),
         so each step is solved by projected SOR: iterate towards the solution, and after every
         sweep clamp the value up to the payoff. (Red-black ordering, so numpy can do each sweep.)
Checks   (1) European put from the same grid vs the Black-Scholes formula.
         (2) American put vs an independent method: a 4,000-step binomial tree.
         (3) Convergence study: halve the grid spacing, see how fast the error shrinks.
             Also tried: two smoothing steps at the start ("Rannacher"), a standard fix for the
             kink at the strike. Result: it did not improve the price here, so it is not needed.
Run:  python p06_american_put.py        (a few seconds)
"""
import numpy as np
from common import A, B, C, FIG, INK, MUTED, bs_call, plt, style

K, R, SIGMA, T, S0 = 100.0, 0.05, 0.20, 1.0, 100.0


def solve(n_s, n_t, american=True, rannacher=True, s_max=3 * K, omega=1.3, tol=1e-9):
    """Returns (S grid, value at t=0, exercise boundary per time step, total PSOR sweeps)."""
    i = np.arange(n_s + 1); S = i * s_max / n_s
    payoff = np.maximum(K - S, 0.0)
    V = payoff.copy()
    al, be, ga = 0.5 * (SIGMA**2 * i**2 - R * i), -(SIGMA**2 * i**2 + R), 0.5 * (SIGMA**2 * i**2 + R * i)
    steps = [(T / n_t / 2, 1.0)] * 4 + [(T / n_t, 0.5)] * (n_t - 2) if rannacher else [(T / n_t, 0.5)] * n_t
    tau, boundary, sweeps = 0.0, [], 0
    red, black = np.arange(1, n_s, 2), np.arange(2, n_s, 2)
    for dt, th in steps:                                       # th = 1: implicit Euler, 0.5: Crank-Nicolson
        tau += dt
        rhs = V.copy()
        rhs[1:-1] += (1 - th) * dt * (al[1:-1] * V[:-2] + be[1:-1] * V[1:-1] + ga[1:-1] * V[2:])
        lo, hi = (K if american else K * np.exp(-R * tau)), 0.0
        V[0], V[-1] = lo, hi
        diag = 1 - th * dt * be
        if american:
            for _ in range(20000):
                change = 0.0
                for idx in (red, black):
                    gs = (rhs[idx] + th * dt * (al[idx] * V[idx - 1] + ga[idx] * V[idx + 1])) / diag[idx]
                    new = np.maximum(payoff[idx], V[idx] + omega * (gs - V[idx]))
                    change = max(change, np.abs(new - V[idx]).max()); V[idx] = new
                sweeps += 1
                if change < tol:
                    break
            ex = np.flatnonzero((V[1:-1] - payoff[1:-1] < 1e-7) & (payoff[1:-1] > 0))
            boundary.append((tau, S[ex.max() + 1] if len(ex) else np.nan))
        else:                                                  # European: plain tridiagonal solve
            M = np.diag(diag[1:-1]) - th * dt * (np.diag(al[2:-1], -1) + np.diag(ga[1:-2], 1))
            b = rhs[1:-1].copy(); b[0] += th * dt * al[1] * lo
            V[1:-1] = np.linalg.solve(M, b)
    return S, V, boundary, sweeps


def binomial_american_put(n):
    dt = T / n; u = np.exp(SIGMA * np.sqrt(dt)); d = 1 / u; q = (np.exp(R * dt) - d) / (u - d); disc = np.exp(-R * dt)
    j = np.arange(n + 1); V = np.maximum(K - S0 * u ** j * d ** (n - j), 0)
    for m in range(n - 1, -1, -1):
        j = np.arange(m + 1)
        V = np.maximum(disc * (q * V[1:] + (1 - q) * V[:-1]), K - S0 * u ** j * d ** (m - j))
    return V[0]


def at_s0(S, V):
    return float(np.interp(S0, S, V))


if __name__ == "__main__":
    bs_put = float(bs_call(S0, K, R, T, SIGMA)) - S0 + K * np.exp(-R * T)
    S, V, _, _ = solve(300, 150, american=False)
    print(f"(1) European put: grid {at_s0(S, V):.4f}   Black-Scholes formula {bs_put:.4f}   error {at_s0(S, V) - bs_put:+.4f}")
    ref = 2 * binomial_american_put(4000) - binomial_american_put(2000)        # Richardson-extrapolated tree
    print(f"(2) American put reference (binomial tree): {ref:.4f}   early-exercise premium {ref - bs_put:.4f}")
    print("(3) Convergence of the PDE price (error vs the tree)")
    print(f"    {'grid (S x t)':>14s} {'no smoothing':>13s} {'ratio':>6s} {'Rannacher':>11s} {'ratio':>6s} {'PSOR sweeps':>12s}")
    prev, errs = (None, None), []
    for n in (75, 150, 300, 600):
        e0 = at_s0(*solve(n, n // 2, rannacher=False)[:2]) - ref
        S, V, bnd, sw = solve(n, n // 2)
        e1 = at_s0(S, V) - ref
        r0 = f"{prev[0] / e0:6.1f}" if prev[0] else "      "; r1 = f"{prev[1] / e1:6.1f}" if prev[1] else "      "
        print(f"    {n:>7d} x {n // 2:<4d} {e0:+13.5f} {r0} {e1:+11.5f} {r1} {sw:12d}")
        prev = (e0, e1); errs.append((n, abs(e0), abs(e1)))
    print("    (ratio 4 = second-order accurate, 2 = first-order). Smoothing did not improve the price.")
    print(f"    exercise boundary: exercise now if S < {bnd[-1][1]:.1f} with a year left; rises to K = {K:.0f} at expiry")

    p = plt()
    fig, axes = p.subplots(1, 3, figsize=(14, 3.8))
    ax = axes[0]
    ax.plot(S, np.maximum(K - S, 0), color=B, lw=1.2, label="exercise now (payoff)")
    Se, Ve, _, _ = solve(300, 150, american=False)
    ax.plot(Se, Ve, color=MUTED, lw=1.2, ls="--", label="European put")
    ax.plot(S, V, color=A, lw=1.6, label="American put")
    ax.set_xlim(50, 150); ax.set_ylim(0, 50); ax.set_xlabel("stock price S", fontsize=8, color=MUTED)
    ax.set_title("The American put never drops below its payoff", loc="left", fontsize=10, color=INK)
    ax.legend(frameon=False, fontsize=8); style(ax)
    ax = axes[1]
    tau, sb = np.array(bnd).T
    ax.plot(T - tau, sb, color=C, lw=1.6); ax.fill_between(T - tau, 60, sb, color=C, alpha=0.08)
    ax.text(0.05, 68, "exercise", color=C, fontsize=9); ax.text(0.05, 96, "hold", color=INK, fontsize=9)
    ax.set_ylim(60, 102); ax.set_xlabel("time (years; expiry at 1.0)", fontsize=8, color=MUTED)
    ax.set_title("Free boundary: the price below which you exercise", loc="left", fontsize=10, color=INK); style(ax)
    ax = axes[2]
    n, e0, e1 = np.array(errs).T
    ax.loglog(n, e0, "o-", color=B, label="Crank-Nicolson"); ax.loglog(n, e1, "o-", color=A, label="with Rannacher smoothing")
    ax.loglog(n, e1[0] * (n[0] / n) ** 2, ls=":", color=MUTED, label="slope for 2nd order")
    ax.set_xlabel("grid points in S", fontsize=8, color=MUTED); ax.set_ylabel("|error| vs binomial tree", fontsize=8, color=MUTED)
    ax.set_title("Convergence study", loc="left", fontsize=10, color=INK); ax.legend(frameon=False, fontsize=8); style(ax)
    fig.tight_layout(); fig.savefig(f"{FIG}/p06_american_put.png", dpi=130)
