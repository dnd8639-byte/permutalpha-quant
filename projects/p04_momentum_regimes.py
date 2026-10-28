"""Project 4: regime-dependent momentum ("momentum crashes", Daniel & Moskowitz 2016).

Idea     Momentum (buy past winners, short past losers) earns a premium on average but crashes
         when a beaten-down market rebounds: the losers it is short bounce hardest.
Data     Kenneth French Data Library: US momentum factor and market return, monthly, 1927 -> 2026.
Regime   BEAR = the market's total return over the previous 24 months is negative (known at the
         start of the month, so no look-ahead).
Checks   (1) Momentum's return, volatility and skew in bear vs normal markets.
         (2) The "written call" test: regress momentum on the market, letting its beta differ in
             bear markets and again when the market is UP in a bear market. A big negative extra
             beta there = momentum is effectively short a call option on a rebound.
         (3) A fix from the literature: scale exposure by 1 / trailing volatility of momentum
             (Barroso & Santa-Clara 2015), using only past data. Does it help, and does it still
             help AFTER those papers were published (2016 ->)?
Run:  python p04_momentum_regimes.py
"""
import numpy as np
import pandas as pd
from common import A, B, C, FIG, INK, MUTED, french, ols, plt, style


def stats(r):
    cum = (1 + r).cumprod()
    return dict(ann_ret=r.mean() * 12, ann_vol=r.std() * 12 ** 0.5, sharpe=r.mean() / r.std() * 12 ** 0.5,
                skew=r.skew(), worst=r.min(), max_dd=(cum / cum.cummax() - 1).min())


def line(name, r):
    s = stats(r)
    return (f"  {name:22s} {len(r):5d} {s['ann_ret']:7.1%} {s['ann_vol']:7.1%} {s['sharpe']:7.2f} {s['skew']:6.2f} "
            f"{s['worst']:8.1%} {s['max_dd']:8.1%}")


if __name__ == "__main__":
    d = french("F-F_Momentum_Factor").join(french("F-F_Research_Data_Factors"), how="inner")
    d["mkt"] = d["Mkt-RF"] + d["RF"]
    past24 = (1 + d["mkt"]).rolling(24).apply(np.prod, raw=True).shift(1) - 1     # known before the month
    d = d[past24.notna()].assign(bear=(past24 < 0).astype(float)[past24.notna()])
    d["up"] = (d["Mkt-RF"] > 0).astype(float)
    mom = d["Mom"]
    print(f"US momentum factor, {d.index[0]:%Y-%m} .. {d.index[-1]:%Y-%m}; bear months: {d.bear.mean():.0%}\n")
    head = f"  {'':22s} {'months':>5s} {'return':>7s} {'vol':>7s} {'Sharpe':>7s} {'skew':>6s} {'worst mo':>8s} {'max DD':>8s}"
    print("(1) Momentum by market regime\n" + head)
    print(line("normal markets", mom[d.bear == 0])); print(line("bear markets", mom[d.bear == 1]))

    X = np.column_stack([d.bear, d["Mkt-RF"], d.bear * d["Mkt-RF"], d.bear * d.up * d["Mkt-RF"]])
    b, t, r2 = ols(mom, X, lags=6)
    print("\n(2) Written-call regression (Newey-West t-stats)")
    for nm, bb, tt in zip(["alpha (monthly)", "extra alpha in bear", "market beta", "extra beta in bear", "extra beta, bear AND market up"], b, t):
        print(f"  {nm:32s} {bb:8.3f}   t = {tt:6.2f}")
    print(f"  => momentum's beta: {b[2]:+.2f} normally, {b[2] + b[3]:+.2f} in a falling bear market, "
          f"{b[2] + b[3] + b[4]:+.2f} in a rebounding bear market")

    vol = mom.rolling(6).std().shift(1)                       # trailing 6-month volatility, lagged
    target = mom.std()
    scaled = (mom * (target / vol).clip(upper=3)).dropna()    # leverage capped at 3x
    raw = mom.loc[scaled.index]
    scaled *= raw.std() / scaled.std()                        # same overall volatility, for a fair picture
    print("\n(3) Volatility-scaled momentum vs plain\n" + head)
    for label, sl in [("full sample", slice(None)), ("before 2016", slice(None, "2015-12")), ("2016 -> (published)", slice("2016-01", None))]:
        print(f"  -- {label}"); print(line("plain", raw.loc[sl])); print(line("vol-scaled", scaled.loc[sl]))
    worst = raw.nsmallest(5)
    print("\nFive worst months for plain momentum (market return, bear flag):")
    for dt, v in worst.items():
        print(f"  {dt:%Y-%m}  momentum {v:7.1%}   market {d.mkt[dt]:+6.1%}   bear={int(d.bear[dt])}")

    p = plt()
    fig, (ax1, ax2) = p.subplots(1, 2, figsize=(12, 4))
    for flag, col, lab in [(0, B, "normal market"), (1, C, "bear market")]:
        m = d.bear == flag
        ax1.scatter(d["Mkt-RF"][m] * 100, mom[m] * 100, s=8, color=col, alpha=0.7, label=lab)
    xs = np.linspace(-30, 40, 100)
    ax1.plot(xs, 100 * (b[0] + b[1]) + (b[2] + b[3]) * xs + b[4] * np.maximum(xs, 0), color=C, lw=1.5)
    ax1.plot(xs, 100 * b[0] + b[2] * xs, color=MUTED, lw=1.2)
    ax1.set_xlabel("market excess return, % per month", fontsize=8, color=MUTED); ax1.set_ylabel("momentum return, %", fontsize=8, color=MUTED)
    ax1.set_title("In bear markets, momentum loses when the market rebounds", loc="left", fontsize=10, color=INK)
    ax1.legend(frameon=False, fontsize=8); style(ax1)
    ax2.plot(np.log((1 + raw).cumprod()), color=MUTED, lw=1.1, label="plain momentum")
    ax2.plot(np.log((1 + scaled).cumprod()), color=A, lw=1.3, label="volatility-scaled (past data only)")
    ax2.axvline(pd.Timestamp("2016-01-01"), color=C, ls="--", lw=1); ax2.text(pd.Timestamp("2016-06-01"), 0.2, "published", fontsize=8, color=C)
    ax2.set_ylabel("cumulative log return", fontsize=8, color=MUTED)
    ax2.set_title("Scaling by trailing volatility helps, but 1932 still hurts", loc="left", fontsize=10, color=INK)
    ax2.legend(frameon=False, fontsize=8); style(ax2)
    fig.tight_layout(); fig.savefig(f"{FIG}/p04_momentum_regimes.png", dpi=130)
