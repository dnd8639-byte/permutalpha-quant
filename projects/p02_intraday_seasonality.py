"""Project 2: intraday seasonality map. Does the hour of the week matter for Bitcoin?

Two different questions, with very different answers:
  VOLATILITY  Is the market reliably more active at certain hours?   (size of moves)
  DIRECTION   Does price reliably go UP or DOWN at certain hours?    (sign of moves)
Method   Hourly BTC/USDT bars 2018 -> now, timestamps in UTC. Build a 7 x 24 map (day of week x hour).
Test     A pattern in a heatmap proves nothing: 168 cells of pure noise also make a pattern.
         Statistic = spread of the 168 cell averages (their standard deviation).
         Null = shuffle the hour-of-week labels within each week, 500 times (keeps each week's
         returns, destroys only WHEN in the week they happened).  p = share of shuffles >= real.
Stability  Fit the map on 2018-2021, then check its correlation with the 2022+ map (out-of-sample).
Run:  python p02_intraday_seasonality.py
"""
import numpy as np
import pandas as pd
from common import FIG, INK, MUTED, btc_hourly, plt, style

DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def cell_means(values, slot):
    return np.bincount(slot, weights=values, minlength=168) / np.bincount(slot, minlength=168)


def perm_test(values, slot, week, n=500, seed=0):
    """Shuffle slot labels within each week; statistic = std of the 168 cell means."""
    rng = np.random.default_rng(seed)
    real = cell_means(values, slot).std()
    order = np.argsort(week, kind="stable")
    v, s, w = values[order], slot[order], week[order]
    starts = np.r_[0, np.flatnonzero(np.diff(w)) + 1, len(w)]
    null = np.empty(n)
    for i in range(n):
        sh = s.copy()
        for a, b in zip(starts[:-1], starts[1:]):
            sh[a:b] = rng.permutation(sh[a:b])
        null[i] = cell_means(v, sh).std()
    return real, null, (1 + (null >= real).sum()) / (n + 1)


if __name__ == "__main__":
    d = btc_hourly()
    r = np.log(d["close"]).diff().dropna()
    slot = (r.index.dayofweek * 24 + r.index.hour).to_numpy()
    week = ((r.index - pd.Timestamp("2018-01-01")).days // 7).to_numpy()
    ret, absr = r.to_numpy(), r.abs().to_numpy()
    print(f"BTC hourly, {r.index[0]:%Y-%m-%d} .. {r.index[-1]:%Y-%m-%d}, {len(r):,} bars (UTC)\n")

    out = {}
    for name, v in [("VOLATILITY (|return|)", absr), ("DIRECTION (return)", ret)]:
        real, null, p = perm_test(v, slot, week)
        split = r.index < "2022-01-01"
        early, late = cell_means(v[split], slot[split]), cell_means(v[~split], slot[~split])
        oos = np.corrcoef(early, late)[0, 1]
        out[name] = (cell_means(v, slot).reshape(7, 24), p, oos)
        print(f"{name:22s} spread of cells = {real / np.median(null):.2f}x the shuffled median   p = {p:.3f}   "
              f"2018-21 vs 2022+ map correlation = {oos:+.2f}")

    vol = out["VOLATILITY (|return|)"][0]
    hr = vol.mean(0)
    print(f"\nMost active hour (UTC): {hr.argmax():02d}:00, {hr.max() / hr.min():.1f}x the quietest ({hr.argmin():02d}:00). "
          f"Weekend volatility = {vol[5:].mean() / vol[:5].mean():.0%} of weekday.")

    p_ = plt()
    fig, axes = p_.subplots(1, 2, figsize=(13, 3.8))
    for ax, (name, (m, p, oos)), cmap in zip(axes, out.items(), ["magma", "RdBu"]):
        scale = 1e4
        kw = dict(vmin=-np.abs(m).max() * scale, vmax=np.abs(m).max() * scale) if cmap == "RdBu" else {}
        im = ax.imshow(m * scale, aspect="auto", cmap=cmap, **kw)
        ax.set_yticks(range(7)); ax.set_yticklabels(DAYS); ax.set_xticks(range(0, 24, 3))
        ax.set_xlabel("hour (UTC)", fontsize=8, color=MUTED)
        ax.set_title(f"{name}: p = {p:.3f}, out-of-sample correlation {oos:+.2f}", loc="left", fontsize=10, color=INK)
        cb = fig.colorbar(im, ax=ax); cb.set_label("basis points per hour", fontsize=8); cb.ax.tick_params(labelsize=7)
        style(ax)
    fig.suptitle("BTC intraday seasonality: WHEN the market moves repeats out of sample; WHICH WAY does not",
                 x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93)); fig.savefig(f"{FIG}/p02_intraday_seasonality.png", dpi=130)
