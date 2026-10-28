"""Rebuild every figure in results/figures/ from the saved results (a few minutes).

  permutation_explainer.png  what a permutation is (real BTC vs shuffled copies)
  all_tests.png              every permutation test: noise histogram vs the real result
  scorecard.png              every p-value in the ledger, grouped by strategy and phase
  hawkes_story.png           VolatilityHawkes: equity by phase, profit factor by year, cost sensitivity
  decay_story.png            10 published strategies: in-sample vs after publication vs after costs
  decay_growth.png           growth of $1 from each publication date, before and after costs
  calendar_costs.png         Monday and turn-of-the-month effects vs the cost of trading them, 1926-2026
  decay_tests.png            the publication-decay permutation (sign-flip) tests
  (validation.png is written by validation/validate.py)

Run:  python scripts/make_figures.py
"""
import glob
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mcptlab import (HawkesVolatility, load_ohlc, load_sample_btc, optimize, permute_ohlc, plot_grid,
                     profit_factor, sharpe, strategy_returns, walk_forward)
from mcptlab.data import DATA_DIR
from mcptlab.plotting import GRID, INK, MUTED, NOISE, REAL

FIG = os.path.join(ROOT, "results", "figures")
PASS, WEAK, FAIL = "#1f7a4d", "#c98a1a", "#a33a3a"
os.makedirs(FIG, exist_ok=True)


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(labelsize=8, colors=MUTED)


# --------------------------------------------------------------------------- 1. explainer
def explainer():
    d = load_sample_btc(end="2020-12-31")[["open", "high", "low", "close"]]
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 3.8), gridspec_kw=dict(width_ratios=[2.2, 1]))
    for s in range(1, 4):
        p = permute_ohlc(d, seed=s)
        a.plot(p.index, p["close"], color=NOISE, lw=0.8, label="shuffled copies" if s == 1 else None)
    a.plot(d.index, d["close"], color=REAL, lw=1.2, label="real BTC/USDT")
    a.set_yscale("log"); a.set_title("Same hourly moves, different order", loc="left", fontsize=10, color=INK)
    a.legend(frameon=False, fontsize=8); style(a)
    r = np.log(d["close"]).diff().dropna()
    rp = np.log(permute_ohlc(d, seed=1)["close"]).diff().dropna()
    bins = np.linspace(-0.03, 0.03, 61)
    b.hist(r, bins=bins, color=REAL, alpha=0.8, label="real")
    b.hist(rp, bins=bins, histtype="step", color=NOISE, lw=1.5, label="shuffled")
    b.set_title("...so the distribution of moves is identical", loc="left", fontsize=10, color=INK)
    b.set_xlabel("hourly log return", fontsize=8, color=MUTED); b.legend(frameon=False, fontsize=8); style(b)
    fig.suptitle("A permutation keeps volatility, fat tails and the start/end price, and destroys only order. "
                 "An edge that needs order should vanish.", x=0.01, ha="left", fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93)); fig.savefig(os.path.join(FIG, "permutation_explainer.png"), dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------- 2. all tests
def _nulls(first, last):
    files = sorted(glob.glob(os.path.join(ROOT, "results", "nulls", "*.csv")))
    return [f for f in files if first <= int(os.path.basename(f)[:2]) <= last]


def all_tests():
    plot_grid(_nulls(1, 12), os.path.join(FIG, "all_tests.png"), cols=4,
              suptitle="The 12 strategy permutation tests: gray = best result on shuffled markets, "
                       "dark line = the real market")


def decay_tests():
    short = lambda t: (t.replace("Halloween / Sell in May", "Halloween").replace(" after publication, net of costs · sign-flip", ", after costs")
                         .replace(" after publication", "").replace(" · sign-flip", ", before costs"))
    plot_grid(_nulls(13, 30), os.path.join(FIG, "decay_tests.png"), cols=3, title_fn=short,
              xlabel="mean monthly return (0.002 = 0.2% per month)", fmt="{:+.4f}",
              suptitle="Is anything left after publication? Gray = the same months with random signs; line = real.\n"
                       "Rows 1-3: before costs. Rows 4-6: after trading costs and slippage.")


# --------------------------------------------------------------------------- 3. scorecard
ROWS = [  # (ledger test name, strategy group, label)
    ("in_sample_mcpt|donchian", "Donchian breakout\n(ES futures, daily)", "in-sample"),
    ("walk_forward_mcpt|donchian", "Donchian breakout\n(ES futures, daily)", "walk-forward"),
    ("in_sample_mcpt|hawkes_vol", "Volatility Hawkes\n(BTC, hourly)", "in-sample"),
    ("in_sample_signflip|hawkes_vol", "Volatility Hawkes\n(BTC, hourly)", "sign-flip (stricter)"),
    ("walk_forward_mcpt|hawkes_vol", "Volatility Hawkes\n(BTC, hourly)", "walk-forward"),
    ("postpub_fixed_mcpt|hawkes_vol", "Volatility Hawkes\n(BTC, hourly)", "post-publication"),
    ("trendline_A1_insample_2018_2019|trendline_base", "Trendline breakout\n(BTC, hourly)", "in-sample"),
    ("trendline_A2_oos_2020_2022|trendline_base", "Trendline breakout\n(BTC, hourly)", "out-of-sample"),
    ("trendline_A3_postpub_2023_now|trendline_base", "Trendline breakout\n(BTC, hourly)", "post-publication"),
    ("trendline_B1B2_2020_2022_wfmcpt|trendline_meta", "Trendline + ML filter\n(BTC, hourly)", "walk-forward"),
    ("trendline_B1B2_2020_2022_labelshuffle|trendline_meta", "Trendline + ML filter\n(BTC, hourly)", "label-shuffle"),
    ("trendline_B3_postpub_2023_now_labelshuffle|trendline_meta", "Trendline + ML filter\n(BTC, hourly)", "post-publication"),
]


def scorecard():
    L = pd.read_csv(os.path.join(ROOT, "results", "LEDGER.csv"))
    L["key"] = L["test"] + "|" + L["strategy"]
    fig, ax = plt.subplots(figsize=(10, 6.2))
    y, yt, yl, groups = 0, [], [], []
    last = None
    for key, group, label in ROWS:
        row = L[L.key == key].iloc[-1]
        if group != last:
            if last is not None:
                y += 0.8
            groups.append((y, group)); last = group
        p = row.p_value
        c = PASS if p < 0.01 else WEAK if p < 0.05 else FAIL
        ax.axhline(y, color="#eef1f5", lw=0.8, zorder=0)
        ax.scatter(p, y, s=70, color=c, zorder=3)
        ax.text(p * (0.8 if p > 0.3 else 1.25), y, f"p = {p:.3f}   PF {row.real:.3f}", va="center", fontsize=8,
                color=INK, ha="right" if p > 0.3 else "left")
        yt.append(y); yl.append(label); y += 1
    ax.set_xscale("log"); ax.set_xlim(1e-3 * 0.8, 1.3); ax.invert_yaxis()
    ax.set_yticks(yt); ax.set_yticklabels(yl, fontsize=8, color=INK)
    for x, t in ((0.05, "0.05"), (0.01, "0.01")):
        ax.axvline(x, color=GRID, ls="--", lw=1)
        ax.text(x, 1.0, f"p = {t}", transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=8, color=MUTED)
    for gy, g in groups:
        ax.text(1.02, gy - 0.1, g, transform=ax.get_yaxis_transform(), fontsize=9, color=INK, va="top",
                fontweight="bold")
    ax.set_xlabel("p-value (log scale): chance that shuffled data does at least this well. Smaller = stronger evidence.",
                  fontsize=8, color=MUTED)
    style(ax)
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=l) for c, l in
               ((PASS, "p < 0.01"), (WEAK, "0.01 - 0.05 (needs confirmation)"), (FAIL, "p > 0.05 (not distinguishable from noise)"))]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.0, -0.06), fontsize=8, frameon=False)
    ax.set_title(f"Scorecard: the {len(ROWS)} strategy tests (the other {len(L) - len(ROWS)} ledger rows are the "
                 "publication-decay study). Every test counts.",
                 loc="left", fontsize=11, color=INK, pad=18)
    fig.tight_layout(rect=(0, 0, 0.8, 1)); fig.savefig(os.path.join(FIG, "scorecard.png"), dpi=130); plt.close(fig)


# --------------------------------------------------------------------------- 4. hawkes story
def hawkes_story():
    fresh_path = os.path.join(DATA_DIR, "btc_1h_binanceus.csv")
    if not os.path.exists(fresh_path):
        print("hawkes_story skipped: needs data/btc_1h_binanceus.csv (python data/download_binanceus.py)")
        return
    COST = 5.0
    strat = HawkesVolatility()
    author = load_sample_btc()[["open", "high", "low", "close"]]
    train = author.loc[:"2020-12-31"]
    p_is, _ = optimize(strat, train, COST)
    r_is = strategy_returns(train, strat.positions(train, p_is), COST)
    wf = walk_forward(strat, author, 24 * 365 * 3, 24 * 90, COST)
    r_wf = wf["returns"].loc["2021-01-01":]
    p_pp, _ = optimize(strat, author, COST)
    fresh = load_ohlc(fresh_path)[["open", "high", "low", "close"]].loc["2022-10-01":]
    pos_pp = strat.positions(fresh, p_pp)
    r_pp = strategy_returns(fresh, pos_pp, COST).loc["2023-01-01":]
    phases = [("in-sample, tuned\n2018-2020", r_is, "#9aa7b8"), ("walk-forward\n2021-2022", r_wf, "#4d6b8a"),
              ("post-publication, frozen\n2023-now", r_pp, REAL)]
    btc = pd.concat([np.log(author["close"]).diff().loc[:"2022-12-31"], np.log(fresh["close"]).diff().loc["2023-01-01":]])

    fig = plt.figure(figsize=(13, 7.5))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.35, 1])
    a = fig.add_subplot(gs[0, :])
    level = 0.0
    for name, r, c in phases:
        eq = level + r.cumsum()
        a.plot(eq.index, eq, color=c, lw=1.3)
        a.axvspan(r.index[0], r.index[-1], color=c, alpha=0.07)
        a.text(r.index[0], 0.97, " " + name, transform=a.get_xaxis_transform(), fontsize=8, color=INK, va="top")
        level = eq.iloc[-1]
    a.plot([], [], color=REAL, lw=1.3, label="strategy, long/short, after 5 bps costs")
    bh = btc.cumsum()
    a.plot(bh.index, bh, color=GRID, lw=1, label="buy & hold BTC (for scale)")
    a.axhline(0, color=GRID, lw=0.8)
    a.set_ylabel("cumulative log return", fontsize=8, color=MUTED); a.legend(frameon=False, fontsize=8, loc="lower right")
    style(a)
    a.set_title("VolatilityHawkes on BTC, after 5 bps costs: strong when tuned, weaker on unseen data, "
                "weaker still after publication", loc="left", fontsize=10, color=INK)

    b = fig.add_subplot(gs[1, 0])
    allr = pd.concat([r_is, r_wf, r_pp])
    yearly = allr.groupby(allr.index.year).apply(profit_factor)
    cols = ["#9aa7b8" if y <= 2020 else "#4d6b8a" if y <= 2022 else REAL for y in yearly.index]
    b.bar(yearly.index.astype(str), yearly.values - 1, bottom=1, color=cols)
    b.axhline(1, color=INK, lw=0.8); b.set_title("Profit factor by year (1.0 = break-even)", loc="left", fontsize=9, color=INK)
    b.tick_params(axis="x", rotation=45); style(b)

    c = fig.add_subplot(gs[1, 1])
    costs = [0, 5, 10, 20]
    sh = [sharpe(strategy_returns(fresh, pos_pp, k).loc["2023-01-01":], 8760) for k in costs]
    c.bar([f"{k} bps" for k in costs], sh, color=[REAL if k == 5 else NOISE for k in costs])
    for i, v in enumerate(sh):
        c.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=8, color=INK)
    c.set_title("Post-publication Sharpe vs trading cost", loc="left", fontsize=9, color=INK); style(c)

    e = fig.add_subplot(gs[1, 2])
    labels = ["in-sample", "sign-flip", "walk-fwd", "post-pub"]
    L = pd.read_csv(os.path.join(ROOT, "results", "LEDGER.csv")).query("strategy == 'hawkes_vol'").set_index("test")
    ps = [L.p_value[t] for t in ("in_sample_mcpt", "in_sample_signflip", "walk_forward_mcpt", "postpub_fixed_mcpt")]
    e.bar(labels, [-np.log10(p) for p in ps], color=[PASS, PASS, WEAK, WEAK])
    e.axhline(-np.log10(0.05), color=FAIL, ls="--", lw=1); e.text(3.4, -np.log10(0.05) + 0.05, "p = 0.05", fontsize=7, color=FAIL, ha="right")
    for i, p in enumerate(ps):
        e.text(i, -np.log10(p) + 0.05, f"{p:.3f}", ha="center", fontsize=8, color=INK)
    e.set_ylabel("-log10(p)  (higher = stronger)", fontsize=8, color=MUTED)
    e.set_title("Permutation-test p-values by phase", loc="left", fontsize=9, color=INK); style(e)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "hawkes_story.png"), dpi=130); plt.close(fig)
    print(f"hawkes: IS {p_is} PF {profit_factor(r_is):.3f} | WF PF {profit_factor(r_wf):.3f} | "
          f"post-pub {p_pp} PF {profit_factor(r_pp):.3f} Sharpe {sharpe(r_pp, 8760):.2f} | costs {np.round(sh, 2)}")


# --------------------------------------------------------------------------- 5-7. publication decay
STAGE = ("#9aa7b8", "#4d6b8a", "#1f2d3d")          # in-sample, after publication, after publication + costs
EFFECT, COST = "#1f6fb5", "#c2552b"


def _decay_modules():
    if not os.path.isdir(os.path.join(DATA_DIR, "french")):
        print("decay figures skipped: needs data/french/ (python data/download_french.py)")
        return None
    import importlib
    sys.path.insert(0, os.path.join(ROOT, "examples"))
    return importlib.import_module("06_publication_decay"), importlib.import_module("07_decay_after_costs")


def decay_story():
    W = pd.read_csv(os.path.join(ROOT, "results", "decay", "window_stats.csv"))
    N = pd.read_csv(os.path.join(ROOT, "results", "decay", "net_of_costs.csv")).query("run == '1x'")
    P = pd.read_csv(os.path.join(ROOT, "results", "decay", "pooled_primary.csv"), index_col=0)["value"]
    names = list(dict.fromkeys(W.strategy))
    g = lambda df, n, w, col: df[(df.strategy == n) & (df.window == w)][col].iloc[0]
    rows = [(n, W[W.strategy == n].paper.iloc[0], g(W, n, "IS", "mean_pct"), g(W, n, "POST", "mean_pct"),
             g(N, n, "POST", "net_pct"), bool(W[W.strategy == n].replicated_IS.iloc[0])) for n in names]
    rows.sort(key=lambda r: r[3] / r[2], reverse=True)

    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 6.4), gridspec_kw=dict(width_ratios=[1.7, 1]))
    h = 0.26
    for i, (n, paper, is_, po, net, ok) in enumerate(rows):
        for j, (v, c) in enumerate(zip((is_, po, net), STAGE)):
            y = i + (j - 1) * h
            a.barh(y, v, height=h * 0.86, color=c)
            a.text(v + (0.02 if v >= 0 else -0.02), y, f"{v:+.2f}", va="center", fontsize=7, color=INK,
                   ha="left" if v >= 0 else "right")
    a.set_yticks(range(len(rows)))
    a.set_yticklabels([f"{n}\n{p}" + ("" if ok else "  (not pooled)") for n, p, *_, ok in rows], fontsize=8, color=INK)
    a.invert_yaxis(); a.axvline(0, color=INK, lw=0.8)
    a.set_xlabel("average return, % per month (long-short or calendar strategy)", fontsize=8, color=MUTED)
    a.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for c in STAGE],
             labels=["in the authors' own sample", "after publication", "after publication and trading costs"],
             frameon=False, fontsize=8, loc="lower left")
    a.set_xlim(-1.9, 1.35)
    a.text(-1.88, len(rows) - 0.55, "reversal net: -1.51", fontsize=7, color=MUTED) if min(r[4] for r in rows) < -1.9 else None
    a.set_title("Each strategy: what the authors found, what was left after publication, what was left after costs",
                loc="left", fontsize=10, color=INK); style(a)

    ratio = [100 * r[3] / r[2] for r in rows]
    b.scatter(ratio, range(len(rows)), s=60, color=[STAGE[1] if r[5] else "white" for r in rows],
              edgecolors=STAGE[1], linewidths=1.5, zorder=3)
    for i, v in enumerate(ratio):
        b.text(v + 3, i, f"{v:.0f}%", va="center", fontsize=8, color=INK)
    pooled = 100 * (1 + P["b_post"])
    lo, hi = 100 * (1 + P["b_post"] - 1.96 * P["se_post"]), 100 * (1 + P["b_post"] + 1.96 * P["se_post"])
    b.axvspan(lo, hi, color=STAGE[0], alpha=0.25, lw=0)
    b.axvline(pooled, color=INK, lw=1.2)
    b.text(pooled + 1.5, len(rows) - 1.45, f"pooled: {pooled:.0f}% remains\n95% CI {lo:.0f}-{hi:.0f}%", fontsize=8, color=INK)
    b.axvline(100, color=GRID, ls="--", lw=1)
    b.text(99, len(rows) - 1.2, "100% =\nno decay", fontsize=7, color=MUTED, ha="right")
    b.set_yticks(range(len(rows))); b.set_yticklabels([]); b.invert_yaxis(); b.set_xlim(0, 110)
    b.set_xlabel("after-publication return as % of in-sample (before costs)", fontsize=8, color=MUTED)
    b.set_title("Share of the edge that survived publication", loc="left", fontsize=10, color=INK); style(b)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "decay_story.png"), dpi=130); plt.close(fig)


def decay_growth():
    mods = _decay_modules()
    if mods is None:
        return
    v6, v7 = mods
    from mcptlab.decay import load_french, monthly_cost, windows
    series, _, end = v6.build_series()
    daily_idx = load_french("ff3_daily").index
    fig, axes = plt.subplots(3, 3, figsize=(13, 8.4), sharex=False)
    for ax, sp in zip(axes.ravel(), v6.SPECS):
        g = series[sp.key]
        kw = dict(daily_index=daily_idx) if sp.key == "monday" else {}
        net = g - monthly_cost(sp.key, g.index, **kw)
        post = windows(g, sp) == "POST"
        x = g.index[post].to_timestamp()
        wg, wn = (1 + g[post]).cumprod(), (1 + net[post]).cumprod()
        ax.plot(x, wg.values, color=STAGE[1], lw=1.4)
        ax.plot(x, wn.values, color=STAGE[2], lw=2)
        ax.axhline(1, color=GRID, lw=0.8)
        ax.set_yscale("log")
        from matplotlib.ticker import FuncFormatter, LogLocator
        span = np.log10(max(wg.max(), wn.max()) / min(wg.min(), wn.min()))
        subs = (1,) if span > 1.5 else (1, 2, 5) if span > 0.6 else (1, 1.25, 1.5, 2, 3, 5, 7)
        ax.yaxis.set_major_locator(LogLocator(base=10, subs=subs))
        ax.yaxis.set_minor_locator(LogLocator(base=10, subs=()))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v:g}"))
        ends = [wg.iloc[-1], wn.iloc[-1]]
        close = abs(np.log(ends[0] / ends[1])) < 0.08
        for k, (w, va) in enumerate(((wg, "bottom" if ends[0] >= ends[1] else "top"),
                                     (wn, "top" if ends[0] >= ends[1] else "bottom"))):
            v = w.iloc[-1]
            ax.text(x[-1], v, f" ${v:.2f}" if v >= 0.01 else f" ${v:.4f}", fontsize=8, color=INK,
                    va=va if close else "center")
        ax.set_title(f"{sp.label} · {sp.paper}", loc="left", fontsize=9, color=INK)
        ax.set_xlim(x[0], x[-1] + (x[-1] - x[0]) * 0.18); style(ax)
    fig.legend(handles=[plt.Line2D([], [], color=STAGE[1], lw=1.4), plt.Line2D([], [], color=STAGE[2], lw=2)],
               labels=["before costs", "after trading costs and slippage"], loc="upper right", frameon=False, fontsize=9)
    fig.suptitle(f"$1 invested in each strategy the month after its paper was published, through {end} (log scale)",
                 x=0.01, ha="left", fontsize=11, color=INK)
    fig.text(0.01, 0.005, "Monday and turn-of-the-month are relative measures (anomaly days minus normal days), compounded "
             "the same way for comparison; they are not literal fund balances.\nCosts: Novy-Marx & Velikov (2016) for "
             "stock portfolios; era-based market costs for calendar strategies.", fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.035, 1, 0.95)); fig.savefig(os.path.join(FIG, "decay_growth.png"), dpi=130); plt.close(fig)


def calendar_costs():
    mods = _decay_modules()
    if mods is None:
        return
    v6, _ = mods
    from mcptlab.decay import load_french, monthly_cost
    series, _, _ = v6.build_series()
    daily_idx = load_french("ff3_daily").index
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
    for ax, key in zip(axes, ("monday", "tom")):
        sp = next(s for s in v6.SPECS if s.key == key)
        g = series[key]
        kw = dict(daily_index=daily_idx) if key == "monday" else {}
        cost = monthly_cost(key, g.index, **kw)
        roll = lambda z: 100 * z.rolling(120, min_periods=60).mean()
        x = g.index.to_timestamp()
        rg, rc = roll(g), roll(cost)
        ax.fill_between(x, rc, rg, where=rg > rc, color=EFFECT, alpha=0.12, lw=0)
        ax.plot(x, rg, color=EFFECT, lw=2)
        ax.plot(x, rc, color=COST, lw=2)
        ax.text(pd.Timestamp("1945-01-01"), rc.loc["1945-01"] * 1.25, "cost of trading it", fontsize=8, color=INK, va="bottom")
        ax.text(pd.Timestamp("1945-01-01"), rg.loc["1945-01"] * 0.55, "the effect, before costs", fontsize=8, color=INK, va="top")
        ax.axhline(0, color=INK, lw=0.8)
        ax.axvline(pd.Period(sp.pub, "M").to_timestamp(), color=INK, ls="--", lw=1)
        ax.text(pd.Period(sp.pub, "M").to_timestamp(), 0.97, f" published {sp.pub[:4]}", transform=ax.get_xaxis_transform(),
                fontsize=8, color=INK, va="top")
        for yr, t in ((1975, "fixed commissions end"), (1982, "S&P futures"), (2001, "decimal prices")):
            ax.axvline(pd.Timestamp(f"{yr}-05-01"), color=GRID, lw=0.8)
            ax.text(pd.Timestamp(f"{yr}-05-01"), 0.02, f" {t}", transform=ax.get_xaxis_transform(), rotation=90,
                    fontsize=7, color=MUTED, va="bottom")
        ax.set_yscale("symlog", linthresh=0.1, linscale=0.6); ax.set_ylim(-1, 15)
        ax.set_yticks([-1, -0.1, 0, 0.1, 1, 10]); ax.set_yticklabels(["-1", "-0.1", "0", "0.1", "1", "10"])
        ax.set_title(f"{sp.label} ({sp.paper})", loc="left", fontsize=10, color=INK); style(ax)
    axes[0].set_ylabel("% per month, 10-year rolling average\n(log-like scale; 0 and negatives shown)", fontsize=8, color=MUTED)
    fig.suptitle("Calendar effects vs the cost of trading them. Shaded = the effect beat its cost. Before 1975 neither was "
                 "tradable; once trading got cheap, the effects were smaller and less reliable.",
                 x=0.01, ha="left", fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93)); fig.savefig(os.path.join(FIG, "calendar_costs.png"), dpi=130); plt.close(fig)


if __name__ == "__main__":
    only = sys.argv[1:]
    for f in (explainer, all_tests, scorecard, hawkes_story, decay_story, decay_growth, calendar_costs, decay_tests):
        if only and f.__name__ not in only:
            continue
        f(); print("done:", f.__name__)
