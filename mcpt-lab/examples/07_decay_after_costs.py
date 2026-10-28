"""The v6 publication-decay strategies after trading costs and slippage  (PREREGISTRATION_v7.md).

Costs: measured effective spreads for the stock-portfolio strategies (Novy-Marx & Velikov 2016),
and era-dependent costs of trading the whole market for the calendar strategies (stock basket before
1982, S&P 500 futures / ETF after). Effective spreads already include a small trader's slippage;
the 2x run stands in for market impact.

    python data/download_french.py                 (once)
    python examples/07_decay_after_costs.py        (~1 minute)
"""
import importlib
import os
import sys

import _path  # noqa: F401
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
os.environ.setdefault("MCPT_LEDGER", os.path.join(RESULTS, "LEDGER.csv"))
os.environ.setdefault("MCPT_RUNS", os.path.join(RESULTS, "decay", "runs"))

from mcptlab import log_test, permutation_test, save_null  # noqa: E402
from mcptlab.decay import (faber_timing, load_french, max_drawdown, mean_stat, monthly_cost,  # noqa: E402
                           monthly_units, signflip_series, windows)

v6 = importlib.import_module("06_publication_decay")
SPECS, FABER, _Named = v6.SPECS, v6.FABER, v6._Named
N_PERM = 10_000
RUNS = {"1x": dict(mult=1.0), "0.5x": dict(mult=0.5), "2x": dict(mult=2.0), "era": dict(mult=1.0, era_scaled=True)}


def _stats(x):
    x = x.dropna()
    return x.mean(), x.mean() / x.std() * np.sqrt(len(x)), x.mean() / x.std() * np.sqrt(12)


def main():
    series, ff3, end = v6.build_series()
    daily_idx = load_french("ff3_daily").index
    out = os.path.join(RESULTS, "decay")
    os.makedirs(out, exist_ok=True)
    print(f"Data through {end}. Net = gross - cost. Returns in %/month.\n")

    rows, post_net = [], {}
    for sp in SPECS:
        g = series[sp.key]
        lab = windows(g, sp)
        kw = dict(daily_index=daily_idx) if sp.key == "monday" else {}
        units = monthly_units(sp.key, g.index, **kw)
        for run, opts in RUNS.items():
            net = g - monthly_cost(sp.key, g.index, **opts, **kw)
            for w in ("IS", "OOS", "POST"):
                m = lab == w
                mu, t, shp = _stats(net[m])
                rows.append(dict(strategy=sp.label, key=sp.key, run=run, window=w, months=int(m.sum()),
                                 gross_pct=100 * g[m].mean(), cost_pct=100 * (g[m] - net[m]).mean(),
                                 net_pct=100 * mu, net_t=t, net_sharpe=shp,
                                 breakeven_oneway_pct=100 * g[m].mean() / units[m].mean()))
            if run == "1x":
                post_net[sp.key] = net[lab == "POST"]
    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(out, "net_of_costs.csv"), index=False)

    base = tab[tab.run == "1x"].set_index(["key", "window"])
    print(f"{'strategy':26s} {'IS gross':>8s} {'IS net':>7s} | {'POST gross':>10s} {'cost':>6s} {'net':>7s} {'t':>5s}"
          f" | {'POST net 0.5x':>13s} {'2x':>7s} {'era':>7s} | {'break-even':>10s}")
    for sp in SPECS:
        b_is, b_po = base.loc[(sp.key, "IS")], base.loc[(sp.key, "POST")]
        alt = {r: tab[(tab.key == sp.key) & (tab.run == r) & (tab.window == "POST")].net_pct.iloc[0]
               for r in ("0.5x", "2x", "era")}
        print(f"{sp.label:26s} {b_is.gross_pct:8.3f} {b_is.net_pct:7.3f} | {b_po.gross_pct:10.3f} "
              f"{b_po.cost_pct:6.3f} {b_po.net_pct:7.3f} {b_po.net_t:5.1f} | {alt['0.5x']:13.3f} "
              f"{alt['2x']:7.3f} {alt['era']:7.3f} | {b_po.breakeven_oneway_pct:9.3f}%")
    print("  (break-even = one-way cost per unit traded at which the POST net return is zero)")

    # ---- POST net sign-flip tests (trials 36-44)
    print(f"\nAnything left after publication AND costs? Sign-flip on POST net returns ({N_PERM:,} draws):")
    num = 22
    for sp in SPECS:
        x = post_net[sp.key]
        pt = permutation_test(mean_stat, x.to_numpy(), n_perm=N_PERM, null=signflip_series, procs=1)
        pt["real"] = float(pt["real"])
        print(f"  {sp.label:26s} POST net {100 * x.mean():+.3f}%/mo   p = {pt['p']:.4f}")
        save_null(pt, os.path.join(RESULTS, "nulls", f"{num:02d}_decay_{sp.key}_post_net_signflip.csv"),
                  title=f"{sp.label} after publication, net of costs · sign-flip")
        log_test(dict(test="post_pub_net_signflip", real=pt["real"], p=pt["p"]), _Named(sp.key),
                 "FF_US_equity", x.to_timestamp(), "NMV/era", "mean", N_PERM,
                 f"PREREG v7 descriptive; net of costs; POST from {x.index[0]}")
        num += 1
    print(f"  Bonferroni bar for 9 tests: p < {0.05 / 9:.4f}")

    # ---- Faber, net of era costs
    fb = faber_timing(ff3["Mkt-RF"] + ff3["RF"], ff3["RF"], cost=0.0)
    fcost = monthly_cost("faber", fb.index, faber_pos=fb["pos"])
    fb["timed"] = fb["timed"] - fcost
    lab = windows(fb["timed"], FABER)
    frows = []
    for w in ("IS", "OOS", "POST"):
        x = fb[lab == w]
        sh = lambda e: e.mean() / e.std() * np.sqrt(12)
        frows.append(dict(window=w, months=len(x), sharpe_timed=sh(x["timed"] - x["rf"]),
                          sharpe_buyhold=sh(x["bh"] - x["rf"]), mdd_timed=max_drawdown(x["timed"]),
                          mdd_buyhold=max_drawdown(x["bh"]), cost_pct=100 * fcost[lab == w].mean(),
                          switches_per_year=12 * monthly_units("faber", x.index, faber_pos=fb["pos"]).mean()))
    ftab = pd.DataFrame(frows)
    ftab.to_csv(os.path.join(out, "faber_net.csv"), index=False)
    print("\nFaber 10-month MA timing, net of era costs:")
    for _, f in ftab.iterrows():
        print(f"  {f.window:4s} Sharpe {f.sharpe_timed:.2f} vs buy-and-hold {f.sharpe_buyhold:.2f}   max DD "
              f"{f.mdd_timed:.0%} vs {f.mdd_buyhold:.0%}   cost {f.cost_pct:.3f}%/mo   {f.switches_per_year:.1f} switches/yr")


if __name__ == "__main__":
    sys.exit(main())
