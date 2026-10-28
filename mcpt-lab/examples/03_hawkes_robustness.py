"""Two harder questions for a strategy that passed all four steps (VolatilityHawkes, BTC 1h).

  A. SIGN-FLIP NULL. A volatility strategy might pass the bar permutation only because shuffling
     destroys volatility clustering. The sign-flip null keeps every bar's SIZE in its real place and
     randomizes only its DIRECTION. Passing it means the strategy times direction, not just volatility.

  B. POST-PUBLICATION. Take the parameters someone would have picked from the author's 2018-2022 data,
     freeze them, and trade 2023 -> today, data that did not exist when the strategy was published.
     Needs data/btc_1h_binanceus.csv:  python data/download_binanceus.py

Run:  python examples/03_hawkes_robustness.py [--quick]
"""
import os, sys
import _path  # noqa: F401
import numpy as np
import pandas as pd
from mcptlab import (HawkesVolatility, fixed_param_mcpt, in_sample_mcpt, load_ohlc, load_sample_btc,
                     optimize, profit_factor)
from mcptlab.data import DATA_DIR

COST = 5.0

if __name__ == "__main__":
    n = 100 if "--quick" in sys.argv else 1000
    author = load_sample_btc()[["open", "high", "low", "close"]]
    strat = HawkesVolatility()

    a = in_sample_mcpt(strat, author.loc[:"2020-12-31"], n_perm=n, cost_bps=COST, null="signflip",
                       market="BTCUSDT_1h", note="sign-flip null (volatility kept)")
    print(f"A. sign-flip null 2018-2020 ({n}): p = {a['p']:.3f}")

    fresh_path = os.path.join(DATA_DIR, "btc_1h_binanceus.csv")
    if not os.path.exists(fresh_path):
        print("B. skipped: run  python data/download_binanceus.py  first."); sys.exit()
    params, is_pf = optimize(strat, author, COST)
    fresh = load_ohlc(fresh_path)[["open", "high", "low", "close"]].loc["2022-10-01":]   # 3 months warm-up
    b = fixed_param_mcpt(strat, fresh, params, "2023-01-01", n_perm=n, cost_bps=COST,
                         market="BTCUSDT_1h_binanceus", note="post-publication, params fixed on 2018-2022")
    yearly = b["returns"].groupby(b["returns"].index.year).apply(profit_factor)
    print(f"B. post-publication {b['returns'].index[0]:%Y-%m-%d}..{b['returns'].index[-1]:%Y-%m-%d}, params {params} "
          f"(chosen in-sample, PF {is_pf:.3f}):\n   PF {b['pf']:.3f}, Sharpe {b['sharpe']:.2f}, p = {b['p']:.3f}")
    print("   PF by year: " + "  ".join(f"{y}: {v:.3f}" for y, v in yearly.items()))
