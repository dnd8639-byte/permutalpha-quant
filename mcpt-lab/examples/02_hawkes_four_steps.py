"""All four steps on a real published strategy: neurotrader888's VolatilityHawkes, BTC/USDT 1-hour
bars 2018-2022 (bundled in data/). Settings were fixed before the original run:
  in-sample 2018-2020, 25-combination grid, profit factor, 5 bps cost per unit traded,
  1000 in-sample permutations; walk-forward: 3-year training window, re-optimized every 90 days,
  out-of-sample 2021-2022; 200 walk-forward permutations.

Run:  python examples/02_hawkes_four_steps.py            (full: ~1-2 hours on a laptop)
      python examples/02_hawkes_four_steps.py --quick    (100 / 50 permutations, ~10-15 minutes)
Every test is appended to ./LEDGER.csv and its histogram saved in ./mcpt_runs/.
"""
import sys
import _path  # noqa: F401
import numpy as np
from mcptlab import (HawkesVolatility, in_sample_excellence, in_sample_mcpt, load_sample_btc,
                     walk_forward, walk_forward_mcpt)

COST, TRAIN, STEP = 5.0, 24 * 365 * 3, 24 * 90

if __name__ == "__main__":
    quick = "--quick" in sys.argv
    n_is, n_wf = (100, 50) if quick else (1000, 200)
    data = load_sample_btc()[["open", "high", "low", "close"]]
    train = data.loc[:"2020-12-31"]
    strat = HawkesVolatility()
    print(f"VolatilityHawkes | BTC 1h {data.index[0]:%Y-%m-%d}..{data.index[-1]:%Y-%m-%d} | {len(strat.grid)} parameter sets | {COST} bps")

    s1 = in_sample_excellence(strat, train, COST)
    print(f"STEP 1  in-sample 2018-2020: best {s1['best_param']}, PF {s1['pf']:.3f}, Sharpe {s1['sharpe']:.2f}")

    s2 = in_sample_mcpt(strat, train, n_perm=n_is, cost_bps=COST, market="BTCUSDT_1h", note="example 02")
    print(f"STEP 2  in-sample MCPT ({n_is}): p = {s2['p']:.3f}   noise's best PF: median {np.median(s2['null']):.3f}, 95th pct {np.percentile(s2['null'], 95):.3f}")
    if s2["p"] > 0.05:
        print("        Step 2 failed -> stop here. (Do not look at out-of-sample results for a failed idea.)")
        sys.exit()

    s3 = walk_forward(strat, data, TRAIN, STEP, COST)
    print(f"STEP 3  walk-forward, out-of-sample {s3['returns'].index[0]:%Y-%m-%d}..: PF {s3['pf']:.3f}, Sharpe {s3['sharpe']:.2f}")

    s4 = walk_forward_mcpt(strat, data, TRAIN, STEP, n_perm=n_wf, cost_bps=COST, market="BTCUSDT_1h", note="example 02")
    print(f"STEP 4  walk-forward MCPT ({n_wf}): p = {s4['p']:.3f}")
