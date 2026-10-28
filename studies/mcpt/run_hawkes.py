"""neurotrader VolatilityHawkes through the four-step MCPT process.
Settings fixed BEFORE running (2026-09-24):
  data   : author's BTCUSDT 1h bars, 2018-01-01 .. 2022-12-31 (from the VolatilityHawkes repo)
  IS     : 2018-2020, grid = author's 25 (kappa x lookback) combos, metric = profit factor
  costs  : 5 bps per unit of position change (~perpetual-futures taker fee); step 1 also shown at 0
  MCPT   : 1000 in-sample permutations (neurotrader bar permutation, OHLC)
  WF     : train 3 years (26280 bars), re-optimize every 90 days (2160 bars), OOS 2021-2022
  WF-MCPT: 200 permutations
Usage: python run_hawkes.py [path/to/BTCUSDT3600.csv]"""
import os, sys
import numpy as np
import pandas as pd
from mcpt import (in_sample_excellence, in_sample_mcpt, walk_forward, walk_forward_mcpt,
                  plot_null, save_null, ledger_count)
from strategies import HawkesVolatility

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "..", "..", "Downloads",
                                                          "VolatilityHawkes-main", "BTCUSDT3600.csv")
COST = 5.0
TRAIN, STEP = 24 * 365 * 3, 24 * 90

df = pd.read_csv(CSV).dropna()
df["date"] = df["date"].astype("datetime64[s]")
df = df.set_index("date")[["open", "high", "low", "close"]]
train = df.loc[:"2020-12-31"]
strat = HawkesVolatility()
out = []
say = lambda s: (print(s, flush=True), out.append(s))

say(f"VolatilityHawkes | BTCUSDT 1h {df.index[0]} .. {df.index[-1]} ({len(df)} bars) | cost {COST} bps | grid {len(strat.grid)}")
s0 = in_sample_excellence(strat, train, 0.0)
s1 = in_sample_excellence(strat, train, COST)
say(f"STEP 1  In-sample 2018-2020: best {s1['best_param']} PF {s1['pf']:.3f} Sharpe {s1['sharpe']:.2f}   (zero cost: best {s0['best_param']} PF {s0['pf']:.3f})")

s2 = in_sample_mcpt(strat, train, n_perm=1000, cost_bps=COST, market="BTCUSDT_1h", note="neurotrader hawkes")
plot_null(s2, os.path.join(HERE, "hawkes_step2.png"), "In-sample MCPT, VolatilityHawkes on BTC 1h")
save_null(s2, os.path.join(HERE, "hawkes_step2_null.csv"))
say(f"STEP 2  In-sample MCPT (1000): p = {s2['p']:.3f}   noise best-PF median {np.median(s2['null']):.3f}, 95th {np.percentile(s2['null'], 95):.3f}")

s3 = walk_forward(strat, df, TRAIN, STEP, COST)
say(f"STEP 3  Walk-forward OOS {s3['returns'].index[0].date()}..{s3['returns'].index[-1].date()}: PF {s3['pf']:.3f} Sharpe {s3['sharpe']:.2f}")
say("        params chosen: " + ", ".join(f"{d.date()}:{p}" for d, p in s3["params"]))

s4 = walk_forward_mcpt(strat, df, TRAIN, STEP, n_perm=200, cost_bps=COST, market="BTCUSDT_1h", note="neurotrader hawkes")
plot_null(s4, os.path.join(HERE, "hawkes_step4.png"), "Walk-forward MCPT, VolatilityHawkes on BTC 1h")
save_null(s4, os.path.join(HERE, "hawkes_step4_null.csv"))
say(f"STEP 4  Walk-forward MCPT (200): p = {s4['p']:.3f}")
say(f"\nLedger: {ledger_count()} tests logged.")
open(os.path.join(HERE, "hawkes_output.txt"), "w").write("\n".join(out) + "\n")
