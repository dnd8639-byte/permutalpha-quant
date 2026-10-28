"""Template: test YOUR idea on YOUR data. Copy this file and edit the two marked places.

Rules that make the p-value mean something:
  1. Write down WHY the idea should work before you run it.
  2. Keep the grid small. Costs always on.
  3. Run steps 1-2 on the early part of your data only. Look at later data only if step 2 passes.
  4. Every run goes in LEDGER.csv. Judge p against how many ideas you've tried.

Run:  python examples/05_your_own_strategy.py path/to/prices.csv
      (CSV needs a date column and a close column; open/high/low optional)
"""
import sys
import _path  # noqa: F401
import numpy as np
import pandas as pd
from mcptlab import Strategy, in_sample_excellence, in_sample_mcpt, ledger_count, load_ohlc, synthetic


# ---- EDIT 1: your idea -------------------------------------------------------------------------
class RSIReversion(Strategy):
    """WHY: (write your reason here) e.g. 'after sharp drops, forced sellers push price below value'.
    Long when a short RSI is below the threshold, flat otherwise."""
    name = "rsi_reversion"

    def __init__(self):
        self.grid = [(n, lvl) for n in (2, 5, 14) for lvl in (10, 20, 30)]

    def positions(self, data, param):
        close = data["close"] if isinstance(data, pd.DataFrame) else data
        n, lvl = param
        d = close.diff()
        up, dn = d.clip(lower=0).ewm(alpha=1 / n).mean(), (-d.clip(upper=0)).ewm(alpha=1 / n).mean()
        rsi = 100 - 100 / (1 + up / dn)
        return (rsi < lvl).astype(float)          # uses only data up to each bar -- never shift(-1)


# ---- EDIT 2: data, cost, split -----------------------------------------------------------------
COST_BPS = 5.0
TRAIN_FRACTION = 0.6

if __name__ == "__main__":
    data = load_ohlc(sys.argv[1])["close"] if len(sys.argv) > 1 else synthetic("noise", seed=11)
    train = data.iloc[: int(len(data) * TRAIN_FRACTION)]
    strat = RSIReversion()
    s1 = in_sample_excellence(strat, train, COST_BPS)
    print(f"STEP 1  best {s1['best_param']}: PF {s1['pf']:.3f}, Sharpe {s1['sharpe']:.2f}")
    s2 = in_sample_mcpt(strat, train, n_perm=500, cost_bps=COST_BPS, market="my_data")
    print(f"STEP 2  p = {s2['p']:.3f}   (noise's best PF median {np.median(s2['null']):.3f})")
    print(f"Ledger: {ledger_count()} tests so far. A p-value only means something next to that number.")
