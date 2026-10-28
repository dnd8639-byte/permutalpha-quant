"""Worked example: Donchian breakout on ES (S&P 500 futures) through all four steps.
Run from the mcpt folder:  python example_donchian.py  [MARKET]   (default ES)"""
import os, sys
import numpy as np
import pandas as pd
from mcpt import (in_sample_excellence, in_sample_mcpt, walk_forward, walk_forward_mcpt,
                  plot_null, save_null, ledger_count)
from strategies import DonchianBreakout

MARKET = sys.argv[1] if len(sys.argv) > 1 else "ES"
COST_BPS = {"ES": 2, "NQ": 2, "ZN": 2, "CL": 5, "GC": 5}.get(MARKET, 5)
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "state", "v3", "clean_daily.csv")


def load_close(market):
    r = pd.read_csv(DATA, index_col=0, parse_dates=True)[market].fillna(0.0)
    close = 100 * (1 + r).cumprod()
    return close[close.index.dayofweek < 5]          # drop Sunday-session stubs (their move rolls into Monday)


if __name__ == "__main__":
    close = load_close(MARKET)
    strat = DonchianBreakout()
    train = close.loc[:"2018-12-31"]
    out = []
    say = lambda s: (print(s, flush=True), out.append(s))

    say(f"Donchian breakout on {MARKET}  |  {len(close)} daily bars {close.index[0].date()}..{close.index[-1].date()}  |  cost {COST_BPS} bps/unit")
    say(f"Grid: {len(strat.grid)} lookbacks ({strat.grid[0]}..{strat.grid[-1]})\n")

    s1 = in_sample_excellence(strat, train, COST_BPS)
    say(f"STEP 1  In-sample excellence (2010-2018): best lookback {s1['best_param']}, PF {s1['pf']:.3f}, Sharpe {s1['sharpe']:.2f}")

    s2 = in_sample_mcpt(strat, train, n_perm=1000, cost_bps=COST_BPS, market=MARKET, note="example")
    plot_null(s2, os.path.join(HERE, f"step2_{MARKET}.png"), f"In-sample MCPT, Donchian on {MARKET}")
    save_null(s2, os.path.join(HERE, f"step2_{MARKET}_null.csv"))
    say(f"STEP 2  In-sample MCPT (1000 perms): p = {s2['p']:.3f}   (noise's best PF: median {np.median(s2['null']):.3f}, 95th pct {np.percentile(s2['null'], 95):.3f})")

    s3 = walk_forward(strat, close, train_bars=1000, step_bars=63, cost_bps=COST_BPS)
    say(f"STEP 3  Walk-forward (4y train, re-optimize quarterly), out-of-sample {s3['returns'].index[0].date()}..: PF {s3['pf']:.3f}, Sharpe {s3['sharpe']:.2f}")

    s4 = walk_forward_mcpt(strat, close, 1000, 63, n_perm=200, cost_bps=COST_BPS, market=MARKET, note="example")
    plot_null(s4, os.path.join(HERE, f"step4_{MARKET}.png"), f"Walk-forward MCPT, Donchian on {MARKET}")
    save_null(s4, os.path.join(HERE, f"step4_{MARKET}_null.csv"))
    say(f"STEP 4  Walk-forward MCPT (200 perms): p = {s4['p']:.3f}")

    say(f"\nLedger now holds {ledger_count()} tests. Every one counts against you when judging the next p-value.")
    open(os.path.join(HERE, f"example_{MARKET}_output.txt"), "w").write("\n".join(out) + "\n")
