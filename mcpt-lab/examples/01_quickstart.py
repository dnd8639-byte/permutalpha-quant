"""Quickstart (about 1-2 minutes, no downloads): does the test tell noise from a real edge?

We generate fake markets where we KNOW the answer:
  * noise : realistic random prices (fat tails, volatility clustering) -- no strategy can have an edge
  * trend : prices with persistent up/down regimes -- a breakout strategy SHOULD have an edge
For 5 markets of each kind, run step 1 (tune the strategy) and step 2 (in-sample permutation test).

Run:  python examples/01_quickstart.py
"""
import _path  # noqa: F401
from mcptlab import DonchianBreakout, in_sample_mcpt, plot_null, synthetic, strategy_returns, profit_factor

if __name__ == "__main__":
    strat = DonchianBreakout(range(10, 201, 10))
    print("market     best lookback   tuned profit factor   permutation p")
    for kind in ("noise", "trend"):
        for seed in range(1, 6):
            prices = synthetic(kind, seed=seed)
            res = in_sample_mcpt(strat, prices, n_perm=100, cost_bps=0, log=False)
            flag = "  <- p < 0.05" if res["p"] < 0.05 else ""
            print(f"{kind} #{seed}   {res['best_param']:>8d}          {res['real']:.3f}               {res['p']:.3f}{flag}")
        plot_null(res, f"quickstart_{kind}.png", f"Donchian on synthetic {kind} #{seed}")
    print("\nTuning finds a 'profitable' setting (profit factor > 1) in most noise markets too.")
    print("The permutation test separates them. On noise about 1 run in 20 still lands below 0.05 by")
    print("chance -- which is why every test goes in the ledger and single results need confirmation.")
