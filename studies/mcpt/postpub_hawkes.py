"""Post-publication test of VolatilityHawkes. Design fixed 2026-09-24 BEFORE any 2023+ data was downloaded.

  1. Parameters = the best (kappa, lookback) by profit factor, 5 bps cost, on the author's FULL
     2018-2022 file -- exactly what someone watching the video in 2023 would have chosen.
  2. Trade those fixed parameters on 2023-01-01 -> latest (Binance.US BTC/USDT 1h), 5 bps.
  3. Permutation test with FIXED parameters: shuffle only the 2023+ bars (1000 permutations),
     p = share of shuffles with PF >= real. (No re-optimization: nothing is being tuned.)
  4. Reported: 2022 overlap check between the two data sources (correlation of hourly returns).
Usage: python postpub_hawkes.py <author BTCUSDT3600.csv> <btc_1h_binanceus.csv>"""
import os, sys
import numpy as np
import pandas as pd
from multiprocessing import Pool
from mcpt import optimize, strategy_returns, profit_factor, sharpe, _ledger, plot_null, save_null
from permute import permute_ohlc
from strategies import HawkesVolatility

HERE = os.path.dirname(os.path.abspath(__file__))
COST, N_PERM, SPLIT = 5.0, 1000, pd.Timestamp("2023-01-01")


def load(path):
    d = pd.read_csv(path).dropna()
    d["date"] = d["date"].astype("datetime64[s]")
    return d.set_index("date")[["open", "high", "low", "close"]].sort_index()


author, fresh = load(sys.argv[1]), load(sys.argv[2])
strat = HawkesVolatility()
params, is_pf = optimize(strat, author, COST)

ov = pd.concat([np.log(author.close).diff(), np.log(fresh.close).diff()], axis=1, keys=["a", "f"]).dropna()
ov = ov.loc["2022-01-01":"2022-12-31"]

# warm-up: the strategy needs ~2 weeks of history (ATR 336 bars) before 2023 -> use fresh data from 2022-10
data = fresh.loc["2022-10-01":]
start = data.index.get_indexer([SPLIT], method="bfill")[0]
pos = strat.positions(data, params)
rets = strategy_returns(data, pos, COST).loc[SPLIT:]
real_pf = profit_factor(rets)


def job(seed):
    perm = permute_ohlc(data, start_index=start, seed=seed)
    return profit_factor(strategy_returns(perm, strat.positions(perm, params), COST).loc[SPLIT:])


if __name__ == "__main__":
    with Pool() as pool:
        null = np.array(pool.map(job, range(1, N_PERM)))
    p = (1 + np.sum(null >= real_pf)) / N_PERM
    res = dict(test="postpub_fixed_mcpt", real=real_pf, null=null, p=p, best_param=params)
    _ledger(res, strat, "BTCUSDT_1h_binanceus", data.loc[SPLIT:]["close"], COST, "pf", N_PERM,
            "post-publication, params fixed from author's 2018-2022 data")
    plot_null(res, os.path.join(HERE, "hawkes_postpub.png"), "Post-publication (2023+), fixed params")
    save_null(res, os.path.join(HERE, "hawkes_postpub_null.csv"))
    yearly = rets.groupby(rets.index.year).apply(profit_factor)
    lines = [
        f"Parameters chosen on author's 2018-2022 data: {params} (in-sample PF {is_pf:.3f})",
        f"Source check, 2022 hourly returns: correlation {ov['a'].corr(ov['f']):.4f} over {len(ov)} bars",
        f"POST-PUBLICATION {rets.index[0].date()}..{rets.index[-1].date()}: PF {real_pf:.3f}  Sharpe {sharpe(rets, 8760):.2f}  "
        f"time in market {(pos.loc[SPLIT:] != 0).mean():.0%}",
        "PF by year: " + "  ".join(f"{y}: {v:.3f}" for y, v in yearly.items()),
        f"Permutation test (fixed params, 2023+ shuffled, {N_PERM}): p = {p:.3f}",
    ]
    print("\n".join(lines))
    open(os.path.join(HERE, "hawkes_postpub_output.txt"), "w").write("\n".join(lines) + "\n")
