"""Secondary check (declared 2026-09-24 after step 2 passed, BEFORE running): is the in-sample edge
DIRECTIONAL, or only the presence of volatility clustering? Same optimization as step 2 on 1000
sign-flipped copies of 2018-2020 (volatility pattern kept exactly, direction randomized).
Reported alongside step 2; it does not replace it."""
import os, sys
import numpy as np, pandas as pd
from multiprocessing import Pool
from mcpt import optimize, plot_null, save_null, _ledger
from permute import signflip_ohlc
from strategies import HawkesVolatility

HERE = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(sys.argv[1]).dropna(); df["date"] = df["date"].astype("datetime64[s]")
train = df.set_index("date")[["open", "high", "low", "close"]].loc[:"2020-12-31"]
strat = HawkesVolatility()
real_p, real_v = optimize(strat, train, 5.0)


def job(seed):
    return optimize(strat, signflip_ohlc(train, seed=seed), 5.0)[1]


if __name__ == "__main__":
    with Pool() as pool:
        null = np.array(pool.map(job, range(1, 1000)))
    p = (1 + np.sum(null >= real_v)) / 1000
    res = dict(test="in_sample_signflip", real=real_v, null=null, p=p, best_param=real_p)
    _ledger(res, strat, "BTCUSDT_1h", train["close"], 5.0, "pf", 1000, "sign-flip null (vol clustering kept)")
    plot_null(res, os.path.join(HERE, "hawkes_signflip.png"), "In-sample sign-flip test (volatility kept)")
    save_null(res, os.path.join(HERE, "hawkes_signflip_null.csv"))
    msg = (f"SIGN-FLIP (volatility kept, direction random), 1000: p = {p:.3f}   "
           f"null best-PF median {np.median(null):.3f}, 95th {np.percentile(null, 95):.3f}   real {real_v:.3f}")
    print(msg); open(os.path.join(HERE, "hawkes_signflip_output.txt"), "w").write(msg + "\n")
