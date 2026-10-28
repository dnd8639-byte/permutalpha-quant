"""neurotrader888's TrendlineBreakoutMetaLabel: a breakout strategy + a machine-learning filter
(random forest) that decides which breakouts to take. Needs scikit-learn.

Design fixed before the original run (author's parameters throughout; 5 bps cost):
  A  BASE BREAKOUT, every trade, fixed parameters -- bar-permutation test (1000)
       A1 2018-2019 in-sample   A2 2020-2022 out-of-sample   A3 2023 -> now (post-publication)
  B  META-LABEL FILTER
       B1 walk-forward MCPT 2020-2022: trades rebuilt + forests retrained on every permuted market (200)
       B2 label-shuffle null 2020-2022: same trades, training labels scrambled (200).
          Does the filter beat a filter that learned nothing?
       B3 label-shuffle null, 2023 -> now (post-publication)
A3 and B3 need data/btc_1h_binanceus.csv  (python data/download_binanceus.py).

Run:  python examples/04_trendline_metalabel.py [A|B|ALL] [--quick]     (full run: several hours)
"""
import os, sys
from functools import partial
import _path  # noqa: F401
import numpy as np
import pandas as pd
from mcptlab import load_ohlc, load_sample_btc, p_value, permute_ohlc
from mcptlab.core import parallel_map
from mcptlab.data import DATA_DIR
from mcptlab.trendline import build_trades, meta_label_probs, net_returns, pf, trade_signals

COST, TRAIN, STEP = 5.0, 365 * 24 * 2, 365 * 24


def stitch(author, fresh):
    """Author's file through 2022 + Binance.US from 2023, rescaled to join without a price jump.
    Volume is rescaled by the ratio of 2022 medians (the volume feature is relative to its own median)."""
    a = author.loc[:"2022-12-31 23:00:00"]
    f = fresh.loc["2023-01-01":].copy()
    f[["open", "high", "low", "close"]] *= a["close"].iloc[-1] / fresh["close"].asof(a.index[-1])
    f["volume"] *= author["volume"].loc["2022"].median() / fresh["volume"].loc["2022"].median()
    return pd.concat([a, f])


def permuted(d, start_i, seed):
    out = permute_ohlc(d[["open", "high", "low", "close"]], start_index=start_i, seed=seed)
    out["volume"] = d["volume"].values                     # volume path kept as is
    return out


def base_pf(d, start_ts, cost=COST):
    T = build_trades(d)
    sig = np.zeros(len(d))
    for tr in T.itertuples():
        sig[int(tr.entry_i):int(tr.exit_i) if np.isfinite(tr.exit_i) else len(d)] = 1
    return pf(net_returns(np.log(d["close"].to_numpy()), sig, cost)[d.index >= start_ts])


def meta_pf(d, start_ts, cost=COST, shuffle_seed=None, T=None):
    T = build_trades(d) if T is None else T
    logc = np.log(d["close"].to_numpy())
    sig, allsig = trade_signals(logc, T, meta_label_probs(T, len(d), TRAIN, STEP, shuffle_labels_seed=shuffle_seed))
    m = d.index >= start_ts
    return pf(net_returns(logc, sig, cost)[m]), pf(net_returns(logc, allsig, cost)[m])


def _a_job(args):
    d, start_i, start_ts, seed = args
    return base_pf(permuted(d, start_i, seed), start_ts)


def _b1_job(args):
    d, start_i, start_ts, seed = args
    return meta_pf(permuted(d, start_i, seed), start_ts)[0]


def _b2_job(args):
    d, T, start_ts, seed = args
    return meta_pf(d, start_ts, shuffle_seed=seed, T=T)[0]


def _start(d, ts):
    return max(0, d.index.get_indexer([pd.Timestamp(ts)], method="bfill")[0] - 1)


def test_A(name, d, start_ts, n):
    ts = pd.Timestamp(start_ts)
    real = base_pf(d, ts)
    null = np.array(parallel_map(_a_job, [(d, _start(d, ts), ts, s) for s in range(1, n)]))
    return f"{name}: base breakout PF {real:.3f}   p = {p_value(real, null):.3f}"


def test_B(name, d, start_ts, n, full_pipeline):
    ts = pd.Timestamp(start_ts)
    T = build_trades(d)
    real_f, real_all = meta_pf(d, ts, T=T)
    out = [f"{name}: filtered PF {real_f:.3f} vs every-trade PF {real_all:.3f}"]
    if full_pipeline:
        null = np.array(parallel_map(_b1_job, [(d, _start(d, ts), ts, s) for s in range(1, n)]))
        out.append(f"   walk-forward MCPT, whole pipeline ({n}): p = {p_value(real_f, null):.3f}")
    null2 = np.array(parallel_map(_b2_job, [(d, T, ts, s) for s in range(1, n)]))
    out.append(f"   label-shuffle null ({n}): p = {p_value(real_f, null2):.3f}")
    return "\n".join(out)


if __name__ == "__main__":
    which = next((a for a in sys.argv[1:] if a in ("A", "B", "ALL")), "ALL")
    nA, nB = (100, 40) if "--quick" in sys.argv else (1000, 200)
    author = load_sample_btc()
    fresh_path = os.path.join(DATA_DIR, "btc_1h_binanceus.csv")
    full = stitch(author, load_ohlc(fresh_path)) if os.path.exists(fresh_path) else None
    if which in ("A", "ALL"):
        print(test_A("A1 2018-2019 in-sample", author.loc[:"2019-12-31"], "2018-01-01", nA))
        print(test_A("A2 2020-2022 out-of-sample", author, "2020-01-01", nA))
        if full is not None:
            print(test_A("A3 2023-now post-publication", full, "2023-01-01", nA))
    if which in ("B", "ALL"):
        print(test_B("B1/B2 2020-2022", author, "2020-01-01", nB, full_pipeline=True))
        if full is not None:
            print(test_B("B3 2023-now post-publication", full, "2023-01-01", nB, full_pipeline=False))
    if full is None:
        print("(post-publication tests skipped: run  python data/download_binanceus.py)")
