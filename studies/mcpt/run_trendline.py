"""neurotrader TrendlineBreakoutMetaLabel through the MCPT process.
DESIGN FIXED 2026-09-24 ~17:40 ET, BEFORE ANY TEST BELOW WAS RUN. Author's parameters throughout
(lookback 72, TP/SL 3 ATR, hold 12, ATR 168; RF 1000 trees depth 3; train 2y, retrain yearly; P>0.5).
Costs: 5 bps per unit of position change (0-cost numbers also printed).

  A. BASE BREAKOUT (every trade, no filter), fixed parameters, long-only
     A1  2018-2019 in-sample                 1000 bar permutations of 2018-2019
     A2  2020-2022 out-of-sample             1000 permutations of bars from 2020 (earlier bars real)
     A3  2023-01 -> now, post-publication    1000 permutations of bars from 2023
  B. META-LABEL FILTER
     B1  walk-forward MCPT 2020-2022, full pipeline (trades rebuilt + forests retrained on each
         permutation of bars from 2020), 200 permutations. Statistic: filtered PF.
     B2  label-shuffle null 2020-2022: same trades & features, training labels randomly permuted,
         200 runs. Does the filter beat a filter that learned nothing? Statistic: filtered PF.
     B3  post-publication 2023 -> now: same walk-forward continued (first 2023 model trained on
         2021-2022), filtered PF + label-shuffle null (200).
  Post-publication data = author file through 2022-12-31 23:00 + Binance.US from 2023-01-01,
  prices rescaled to join without a jump; Binance.US volume rescaled by the ratio of 2022 medians
  (the volume feature is a ratio to its own rolling median, so only the join week is affected).
Usage: python run_trendline.py <author BTCUSDT3600.csv> <btc_1h_binanceus.csv> [A|B|ALL]"""
import os, sys
import numpy as np
import pandas as pd
from multiprocessing import Pool
from permute import permute_ohlc
from trendline_meta import build_trades, meta_label_probs, trade_signals, net_returns, pf
from mcpt import _ledger

HERE = os.path.dirname(os.path.abspath(__file__))
COST, TRAIN, STEP = 5.0, 365 * 24 * 2, 365 * 24


class _S:                                                     # minimal object for the ledger writer
    def __init__(self, name, grid): self.name, self.grid = name, [None] * grid


def load(path):
    d = pd.read_csv(path).dropna()
    d["date"] = d["date"].astype("datetime64[s]")
    return d.set_index("date")[["open", "high", "low", "close", "volume"]].sort_index()


def stitch(author, fresh):
    a = author.loc[:"2022-12-31 23:00:00"]
    f = fresh.loc["2023-01-01":].copy()
    k = a["close"].iloc[-1] / fresh["close"].asof(a.index[-1])
    f[["open", "high", "low", "close"]] *= k
    va = author["volume"].loc["2022"].median(); vf = fresh["volume"].loc["2022"].median()
    f["volume"] *= va / vf
    return pd.concat([a, f])


def with_volume(perm_ohlc, real):
    out = perm_ohlc.copy()
    out["volume"] = real["volume"].values                     # volume path kept as is (not a price series)
    return out


# ---------------------------------------------------------------- A: base breakout
def base_pf(d, start_ts, end_ts=None, cost=COST):
    T = build_trades(d)
    logc = np.log(d["close"].to_numpy())
    allsig = np.zeros(len(d))
    for tr in T.itertuples():
        x = int(tr.exit_i) if np.isfinite(tr.exit_i) else len(d)
        allsig[int(tr.entry_i):x] = 1
    r = net_returns(logc, allsig, cost)
    m = (d.index >= start_ts) & ((d.index <= end_ts) if end_ts is not None else True)
    return pf(r[m]), T


def _a_job(args):
    d, start_i, start_ts, end_ts, seed = args
    p = with_volume(permute_ohlc(d[["open", "high", "low", "close"]], start_index=start_i, seed=seed), d)
    return base_pf(p, start_ts, end_ts)[0]


def test_A(name, d, start_ts, end_ts, n=1000):
    start_i = max(0, d.index.get_indexer([pd.Timestamp(start_ts)], method="bfill")[0] - 1)
    real, T = base_pf(d, start_ts, end_ts)
    real0, _ = base_pf(d, start_ts, end_ts, cost=0.0)
    with Pool() as pool:
        null = np.array(pool.map(_a_job, [(d, start_i, pd.Timestamp(start_ts), end_ts, s) for s in range(1, n)]))
    p = (1 + np.sum(null >= real)) / n
    _ledger(dict(test=f"trendline_{name}", real=real, p=p, best_param="lb72 fixed"), _S("trendline_base", 1),
            "BTCUSDT_1h", d.loc[start_ts:end_ts]["close"], COST, "pf", n, "fixed author params")
    ntr = ((T.entry_i >= d.index.get_indexer([pd.Timestamp(start_ts)], method="bfill")[0])).sum()
    return f"{name}: PF {real:.3f} (0-cost {real0:.3f})  p = {p:.3f}  [noise median {np.median(null):.3f}, 95th {np.percentile(null, 95):.3f}]"


# ---------------------------------------------------------------- B: meta-label
def meta_pf(d, start_ts, end_ts=None, cost=COST, shuffle_seed=None, T=None):
    T = build_trades(d) if T is None else T
    logc = np.log(d["close"].to_numpy())
    prob = meta_label_probs(T, len(d), TRAIN, STEP, shuffle_labels_seed=shuffle_seed)
    sig, allsig = trade_signals(logc, T, prob)
    m = (d.index >= start_ts) & ((d.index <= end_ts) if end_ts is not None else True)
    return pf(net_returns(logc, sig, cost)[m]), pf(net_returns(logc, allsig, cost)[m]), T, (sig[m] > 0).mean()


def _b1_job(args):
    d, start_i, start_ts, end_ts, seed = args
    p = with_volume(permute_ohlc(d[["open", "high", "low", "close"]], start_index=start_i, seed=seed), d)
    return meta_pf(p, start_ts, end_ts)[0]


def _b2_job(args):
    d, T, start_ts, end_ts, seed = args
    return meta_pf(d, start_ts, end_ts, shuffle_seed=seed, T=T)[0]


def test_B(name, d, start_ts, end_ts, n=200, full_pipeline=True):
    real_f, real_all, T, tim = meta_pf(d, start_ts, end_ts)
    f0, a0, _, _ = meta_pf(d, start_ts, end_ts, cost=0.0, T=T)
    lines = [f"{name}: filtered PF {real_f:.3f} (0-cost {f0:.3f}) vs all-trades PF {real_all:.3f} (0-cost {a0:.3f}); "
             f"time in market filtered {tim:.0%}"]
    with Pool() as pool:
        if full_pipeline:
            start_i = max(0, d.index.get_indexer([pd.Timestamp(start_ts)], method="bfill")[0] - 1)
            null = np.array(pool.map(_b1_job, [(d, start_i, pd.Timestamp(start_ts), end_ts, s) for s in range(1, n)]))
            p = (1 + np.sum(null >= real_f)) / n
            _ledger(dict(test=f"trendline_{name}_wfmcpt", real=real_f, p=p, best_param="author"), _S("trendline_meta", 1),
                    "BTCUSDT_1h", d.loc[start_ts:end_ts]["close"], COST, "pf", n, "full pipeline permuted")
            lines.append(f"   full-pipeline WF-MCPT ({n}): p = {p:.3f}  [noise median {np.median(null):.3f}, 95th {np.percentile(null, 95):.3f}]")
        null2 = np.array(pool.map(_b2_job, [(d, T, pd.Timestamp(start_ts), end_ts, s) for s in range(1, n)]))
    p2 = (1 + np.sum(null2 >= real_f)) / n
    _ledger(dict(test=f"trendline_{name}_labelshuffle", real=real_f, p=p2, best_param="author"), _S("trendline_meta", 1),
            "BTCUSDT_1h", d.loc[start_ts:end_ts]["close"], COST, "pf", n, "label-shuffle null")
    lines.append(f"   label-shuffle null ({n}): p = {p2:.3f}  [shuffled-label filter PF median {np.median(null2):.3f}, 95th {np.percentile(null2, 95):.3f}]")
    return "\n".join(lines)


if __name__ == "__main__":
    author, fresh = load(sys.argv[1]), load(sys.argv[2])
    which = sys.argv[3] if len(sys.argv) > 3 else "ALL"
    full = stitch(author, fresh)
    out = []
    say = lambda s: (print(s, flush=True), out.append(s))
    if which in ("A", "ALL"):
        say("A. BASE BREAKOUT (all trades, fixed params, 5 bps)")
        say("  " + test_A("A1_insample_2018_2019", author.loc[:"2019-12-31"], "2018-01-01", None))
        say("  " + test_A("A2_oos_2020_2022", author, "2020-01-01", None))
        say("  " + test_A("A3_postpub_2023_now", full, "2023-01-01", None))
    if which in ("B", "ALL"):
        say("B. META-LABEL FILTER (5 bps)")
        say("  " + test_B("B1B2_2020_2022", author, "2020-01-01", None, full_pipeline=True))
        say("  " + test_B("B3_postpub_2023_now", full, "2023-01-01", None, full_pipeline=False))
    open(os.path.join(HERE, f"trendline_output_{which}.txt"), "w").write("\n".join(out) + "\n")
