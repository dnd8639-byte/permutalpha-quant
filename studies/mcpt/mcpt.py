"""The four-step strategy development process (after neurotrader888/mcpt), with trading costs
and an automatic trial ledger.

  1. in_sample_excellence : optimize on training data. Is the best version any good?
  2. in_sample_mcpt       : re-run the SAME optimization on N permuted copies of the training data.
                            p = share of permutations whose best result >= the real best result.
                            Small p -> the optimizer found structure that noise can't produce.
  3. walk_forward         : re-optimize on a rolling training window, trade the next block
                            out-of-sample, repeat. Only out-of-sample results count.
  4. walk_forward_mcpt    : the walk-forward on N permutations (permuting only the out-of-sample
                            part). Small p -> the out-of-sample result isn't luck.

Conventions: position decided at bar t's close is applied to the return from t to t+1.
Costs: `cost_bps` per unit of position change (a full long->short flip = 2 units).
Every call to steps 2 and 4 is appended to LEDGER.csv -- that's your honest count of trials.
"""
import os, time, csv
import numpy as np
import pandas as pd
from multiprocessing import Pool

from permute import permute_close, permute_ohlc

HERE = os.path.dirname(os.path.abspath(__file__))
LEDGER = os.path.join(HERE, "LEDGER.csv")


# ----------------------------------------------------------------------------- data helpers
def _close(data):
    return data["close"] if isinstance(data, pd.DataFrame) else data


def _permute(data, start_index=0, seed=None):
    """OHLC DataFrame -> neurotrader bar permutation; close Series -> return shuffle."""
    if isinstance(data, pd.DataFrame) and {"open", "high", "low", "close"} <= set(data.columns):
        return permute_ohlc(data[["open", "high", "low", "close"]], start_index=start_index, seed=seed)
    return permute_close(data, start_index=start_index, seed=seed)


def _bpy(strategy):
    return getattr(strategy, "bars_per_year", 252)


# ----------------------------------------------------------------------------- metrics
def strategy_returns(data, pos, cost_bps=0.0):
    close = _close(data)
    r = np.log(close).diff().shift(-1)                      # next-bar log return
    pos = pos.reindex(close.index).fillna(0.0)
    cost = pos.diff().abs().fillna(pos.abs()) * cost_bps / 1e4
    return (pos * r - cost).iloc[:-1]


def profit_factor(rets):
    rets = rets.dropna()
    gain, loss = rets[rets > 0].sum(), -rets[rets < 0].sum()
    return gain / loss if loss > 0 else np.inf


def sharpe(rets, bars_per_year=252):
    rets = rets.dropna()
    return rets.mean() / rets.std() * np.sqrt(bars_per_year) if rets.std() > 0 else 0.0


METRICS = {"pf": profit_factor, "sharpe": sharpe}


# ----------------------------------------------------------------------------- step 1
def optimize(strategy, close, cost_bps=0.0, metric="pf"):
    f = METRICS[metric]
    best, best_p = -np.inf, None
    for p in strategy.grid:
        v = f(strategy_returns(close, strategy.positions(close, p), cost_bps))
        if np.isfinite(v) and v > best:
            best, best_p = v, p
    return best_p, best


def in_sample_excellence(strategy, close, cost_bps=0.0, metric="pf"):
    p, v = optimize(strategy, close, cost_bps, metric)
    rets = strategy_returns(close, strategy.positions(close, p), cost_bps)
    return dict(best_param=p, metric=v, pf=profit_factor(rets), sharpe=sharpe(rets, _bpy(strategy)), returns=rets)


# ----------------------------------------------------------------------------- step 2
def _is_job(a):
    strategy, close, cost_bps, metric, seed = a
    return optimize(strategy, _permute(close, seed=seed), cost_bps, metric)[1]


def in_sample_mcpt(strategy, close, n_perm=1000, cost_bps=0.0, metric="pf", seed=0,
                   market="?", note="", procs=None, log=True):
    real_p, real_v = optimize(strategy, close, cost_bps, metric)
    jobs = [(strategy, close, cost_bps, metric, seed + k) for k in range(1, n_perm)]
    with Pool(procs) as pool:
        null = np.array(pool.map(_is_job, jobs))
    pval = (1 + np.sum(null >= real_v)) / n_perm            # neurotrader's convention: count the real one
    res = dict(test="in_sample_mcpt", best_param=real_p, real=real_v, null=null, p=pval)
    if log:
        _ledger(res, strategy, market, close, cost_bps, metric, n_perm, note)
    return res


# ----------------------------------------------------------------------------- step 3
def walk_forward(strategy, close, train_bars, step_bars, cost_bps=0.0, metric="pf"):
    n = len(close)
    pos = pd.Series(np.nan, index=_close(close).index)
    params = []
    for start in range(train_bars, n, step_bars):
        p, _ = optimize(strategy, close.iloc[start - train_bars:start], cost_bps, metric)
        full = strategy.positions(close.iloc[:min(start + step_bars, n)], p)   # causal: uses data <= each bar
        pos.iloc[start:start + step_bars] = full.iloc[start:start + step_bars].values
        params.append((close.index[start], p))
    oos = strategy_returns(close, pos.fillna(0.0), cost_bps).iloc[train_bars:]
    return dict(returns=oos, params=params, pf=profit_factor(oos), sharpe=sharpe(oos, _bpy(strategy)),
                metric=METRICS[metric](oos))


# ----------------------------------------------------------------------------- step 4
def _wf_job(a):
    strategy, close, train_bars, step_bars, cost_bps, metric, seed = a
    perm = _permute(close, start_index=train_bars, seed=seed)
    return walk_forward(strategy, perm, train_bars, step_bars, cost_bps, metric)["metric"]


def walk_forward_mcpt(strategy, close, train_bars, step_bars, n_perm=200, cost_bps=0.0,
                      metric="pf", seed=0, market="?", note="", procs=None, log=True):
    real = walk_forward(strategy, close, train_bars, step_bars, cost_bps, metric)
    jobs = [(strategy, close, train_bars, step_bars, cost_bps, metric, seed + k) for k in range(1, n_perm)]
    with Pool(procs) as pool:
        null = np.array(pool.map(_wf_job, jobs))
    pval = (1 + np.sum(null >= real["metric"])) / n_perm
    res = dict(test="walk_forward_mcpt", real=real["metric"], null=null, p=pval, wf=real)
    if log:
        _ledger(res, strategy, market, close, cost_bps, metric, n_perm,
                f"train={train_bars} step={step_bars} {note}")
    return res


# ----------------------------------------------------------------------------- ledger + plots
def _ledger(res, strategy, market, close, cost_bps, metric, n_perm, note):
    new = not os.path.exists(LEDGER)
    with open(LEDGER, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["time", "test", "strategy", "grid_size", "market", "start", "end", "bars",
                        "metric", "real", "p_value", "n_perm", "cost_bps", "best_param", "note"])
        w.writerow([time.strftime("%Y-%m-%d %H:%M"), res["test"], strategy.name, len(strategy.grid), market,
                    close.index[0].date(), close.index[-1].date(), len(close), metric,
                    round(float(res["real"]), 4), round(float(res["p"]), 4), n_perm, cost_bps,
                    res.get("best_param", ""), note])


def ledger_count():
    return len(pd.read_csv(LEDGER)) if os.path.exists(LEDGER) else 0


def save_null(res, path):
    pd.Series(res["null"], name=f"{res['test']}_null").to_csv(path, index=False)


def plot_null(res, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(res["null"], bins=40, color="#8aa1c1", label="permutations (noise)")
    ax.axvline(res["real"], color="#c0392b", lw=2, label=f"real = {res['real']:.3f}")
    ax.set_title(f"{title}   p = {res['p']:.3f}")
    ax.set_xlabel("best in-sample profit factor" if res["test"].startswith("in_sample")
                  else "walk-forward (out-of-sample) profit factor")
    ax.legend(); fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)
