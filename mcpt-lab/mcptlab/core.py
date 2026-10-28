"""The four-step strategy test (after neurotrader888/mcpt), with trading costs and a trial ledger.

    1. in_sample_excellence  Optimize on training data. Is the best version any good?
    2. in_sample_mcpt        Re-run the SAME optimization on N permuted copies of the training data.
                             p = share of permutations whose best result >= the real best result.
    3. walk_forward          Re-optimize on a rolling window, trade the next block out-of-sample.
    4. walk_forward_mcpt     The walk-forward on N permutations (only the out-of-sample part is
                             permuted). Small p -> the out-of-sample result isn't luck.

Plus two tests for strategies whose parameters are already fixed:
    fixed_param_mcpt         Trade fixed parameters on new data; permute only the new data.
    permutation_test         Bring-your-own statistic + null (bars, sign-flip, or a custom function).

Conventions
    * A strategy's position decided at bar t's close is applied to the return from t to t+1.
    * Costs: `cost_bps` per unit of position change (a full long -> short flip is 2 units).
    * p = (1 + #{null >= real}) / n_perm  -- the real result counts as one of the n_perm draws,
      so p can never be 0 (smallest possible p = 1 / n_perm).
    * Every test run with log=True is appended to the ledger (default ./LEDGER.csv, or the path in
      the MCPT_LEDGER environment variable) and its null distribution + histogram are saved to
      ./mcpt_runs/. The ledger is your honest count of how many ideas you have tried.
"""
from __future__ import annotations

import csv
import os
import time
from multiprocessing import Pool
from typing import Callable

import numpy as np
import pandas as pd

from .permute import permute_close, permute_ohlc, signflip_ohlc


# ============================================================================ data helpers
def _close(data):
    return data["close"] if isinstance(data, pd.DataFrame) else data


def _is_ohlc(data):
    return isinstance(data, pd.DataFrame) and {"open", "high", "low", "close"} <= set(data.columns)


def permute(data, start_index=0, seed=None, null="bars"):
    """Return one null copy of `data`.
    null="bars"     : OHLC -> neurotrader bar permutation; close Series -> return shuffle.
    null="signflip" : OHLC only; keeps every bar's size in place, randomizes its direction.
    null=callable   : your own function f(data, start_index, seed) -> data."""
    if callable(null):
        return null(data, start_index, seed)
    if null == "signflip":
        if not _is_ohlc(data):
            raise ValueError("signflip needs OHLC data")
        return signflip_ohlc(data[["open", "high", "low", "close"]], start_index=start_index, seed=seed)
    if null != "bars":
        raise ValueError(f"unknown null {null!r}")
    if _is_ohlc(data):
        return permute_ohlc(data[["open", "high", "low", "close"]], start_index=start_index, seed=seed)
    return permute_close(data, start_index=start_index, seed=seed)


def _bpy(strategy):
    return getattr(strategy, "bars_per_year", 252)


def p_value(real, null):
    """neurotrader's convention: the real result counts as one draw."""
    null = np.asarray(null, float)
    return (1 + np.sum(null >= real)) / (len(null) + 1)


def parallel_map(fn, items, procs=None):
    """Pool.map, or a plain loop when procs == 1 (handy for debugging and tests)."""
    items = list(items)
    if procs == 1:
        return [fn(x) for x in items]
    with Pool(procs) as pool:
        return pool.map(fn, items)


# ============================================================================ metrics
def strategy_returns(data, pos, cost_bps=0.0):
    """Per-bar log returns of holding `pos[t]` from bar t to t+1, minus trading costs."""
    close = _close(data)
    r = np.log(close).diff().shift(-1)                      # next-bar log return
    pos = pos.reindex(close.index).fillna(0.0)
    cost = pos.diff().abs().fillna(pos.abs()) * cost_bps / 1e4
    return (pos * r - cost).iloc[:-1]


def profit_factor(rets):
    """Sum of winning bars / sum of losing bars. 1.0 = break-even."""
    rets = pd.Series(rets).dropna()
    gain, loss = rets[rets > 0].sum(), -rets[rets < 0].sum()
    return gain / loss if loss > 0 else np.inf


def sharpe(rets, bars_per_year=252):
    """Annualized Sharpe ratio (no risk-free rate)."""
    rets = pd.Series(rets).dropna()
    return rets.mean() / rets.std() * np.sqrt(bars_per_year) if rets.std() > 0 else 0.0


METRICS = {"pf": profit_factor, "sharpe": sharpe}


# ============================================================================ step 1
def optimize(strategy, data, cost_bps=0.0, metric="pf"):
    """Try every value in strategy.grid; return (best_param, best_metric)."""
    f = METRICS[metric]
    best, best_p = -np.inf, None
    for p in strategy.grid:
        v = f(strategy_returns(data, strategy.positions(data, p), cost_bps))
        if np.isfinite(v) and v > best:
            best, best_p = v, p
    return best_p, best


def in_sample_excellence(strategy, data, cost_bps=0.0, metric="pf"):
    p, v = optimize(strategy, data, cost_bps, metric)
    rets = strategy_returns(data, strategy.positions(data, p), cost_bps)
    return dict(test="in_sample_excellence", best_param=p, metric=v, pf=profit_factor(rets),
                sharpe=sharpe(rets, _bpy(strategy)), returns=rets)


# ============================================================================ step 2
def _is_job(a):
    strategy, data, cost_bps, metric, seed, null = a
    return optimize(strategy, permute(data, seed=seed, null=null), cost_bps, metric)[1]


def in_sample_mcpt(strategy, data, n_perm=1000, cost_bps=0.0, metric="pf", seed=0, null="bars",
                   market="?", note="", procs=None, log=True):
    """Step 2. Optimize on real data and on n_perm - 1 permutations; compare the best results."""
    real_p, real_v = optimize(strategy, data, cost_bps, metric)
    jobs = [(strategy, data, cost_bps, metric, seed + k, null) for k in range(1, n_perm)]
    null_vals = np.array(parallel_map(_is_job, jobs, procs))
    test = "in_sample_mcpt" if null == "bars" else f"in_sample_{null if isinstance(null, str) else 'custom'}"
    res = dict(test=test, best_param=real_p, real=real_v, null=null_vals, p=p_value(real_v, null_vals))
    if log:
        log_test(res, strategy, market, data, cost_bps, metric, n_perm, note)
    return res


# ============================================================================ step 3
def walk_forward(strategy, data, train_bars, step_bars, cost_bps=0.0, metric="pf"):
    """Step 3. Every step_bars, re-optimize on the previous train_bars and trade the next block."""
    n = len(data)
    pos = pd.Series(np.nan, index=_close(data).index)
    params = []
    for start in range(train_bars, n, step_bars):
        p, _ = optimize(strategy, data.iloc[start - train_bars:start], cost_bps, metric)
        full = strategy.positions(data.iloc[:min(start + step_bars, n)], p)   # causal: data <= each bar
        pos.iloc[start:start + step_bars] = full.iloc[start:start + step_bars].values
        params.append((data.index[start], p))
    oos = strategy_returns(data, pos.fillna(0.0), cost_bps).iloc[train_bars:]
    return dict(test="walk_forward", returns=oos, params=params, pf=profit_factor(oos),
                sharpe=sharpe(oos, _bpy(strategy)), metric=METRICS[metric](oos))


# ============================================================================ step 4
def _wf_job(a):
    strategy, data, train_bars, step_bars, cost_bps, metric, seed = a
    perm = permute(data, start_index=train_bars, seed=seed)
    return walk_forward(strategy, perm, train_bars, step_bars, cost_bps, metric)["metric"]


def walk_forward_mcpt(strategy, data, train_bars, step_bars, n_perm=200, cost_bps=0.0,
                      metric="pf", seed=0, market="?", note="", procs=None, log=True):
    """Step 4. The whole walk-forward on permutations of the out-of-sample bars."""
    real = walk_forward(strategy, data, train_bars, step_bars, cost_bps, metric)
    jobs = [(strategy, data, train_bars, step_bars, cost_bps, metric, seed + k) for k in range(1, n_perm)]
    null_vals = np.array(parallel_map(_wf_job, jobs, procs))
    res = dict(test="walk_forward_mcpt", real=real["metric"], null=null_vals,
               p=p_value(real["metric"], null_vals), wf=real)
    if log:
        log_test(res, strategy, market, data, cost_bps, metric, n_perm,
                 f"train={train_bars} step={step_bars} {note}".strip())
    return res


# ============================================================================ fixed-parameter tests
def _fixed_job(a):
    strategy, data, param, start, cost_bps, metric, seed, null = a
    perm = permute(data, start_index=start, seed=seed, null=null)
    rets = strategy_returns(perm, strategy.positions(perm, param), cost_bps).iloc[start:]
    return METRICS[metric](rets)


def fixed_param_mcpt(strategy, data, param, test_start, n_perm=1000, cost_bps=0.0, metric="pf",
                     seed=0, null="bars", market="?", note="", procs=None, log=True):
    """For parameters chosen BEFORE seeing the test period (e.g. a published strategy tested on
    data after its publication date). Nothing is re-optimized; only bars from test_start on are
    permuted, so the warm-up history stays real."""
    start = data.index.get_indexer([pd.Timestamp(test_start)], method="bfill")[0]
    rets = strategy_returns(data, strategy.positions(data, param), cost_bps).iloc[start:]
    real_v = METRICS[metric](rets)
    jobs = [(strategy, data, param, start, cost_bps, metric, seed + k, null) for k in range(1, n_perm)]
    null_vals = np.array(parallel_map(_fixed_job, jobs, procs))
    res = dict(test="fixed_param_mcpt", best_param=param, real=real_v, null=null_vals,
               p=p_value(real_v, null_vals), returns=rets, pf=profit_factor(rets),
               sharpe=sharpe(rets, _bpy(strategy)))
    if log:
        log_test(res, strategy, market, data.iloc[start:], cost_bps, metric, n_perm, note)
    return res


def _custom_job(a):
    statistic, data, start, seed, null = a
    return statistic(permute(data, start_index=start, seed=seed, null=null))


def permutation_test(statistic: Callable, data, n_perm=1000, start_index=0, seed=0, null="bars",
                     procs=None):
    """Most general form: `statistic(data) -> float` (must be a top-level function or a
    functools.partial so it can be sent to worker processes). Returns dict(real, null, p).
    Log it yourself with log_test() if it's a real trial."""
    real = statistic(data)
    jobs = [(statistic, data, start_index, seed + k, null) for k in range(1, n_perm)]
    null_vals = np.array(parallel_map(_custom_job, jobs, procs))
    return dict(test="permutation_test", real=real, null=null_vals, p=p_value(real, null_vals))


# ============================================================================ ledger + autosave
def ledger_path():
    return os.environ.get("MCPT_LEDGER", os.path.join(os.getcwd(), "LEDGER.csv"))


def runs_dir():
    return os.environ.get("MCPT_RUNS", os.path.join(os.getcwd(), "mcpt_runs"))


LEDGER_COLUMNS = ["time", "test", "strategy", "grid_size", "market", "start", "end", "bars",
                  "metric", "real", "p_value", "n_perm", "cost_bps", "best_param", "note"]


def log_test(res, strategy, market, data, cost_bps, metric, n_perm, note=""):
    """Append one row to the ledger and save the null distribution + histogram."""
    _autosave(res, strategy, market)
    path = ledger_path()
    new = not os.path.exists(path)
    idx = _close(data).index
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(LEDGER_COLUMNS)
        w.writerow([time.strftime("%Y-%m-%d %H:%M"), res["test"], getattr(strategy, "name", str(strategy)),
                    len(getattr(strategy, "grid", [None])), market, _d(idx[0]), _d(idx[-1]), len(idx),
                    metric, round(float(res["real"]), 4), round(float(res["p"]), 4), n_perm, cost_bps,
                    res.get("best_param", ""), note])


def _d(x):
    return x.date() if hasattr(x, "date") else x


def ledger_count():
    path = ledger_path()
    return len(pd.read_csv(path)) if os.path.exists(path) else 0


def _autosave(res, strategy, market):
    try:
        from .plotting import plot_null
        d = runs_dir()
        os.makedirs(d, exist_ok=True)
        stem = f"{time.strftime('%Y%m%d_%H%M%S')}_{getattr(strategy, 'name', 'custom')}_{market}_{res['test']}"
        if "null" in res:
            save_null(res, os.path.join(d, stem + "_null.csv"),
                      title=f"{getattr(strategy, 'name', 'custom')} {market} · {res['test']}")
            plot_null(res, os.path.join(d, stem + ".png"),
                      f"{getattr(strategy, 'name', 'custom')} on {market} · {res['test']}")
    except Exception as e:                                    # plotting must never break a test
        print("autosave failed:", e)


def save_null(res, path, title=None):
    """CSV with one column; the header is 'title|real' so plots can be rebuilt later."""
    title = title or res.get("test", "test")
    pd.Series(res["null"], name=f"{title}|{float(res['real'])}").to_csv(path, index=False)


def load_null(path):
    """Inverse of save_null -> (title, real, null array)."""
    head = pd.read_csv(path, nrows=0).columns[0]
    title, real = head.rsplit("|", 1)
    return title, float(real), pd.read_csv(path).iloc[:, 0].to_numpy()
