"""edge-engine: the safe search loop, the one-shot holdout, and permutation tests of the WHOLE search.

  1. Split each market's days: first 70% = development, last 30% = holdout (never seen by the search).
  2. Inside development, walk-forward: the ER model is retrained on all earlier days and predicts the next block.
     Every strategy is scored only on those out-of-sample development blocks.
  3. The loop: start from random strategies; each round, mutate the best ones and add new random ones; keep the
     best few found so far (elitism) and drop the rest. Fixed number of rounds. Every strategy tried is logged.
  4. The finalist is run ONCE on the holdout.
  5. Permutation tests: (a) run the entire search on shuffled development data, many times, and compare its best
     score with the real search's best score; (b) run the frozen finalist on shuffled holdout data.
"""
from __future__ import annotations

import copy
import json
import os
import sys
from dataclasses import dataclass, field
import multiprocessing as mp
from statistics import NormalDist

import numpy as np
import pandas as pd

from core import GATES, RULES, SIDES, ERModel, Market, execute, positions, sharpe

ND = NormalDist()
for _p in (os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "mcpt-lab"),):
    if os.path.isdir(os.path.join(_p, "mcptlab")):
        sys.path.insert(0, _p)


@dataclass
class Settings:
    dev_frac: float = 0.70
    min_train_days: int = 365
    step_days: int = 90
    rounds: int = 12
    pop_init: int = 24
    children: int = 18
    fresh: int = 6
    elite: int = 6
    min_trades_per_year: float = 4.0
    both_sides: bool = True        # only strategies that trade long AND short (low market exposure)
    seed: int = 0


# ---------------------------------------------------------------------------------------------- genomes
def random_genome(rng, both_sides=False):
    rule = rng.choice(list(RULES))
    params = {k: int(rng.choice(v)) for k, v in RULES[rule].items()}
    if rule == "ma_cross" and params["fast"] >= params["slow"]:
        params["fast"], params["slow"] = 5, 40
    side = "both" if both_sides else str(rng.choice(SIDES))
    return dict(rule=str(rule), params=params, gate=str(rng.choice(list(GATES))), side=side)


def mutate(g, rng, both_sides=False):
    g = copy.deepcopy(g)
    what = rng.choice(["param", "gate", "side", "rule"], p=[0.5, 0.25, 0.15, 0.10])
    if both_sides and what == "side":
        what = "gate"
    if what == "param":
        k = rng.choice(list(g["params"]))
        g["params"][k] = int(rng.choice(RULES[g["rule"]][k]))
        if g["rule"] == "ma_cross" and g["params"]["fast"] >= g["params"]["slow"]:
            return mutate(g, rng, both_sides)
    elif what == "gate":
        g["gate"] = str(rng.choice(list(GATES)))
    elif what == "side":
        g["side"] = str(rng.choice(SIDES))
    else:
        g = {**random_genome(rng, both_sides), "gate": g["gate"], "side": g["side"]}
    return g


def key(g):
    return json.dumps(g, sort_keys=True)


# ---------------------------------------------------------------------------------------------- walk-forward ER tiers
def split_day(m: Market, s: Settings):
    return m.D.index[int(len(m.D) * s.dev_frac)]


def walk_forward_tiers(m: Market, s: Settings, start, end, seed=0):
    """ER-model tiers for days in [start, end), each block predicted by a model trained only on earlier days.
    Also returns out-of-sample predictions for model diagnostics."""
    F = m.F
    tiers = np.full(len(F), -1)
    preds = np.full(len(F), np.nan)
    days = F.index
    first = max(days.searchsorted(start), s.min_train_days)
    stop = days.searchsorted(end)
    for b in range(first, stop, s.step_days):
        train = F.iloc[:b - 1]                                     # drop the last training row: its target is in the test block
        model = ERModel(seed).fit(train)
        blk = slice(b, min(b + s.step_days, stop))
        p = model.predict(F.iloc[blk])
        preds[blk], tiers[blk] = p, model.tier(p)
    return tiers, preds


@dataclass
class Prepared:
    markets: list
    tiers: dict = field(default_factory=dict)
    preds: dict = field(default_factory=dict)
    windows: dict = field(default_factory=dict)       # market -> (first OOS day, last day) for scoring


def prepare(markets, s: Settings, phase="dev"):
    P = Prepared(markets)
    for m in markets:
        cut = split_day(m, s)
        if phase == "dev":
            t, p = walk_forward_tiers(m, s, m.D.index[0], cut, s.seed)
            oos = m.D.index[np.where(t >= 0)[0]]
            P.windows[m.name] = (oos[0], oos[-1]) if len(oos) else None
        else:
            t, p = walk_forward_tiers(m, s, cut, m.D.index[-1] + pd.Timedelta(days=1), s.seed)
            P.windows[m.name] = (cut, m.D.index[-2])
        P.tiers[m.name], P.preds[m.name] = t, p
    return P


# ---------------------------------------------------------------------------------------------- scoring
def strategy_returns(P: Prepared, g, **exec_kw):
    """Equal-weight portfolio of the genome run on every market, over each market's scoring window."""
    series, trades = [], 0
    for m in P.markets:
        w = P.windows.get(m.name)
        if not w:
            continue
        pos = positions(m.D, g, P.tiers[m.name])
        mask = (m.D.index >= w[0]) & (m.D.index <= w[1])
        days, dpos = m.D.index[mask], pos[mask]
        trades += int(np.abs(np.diff(dpos, prepend=0)).astype(bool).sum())
        series.append(execute(m, dpos, days, **exec_kw).rename(m.name))
    R = pd.concat(series, axis=1).fillna(0.0)
    return R.mean(axis=1), R, trades


def score(P, g, s: Settings):
    port, R, trades = strategy_returns(P, g)
    years = len(port) / 365
    if trades / max(years, 1e-9) < s.min_trades_per_year * len(P.markets):
        return -9.0                                                 # too few trades to judge
    return sharpe(port)


# ---------------------------------------------------------------------------------------------- the safe loop
def search(P: Prepared, s: Settings, log=None):
    rng = np.random.default_rng(s.seed)
    tried = {}

    def ev(g):
        k = key(g)
        if k not in tried:
            tried[k] = (score(P, g, s), g)
            if log is not None:
                log.append(dict(round=rnd, score=tried[k][0], **{"genome": k}))
        return tried[k][0]

    rnd = 0
    for _ in range(s.pop_init):
        ev(random_genome(rng, s.both_sides))
    history = []
    for rnd in range(1, s.rounds + 1):
        elite = sorted(tried.values(), key=lambda x: -x[0])[:s.elite]
        for _ in range(s.children):
            ev(mutate(elite[rng.integers(len(elite))][1], rng, s.both_sides))
        for _ in range(s.fresh):
            ev(random_genome(rng, s.both_sides))
        history.append(max(v[0] for v in tried.values()))
    best_score, best = max(tried.values(), key=lambda x: x[0])
    scores = np.array([v[0] for v in tried.values() if v[0] > -9])
    return dict(best=best, best_score=best_score, history=history, n_trials=len(tried), trial_scores=scores)


# ---------------------------------------------------------------------------------------------- statistics
def deflated_sharpe(daily, n_trials, per_year=365):
    """Probability that the true Sharpe beats the best Sharpe expected from n_trials strategies with no skill
    (Bailey & Lopez de Prado). The spread of a no-skill Sharpe estimate is taken from the track length,
    sd ~ 1/sqrt(T), rather than from the trials' own spread: when some trials have real skill, their spread
    overstates the noise and makes the test too harsh."""
    r = np.asarray(daily, float); r = r[np.isfinite(r)]
    if len(r) < 10 or r.std() == 0:
        return float("nan")
    sr = r.mean() / r.std(ddof=1)
    z = (r - r.mean()) / r.std()
    skew, kurt = (z ** 3).mean(), (z ** 4).mean()
    sd = 1 / np.sqrt(len(r))
    g = 0.5772156649
    sr0 = 0.0 if n_trials < 2 else sd * ((1 - g) * ND.inv_cdf(1 - 1 / n_trials) + g * ND.inv_cdf(1 - 1 / (n_trials * np.e)))
    return ND.cdf((sr - sr0) * np.sqrt(len(r) - 1) / np.sqrt(max(1 - skew * sr + (kurt - 1) / 4 * sr ** 2, 1e-12)))


def model_quality(P: Prepared):
    """Out-of-sample: does the ER model beat 'tomorrow = recent average' and 'tomorrow = training mean'?"""
    out = {}
    for m in P.markets:
        y, p = m.F["er_next"].to_numpy(), P.preds[m.name]
        ok = np.isfinite(y) & np.isfinite(p)
        if ok.sum() < 30:
            continue
        base = m.F["er_20"].to_numpy()[ok]
        mse = lambda a: float(np.mean((y[ok] - a) ** 2))
        out[m.name] = dict(n=int(ok.sum()), corr=float(np.corrcoef(y[ok], p[ok])[0, 1]),
                           skill_vs_mean=1 - mse(p[ok]) / mse(np.full(ok.sum(), y[ok].mean())),
                           skill_vs_persistence=1 - mse(p[ok]) / mse(base))
    return out


def side_and_beta(P: Prepared, g):
    """Bidirectionality and market exposure of the finalist."""
    both, R, _ = strategy_returns(P, g)
    longs, _, _ = strategy_returns(P, {**g, "side": "long"}) if g["side"] == "both" else (None, None, None)
    shorts, _, _ = strategy_returns(P, {**g, "side": "short"}) if g["side"] == "both" else (None, None, None)
    mk = []
    for m in P.markets:
        w = P.windows[m.name]
        mk.append(m.D["ret"].shift(-1).loc[w[0]:w[1]].rename(m.name))          # next-day market return, aligned like execute
    M = pd.concat(mk, axis=1).fillna(0.0).mean(axis=1)
    M.index = M.index + pd.Timedelta(days=1)
    j = pd.concat([both.rename("s"), M.rename("m")], axis=1).dropna()
    beta = float(np.cov(j.s, j.m)[0, 1] / j.m.var()) if j.m.var() > 0 else float("nan")
    return dict(sharpe=sharpe(both), ann_vol=float(both.std() * np.sqrt(365)), beta=beta,
                corr_market=float(j.s.corr(j.m)), sharpe_long=sharpe(longs) if longs is not None else None,
                sharpe_short=sharpe(shorts) if shorts is not None else None)


def cost_monte_carlo(P: Prepared, g, n=200, seed=0):
    """Re-run the finalist with random costs (x0.5 to x3 the assumption) and random fill delays (0-3 hours)."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        mult = float(np.exp(rng.normal(0.25, 0.45)))
        delay = int(rng.integers(0, 4))
        port, _, _ = strategy_returns_scaled(P, g, mult, delay)
        out.append(sharpe(port))
    return np.array(out)


def strategy_returns_scaled(P, g, mult, delay):
    series = []
    for m in P.markets:
        w = P.windows[m.name]
        pos = positions(m.D, g, P.tiers[m.name])
        mask = (m.D.index >= w[0]) & (m.D.index <= w[1])
        series.append(execute(m, pos[mask], m.D.index[mask], cost_bps=m.cost_bps * mult, delay_h=delay).rename(m.name))
    R = pd.concat(series, axis=1).fillna(0.0)
    return R.mean(axis=1), R, 0


# ---------------------------------------------------------------------------------------------- permutations
def _pool(procs):
    # 'spawn', not 'fork': the tree model uses OpenMP threads, and forking a process that has started them can hang.
    os.environ["OMP_NUM_THREADS"] = "1"                       # one thread per worker process: no oversubscription
    return mp.get_context("spawn").Pool(procs)


def permuted_market(m: Market, start_day, seed, event_days=()):
    """Shuffle the hourly bars from start_day on (neurotrader bar permutation); earlier bars stay real."""
    from mcptlab.permute import permute_ohlc
    h = m.h
    start_i = max(0, h.index.searchsorted(start_day) - 1)
    p = permute_ohlc(h[["open", "high", "low", "close"]], start_index=start_i, seed=seed)
    if "volume" in h:
        p["volume"] = h["volume"].values
    return Market(m.name, p, m.cost_bps, m.delay_h, event_days)


def _search_null(args):
    markets, s, seed, events = args
    pm = [permuted_market(m, m.D.index[0], seed, events.get(m.name, ())) for m in markets]
    return search(prepare(pm, s, "dev"), s)["best_score"]


def search_permutation_test(markets, s, real_best, n=40, events=None, procs=None):
    """The whole search, every round, rerun on shuffled development data. Only development bars are used."""
    events = events or {}
    dev_only = []
    for m in markets:
        cut = split_day(m, s)
        dev_only.append(Market(m.name, m.h.loc[:cut - pd.Timedelta(hours=1)], m.cost_bps, m.delay_h, events.get(m.name, ())))
    with _pool(procs) as pool:
        null = np.array(pool.map(_search_null, [(dev_only, s, 1000 + k, events) for k in range(n)]))
    return (1 + np.sum(null >= real_best)) / (n + 1), null


def _holdout_null(args):
    markets, s, g, seed, events = args
    pm = [permuted_market(m, split_day(m, s), seed, events.get(m.name, ())) for m in markets]
    port, _, _ = strategy_returns(prepare(pm, s, "holdout"), g)
    return sharpe(port)


def holdout_permutation_test(markets, s, g, real, n=200, events=None, procs=None):
    events = events or {}
    with _pool(procs) as pool:
        null = np.array(pool.map(_holdout_null, [(markets, s, g, 5000 + k, events) for k in range(n)]))
    return (1 + np.sum(null >= real)) / (n + 1), null


def gate_ablation(P: Prepared, g):
    """Same strategy with every gate setting: does the efficiency-ratio gate actually help?"""
    return {gate: sharpe(strategy_returns(P, {**g, "gate": gate})[0]) for gate in GATES}
