"""edge-engine, daily futures edition: 35 CME futures pooled, a two-dimensional gate, and the same safe loop.

The gate
  * Efficiency model: predicts tomorrow's bar efficiency |C-O|/(H-L)  -> low / mid / high tier.
  * Volatility model: predicts tomorrow's range relative to its 20-day average -> low / mid / high tier.
  Both are ONE pooled gradient-boosted model across all markets, refit every January on all earlier days only
  (walk-forward), with tier cut-offs taken from the training predictions.
  A strategy's gate = (allowed efficiency tiers) x (allowed volatility tiers).

Execution (no look-ahead)
  * Prices are roll-adjusted: close = exp(cumsum of same-contract daily returns); open/high/low keep their
    distance from that day's close, so efficiency and range are unchanged and no roll jump ever enters a return.
  * The position for day t+1 is decided at the close of day t. It is filled a fraction `delay` of the way into
    day t+1 (default 1 hour of a 23-hour session), so the old position earns the start of the day and the new one
    the rest. Each change in position pays `cost_bps` per unit (half-spread + commission + slippage).
  * Each market is sized to 1% expected daily volatility (trailing 20-day, or the volatility model's forecast).
  * Portfolio = average of the markets trading that day. Sharpe uses 252 days a year.

Phases
  * dev:     every market is cut at the 70% date BEFORE any feature is computed. Strategies are scored on the
             walk-forward out-of-sample years 2013 -> cut.
  * holdout: run once, after the finalist is frozen.

Usage
  python daily_engine.py dev [N_PERM]       search + whole-search permutation test (default 39 permutations)
  python daily_engine.py holdout            ONE-SHOT: runs the frozen finalist on the last 30%
"""
from __future__ import annotations

import copy
import json
import multiprocessing as mp
import os
import sys
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
for _p in (os.path.join(HERE, "..", "..", "mcpt-lab"), os.path.join(HERE, "..", "mcpt-lab"), "/home/claude/mcpt-lab"):
    if os.path.isdir(os.path.join(_p, "mcptlab")):
        sys.path.insert(0, _p)
        break

from pooled import CLASS, FEATURES, features, load_daily, split_date, panel   # noqa: E402

RUNS = os.path.join(HERE, "runs")
PER_YEAR = 252
TARGET_VOL = 0.01                     # each market sized to ~1% daily volatility
DELAY = 1 / 23                        # fill 1 hour into a 23-hour session

COST_BPS = {"fx": 1.0, "rates": 0.75, "equity": 1.0, "energy": 2.0, "metal": 1.5, "ags": 2.5, "meat": 3.0}
COST_OVERRIDE = {"NG": 3.0, "PA": 4.0, "PL": 3.0, "ZO": 5.0, "NKD": 2.0, "BZ": 2.5}

RULES = {
    "breakout": dict(lookback=[10, 20, 50, 100, 200]),
    "ma_cross": dict(fast=[5, 10, 20, 50], slow=[50, 100, 200]),
    "momentum": dict(lookback=[5, 20, 60, 120, 250]),
    "reversal": dict(lookback=[1, 2, 3, 5]),
}
ER_GATES = {"all": (0, 1, 2), "mid+high": (1, 2), "high": (2,), "low": (0,), "low+mid": (0, 1)}
VOL_GATES = {"all": (0, 1, 2), "low+mid": (0, 1), "low": (0,), "mid+high": (1, 2), "high": (2,)}
SIZING = ("trailing", "model")


def cost_bps(root):
    return COST_OVERRIDE.get(root, COST_BPS[CLASS[root]])


# ---------------------------------------------------------------------------------------------- data
def adjusted(D):
    """Roll-adjusted OHLC from same-contract returns. Efficiency and range are unchanged."""
    r = D["ret"].fillna(0.0)
    lc = np.log(D["close"].iloc[0]) + r.cumsum()
    lo_ = np.log(D["open"] / D["close"]); lh = np.log(D["high"] / D["close"]); ll = np.log(D["low"] / D["close"])
    A = pd.DataFrame({"open": np.exp(lc + lo_), "high": np.exp(lc + lh), "low": np.exp(lc + ll), "close": np.exp(lc),
                      "volume": D["volume"], "contract": D["contract"]}, index=D.index)
    A["ret"] = np.log(A["close"]).diff()
    return A


def load_book(end=None, roots=None):
    book = {}
    for r in roots or sorted(CLASS):
        A = adjusted(load_daily(r))
        book[r] = A[A.index < end] if end is not None else A
    return book


def _onto(A, common):
    """Fold a market's own trading days onto a shared calendar: a day missing from `common` is merged into the
    next shared day (open first, high max, low min, close last), so no price move is lost."""
    pos = common.searchsorted(A.index)
    keep = pos < len(common)
    g = pd.Index(common[pos[keep]], name="date")
    X = A[keep]
    out = pd.DataFrame({"open": X["open"].groupby(g).first(), "high": X["high"].groupby(g).max(),
                        "low": X["low"].groupby(g).min(), "close": X["close"].groupby(g).last(),
                        "volume": X["volume"].groupby(g).sum(), "contract": X["contract"].groupby(g).last()})
    return out


def permute_book(book, seed):
    """Bar permutation of the whole development history, the SAME shuffle for every market, so the null keeps
    the real cross-market correlation (crisis days stay crisis days in every market at once).
    Markets are first folded onto the calendar they all share; a market that starts late (RTY) is shuffled alone."""
    from mcptlab.permute import permute_ohlc
    start = min(A.index[0] for A in book.values())
    early = [r for r, A in book.items() if A.index[0] <= start + pd.Timedelta(days=7)]
    common = None
    for r in early:
        common = book[r].index if common is None else common.intersection(book[r].index)
    out = {}
    folded = {r: _onto(book[r], common) for r in early}
    idx = folded[early[0]].index
    folded = {r: F.reindex(idx).ffill() for r, F in folded.items()}
    perm = permute_ohlc([F[["open", "high", "low", "close"]] for F in folded.values()], start_index=0, seed=seed)
    for (r, F), p in zip(folded.items(), perm):
        p["volume"], p["contract"] = F["volume"].values, F["contract"].values
        out[r] = p
    for i, r in enumerate(x for x in book if x not in early):
        F = _onto(book[r], common)
        p = permute_ohlc(F[["open", "high", "low", "close"]], start_index=0, seed=seed * 100 + i)
        p["volume"], p["contract"] = F["volume"].values, F["contract"].values
        out[r] = p
    for r, p in out.items():
        p["ret"] = np.log(p["close"]).diff()
    return {r: out[r] for r in book}


# ---------------------------------------------------------------------------------------------- the two models
def _gbm(seed=0):
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=200, min_samples_leaf=200,
                                         l2_regularization=1.0, random_state=seed)


def build_panel(book, events):
    parts = []
    for r, A in book.items():
        F = features(A, r, events)
        F["rr_next"] = F["range_ratio"].shift(-1)
        parts.append(F)
    P = pd.concat(parts).sort_index()
    P.index.name = "date"
    return P


def walk_forward(P, first_year=2013, seed=0):
    """Refit both pooled models every January on all earlier rows; predict that year. Returns tiers/preds."""
    out = pd.DataFrame(index=P.index, data={"er_tier": -1, "vol_tier": -1, "er_pred": np.nan, "rr_pred": np.nan})
    out["market"] = P["market"].values
    X_all = P[FEATURES].to_numpy(float)
    okX = ~np.isnan(X_all).any(axis=1)
    dates = P.index
    last = dates.max()
    quality = []
    for y in range(first_year, last.year + 1):
        t0, t1 = pd.Timestamp(f"{y}-01-01"), pd.Timestamp(f"{y + 1}-01-01")
        tr = (dates < t0 - pd.Timedelta(days=1)) & okX
        te = (dates >= t0) & (dates < t1) & okX
        if te.sum() == 0:
            continue
        for tgt, tcol, pcol in (("er_next", "er_tier", "er_pred"), ("rr_next", "vol_tier", "rr_pred")):
            m = tr & P[tgt].notna().to_numpy()
            model = _gbm(seed).fit(X_all[m], P[tgt].to_numpy(float)[m])
            lo, hi = np.quantile(model.predict(X_all[m]), [1 / 3, 2 / 3])
            p = model.predict(X_all[te])
            out.loc[te, pcol] = p
            out.loc[te, tcol] = np.where(p < lo, 0, np.where(p < hi, 1, 2))
        q = P.loc[te, ["er_next", "rr_next"]].assign(er_pred=out.loc[te, "er_pred"].values, rr_pred=out.loc[te, "rr_pred"].values,
                                                     er_tier=out.loc[te, "er_tier"].values, vol_tier=out.loc[te, "vol_tier"].values)
        quality.append(q)
    return out, (pd.concat(quality) if quality else None)


@dataclass
class Prepared:
    arrays: dict            # market -> dict of numpy arrays (aligned on that market's days)
    index: dict             # market -> DatetimeIndex
    quality: pd.DataFrame


def prepare(book, events, score_from=None, score_to=None, seed=0):
    P = build_panel(book, events)
    T, Q = walk_forward(P, seed=seed)
    arrays, index = {}, {}
    for r, A in book.items():
        t = T[T["market"] == r].reindex(A.index)
        Fm = P[P["market"] == r].reindex(A.index)
        c = A["close"].to_numpy(float)
        ret = A["ret"].fillna(0.0).to_numpy(float)
        intra = np.log(A["close"] / A["open"]).to_numpy(float)
        sd20 = A["ret"].rolling(20).std().to_numpy(float)
        # model sigma: scale the trailing sd by predicted range relative to the recent average range
        rr_mean = Fm["range_ratio"].rolling(20).mean().to_numpy(float)
        sd_model = sd20 * np.exp(t["rr_pred"].to_numpy(float) - rr_mean)
        win = t["er_tier"].to_numpy() >= 0
        if score_from is not None:
            win &= (A.index >= score_from)
        if score_to is not None:
            win &= (A.index < score_to)
        arrays[r] = dict(close=c, ret=ret, intra=intra, gap=ret - intra, sd20=sd20, sd_model=sd_model,
                         er_tier=t["er_tier"].fillna(-1).to_numpy(int), vol_tier=t["vol_tier"].fillna(-1).to_numpy(int),
                         win=win, cost=cost_bps(r) / 1e4)
        index[r] = A.index
    return Prepared(arrays, index, Q)


# ---------------------------------------------------------------------------------------------- strategies
def signal(c, rule, p):
    s = pd.Series(c)
    if rule == "breakout":
        L = p["lookback"]
        hi, lo = s.rolling(L).max().shift(1), s.rolling(L).min().shift(1)
        x = pd.Series(np.where(s > hi, 1.0, np.where(s < lo, -1.0, np.nan))).ffill().fillna(0.0)
    elif rule == "ma_cross":
        x = np.sign(s.rolling(p["fast"]).mean() - s.rolling(p["slow"]).mean()).fillna(0.0)
    elif rule == "momentum":
        x = np.sign(s - s.shift(p["lookback"])).fillna(0.0)
    elif rule == "reversal":
        x = -np.sign(s - s.shift(p["lookback"])).fillna(0.0)
    else:
        raise ValueError(rule)
    return x.to_numpy(float)


class Scorer:
    def __init__(self, prep: Prepared):
        self.p = prep
        self._sig = {}

    def sig(self, r, g):
        k = (r, g["rule"], json.dumps(g["params"], sort_keys=True))
        if k not in self._sig:
            self._sig[k] = signal(self.p.arrays[r]["close"], g["rule"], g["params"])
        return self._sig[k]

    def market_returns(self, r, g, cost_mult=1.0, delay=DELAY):
        a = self.p.arrays[r]
        s = self.sig(r, g)
        if g["side"] == "long":
            s = np.maximum(s, 0)
        elif g["side"] == "short":
            s = np.minimum(s, 0)
        allowed = np.isin(a["er_tier"], ER_GATES[g["er_gate"]]) & np.isin(a["vol_tier"], VOL_GATES[g["vol_gate"]])
        sd = a["sd_model"] if g["sizing"] == "model" else a["sd20"]
        size = np.clip(TARGET_VOL / sd, 0, 50)
        pos = np.where(allowed & a["win"] & np.isfinite(size), s * size, 0.0)        # decided at close of day t
        held = np.concatenate([[0.0], pos[:-1]])                                      # in force during day t
        prev = np.concatenate([[0.0], held[:-1]])
        pnl = prev * (a["gap"] + delay * a["intra"]) + held * (1 - delay) * a["intra"] - np.abs(held - prev) * a["cost"] * cost_mult
        live = np.concatenate([[False], a["win"][:-1]])                               # days whose position came from the window
        return pd.Series(np.where(live, pnl, np.nan), index=self.p.index[r]), int((np.diff(pos[a["win"]], prepend=0) != 0).sum())

    def portfolio(self, g, **kw):
        cols, trades = {}, 0
        for r in self.p.arrays:
            x, n = self.market_returns(r, g, **kw)
            cols[r] = x; trades += n
        R = pd.DataFrame(cols)
        R = R[R.notna().any(axis=1)]
        return R.mean(axis=1), R, trades


def sharpe(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return float(x.mean() / x.std() * np.sqrt(PER_YEAR)) if len(x) > 2 and x.std() > 0 else 0.0


# ---------------------------------------------------------------------------------------------- the safe loop
@dataclass
class Settings:
    rounds: int = 12
    pop_init: int = 24
    children: int = 18
    fresh: int = 6
    elite: int = 6
    min_trades_per_year: float = 4.0
    both_sides: bool = True
    seed: int = 0


def random_genome(rng, both=True):
    rule = str(rng.choice(list(RULES)))
    params = {k: int(rng.choice(v)) for k, v in RULES[rule].items()}
    if rule == "ma_cross" and params["fast"] >= params["slow"]:
        params["fast"] = 10
    return dict(rule=rule, params=params, er_gate=str(rng.choice(list(ER_GATES))), vol_gate=str(rng.choice(list(VOL_GATES))),
                sizing=str(rng.choice(SIZING)), side="both" if both else str(rng.choice(["both", "long", "short"])))


def mutate(g, rng, both=True):
    g = copy.deepcopy(g)
    what = rng.choice(["param", "er_gate", "vol_gate", "sizing", "rule"], p=[0.4, 0.2, 0.2, 0.1, 0.1])
    if what == "param":
        k = str(rng.choice(list(g["params"])))
        g["params"][k] = int(rng.choice(RULES[g["rule"]][k]))
        if g["rule"] == "ma_cross" and g["params"]["fast"] >= g["params"]["slow"]:
            return mutate(g, rng, both)
    elif what == "er_gate":
        g["er_gate"] = str(rng.choice(list(ER_GATES)))
    elif what == "vol_gate":
        g["vol_gate"] = str(rng.choice(list(VOL_GATES)))
    elif what == "sizing":
        g["sizing"] = "model" if g["sizing"] == "trailing" else "trailing"
    else:
        n = random_genome(rng, both)
        g = {**n, "er_gate": g["er_gate"], "vol_gate": g["vol_gate"], "sizing": g["sizing"]}
    return g


def key(g):
    return json.dumps(g, sort_keys=True)


def search(prep, s: Settings, log=None):
    sc = Scorer(prep)
    rng = np.random.default_rng(s.seed)
    tried = {}
    n_mk = len(prep.arrays)

    def ev(g, rnd):
        k = key(g)
        if k not in tried:
            port, _, trades = sc.portfolio(g)
            years = len(port) / PER_YEAR
            val = sharpe(port) if trades / max(years, 1e-9) >= s.min_trades_per_year * n_mk else -9.0
            tried[k] = (val, g)
            if log is not None:
                log.append(dict(round=rnd, score=round(val, 4), genome=k))
        return tried[k][0]

    for _ in range(s.pop_init):
        ev(random_genome(rng, s.both_sides), 0)
    history = []
    for rnd in range(1, s.rounds + 1):
        elite = sorted(tried.values(), key=lambda x: -x[0])[:s.elite]
        for _ in range(s.children):
            ev(mutate(elite[rng.integers(len(elite))][1], rng, s.both_sides), rnd)
        for _ in range(s.fresh):
            ev(random_genome(rng, s.both_sides), rnd)
        history.append(max(v[0] for v in tried.values()))
    best_score, best = max(tried.values(), key=lambda x: x[0])
    return dict(best=best, best_score=best_score, history=history, n_trials=len(tried),
                trial_scores=np.array([v[0] for v in tried.values() if v[0] > -9]), scorer=sc)


# ---------------------------------------------------------------------------------------------- diagnostics
def deflated_sharpe(daily, n_trials):
    from engine import deflated_sharpe as ds
    return ds(daily, n_trials)


def model_report(Q):
    def r2(y, p): return 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    q = Q.dropna(subset=["er_next", "rr_next", "er_pred", "rr_pred"])
    er_t = q.groupby("er_tier")["er_next"].mean()
    vol_t = np.exp(q.groupby("vol_tier")["rr_next"].mean())
    return dict(n=len(q), er_r2=r2(q.er_next, q.er_pred), vol_r2=r2(q.rr_next, q.rr_pred),
                er_by_tier=[round(er_t.get(i, np.nan), 4) for i in range(3)],
                range_by_vol_tier=[round(vol_t.get(i, np.nan), 3) for i in range(3)])


def gate_ablation(sc: Scorer, g):
    grid = {}
    for eg in ER_GATES:
        for vg in VOL_GATES:
            grid[(eg, vg)] = sharpe(sc.portfolio({**g, "er_gate": eg, "vol_gate": vg})[0])
    T = pd.Series(grid).unstack()
    return T.reindex(index=list(ER_GATES), columns=list(VOL_GATES))


def cost_monte_carlo(sc: Scorer, g, n=200, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        mult = float(np.exp(rng.normal(0.25, 0.45)))
        delay = float(rng.uniform(0, 3 / 23))
        out.append(sharpe(sc.portfolio(g, cost_mult=mult, delay=delay)[0]))
    return np.array(out)


def side_and_beta(sc: Scorer, g):
    port, R, _ = sc.portfolio(g)
    lon = sc.portfolio({**g, "side": "long"})[0]
    sho = sc.portfolio({**g, "side": "short"})[0]
    # market factor: every market held long at the same 1%-vol size, over the same days
    cols = {}
    for r, a in sc.p.arrays.items():
        size = np.clip(TARGET_VOL / a["sd20"], 0, 50)
        held = np.concatenate([[0.0], np.where(a["win"] & np.isfinite(size), size, 0.0)[:-1]])
        live = np.concatenate([[False], a["win"][:-1]])
        cols[r] = pd.Series(np.where(live, held * a["ret"], np.nan), index=sc.p.index[r])
    M = pd.DataFrame(cols).mean(axis=1)
    es = cols.get("ES")
    j = pd.concat([port.rename("s"), M.rename("m"), (es.rename("es") if es is not None else None)], axis=1).dropna()
    beta = float(np.cov(j.s, j.m)[0, 1] / j.m.var())
    beta_es = float(np.cov(j.s, j.es)[0, 1] / j.es.var()) if "es" in j else float("nan")
    return dict(sharpe=sharpe(port), ann_vol=float(port.std() * np.sqrt(PER_YEAR)), beta_all_markets=beta,
                corr_all_markets=float(j.s.corr(j.m)), beta_sp500=beta_es, sharpe_long=sharpe(lon), sharpe_short=sharpe(sho),
                sharpe_market_long_only=sharpe(M))


# ---------------------------------------------------------------------------------------------- permutation test
def _pool(n):
    os.environ["OMP_NUM_THREADS"] = "1"
    return mp.get_context("spawn").Pool(n)


def _null(args):
    book, events, s, seed = args
    pb = permute_book(book, seed)
    return search(prepare(pb, events, seed=0), s)["best_score"]


def search_permutation_test(book, events, s, real_best, n=39, procs=None):
    procs = procs or max(1, (os.cpu_count() or 2))
    with _pool(procs) as pool:
        null = np.array(pool.map(_null, [(book, events, s, 1000 + k) for k in range(n)]))
    return (1 + np.sum(null >= real_best)) / (n + 1), null


# ---------------------------------------------------------------------------------------------- CLI
def _events():
    return pd.read_csv(os.path.join(HERE, "events_full.csv"), parse_dates=["date"])


def run_dev(n_perm=39):
    os.makedirs(RUNS, exist_ok=True)
    fin = os.path.join(RUNS, "finalist_futures.json")
    if os.path.exists(fin) and json.load(open(fin)).get("holdout_opened"):
        sys.exit("The holdout has already been opened. A new development run would be a new, separately counted test.")
    t_start = time.time()
    cut = split_date(panel())
    events = _events()
    book = load_book(end=cut)                                             # nothing after the cut is ever loaded
    s = Settings()
    prep = prepare(book, events)
    mq = model_report(prep.quality)
    print(f"split {cut.date()} | markets {len(book)} | dev rows {sum(len(a) for a in book.values()):,}")
    print(f"efficiency model OOS R2 {mq['er_r2']:+.4f}, ER by tier {mq['er_by_tier']}")
    print(f"volatility model OOS R2 {mq['vol_r2']:+.3f}, range by tier {mq['range_by_vol_tier']}")
    log = []
    res = search(prep, s, log)
    g, sc = res["best"], res["scorer"]
    port = sc.portfolio(g)[0]
    print(f"\nsearch: {res['n_trials']} strategies tried; best dev Sharpe {res['best_score']:.3f}")
    print("best-so-far by round:", " ".join(f"{h:.2f}" for h in res["history"]))
    print("finalist:", key(g))
    dsr = deflated_sharpe(port.to_numpy(), res["n_trials"])
    sb = side_and_beta(sc, g)
    ab = gate_ablation(sc, g)
    mc = cost_monte_carlo(sc, g)
    print(f"deflated Sharpe (prob. real after {res['n_trials']} trials): {dsr:.3f}")
    print(f"long-only Sharpe {sb['sharpe_long']:.2f} | short-only {sb['sharpe_short']:.2f} | beta to all markets "
          f"{sb['beta_all_markets']:+.2f} | beta to S&P {sb['beta_sp500']:+.2f} | corr {sb['corr_all_markets']:+.2f}")
    print(f"cost/delay Monte Carlo: median Sharpe {np.median(mc):.2f}, 5th pct {np.percentile(mc, 5):.2f}, share > 0 {np.mean(mc > 0):.0%}")
    print("\ngate ablation (rows = efficiency gate, columns = volatility gate), dev Sharpe:")
    print(ab.round(2).to_string())
    print(f"\nwhole-search permutation test, {n_perm} permutations (this re-fits both models and reruns every round)...")
    t0 = time.time()
    p, null = search_permutation_test(book, events, s, res["best_score"], n=n_perm)
    print(f"  p = {p:.3f}   null best Sharpes: median {np.median(null):.2f}, 95th pct {np.percentile(null, 95):.2f}  "
          f"({(time.time() - t0) / 60:.1f} min)")
    out = dict(finalist=g, dev_sharpe=res["best_score"], n_trials=res["n_trials"], dsr=dsr, search_p=p,
               null=null.tolist(), side_beta=sb, cost_mc=dict(median=float(np.median(mc)), p5=float(np.percentile(mc, 5)),
                                                               share_pos=float(np.mean(mc > 0))),
               gate_ablation={f"{i}|{j}": float(ab.loc[i, j]) for i in ab.index for j in ab.columns},
               model=mq, split=str(cut.date()), holdout_opened=False,
               created=time.strftime("%Y-%m-%d %H:%M"))
    json.dump(out, open(fin, "w"), indent=1, default=float)
    pd.DataFrame(log).to_csv(os.path.join(RUNS, "LEDGER_futures_trials.csv"), index=False)
    led = os.path.join(RUNS, "LEDGER_futures.csv")
    row = pd.DataFrame([dict(date=out["created"], test="futures 2-D gate search", finalist=key(g), dev_sharpe=round(res["best_score"], 3),
                             trials=res["n_trials"], search_p=round(p, 3), dsr=round(dsr, 3), holdout="sealed")])
    row.to_csv(led, mode="a", header=not os.path.exists(led), index=False)
    print(f"\nsaved {fin}  ({(time.time() - t_start) / 60:.1f} min total)")


def run_holdout():
    fin = os.path.join(RUNS, "finalist_futures.json")
    F = json.load(open(fin))
    if F.get("holdout_opened"):
        sys.exit(f"Holdout already opened on {F['holdout_opened']}. It is a one-shot test.")
    F["holdout_opened"] = time.strftime("%Y-%m-%d %H:%M")
    json.dump(F, open(fin, "w"), indent=1, default=float)                # lock FIRST, then look
    cut = pd.Timestamp(F["split"])
    book = load_book()
    prep = prepare(book, _events(), score_from=cut)
    sc = Scorer(prep)
    g = F["finalist"]
    port = sc.portfolio(g)[0]
    sb = side_and_beta(sc, g)
    print(f"HOLDOUT {cut.date()} -> {port.index[-1].date()}: Sharpe {sharpe(port):.3f}  (dev was {F['dev_sharpe']:.3f})")
    print(json.dumps(sb, indent=1, default=float))
    F["holdout_sharpe"] = sharpe(port); F["holdout_side_beta"] = sb
    json.dump(F, open(fin, "w"), indent=1, default=float)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "dev"
    if cmd == "dev":
        run_dev(int(sys.argv[2]) if len(sys.argv) > 2 else int(os.environ.get("N_PERM", 39)))
    elif cmd == "holdout":
        run_holdout()
    else:
        sys.exit(__doc__)
