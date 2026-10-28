"""Move the gate from 'last night' to 'after the first k hours of today'. At hour k we already know how today
has started: how straight the first k hours were, how big the range is so far vs normal, how heavy the volume is.
Does THAT predict how efficiently the rest of the day runs?  BTC hourly, development 70% only. Walk-forward by year."""
import os, sys
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from core import load_hourly

h = load_hourly("/home/claude/mcpt-lab/data/btc_hourly_full.csv")
h["day"] = h.index.floor("D"); h["hr"] = h.index.hour
full = h.groupby("day").filter(lambda x: len(x) == 24)
days = full["day"].unique()
cut = days[int(len(days) * 0.7)]
full = full[full["day"] < cut]
COST = 5e-4

def table(k):
    rows = []
    for d, x in full.groupby("day"):
        o = x["open"].iloc[0]; first = x.iloc[:k]; rest = x.iloc[k:]
        pk = first["close"].iloc[-1]
        rng_k = first["high"].max() - first["low"].min()
        rows.append(dict(day=d, er_k=abs(pk - o) / rng_k if rng_k > 0 else 0, move_k=np.log(pk / o), rng_k=np.log(first["high"].max() / first["low"].min()),
                         vol_k=first["volume"].sum(), rest_er=abs(rest["close"].iloc[-1] - pk) / (rest["high"].max() - rest["low"].min()),
                         orb=np.sign(np.log(pk / o)) * np.log(rest["close"].iloc[-1] / pk) - 2 * COST,
                         day_rng=np.log(x["high"].max() / x["low"].min()), day_vol=x["volume"].sum()))
    T = pd.DataFrame(rows).set_index("day")
    T["rng_k_rel"] = T["rng_k"] / T["day_rng"].shift(1).rolling(20).mean()      # range so far vs a normal whole day
    T["vol_k_rel"] = np.log(T["vol_k"] / (T["day_vol"].shift(1).rolling(20).mean() * k / 24))
    T["move_k_rel"] = T["move_k"].abs() / T["day_rng"].shift(1).rolling(20).mean()
    T["er_prev"] = (T["er_k"] * 0).shift(1)                                       # placeholder removed below
    T["day_er_prev"] = np.nan
    return T.drop(columns=["er_prev", "day_er_prev"]).dropna()

FEAT = ["er_k", "rng_k_rel", "vol_k_rel", "move_k_rel"]
for k in (2, 4):
    T = table(k)
    out = []
    for y in range(2019, cut.year + 1):
        t0 = pd.Timestamp(f"{y}-01-01")
        tr, te = T[T.index < t0], T[(T.index >= t0) & (T.index < pd.Timestamp(f"{y + 1}-01-01"))]
        if len(te) == 0 or len(tr) < 200: continue
        m = HistGradientBoostingRegressor(max_depth=2, learning_rate=0.05, max_iter=150, min_samples_leaf=50, random_state=0)
        m.fit(tr[FEAT], tr["rest_er"])
        lo, hi = np.quantile(m.predict(tr[FEAT]), [1 / 3, 2 / 3])
        p = m.predict(te[FEAT])
        out.append(te.assign(pred=p, tier=np.where(p < lo, "low", np.where(p < hi, "mid", "high"))))
    O = pd.concat(out)
    c = np.corrcoef(O["pred"], O["rest_er"])[0, 1]
    g = O.groupby("tier").agg(n=("orb", "size"), rest_er=("rest_er", "mean"), orb_bps=("orb", lambda s: s.mean() * 1e4)).reindex(["low", "mid", "high"])
    hi_ = O.loc[O.tier == "high", "orb"]
    print(f"\nk = {k}h  ({len(O)} out-of-sample days)  forecast corr with rest-of-day ER: {c:+.3f}")
    print(g.round(3).to_string())
    print(f"   ORB on all days {O['orb'].mean()*1e4:+.1f} bps | gated to predicted-high days: Sharpe {hi_.mean()/hi_.std()*np.sqrt(365):+.2f}")
