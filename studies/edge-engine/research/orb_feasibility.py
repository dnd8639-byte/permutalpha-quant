"""Feasibility: is there a strategy whose profit actually depends on tomorrow's efficiency, and how good would
the gate have to be?   Uses BTC hourly DEVELOPMENT data only (first 70%, the part the BTC search already used).

Opening-range breakout (ORB): at hour k of the UTC day, take the direction of the move since the open,
hold it to the day's close. On an efficient day (straight line) the early direction is the final direction.
Cost: 5 bps per side.
"""
import sys, os
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..")); sys.path.insert(0, os.path.join(HERE, "..", "futures"))
from core import load_hourly
from pooled import panel, split_date, FEATURES, features, CLASSES
from sklearn.ensemble import HistGradientBoostingRegressor

h = load_hourly(os.path.join(HERE, "..", "..", "..", "mcpt-lab", "data", "btc_hourly_full.csv")) \
    if os.path.exists(os.path.join(HERE, "..", "..", "..", "mcpt-lab")) else load_hourly("/home/claude/mcpt-lab/data/btc_hourly_full.csv")
day = h.index.floor("D")
g = h.groupby(day)
D = pd.DataFrame({"open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(), "close": g["close"].last(),
                  "volume": g["volume"].sum(), "n": g["close"].count()})
D = D[D["n"] >= 20]
cut = D.index[int(len(D) * 0.7)]
D = D[D.index < cut]; h = h[h.index < cut]
D["ret"] = np.log(D["close"]).diff()
D["er"] = (D["close"] - D["open"]).abs() / (D["high"] - D["low"])
print(f"BTC dev days {len(D)}  ({D.index[0].date()} -> {D.index[-1].date()})")

COST = 5e-4
def orb(k):
    px = h["close"].groupby(h.index.floor("D")).apply(lambda s: s.iloc[k - 1] if len(s) >= k else np.nan)   # close of hour k
    d = np.sign(np.log(px / D["open"].reindex(px.index)))
    pnl = d * np.log(D["close"].reindex(px.index) / px) - 2 * COST * (d != 0)
    return pnl.reindex(D.index)

res = {k: orb(k) for k in (1, 2, 4)}
tier_real = pd.qcut(D["er"], 3, labels=["low", "mid", "high"])
print("\n1) If you KNEW tomorrow's efficiency (oracle): ORB average daily P&L in bps, by realized tier")
print(pd.DataFrame({f"ORB {k}h": (res[k].groupby(tier_real, observed=True).mean() * 1e4).round(1) for k in res}).to_string())
print("   all days:", {f"ORB {k}h": round(res[k].mean() * 1e4, 1) for k in res})

# 2) the real, forecastable version: pooled futures ER model (trained on futures dev data) applied to BTC days
ev = pd.read_csv(os.path.join(HERE, "..", "futures", "events_full.csv"), parse_dates=["date"])
P = panel(); fut_dev = P[P.index < split_date(P)].dropna(subset=FEATURES + ["er_next"])
D2 = D.copy(); D2["contract"] = "BTC"
FB = features(D2.assign(ret=D2["ret"]), "ES", ev)          # compute features; class dummies reset below
for c in CLASSES: FB[f"is_{c}"] = 0.0
out = []
for y in range(2019, cut.year + 1):
    t0 = pd.Timestamp(f"{y}-01-01")
    tr = fut_dev[fut_dev.index < t0 - pd.Timedelta(days=1)]
    m = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=200, min_samples_leaf=200,
                                      l2_regularization=1.0, random_state=0).fit(tr[FEATURES].to_numpy(float), tr["er_next"])
    te = FB[(FB.index >= t0) & (FB.index < pd.Timestamp(f"{y + 1}-01-01"))].dropna(subset=FEATURES)
    if len(te): out.append(pd.Series(m.predict(te[FEATURES].to_numpy(float)), index=te.index))
pred = pd.concat(out)                                   # prediction made at close of day d for day d+1
pred.index = pred.index + pd.Timedelta(days=1)          # align to the day being predicted
pred = pred.reindex(D.index).dropna()
tier_pred = pd.qcut(pred, 3, labels=["low", "mid", "high"])
print(f"\n2) With the gate we actually have (futures ER model, {len(pred)} BTC days): realized ER and ORB P&L by PREDICTED tier")
tab = pd.DataFrame({"realized ER": D["er"].reindex(pred.index).groupby(tier_pred, observed=True).mean().round(3)})
for k in res: tab[f"ORB {k}h bps"] = (res[k].reindex(pred.index).groupby(tier_pred, observed=True).mean() * 1e4).round(1)
print(tab.to_string())

# 3) how strong would a gate need to be?  blend the oracle with noise until the tier ER spread matches a target
print("\n3) Gate strength needed: ORB 2h Sharpe on gated days (high tier only) vs. the gate's ER spread")
rng = np.random.default_rng(0); z = (D["er"] - D["er"].mean()) / D["er"].std()
for rho in (0.05, 0.1, 0.2, 0.3, 0.5, 1.0):
    s = rho * z + np.sqrt(1 - rho ** 2) * pd.Series(rng.standard_normal(len(z)), index=z.index)
    t = pd.qcut(s, 3, labels=[0, 1, 2])
    spread = D["er"][t == 2].mean() - D["er"][t == 0].mean()
    x = res[2][t == 2].dropna()
    print(f"   forecast corr {rho:4.2f}: ER spread {spread:+.3f}   gated ORB Sharpe {x.mean() / x.std() * np.sqrt(365):+.2f}")
