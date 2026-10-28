"""Building the gate up.  Same walk-forward as study_er.py (dev 70% only, yearly test blocks 2013-2021).
Every variant tried is printed and counted, so the number of attempts is on the record.

Gate metric = how far apart the realized next-day ER is between the predicted HIGH and LOW tiers
(tiers cut from training predictions only), with a t-stat from the per-day spread series."""
import sys, os
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pooled import panel_v2, split_date, FEATURES, CROSS, CALENDAR

P = panel_v2()
cut = split_date(P)
SETS = {"A base (last run)": FEATURES,
        "B base + calendar": FEATURES + CALENDAR,
        "C base + cross-market": FEATURES + CROSS,
        "D everything": FEATURES + CALENDAR + CROSS}
ALL = FEATURES + CALENDAR + CROSS
dev = P[P.index < cut].dropna(subset=ALL + ["er_next", "range_next"])
print(f"dev rows {len(dev):,}, split {cut.date()}, holdout not read\n")

def gbm():
    return HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=300, min_samples_leaf=200,
                                         l2_regularization=1.0, random_state=0)

def walk(cols, target):
    out = []
    for y0 in range(2013, cut.year + 1):
        t0, t1 = pd.Timestamp(f"{y0}-01-01"), min(pd.Timestamp(f"{y0 + 1}-01-01"), cut)
        tr = dev[dev.index < t0 - pd.Timedelta(days=1)]; te = dev[(dev.index >= t0) & (dev.index < t1)]
        if len(te) == 0: continue
        m = gbm().fit(tr[cols].to_numpy(float), tr[target].to_numpy(float))
        lo, hi = np.quantile(m.predict(tr[cols].to_numpy(float)), [1 / 3, 2 / 3])
        p = m.predict(te[cols].to_numpy(float))
        out.append(pd.DataFrame({"y": te[target], "p": p, "tier": np.where(p < lo, 0, np.where(p < hi, 1, 2)),
                                 "market": te["market"], "er_next": te["er_next"]}, index=te.index))
    return pd.concat(out)

def spread_t(O):
    d = O.groupby([O.index, "tier"])["er_next"].mean().unstack()
    s = (d[2] - d[0]).dropna()
    return s.mean(), s.mean() / s.std() * np.sqrt(len(s))

def r2(y, p): return 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()

print("ER gate variants                  R2      low    mid    high   high-low   t")
res = {}
for name, cols in SETS.items():
    O = walk(cols, "er_next"); res[name] = O
    m = O.groupby("tier")["y"].mean(); sp, t = spread_t(O)
    print(f"{name:30s} {r2(O.y, O.p):+.4f}   {m[0]:.3f}  {m[1]:.3f}  {m[2]:.3f}   {sp:+.3f}   {t:5.1f}")

# volatility axis: predict tomorrow's range relative to its 20-day average
V = walk(ALL, "range_next")
print(f"\nvolatility model (all features)  R2 {r2(V.y, V.p):.3f}   corr {np.corrcoef(V.y, V.p)[0, 1]:.3f}")

# the 2-D gate: predicted efficiency tier x predicted volatility tier
best = res["D everything"]
G = pd.DataFrame({"er_tier": best["tier"], "vol_tier": V["tier"].reindex(best.index).values,
                  "er_next": best["er_next"], "range_next": V["y"].reindex(best.index).values}, index=best.index)
G = G.dropna()
names = {0: "low", 1: "mid", 2: "high"}
print("\n2-D gate: realized next-day ER  (rows = predicted volatility tier, columns = predicted ER tier)")
tab = G.pivot_table(index="vol_tier", columns="er_tier", values="er_next", aggfunc="mean").rename(index=names, columns=names)
print(tab.round(3).to_string())
print("\nrealized next-day range vs its 20-day average (exp of log ratio)")
tab2 = G.pivot_table(index="vol_tier", columns="er_tier", values="range_next", aggfunc="mean").rename(index=names, columns=names)
print(np.exp(tab2).round(2).to_string())
print("\ncell counts"); print(G.pivot_table(index="vol_tier", columns="er_tier", values="er_next", aggfunc="count").rename(index=names, columns=names).to_string())

# what drives the best ER model
from sklearn.inspection import permutation_importance
tr = dev[dev.index < pd.Timestamp("2020-12-31")]; te = dev[dev.index >= pd.Timestamp("2021-01-01")]
m = gbm().fit(tr[ALL].to_numpy(float), tr["er_next"].to_numpy(float))
pi = permutation_importance(m, te[ALL].to_numpy(float), te["er_next"].to_numpy(float), n_repeats=5, random_state=0, scoring="r2")
print("\nTop drivers of the everything-model (drop in 2021 R2 when shuffled):")
print(pd.Series(pi.importances_mean, index=ALL).sort_values(ascending=False).head(8).round(5).to_string())
best.to_csv("gate_oos_er.csv"); V.to_csv("gate_oos_vol.csv")
