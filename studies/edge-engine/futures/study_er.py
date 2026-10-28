"""Can tomorrow's efficiency be predicted?  Pooled walk-forward study on the DEVELOPMENT 70% only."""
import sys, os
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pooled import panel, split_date, FEATURES

P = panel()
cut = split_date(P)
dev = P[P.index < cut].dropna(subset=FEATURES + ["er_next"])
print(f"panel rows {len(P):,}  dev rows {len(dev):,}  split date {cut.date()}  (holdout not read below)")

def r2(y, p): return 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()

# walk-forward: expanding train, 1-year test blocks, a 1-day gap so the last training target never lies in the test block
years = pd.date_range("2013-01-01", cut, freq="YS")
rows, oos = [], []
for i, t0 in enumerate(years):
    t1 = years[i + 1] if i + 1 < len(years) else cut
    tr = dev[dev.index < t0 - pd.Timedelta(days=1)]
    te = dev[(dev.index >= t0) & (dev.index < t1)]
    if len(te) == 0: continue
    X, y = tr[FEATURES].to_numpy(float), tr["er_next"].to_numpy(float)
    gbm = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=300, min_samples_leaf=200,
                                        l2_regularization=1.0, random_state=0).fit(X, y)
    rid = make_pipeline(StandardScaler(), Ridge(alpha=10.0)).fit(X, y)
    mkt_mean = tr.groupby("market")["er_next"].mean()
    o = pd.DataFrame({"y": te["er_next"], "gbm": gbm.predict(te[FEATURES].to_numpy(float)),
                      "ridge": rid.predict(te[FEATURES].to_numpy(float)), "mean": y.mean(),
                      "mkt_mean": te["market"].map(mkt_mean).to_numpy(), "er20": te["er_20"], "market": te["market"],
                      "train_cut_lo": np.quantile(gbm.predict(X), 1 / 3), "train_cut_hi": np.quantile(gbm.predict(X), 2 / 3)},
                     index=te.index)
    oos.append(o)
    rows.append(dict(test_year=t0.year, n_train=len(tr), n_test=len(te), **{k: round(r2(o.y, o[k]), 4) for k in ["gbm", "ridge", "mkt_mean", "er20"]},
                     corr_gbm=round(np.corrcoef(o.y, o.gbm)[0, 1], 3)))
O = pd.concat(oos)
print("\nOut-of-sample R^2 vs the pooled training mean (0 = no better than the mean), per test year")
print(pd.DataFrame(rows).to_string(index=False))
print("\nALL dev OOS:", {k: round(r2(O.y, O[k]), 4) for k in ["gbm", "ridge", "mkt_mean", "er20"]},
      " corr gbm", round(np.corrcoef(O.y, O.gbm)[0, 1], 3), " n", len(O))

# tercile gates from TRAINING cuts, realized ER by predicted tier
O["tier"] = np.where(O.gbm < O.train_cut_lo, "low", np.where(O.gbm < O.train_cut_hi, "mid", "high"))
t = O.groupby("tier")["y"].agg(["count", "mean"]).reindex(["low", "mid", "high"])
print("\nRealized next-day ER by predicted tier (OOS):\n", t.round(4).to_string())

# what does the full-dev model rely on?  permutation importance on the last block
from sklearn.inspection import permutation_importance
tr = dev[dev.index < years[-1]]; te = dev[dev.index >= years[-1]]
gbm = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=300, min_samples_leaf=200,
                                    l2_regularization=1.0, random_state=0).fit(tr[FEATURES].to_numpy(float), tr["er_next"].to_numpy(float))
pi = permutation_importance(gbm, te[FEATURES].to_numpy(float), te["er_next"].to_numpy(float), n_repeats=5, random_state=0, scoring="r2")
imp = pd.Series(pi.importances_mean, index=FEATURES).sort_values(ascending=False)
print("\nPermutation importance (drop in OOS R^2), top 10:\n", imp.head(10).round(5).to_string())

# events: raw effect, dev data
print("\nMean next-day ER when a release is scheduled tomorrow vs not (dev data):")
for e in ["fomc_next", "cpi_next", "nfp_next", "eia_next"]:
    g = dev.groupby(dev[e])["er_next"].mean()
    print(f"  {e:10s} yes {g.get(1.0, np.nan):.4f}  no {g.get(0.0, np.nan):.4f}  n_yes {int(dev[e].sum())}")
O.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "study_oos.csv"))
