"""Does efficiency become predictable over LONGER windows?  Kaufman ER over the next H days:
|C_{t+H} - C_t| / sum |daily returns|, from the same-contract (roll-free) returns.  Pooled walk-forward on dev only.
Training stops H+1 days before each test block so no training target overlaps the test period."""
import sys, os
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pooled import CLASS, CLASSES, load_daily, split_date, panel

def kaufman(r, H):          # ER over the window (t-H, t]
    return (r.rolling(H).sum().abs() / r.abs().rolling(H).sum())

rows = []
for root in sorted(CLASS):
    D = load_daily(root); r = D["ret"].fillna(0.0)
    F = pd.DataFrame(index=D.index)
    for H in (5, 20, 60):
        F[f"ker_{H}"] = kaufman(r, H)
    F["vol_ratio"] = np.log(r.rolling(20).std() / r.rolling(120).std())
    F["trend_60"] = r.rolling(60).sum() / (r.rolling(60).std() * np.sqrt(60))
    for c in CLASSES: F[f"is_{c}"] = float(CLASS[root] == c)
    for H in (5, 20, 60):
        F[f"y_{H}"] = kaufman(r, H).shift(-H)
    F["market"] = root
    rows.append(F)
P = pd.concat(rows).sort_index()
cut = split_date(panel())
FEAT = ["ker_5", "ker_20", "ker_60", "vol_ratio", "trend_60"] + [f"is_{c}" for c in CLASSES]

def r2(y, p): return 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()
print(f"split {cut.date()}; dev only\n")
print(" H   OOS R2 gbm   OOS R2 market-mean   corr   low/mid/high realized ER")
for H in (5, 20, 60):
    dev = P[(P.index < cut - pd.Timedelta(days=int(H * 1.5)))].dropna(subset=FEAT + [f"y_{H}"])
    out = []
    years = pd.date_range("2013-01-01", cut, freq="YS")
    for i, t0 in enumerate(years):
        t1 = years[i + 1] if i + 1 < len(years) else cut
        tr = dev[dev.index < t0 - pd.Timedelta(days=int(H * 1.5) + 1)]
        te = dev[(dev.index >= t0) & (dev.index < t1)]
        if len(te) == 0: continue
        m = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=300, min_samples_leaf=300,
                                          random_state=0).fit(tr[FEAT].to_numpy(float), tr[f"y_{H}"].to_numpy(float))
        fit = m.predict(tr[FEAT].to_numpy(float)); lo, hi = np.quantile(fit, [1/3, 2/3])
        p = m.predict(te[FEAT].to_numpy(float))
        out.append(pd.DataFrame({"y": te[f"y_{H}"], "p": p, "mm": te["market"].map(tr.groupby("market")[f"y_{H}"].mean()).to_numpy(),
                                 "tier": np.where(p < lo, 0, np.where(p < hi, 1, 2))}, index=te.index))
    O = pd.concat(out)
    t = O.groupby("tier")["y"].mean()
    print(f"{H:2d}   {r2(O.y, O.p):+.4f}      {r2(O.y, O.mm):+.4f}           {np.corrcoef(O.y, O.p)[0,1]:.3f}   "
          f"{t.get(0):.3f} / {t.get(1):.3f} / {t.get(2):.3f}")
