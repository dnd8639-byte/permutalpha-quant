"""Does the daily-bar efficiency |C-O|/(H-L) track the hourly-path efficiency |net|/sum|hourly moves|?"""
import sys, glob, os
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from core import load_hourly, daily_table

def check(name, path):
    h = load_hourly(path)
    D = daily_table(h)
    bar = (D["net"].abs() / (D["hi"] - D["lo"])).where(D["hi"] > D["lo"]).clip(0, 1)
    d = pd.DataFrame({"path": D["er"], "bar": bar}).dropna()
    pear = d.corr().iloc[0, 1]; spear = d.corr("spearman").iloc[0, 1]
    # agreement on terciles
    tp = pd.qcut(d["path"], 3, labels=False); tb = pd.qcut(d["bar"], 3, labels=False)
    agree = (tp == tb).mean()
    ac = lambda s: s.autocorr(1)
    ac20 = lambda s: s.corr(s.rolling(20).mean().shift(1))
    return dict(market=name, days=len(d), pearson=round(pear, 3), spearman=round(spear, 3), tercile_agree=round(agree, 3),
                ac1_path=round(ac(d["path"]), 3), ac1_bar=round(ac(d["bar"]), 3),
                ma20_path=round(ac20(d["path"]), 3), ma20_bar=round(ac20(d["bar"]), 3))

if __name__ == "__main__":
    rows = [check(os.path.basename(p).split("_")[0] if "btc" not in p else "BTC", p) for p in sys.argv[1:]]
    print(pd.DataFrame(rows).to_string(index=False))
