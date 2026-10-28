# H5 canary: do monthly-AVERAGE prices create fake momentum? (Working 1960)
import numpy as np, pandas as pd, backtest_v2 as bt, stats
rng=np.random.default_rng(7); eom=[]; avg=[]
for k in range(60):
    d=pd.DataFrame(np.exp(np.cumsum(0.01*rng.standard_normal((21*240,10)),0)),
                   index=pd.bdate_range("2000-01-03",periods=21*240))
    for how,out in (("last",eom),("mean",avg)):
        px=getattr(d.resample("ME"),how)()
        out.append(stats.sharpe(bt.run_backtest(px,cost_bps=0.0)["returns"]))
for n,x in (("end-of-month prices",eom),("monthly-average prices",avg)):
    x=np.array(x); print(f"{n:24s} mean SR {x.mean():+.3f} (SE {x.std(ddof=1)/np.sqrt(len(x)):.3f})")
