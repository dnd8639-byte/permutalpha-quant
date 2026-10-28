# Does the v1 backtest use information from month t to size the month-t position?
import numpy as np, pandas as pd, sys
sys.path.insert(0,'../research')
from signals import *
CORR_WINDOW=24
def bt(prices, lag_vol):
    rets=prices.pct_change(); vol=realized_vol(rets)
    if lag_vol: vol=vol.shift(1)
    sig=momentum_signal(rets,vol); pos=raw_positions(sig,vol)
    g=pd.DataFrame(index=pos.index,columns=pos.columns,dtype=float)
    for i,t in enumerate(pos.index):
        if i<CORR_WINDOW: continue
        c=rets.iloc[i-CORR_WINDOW:i].corr().fillna(0.0)
        g.loc[t]=pos.loc[t]*portfolio_gearing(pos.loc[t],c,vol.loc[t])
    net=(g*rets).sum(axis=1)-(g-g.shift(1)).abs().sum(axis=1)*1e-3
    r=net.iloc[CORR_WINDOW+2:]
    return r.mean()*12/(r.std()*np.sqrt(12)), r.std()*np.sqrt(12)
rng=np.random.default_rng(0)
out={False:[],True:[]}
for k in range(60):
    T,N=240,10
    # GARCH-ish random walk: zero drift, NO predictability
    sig=np.empty((T,N)); s=np.full(N,0.04); r=np.empty((T,N))
    for t in range(T):
        r[t]=s*rng.standard_t(5,N)/np.sqrt(5/3); s=np.sqrt(0.0004+0.1*r[t]**2+0.85*s**2)
    px=pd.DataFrame(100*np.cumprod(1+r,axis=0),index=pd.date_range('2000-01-31',periods=T,freq='ME'))
    for lv in (False,True): out[lv].append(bt(px,lv))
for lv in (False,True):
    a=np.array(out[lv]); print(f"lag_vol={lv}: mean SR {a[:,0].mean():+.3f}  SR spread(std) {a[:,0].std():.3f}  realized vol {a[:,1].mean():.3f}  (target 0.10)")
