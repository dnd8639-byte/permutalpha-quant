"""What would the daily efficiency look like if prices were a pure random walk?
Simulate a day as 1,380 one-minute steps (a 23-hour Globex session) with fat tails and compare the bar ER
distribution and its persistence to the real futures panel (dev period)."""
import sys, os
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pooled import panel, split_date

rng = np.random.default_rng(0)
n_days, steps = 40000, 1380
er = np.empty(n_days)
vol = np.exp(np.cumsum(rng.normal(0, 0.1, n_days)) * 0.3)        # clustered volatility, no direction memory
for i in range(n_days):
    x = np.cumsum(rng.standard_t(4, steps)) * vol[i]
    x = np.concatenate([[0.0], x])
    er[i] = abs(x[-1]) / (x.max() - x.min())
s = pd.Series(er)
P = panel(); dev = P[P.index < split_date(P)]
real = dev["er"].dropna()
q = [0.1, 0.25, 0.5, 0.75, 0.9]
print("bar ER         mean    " + "  ".join(f"q{int(x*100):02d}" for x in q) + "   lag-1 autocorr")
print(f"random walk   {s.mean():.3f}   " + "  ".join(f"{v:.2f}" for v in s.quantile(q)) + f"   {s.autocorr(1):+.3f}")
ac = dev.groupby("market")["er"].apply(lambda x: x.autocorr(1)).mean()
print(f"35 futures    {real.mean():.3f}   " + "  ".join(f"{v:.2f}" for v in real.quantile(q)) + f"   {ac:+.3f} (avg per market)")
print("\nper class mean ER:", dev.groupby(dev['market'].map(__import__('pooled').CLASS))['er'].mean().round(3).to_dict())
