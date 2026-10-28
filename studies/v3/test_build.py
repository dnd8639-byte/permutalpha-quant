"""H6: does build_clean remove roll gaps? Synthetic contracts with a KNOWN underlying and a
steep contango (each later contract priced higher). The naive Yahoo-style splice should
show the fake roll losses; build_clean should reproduce the true contract returns."""
import numpy as np
import pandas as pd
from build_clean import clean_returns, to_monthly

rng = np.random.default_rng(11)
days = pd.bdate_range("2010-01-04", "2019-12-31")
spot = 50 * np.exp(np.cumsum(0.015 * rng.standard_normal(len(days))))
rows = []
expiries = pd.date_range("2010-02-15", "2020-06-30", freq="MS") + pd.Timedelta(days=14)
for k, exp in enumerate(expiries):
    live = days[(days <= exp) & (days >= exp - pd.Timedelta(days=150))]
    for d in live:
        i = days.get_loc(d)
        ttm = (exp - d).days / 365
        px = spot[i] * np.exp(0.30 * ttm)                # 30%/yr contango
        near = (exp - d).days < 35                       # volume sits in the nearest contract
        rows.append((d, f"X{k:03d}", px, 1000 if near else 50))
df = pd.DataFrame(rows, columns=["date", "symbol", "close", "volume"])

clean, held, rolls = clean_returns(df)
# truth (analytic, independent of the builder): holding any contract earns the spot move
# minus the contango roll-down: (S_t/S_{t-1}) * exp(-0.30 * dt) - 1
close = df.pivot(index="date", columns="symbol", values="close")
S = pd.Series(spot, index=days).reindex(close.index)
dt = close.index.to_series().diff().dt.days / 365
truth = (S / S.shift(1)) * np.exp(-0.30 * dt) - 1
truth = truth.where(held.notna())
# naive splice: always quote the nearest-expiry contract (what Yahoo's =F does)
front = close.apply(lambda row: row.dropna().index.min(), axis=1)
splice = pd.Series([close.iloc[i][front.iloc[i]] for i in range(len(close))], index=close.index)
naive = splice.pct_change()

err = (clean - truth).abs().max()
bias_naive = (to_monthly(naive).mean() - to_monthly(truth).mean()) * 12
bias_clean = (to_monthly(clean).mean() - to_monthly(truth).mean()) * 12
print(f"rolls detected: {rolls}  (expected ~{len(expiries) - 5})")
print(f"max |clean - truth| daily: {err:.2e}")
print(f"annual return bias  naive splice: {bias_naive:+.2%}   clean: {bias_clean:+.2%}")
ok = err < 1e-12 and abs(bias_clean) < 1e-9 and abs(bias_naive) > 0.10 and rolls > 100
print("H6:", "PASS" if ok else "FAIL")
