from data import load_prices
from backtest import run_backtest

px = load_prices().resample("ME").last()
print(f"loaded {px.shape[0]} monthly bars, {px.shape[1]} assets")
print(f"range: {px.index[0].date()} to {px.index[-1].date()}\n")

res = run_backtest(px, cost_bps=10.0)
for k, v in res["metrics"].items():
    print(f"{k:12s} {v: .4f}")
