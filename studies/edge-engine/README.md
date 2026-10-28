# edge-engine

A strategy-development engine built around one idea: **predict how cleanly tomorrow's price will move, not which way**,
and only let a strategy trade on days it suits. Everything is tested so that it cannot fool itself.

* **Target:** the daily efficiency ratio ER = |net move| / total distance travelled (from hourly bars).
  Near 1 = price moved cleanly in one direction; near 0 = it went back and forth.
* **Model:** gradient-boosted trees predict tomorrow's ER from today's data (recent ER, volatility, range, volume,
  day of week, scheduled news tomorrow). Always compared with two simple baselines.
* **Gate:** predictions are split into low / medium / high thirds (cut-offs from training data only). A strategy can
  be switched on only for some thirds.
* **Strategies:** trend breakout, moving-average cross, momentum, reversal; long, short or both; any gate.
* **Costs:** fills happen hours after the decision (delay) at a cost per unit traded (spread + slippage), simulated
  hour by hour. A Monte Carlo re-runs the finalist with costs from 0.5x to 3x and delays of 0-3 hours.

## The safe loop
1. Each market is split: first **70% development**, last **30% holdout**.
2. Inside development, **walk-forward**: the ER model is retrained on all earlier days and predicts the next 90 days.
   Strategies are scored only on those predicted days.
3. The loop starts from random strategies. Each round it mutates the best ones and adds new random ones, and keeps
   the best few found so far. A fixed number of rounds; **every strategy tried is logged** in `runs/LEDGER.csv`.
4. **The whole search is permutation-tested:** the entire loop is re-run on shuffled development data many times.
   If the real search doesn't beat the search on shuffled markets, its "edge" came from searching.
5. The finalist is run **once** on the holdout, with a permutation test, long-only and short-only results, market
   beta, the cost Monte Carlo, and a gate ablation (does the gate actually help?). `run.py` refuses to open the
   same holdout twice.

## Use
```bash
pip install numpy pandas scikit-learn yfinance
python data/download.py            # hourly forex/futures (last ~730 days) + BTC since 2018 + news dates
python validate.py                 # test the engine on fake markets with known answers (~4 minutes)
python run.py dev BTC EURUSD GOLD   # development search + whole-search permutation test
python run.py holdout              # ONCE, when you've decided the finalist deserves it
```
Settings (rounds, population, folds, costs) are in `engine.py` (`Settings`) and `markets.json`.

## Validation (synthetic markets, `validate.py`)
| market | what's planted | what the engine found |
|---|---|---|
| Regime-switching | trend days and choppy days, persistent; trend rules win only on trend days | trend rule gated to mid+high ER; holdout Sharpe 8.2, p = 0.01; gate ablation: high-ER days +7.7, low-ER days -0.5 |
| Pure noise (GARCH-t) | nothing | whole-search p = 0.75; holdout Sharpe -1.05, p = 0.93 |
