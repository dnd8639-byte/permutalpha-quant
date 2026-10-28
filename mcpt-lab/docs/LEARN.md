# Learning strategy development with mcpt-lab

## The one idea behind everything

Any optimizer will find *something* in any price history, even pure noise. So the question is never "does my strategy look good?" It is **"does it look better than what the same process finds in noise?"**

A permutation keeps everything about the prices *except their order*:
- the same average move;
- the same volatility;
- the same fat tails.

If your edge needs order (trends, reversals, patterns), it should disappear when the order is shuffled.

## Your first hour

```bash
python examples/01_quickstart.py              # noise vs. planted trend
python examples/02_hawkes_four_steps.py --quick
python validation/validate.py                 # check the tester yourself
```

## Writing a strategy

```python
from mcptlab import Strategy

class MyIdea(Strategy):
    """WHY this should work: <write the mechanism BEFORE testing>"""
    name = "my_idea"
    bars_per_year = 252                    # 8760 for hourly crypto, 252 for daily stocks/futures

    def __init__(self):
        self.grid = [10, 20, 40, 80]       # values to try: keep this SMALL

    def positions(self, data, param):
        close = data["close"]
        # Return +1 / -1 / 0 (or fractions) for every bar.
        # Use only data up to each bar. Never use .shift(-1) or anything "future".
        ...
```

The framework handles the rest:
- next-bar execution;
- costs;
- tuning;
- permutations;
- the ledger;
- plots.

## House rules

1. **Costs always on.** Rough guide:
   - equity-index and Treasury futures: about 2 bps;
   - energy and metals: about 5 bps;
   - crypto (taker fees): 5–10 bps;
   - small caps: much more.
2. **Small grids.** Each extra value is another chance for noise to win.
3. **The ledger is your conscience.** `LEDGER.csv` records every step-2 and step-4 test. After 20 tests, one p < 0.05 is expected by luck.
4. **Keep a holdout.** Develop on the early data. Touch later data only after step 2 passes.
5. **One change = one new test.** Changing a rule after seeing a result is a new idea. It goes in the ledger too.

## Where good ideas come from

Durable edges usually come from understanding *why* a price should move:
- **Market structure:** rolls, settlement times, index rebalances, funding rates.
- **Who is forced to trade:** margin calls, fund flows, hedging.
- **Cross-market links:** does one market lead another?
- **Conditioning:** a rule that works only in certain volatility regimes.
- **Data other people don't collect.**

Write the reason down *before* the test. A reasoned idea that passes step 2 is worth far more than a pattern with a slightly smaller p-value.

## Exercises

1. Run `MovingAverageCross` through steps 1–2 on the bundled BTC data (2018–2019 only). Predict the result before you run it.
2. In `examples/05_your_own_strategy.py`, replace the RSI idea with one of your own. Write the mechanism in the docstring. Run steps 1–2 only.
3. Take a strategy that passed step 2 and run `fixed_param_mcpt` on a later period. Did it survive?
