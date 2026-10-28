# Learning strategy development with MCPT

This folder is your strategy lab. It uses neurotrader's four-step process, plus trading costs and an automatic trial ledger. Everything runs with numpy, pandas and matplotlib.

## The one idea behind all of it

Any optimizer will find *something* in any price history, even pure noise. The question is never "does my strategy look good?" It is **"does it look better than what the same process finds in noise?"** A permutation keeps everything about the prices *except their order*: same average move, same volatility, same fat tails. If your edge needs order (trends, reversals, patterns), it should disappear when the order is shuffled.

## The four steps (a funnel: stop at the first failure)

| step | question | pass looks like | if it fails |
|---|---|---|---|
| 1. In-sample excellence | Is the best tuned version any good on the training data? | PF clearly > 1 **after costs** | idea is dead, move on |
| 2. In-sample MCPT | Does the optimizer find more in real data than in shuffled data? | p < 0.01 (neurotrader's bar) | it was fitting noise, move on |
| 3. Walk-forward | Re-tune on a rolling window, trade the next block. Does it make money on data it never saw? | PF > 1 out of sample | edge doesn't generalize |
| 4. Walk-forward MCPT | Is that out-of-sample result better than walk-forwards on shuffled data? | p < 0.05 | out-of-sample result was luck |

**Do not look at steps 3–4 until step 2 passes.** Looking at out-of-sample results and then going back to change the idea turns your out-of-sample data into training data. That is the most common way people fool themselves.

## Worked example: Donchian breakout on ES (`example_donchian.py`)

```
STEP 1  best lookback 125, PF 1.110, Sharpe 0.54        <- looks fine...
STEP 2  p = 0.562  (noise's best PF: median 1.117)      <- ...but noise does just as well
STEP 3  out-of-sample PF 0.908, Sharpe -0.50            <- and it loses money on new data
STEP 4  p = 0.985
```

This is the lesson in one screen: **a respectable-looking backtest is exactly what an optimizer produces from noise.** Step 2 caught it before any money or time was spent. See `step2_ES.png`: the red line (real) sits in the middle of the blue pile (noise).

## How to run

```bash
cd ~/Desktop/ahl-node && source .venv/bin/activate && cd mcpt
python example_donchian.py ES        # try CL, GC, ZN, NQ ... (~10 minutes each)
python validate.py                   # re-check the test itself (~7 minutes)
```

## How to write your own strategy

In `strategies.py`, copy `DonchianBreakout` and change two things:

```python
class MyIdea(Strategy):
    name = "my_idea"
    def __init__(self):
        self.grid = [10, 20, 40, 80]          # parameter values to try (keep this SMALL)
    def positions(self, close, param):
        # return +1 (long), -1 (short), 0 (flat) for each bar,
        # using only prices up to that bar. Never use .shift(-1) here.
        ...
```

The framework handles next-bar execution, costs, optimization, permutations and logging.

## House rules (these are what make a p-value mean something)

1. **Costs always on.** Use realistic bps: equity index and Treasury futures about 2, energy and metals about 5, grains 8 or more. Fast strategies that only work at zero cost are not strategies.
2. **Small grids.** Every extra parameter value gives the optimizer more chances to fit noise. Step 2 accounts for this, but a smaller grid gives it more power to see a real edge.
3. **The ledger is your conscience.** `LEDGER.csv` records every step-2 and step-4 test automatically. If you have run 20 tests, one will show p < 0.05 by luck. Judge a result against how many tests you have run: roughly, want p < 0.05 ÷ (number of ideas tried), **or** confirm it on a market you never used while developing.
4. **Keep a holdout.** Develop on 2010–2018. Touch 2019+ only for the walk-forward, and only after step 2 passes.
5. **One change = one new test.** Changing a rule after seeing a result is a new idea. It goes in the ledger too.

## Where new ideas should come from

You're right that published strategies get competed away. Original ideas usually come from **understanding *why* a price should move**, then checking whether it does:

- **Market structure:** settlement times, contract rolls, margin changes, exchange holidays, the Sunday-evening reopen.
- **Cross-market links:** does one market lead another (e.g. copper → equities, oil → CAD)? You have 35 clean markets to explore.
- **Conditioning:** a trend rule that works only in high or low volatility, or only when two markets agree.
- **Your own data:** things other people don't collect. This is where small, durable edges usually live.

Write the reason down **before** testing. An idea with a reason that passes step 2 is worth far more than a pattern with no reason and a slightly smaller p-value.

## Known limitations

- `permute_close` shuffles daily returns. That also destroys **volatility clustering**, so a strategy that times volatility (not direction) can pass by exploiting that. This is real structure, but it's not a directional edge.
- Validation on noise: 4 of 40 runs came out below p = 0.05 (about 2 expected). That's within chance, but treat p between 0.01 and 0.05 as "interesting, needs confirmation," not "found."
- Daily data only, 2010–2026, CME markets. Intraday ideas need intraday data.

## First exercises

1. Run the example on **CL** and **ZN**. Do trend breakouts behave differently in oil and bonds? (Each run is logged.)
2. Run `MovingAverageCross` through steps 1–2 on ES. Before running, predict whether it will pass, and write down why.
3. Design one idea of your own from the list above. Write the *reason* in a comment at the top of the class. Run steps 1–2 only.
