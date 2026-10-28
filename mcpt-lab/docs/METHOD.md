# Method

## The problem: every backtest is an optimization

A backtest that tries 25 parameter settings and reports the best one has done 25 experiments and kept the winner. On pure noise, the best of 25 random strategies will usually look profitable. So "my tuned strategy made money in the past" says very little. The useful question is:

> **Did tuning find more in the real market than the same tuning finds in markets that contain no exploitable pattern?**

A permutation test answers that directly, with no assumptions about how returns are distributed.

## What a permutation keeps and what it destroys

`permute_ohlc` (from neurotrader888) takes every bar of real data and splits it into two parts:

- the **gap**: previous close → this open;
- the **intrabar move**: open → high, low and close.

The gaps and the intrabar moves are shuffled separately, and a new price path is rebuilt from them.

| kept | destroyed |
|---|---|
| the exact set of price moves (mean, volatility, fat tails, skew) | the order of the moves: trends, reversals, momentum, any pattern in time |
| the start and end price (so the overall drift is the same) | volatility clustering (calm and wild periods get mixed together) |
| valid bars (high ≥ open, close ≥ low) | |
| cross-market correlation, when several markets are permuted together | |

If a strategy's edge depends on the order of moves, it should vanish on permuted data. If the strategy does just as well on permuted data, the "edge" came from the tuning, not from the market.

## The four steps

1. **In-sample excellence.** Tune the strategy on the training period *with costs*. If the best version isn't clearly profitable, stop.

2. **In-sample permutation test.** Make N permuted copies of the training data (default 1,000) and run the **same tuning** on each one. Record the best result on each copy.
   - p = (1 + number of copies that did at least as well) / N.
   - Counting the real result as one of the draws means p can never be exactly 0.
   - Tuning is repeated on every copy. That automatically charges the strategy for the size of its parameter grid.

3. **Walk-forward.** Walk through time:
   - tune on a rolling window (for example, 3 years);
   - trade the next block (for example, 90 days) with the chosen parameters;
   - move forward and repeat.

   Only the out-of-sample blocks are scored.

4. **Walk-forward permutation test.** Repeat the whole walk-forward on copies where only the out-of-sample bars are permuted (the first training window stays real).

**Step order matters.** Out-of-sample results are looked at only after step 2 passes. Looking at them and then going back to adjust the idea quietly turns test data into training data.

## Extra nulls in this repo

### Sign-flip (`null="signflip"`)

This null randomly mirrors each bar around its open.

- **Kept exactly:** bar sizes, and so volatility and volatility clustering, in their real order.
- **Randomized:** direction.

Use it for strategies that react to volatility. A bar permutation also scrambles *when* volatility happens, so a volatility strategy could pass step 2 just by exploiting clustering. Passing the sign-flip test shows the strategy also gets direction right.

### Fixed-parameter test (`fixed_param_mcpt`)

For a published strategy:

- take the parameters its author would have chosen;
- freeze them;
- trade only data after the publication date;
- permute only that period.

Nothing is re-tuned, so nothing can be overfit. This is the most direct test of whether an edge survives being public.

### Label-shuffle null (custom)

For machine-learning filters, the training labels are scrambled before each model is fit.

- **Question:** does the real filter beat a filter that learned nothing?
- **Held fixed:** the trades and features are identical, so any difference comes from what the model learned.

## Costs

Costs are charged as `cost_bps` per unit of position change:

- going from flat to long costs 1 unit;
- flipping from long to short costs 2 units.

Costs are part of every test, including every permuted copy.

## Reading a p-value honestly

- **p < 0.01** on the first test of a well-motivated idea is strong evidence.
- **0.01 ≤ p < 0.05** is "interesting, needs confirmation". On pure noise, about 1 test in 20 lands here by chance. `validation/` shows exactly that rate.
- **Multiple testing.** If you have run k tests, the chance that at least one reaches p < 0.05 by luck is about 1 − 0.95^k. That is 64% for k = 20. The ledger exists so that k is never forgotten.
- **Confirm on something you never touched.** Use a different market, a later period, or data published after the idea.

## Limits

- **The permutation test tells you whether an edge is real, not whether it is large.** A real edge can still be too small to cover costs you haven't modeled, such as slippage, funding and taxes.
- **Bar permutation assumes the moves are exchangeable.** Shuffling bars also mixes regimes (a bull market's moves with a bear market's). That is the right null for "is there timing skill", but it is not the only reasonable one.
- **Some results are hard to test.** Results that depend on a single rare event, such as a crash, have little statistical power either way.
