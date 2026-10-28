# Changelog

Changes that can alter what a test measures are marked **(affects results)**. A frozen card records the lab
version it was frozen with, and its gate report shows both versions if they differ.

## 0.9.1
- Published on PyPI as **permutalpha** (`pip install permutalpha`; the import name and the `oql` command are
  unchanged, and `permutalpha` works as a command too). Extras: `[trees]`, `[paper]`, `[all]`.
- `oql demo [DIR]`: a ready lab on synthetic markets with two frozen cards, so a fresh install shows a PASS and a
  FAIL in minutes.

## 0.9.0: bulletproofing
- **(affects results)** No position without a trustworthy volatility estimate. A flat or stale stretch (half of
  the last 20 moves exactly zero: a pegged currency, a frozen feed) had a tiny measured volatility, and dividing by it
  put the market at the 10x leverage cap. Such bars now get no position; normal markets are unaffected.
- **Frozen data means the data at the freeze.** The freeze records each dataset's last timestamp and a fingerprint
  of its rows up to then. Gate and holdout read only those rows, so the daily collector appending new data no longer
  blocks a frozen card, while any change to an existing row (a vendor revision, an edit) is still refused. Cards
  frozen before 0.9 keep the old whole-file check.
- **Tests that cannot work are refused at the freeze:** n_perm too small to reach p < alpha, alpha above 0.2,
  negative costs, a development share outside 30-95%, a fill delay outside [0, 1), unknown rules or gates, a
  dataset listed twice, an experiment.py that does not load.
- **Tamper-evident ledger.** Every row carries a hash chained to the row before; `oql ledger verify` reports any
  edited, removed or reordered row, and `oql ledger head` prints the latest hash to publish (e.g. in a commit
  message). An existing ledger is sealed as it stands the first time a row is added.
- **Safe with several processes.** Board changes happen under a lab lock; a gate or holdout records its result on a
  fresh copy of the board, so cards created or edited while a long gate ran are no longer overwritten; two gate
  runs of the same card cannot overlap; ledger appends cannot fork the chain.
- New test suite `tests/test_robust.py`: no look-ahead across every rule, side and gate setting; engine invariants on
  messy markets (gaps, frozen feeds, fat tails); costs only ever reduce returns; the gate is deterministic across
  process counts; frozen-data and validation behaviour; ledger tampering; concurrency.
- Continuous integration (`.github/workflows/tests.yml`): the full suite on Python 3.10 with pandas 2 and Python
  3.12 with pandas 3, on every push.
- Removed deprecated `Timestamp.utcnow` calls (gone in a future pandas); live volatility ignores zero prices.

## 0.8.0
- **Third null model: block bootstrap** (`nulls=["block"]`). The history is cut into 20-bar blocks (random grid
  offset) and the blocks are shuffled, one order for all markets: volatility clustering, short-term persistence and
  cross-market co-movement survive inside each block; longer patterns are broken.
- Calibration (docs/CALIBRATION.md): false alarms 3% and 2% on no-edge markets; it catches 88% of planted trend
  edges against 96-98% for the other two nulls, because 20-bar blocks keep part of each ~67-bar trend. It is the
  strictest test for trend strategies and only credits structure longer than a block.

## 0.7.1
- **Holdout power analysis.** A holdout FAIL only means something if the holdout could have seen the edge. Every
  holdout now writes `holdout.md` with: the Sharpe's 95% interval, the bar it had to beat, the smallest true Sharpe
  it detects 80% of the time, the chance it would have passed if the edge were as strong as in development, and a
  test of whether the edge shrank. `oql card holdout-power ID` writes the same report for a holdout already run
  (it recomputes the shuffles for the frozen finalist; the recorded verdict, ledger and stage are unchanged).

## 0.7.0
- **Project 11, large search** (Projects tab): the lab's search loop at growing sizes, up to about 3,000 distinct
  strategies from dense grids, on a synthetic book with a known truth. For each size it compares a naive t-test of
  the winner, the Deflated Sharpe Ratio, and the gate. On a no-edge book the naive test calls a ~700-strategy winner
  significant (p 0.03) while the gate does not (p 0.55); on a planted edge the gate passes at every size.
- Power study (docs/CALIBRATION.md): with about 11 years of data, edges whose best strategy shows a Sharpe near 0.4
  are caught 8-15% of the time; edges near Sharpe 1 over decades pass reliably.
- Calibration can run the large-search space (`--kinds big1000-noise`).

## 0.6.2
- `oql data pull-long-history-excess` **(fixes a data flaw found by card 0003)**: the long histories as returns a
  financed position actually earns, under `fred-xs/`. Treasuries earn yield minus the 3-month T-bill rate (the
  0.6.1 bond indexes credited the full yield, so a leveraged low-volatility bond looked like a riskless trend);
  currencies earn the spot move plus the interest-rate difference (OECD 3-month rates, used one month late), and
  start when that rate is known. Oil is left out: its tradable futures return cannot be built from the spot price.
  The 0.6.1 `fred-bars/` datasets are kept unchanged, because card 0003 was frozen on them.

## 0.6.1
- `oql data pull-long-history`: free FRED daily series turned into tradable bars under `fred-bars/`: 12 currencies
  (from 1971, as the foreign currency's price in dollars), constant-maturity Treasuries as total-return indexes
  (from 1962; -duration x yield change + carry) and WTI oil (from 1986). Close-only, so open = previous close.
  Decades of history give tests the statistical power that 2010-onward data cannot.
- Calibration can run power studies on weaker planted trends (`--kinds drift0.0001,...`).

## 0.6.0
- **Second null model** (`Experiment(nulls=["bars", "vol_matched"])`; `write_experiment(nulls=...)`). The
  volatility-matched null divides each bar by the market's volatility at its time, shuffles the standardized bars,
  and multiplies each by the volatility of the slot it lands in: calm periods stay calm and crises stay at their
  dates, and only the order of the moves is destroyed. With several nulls declared, the gate and the holdout PASS
  only if p < alpha under every one; the report shows each. Default stays `["bars"]`, so existing cards are unchanged.
- **Calibration study** (`python -m oql.calibration`, docs/CALIBRATION.md): 300 full gates on synthetic books with a
  known truth. Both nulls keep false positives at or below 5% (3-4%; 1% under strong volatility clustering) and pass
  96-98% of real edges. Under strong clustering the bar shuffle is too strict (p-values pile up near 1); the
  volatility-matched null is better calibrated there.

## 0.5.2
- **Rebalance band** (opt-in, `Experiment(rebalance_band=0.25)`; `write_experiment(rebalance_band=...)` for the
  assistant; "Resizing" in the gate check). A position keeps its size until the volatility-based target has moved more
  than the band away, then jumps to the target, so a strategy stops paying to resize every bar while staying within
  the band of its risk target. Paper trading puts the catalog's history in front of the live bars so its sizes match
  the backtest's. On synthetic hourly data a 25% band cut the cost bill of a slow rule (250-hour momentum) by about
  30% and of a fast one (20-hour) by about 8%: for fast rules most of the cost is direction changes, not resizing.
  Experiments without it are unchanged.

## 0.5.1
- The gate report counts efficiency studies and gate checks on the card's datasets made before its freeze
  ("Efficiency-lab explorations on these datasets before the freeze"). They shape which ideas get proposed, so they
  belong next to the result. Recorded at the freeze; cards frozen earlier say "not recorded".
- A project whose data is not in the catalog says what is missing and how to add it, instead of failing.
- `oql serve` on a port already in use says so (it was reported as a network error).
- No more "All-NaN slice" warning when a feature has no values yet early in a study.

## 0.5.0
- **Efficiency lab** (`oql/efficiency.py`, `oql efficiency study`, the dashboard's Efficiency tab, the assistant's
  `efficiency_study` tool). Is the next day's (hourly bars) or hour's (minute bars) efficiency ratio predictable?
  Walk-forward ridge (or boosted trees) on development data only, from price features plus optional calendar,
  funding, perp-spot basis, VIX and yield slope, CFTC positioning and Pi order-book groups, all joined as of the
  period start. The order-book data from the micro study's lock date on stays sealed. Each run first checks itself on
  a synthetic market with planted efficiency regimes and on a random walk, and is counted in the ledger.
- **Efficiency gates in the search** (`oql/gates.py`). An experiment can declare `gates=["none", "er_trailing",
  "er_forecast"]`; the search may then switch any strategy on only in the top (or bottom) third or half of trailing
  efficiency or of the efficiency forecast. Gates are computed from the bars inside the search, so the gate test
  reruns them on every shuffled copy. Switching pays costs. Paper trading computes the gate the same way, with the
  catalog's history in front of the live journal. Experiments without `gates` are unchanged.
- **Gate check** (`oql efficiency gates`, the Efficiency tab): all 21 gate settings on one strategy, P&L by forecast
  third, and a simulation of how strong a forecast would need to be. Counted in the ledger as 21 explorations.
- New rule `orb` (opening-range breakout on intraday bars), available by name; not in the default search.
- Assistant tools: `explore` and `permutation_check` take an optional gate; `write_experiment` takes `gates` and
  `gate_features` and counts gate settings in the search size; new `efficiency_study`.
- Ledger counts (command line, dashboard, assistant) add up the settings a row stands for, so a gate check counts 21.
- Charts: ranked horizontal bars for long category names.

## 0.4.0
- **Live dashboard.** `oql serve` can run any computation in the background and stream its results to the page as they
  arrive (`oql/live.py`, Server-Sent Events); charts redraw as the work happens, with hover readouts and a validated
  colour palette for light and dark mode (`oql/web/live.js`).
- **Projects tab.** The ten quant mini-projects (tail index, intraday seasonality, overfitting detector, momentum
  crashes, factor neutrality, American put, autodiff Greeks, implied distribution, Hawkes trades, news diffusion) run
  live on the lab's catalog data (`oql/projects/`). Each runs its known-answer checks first and shows them passing or
  failing; the four projects with published real-data results reproduce them.
- **Live market tab.** Public exchange data (Kraken, Coinbase, Binance.US; no account) polled by the server: price,
  the rolling one-hour efficiency ratio, realized volatility and the bid-ask spread, updating on their own
  (`oql/market.py`).
- New data commands: `oql data pull-french` (Fama-French factor tables) and `oql data import-projects DIR` (the
  projects' SPY option chain, licensed and kept local, and a day of Binance.US trades).
- The ledger tab counts tests (gate and holdout runs) separately from explorations.
- Packaging includes the `oql.projects` package and the page script, so a regular `pip install` works.

## 0.3.4
Found in the first dry run with an AI assistant connected over MCP:
- `write_experiment` and `explore` now say that every strategy trades all of an experiment's datasets together as one
  equal-risk portfolio. The assistant had counted 4 lookbacks on 2 coins as 8 strategies.
- Editing a card after asking for review marks the review summary as out of date, on the card and on the dashboard,
  until the assistant sends a fresh one.
- The gate report explains a result worse than every shuffled run: the signal pointed the wrong way more often than
  chance, and testing the opposite idea would be a new card and a new counted test.

## 0.3.3
- `oql data check` compares prices with a relative tolerance. Prices stored as decimals round slightly, and at large
  levels (Nikkei futures in the tens of thousands) that rounding was being reported as broken bars.
- Large one-bar moves are split in two: spikes that reverse on the next bar (the mark of a bad price) and moves that
  stick (real events, or an unadjusted futures roll). Each warning names a date to look at, as does the gap warning.
  This does not change any test result.

## 0.3.2
- **(affects results)** A trade is now a change of direction (enter, exit or flip). Positions are resized every
  bar to hold volatility steady; those resizes are still charged as costs, but they were also being counted as
  trades, so every strategy "traded" every bar and the minimum-trades rule never rejected anything. The minimum
  is now 1 direction change per year per market (`min_trades_per_year=1.0`): it rejects strategies that almost
  never change position, and keeps slow trend systems such as 12-month momentum on daily data.
- Exploration (`explore`, `permutation_check`) now uses the costs and fill delay declared in the card's
  `experiment.py`, so an AI assistant sees what the gate will see. Results report `cost_bps` and
  `direction_changes_per_year`.
- `oql card freeze` records the lab version; the gate report shows the version at the freeze and at the gate.
- `oql ledger` separates tests (gate, holdout) from explorations in its summary.
- `oql data check` warns about bars that are not on the dataset's time grid (found in real Binance data: 43 hourly
  bars after the February 2018 outage opened 28 minutes past the hour).
- The code repository ignores lab state (`cards/`, `board.json`, `LEDGER.csv`): run a lab in its own folder.
- `demo/check_parity.py` checks that the live demo page computes the same numbers as this code.

## 0.3.1
- Binance futures files start with a header row; the parser now reads them, and number columns stored as text
  are converted before saving.

## 0.3.0
- Phase 5: `oql mcp`, an MCP server for AI research assistants. Every exploration is counted and reported in the
  gate report.

## 0.2.0
- Data catalog, validation gate, research board, dashboard (`oql serve`), collector, paper trading.
