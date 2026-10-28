# Paper trader

Runs the one strategy that survived testing (VolatilityHawkes, frozen parameters) on live hourly Bitcoin prices, **with no real money**. It reads public prices and writes decisions to a local database. There is no order-placing code in it.

Its job is to answer one engineering question: **does the live system make exactly the decisions the backtest says it should?**

## Every hour, in this order
1. **Reconcile first.** Pull the latest bars, repair gaps, and record any bar the exchange has changed.
2. **Risk gate.** Stand aside (go flat) if the data is stale, incomplete or absurd, if the clock is wrong, or if the kill switch is on.
3. **Decide.** Run the same strategy class the backtests use, on all stored bars.
4. **Record.** Write the decision once. Update the paper profit and loss.
5. **Audit.** Recompute every past decision from stored data and compare it with what was logged.

## Commands
```bash
python replay.py        # prove it offline first: replays 30 days of history with faults injected (about 2 minutes)
python run.py once      # one live cycle now (first run downloads history since 1 Jan 2026)
python run.py           # run forever, once an hour; Control + C stops it
python run.py status    # print the current status
touch KILL              # kill switch: forces the bot flat.   rm KILL  turns it back on
```
The status page is `state/status.html` (refreshes itself every minute).

## Files
| file | what it does |
|---|---|
| `config.py` | every setting: symbol, frozen parameters, cost, risk limits |
| `feed.py` | live prices from the exchange, or replayed history for tests |
| `store.py` | the SQLite database: bars, revisions, decisions, ledger, audits, heartbeat |
| `engine.py` | the five-step hourly cycle |
| `status.py` | text summary and status page |
| `oled.py` | draws the status on a small 128 x 64 OLED screen (`python oled.py preview` saves a picture of it) |
| `run.py` | starts it |
| `replay.py` | the test: clean replay, power cut, bad data, revised data |
| `paper-trader.service` | starts it at boot on the Raspberry Pi |

It needs the `mcptlab` folder from mcpt-lab (it looks in `../mcpt-lab` or `./mcpt-lab`), plus numpy, pandas and ccxt.
