# micro: order-book shape → next-hour efficiency

This ties the Pi's order-book recorder to the edge-engine research.

The edge-engine showed that tomorrow's efficiency (net move ÷ distance travelled) cannot be predicted from past prices. This module asks whether the **order book** can predict it, using the data the Pi records every 10 seconds on three exchanges. The rules of the test are fixed in advance in [PREREGISTRATION.md](PREREGISTRATION.md).

```
Pi:   recorder (every 10 s) ──> ~/recorder/data/book_YYYY-MM.db
      extract.py (hourly timer) ──> ~/micro/data/hourly.csv      one row per hour, ~30 numbers
Mac:  scp the CSV ──> study.py status | pilot | freeze | confirm
```

| file | what it does |
|---|---|
| `extract.py` | Runs on the Pi. Reads the recorder's files read-only and writes one row per complete hour: depth within 2, 10 and 25 bps (where the recording reaches), book imbalance, microprice offset, trade-flow imbalance, spread, cross-exchange price dispersion, and the hour's efficiency and range. |
| `study.py` | Runs on the Mac. `status` shows data counts only. `pilot` explores the hours before the lock date. `freeze` fingerprints the rules. `confirm` is the one-shot test. |
| `simulate.py` | Writes fake recordings in the recorder's exact format. One version has a planted order-flow effect and one is pure noise; the pipeline must find the first and reject the second. |
| `selftest.sh` | Runs that check from start to finish. |
| `micro-extract.service/.timer` | Run `extract.py` on the Pi at 7 minutes past every hour. |

**Timeline:** pilot until 2026-10-20 00:00 UTC, then freeze and push to GitHub. The confirmation test opens at 1,000 hours, around the start of December.

**Freeze record:** frozen 2026-10-20 00:47 UTC (`runs/freeze.json`), 47 minutes after the lock and before the first
confirmation hour (00:00-01:00 UTC) was complete, so no confirmation data existed when the rules were fixed. The
fingerprints cover `study.py` and `PREREGISTRATION.md`; the confirmatory test refuses to run if either changes.
The data (`data/`) is not published: it is the Pi's own recording, and the confirmation hours stay unseen until the test.
