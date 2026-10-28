# Order-book recorder

Records the top 50 levels of the BTC order book and every trade from several exchanges, every 10 seconds, into one SQLite file per month. Read-only public data: no account, no API key, and no code that can place an order.

```bash
python recorder.py test     # one cycle; shows what each venue returned
python recorder.py          # record forever
python check.py             # what has been collected, and each venue's success rate
```

Each venue is independent. If one is slow or down, its failure is logged in the `health` table and the others carry on. Edit the venue list in `config.py`.

Expect roughly 30-40 MB of data per day.
