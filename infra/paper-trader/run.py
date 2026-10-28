"""Run the paper trader.

    python run.py once      one cycle now, print the status
    python run.py           run forever: one cycle 20 seconds after every hour closes
    python run.py status    print the current status without doing anything
PAPER TRADING ONLY. It reads public prices and writes to a local database. It cannot place orders.
"""
import sys, time, traceback

import config as C
import engine, oled, status
from feed import LiveFeed
from store import Store


def one(store, feed):
    out = engine.cycle(store, feed)
    d = status.summary(store)
    status.write_html(d)
    oled.show(d)                                               # does nothing if no screen is attached
    print(status.text(d), flush=True)
    return out


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "loop"
    store = Store(C.DB_PATH)
    if mode == "status":
        print(status.text(status.summary(store))); sys.exit()
    feed = LiveFeed(C.EXCHANGE, C.SYMBOL)
    if mode == "once":
        one(store, feed); sys.exit()
    print("Paper trader started. Press Control + C to stop.", flush=True)
    while True:
        try:
            one(store, feed)
        except Exception:                                      # log it and keep going: the next hour is a fresh start
            traceback.print_exc()
        now = time.time()
        nxt = (now // 3600 + 1) * 3600 + C.RUN_AT_SECOND
        time.sleep(max(5, nxt - now))
