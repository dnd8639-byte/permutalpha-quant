"""Prove the bot before it runs live: replay real history through the SAME engine, one hour at a time.

  TEST 1  clean replay      every live decision must equal the backtest, and paper P&L must equal backtest P&L
  TEST 2  power cut         stop the bot for 30 hours mid-run, restart from the database: no state lost,
                            missed hours are labelled MISSED, and it resumes matching
  TEST 3  bad data          a missing bar, a late bar, an absurd price, a wrong clock, the kill switch:
                            the risk gate must go flat each time and recover by itself
  TEST 4  revised data      the exchange changes a bar after the bot acted on it: the audit must notice
Run:  python replay.py [path/to/btc_1h_binanceus.csv]      (a few minutes)
"""
import os, sys
import numpy as np
import pandas as pd

import config as C
import engine
from feed import ReplayFeed
from store import Store

H = C.BAR_MS


def load_rows(path, start, end):
    d = pd.read_csv(path).dropna()
    d["ts"] = [int(t.timestamp() * 1000) for t in pd.to_datetime(d["date"])]
    d = d[(d.ts >= start) & (d.ts < end)].sort_values("ts")
    return [[int(r[0])] + list(r[1:]) for r in d[["ts", "open", "high", "low", "close", "volume"]].values.tolist()]


def run(rows, first_live, last_live, setup=None, off=()):
    """Step the clock hour by hour, using a real database file. `off` = (start, end) window when the bot
    is switched off; afterwards the database is closed and reopened, exactly like a restart."""
    import tempfile
    path = os.path.join(tempfile.mkdtemp(), "replay.db")
    store, feed = Store(path), ReplayFeed(rows)
    if setup:
        setup(feed)
    out, was_off = None, False
    for t in range(first_live, last_live + 1, H):
        if off and off[0] <= t < off[1]:
            was_off = True
            continue
        if was_off:
            store.db.close(); store = Store(path); was_off = False
        feed.clock = t + H + C.RUN_AT_SECOND * 1000              # 20 seconds after bar t closes
        out = engine.cycle(store, feed)
    return store, out


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}  {detail}")
    return bool(cond)


if __name__ == "__main__":
    default = next((p for p in (os.path.join(C.HERE, "..", "mcpt-lab", "data", "btc_1h_binanceus.csv"),
                                os.path.join(C.HERE, "mcpt-lab", "data", "btc_1h_binanceus.csv")) if os.path.exists(p)), None)
    path = sys.argv[1] if len(sys.argv) > 1 else default
    if os.path.exists(C.KILL_FILE):
        raise SystemExit("remove the KILL file before running the replay")
    anchor = engine.ANCHOR_MS
    rows = load_rows(path, anchor, anchor + 10 ** 13)
    N_LIVE = 720                                               # 30 days of hourly decisions
    first = int(rows[C.WARMUP_BARS + 200][0]); last = first + (N_LIVE - 1) * H
    idx = pd.to_datetime([first, last], unit="ms")
    print(f"Replaying {N_LIVE} hourly decisions, {idx[0]:%Y-%m-%d %H:%M} .. {idx[1]:%Y-%m-%d %H:%M} UTC, history from {C.ANCHOR[:10]}\n")
    ok = True

    print("TEST 1  clean replay")
    store, out = run(rows, first, last)
    ok &= check("every live decision equals the backtest", out["mismatched"] == 0 and out["live"] == N_LIVE, f"({out['matched']}/{out['live']} matched)")
    ok &= check("paper P&L equals backtest P&L", abs(out["equity_live"] - out["equity_backtest"]) < 1e-9,
                f"(live {out['equity_live']:+.6f}, backtest {out['equity_backtest']:+.6f} log return)")
    trades = store.q("SELECT COUNT(*) FROM ledger WHERE cost>0")[0][0]
    ok &= check("decisions were not all the same", trades > 0, f"({trades} position changes)")

    print("TEST 2  power cut for 30 hours, then restart from the saved database")
    cut = (first + 200 * H, first + 230 * H)
    store, out = run(rows, first, last, off=cut)
    ok &= check("missed hours are labelled, not invented", out["missed"] == 30, f"({out['missed']} MISSED)")
    ok &= check("after restart, live decisions match again", out["mismatched"] == 0 and out["live"] == N_LIVE - 30, f"({out['matched']}/{out['live']})")

    print("TEST 3  bad data and operator controls: the risk gate must go flat, then recover")
    t_gap, t_late, t_spike, t_clock, t_kill = (first + k * H for k in (100, 250, 400, 500, 600))

    def faults(feed):
        feed.hide = {t_gap - 2 * H}                            # a bar the exchange never serves...
        feed.delay = {t_late: 90_000}                          # a bar published 90 seconds late
        feed.spike = {t_spike: 1.0}                            # close of $1: absurd

    class ClockFeed(ReplayFeed):
        def exchange_time_ms(self):
            return self.clock - (120_000 if self.clock // H * H - H == t_clock else 0)
    store, feed = Store(":memory:"), ClockFeed(rows); faults(feed)
    for t in range(first, last + 1, H):
        feed.clock = t + H + C.RUN_AT_SECOND * 1000
        if t == t_kill: open(C.KILL_FILE, "w").close()
        elif os.path.exists(C.KILL_FILE): os.remove(C.KILL_FILE)
        if t == t_gap + 30 * H: feed.hide = set()              # ...until 30 hours later
        out = engine.cycle(store, feed)
    why = {r[0]: r[1] for r in store.q("SELECT bar_ts, reasons FROM signals WHERE status='HALT'")}
    for nm, t, word in [("missing bar", t_gap, "GAP"), ("late bar", t_late, "STALE"), ("absurd price", t_spike, "INSANE"),
                        ("clock 2 minutes off", t_clock, "CLOCK"), ("kill switch", t_kill, "KILL")]:
        st = store.q("SELECT status, target, reasons FROM signals WHERE bar_ts=?", (t,))
        hit = (st and st[0][0] != "LIVE" and st[0][1] == 0.0 and word in st[0][2]) or (word == "STALE" and not st) or \
              (word == "STALE" and st and st[0][0] == "MISSED")
        ok &= check(f"{nm:20s} -> flat or no decision", hit, f"({st[0][0] if st else 'no decision that hour'})")
    after = store.q("SELECT status FROM signals WHERE bar_ts=?", (last,))[0][0]
    ok &= check("recovers by itself", after == "LIVE" and out["status"] == "OK", f"(last decision {after}; {out['halted']} HALT hours in total)")

    print("TEST 4  the exchange revises a bar after the bot acted on it")
    t_rev = first + 300 * H
    rev_row = next(r for r in rows if int(r[0]) == t_rev)
    first_shown = (rev_row[1] + rev_row[4]) / 2             # the close first shown is wrong, but still a valid bar
    store, out = run(rows, first, last, setup=lambda f: setattr(f, "revise", {t_rev: (first_shown, t_rev + 3 * H)}))
    n_rev = store.q("SELECT COUNT(*) FROM revisions")[0][0]
    ok &= check("revision is recorded", n_rev >= 1, f"({n_rev} revision logged)")
    st = store.q("SELECT status, reasons FROM signals WHERE status!='LIVE'")
    print(f"        audit after the revision: {out['mismatched']} of {out['live']} live decisions differ from a backtest on the corrected data")

    print("\nALL TESTS PASSED" if ok else "\nSOME TESTS FAILED")
