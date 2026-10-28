"""One cycle of the paper trader. The order never changes:

    1. RECONCILE FIRST  pull bars, repair gaps, note any bar the exchange has revised
    2. RISK GATE        is the data fresh, complete and sane? is the clock right? kill switch?
    3. DECIDE           run the frozen strategy on ALL stored bars; take the newest bar's position
    4. RECORD           write the decision once (never rewritten), update the paper ledger
    5. AUDIT            recompute every past decision from stored data and compare with what was logged

There is no order-placing code anywhere in this program. "Position" means a number in a database.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

import config as C

for _p in (os.path.join(C.HERE, "mcpt-lab"), os.path.join(C.HERE, "..", "mcpt-lab")):
    if os.path.isdir(os.path.join(_p, "mcptlab")):
        sys.path.insert(0, _p)
from mcptlab.strategies import HawkesVolatility   # the SAME class the backtests use: one source of truth

STRATEGY = HawkesVolatility()
ANCHOR_MS = int(pd.Timestamp(C.ANCHOR).timestamp() * 1000)


def model_positions(bars):
    """Strategy position for every stored bar (decided at that bar's close)."""
    return STRATEGY.positions(bars[["open", "high", "low", "close"]], C.PARAMS)


def risk_gate(store, feed, bars, now_ms):
    """Returns a list of reasons to stand aside. Empty list = OK."""
    reasons = []
    expected = (now_ms // C.BAR_MS) * C.BAR_MS - C.BAR_MS           # the bar that just closed
    last = int(bars.index[-1].timestamp() * 1000) if len(bars) else None
    if last != expected:
        reasons.append("STALE: newest bar is not the hour that just closed")
    if len(bars) < C.WARMUP_BARS:
        reasons.append(f"WARMUP: {len(bars)} of {C.WARMUP_BARS} bars")
    recent = pd.Series(bars.index[-C.GAP_LOOKBACK_BARS:])
    if len(recent) > 1 and (recent.diff().dropna() != pd.Timedelta(milliseconds=C.BAR_MS)).any():
        reasons.append("GAP: a bar is missing in the last day")
    if len(bars) >= 2:
        b = bars.iloc[-1]
        move = abs(np.log(b.close / bars.close.iloc[-2]))
        if not (b.low <= min(b.open, b.close) and b.high >= max(b.open, b.close) and b.low > 0):
            reasons.append("INSANE: last bar's open/high/low/close contradict each other")
        elif move > C.MAX_BAR_MOVE:
            reasons.append(f"INSANE: last bar moved {move:.0%} in one hour")
    try:
        drift = abs(feed.now_ms() - feed.exchange_time_ms()) / 1000
        if drift > C.MAX_CLOCK_DRIFT_S:
            reasons.append(f"CLOCK: this computer is {drift:.0f}s off the exchange")
    except Exception as e:
        reasons.append(f"CLOCK: could not read exchange time ({type(e).__name__})")
    if os.path.exists(C.KILL_FILE):
        reasons.append("KILL: kill-switch file present")
    return reasons


def cycle(store, feed):
    now = feed.now_ms()
    # 1. reconcile data: refetch from 3 bars before the newest stored bar (catches late revisions)
    last = store.last_bar_ts()
    since = ANCHOR_MS if last is None else last - 3 * C.BAR_MS
    try:
        fetched = feed.closed_bars(since, now, C.BAR_MS)
        store.upsert_bars(fetched, now)
        if last is not None:                                    # also try to repair any older gap in the last day
            ts = [r[0] for r in store.q("SELECT ts FROM bars WHERE ts>=? ORDER BY ts", (last - C.GAP_LOOKBACK_BARS * C.BAR_MS,))]
            if any(b - a != C.BAR_MS for a, b in zip(ts, ts[1:])):
                store.upsert_bars(feed.closed_bars(ts[0], now, C.BAR_MS), now)
        fetch_error = None
    except Exception as e:                                      # network down: carry on with what we have
        fetch_error = f"FEED: {type(e).__name__}: {e}"
    bars = store.bars()
    if not len(bars):
        store.x("INSERT INTO heartbeat VALUES(?,?,?,?)", (now, None, "HALT", fetch_error or "no data"))
        return dict(status="HALT", reasons=[fetch_error or "no data"])

    # 2. gate
    reasons = risk_gate(store, feed, bars, now) + ([fetch_error] if fetch_error else [])
    newest = int(bars.index[-1].timestamp() * 1000)

    # 3-4. decide and record. Bars that closed while the bot was off are marked MISSED (position carried).
    known = {r[0] for r in store.q("SELECT bar_ts FROM signals")}
    pos = model_positions(bars)
    first_live = store.q("SELECT MIN(bar_ts) FROM signals")[0][0]
    prev_target = (store.q("SELECT target FROM signals ORDER BY bar_ts DESC LIMIT 1") or [(0.0,)])[0][0]
    for ts_dt, model in pos.items():
        ts = int(ts_dt.timestamp() * 1000)
        if ts in known or (first_live is None and ts != newest) or (first_live is not None and ts < first_live):
            continue
        if ts == newest and not any(r.startswith("STALE") for r in reasons):
            status, target = ("LIVE", float(model)) if not reasons else ("HALT", 0.0)
        else:
            status, target = "MISSED", prev_target
        store.x("INSERT INTO signals VALUES(?,?,?,?,?,?,?)",
                (ts, now, status, target, float(model), float(bars.close[ts_dt]), json.dumps(reasons) if status != "LIVE" else ""))
        prev_target = target
    _update_ledger(store, bars)

    # 5. audit
    rec = audit(store, bars, pos, now)
    status = "OK" if not reasons else "HALT"
    store.x("INSERT INTO heartbeat VALUES(?,?,?,?)", (now, newest, status, "; ".join(reasons)))
    return dict(status=status, reasons=reasons, bar=bars.index[-1], target=prev_target, **rec)


def _update_ledger(store, bars):
    """Paper P&L, bar by bar: hold `position` from one close to the next, pay cost on changes."""
    sig = store.q("SELECT bar_ts, target FROM signals ORDER BY bar_ts")
    done = {r[0] for r in store.q("SELECT bar_ts FROM ledger")}
    close = {int(t.timestamp() * 1000): c for t, c in bars.close.items()}
    equity = (store.q("SELECT equity FROM ledger ORDER BY bar_ts DESC LIMIT 1") or [(0.0,)])[0][0]
    prev = 0.0
    for i, (ts, target) in enumerate(sig):
        nxt = ts + C.BAR_MS
        if ts not in done and nxt in close and ts in close:     # the next bar has closed: this row is final
            ret = float(np.log(close[nxt] / close[ts])); cost = abs(target - prev) * C.COST_BPS / 1e4
            pnl = target * ret - cost; equity += pnl
            store.x("INSERT INTO ledger VALUES(?,?,?,?,?,?)", (ts, target, ret, cost, pnl, equity))
        prev = target


def audit(store, bars, pos, now):
    """Compare every logged decision with the strategy recomputed on today's stored data."""
    sig = store.q("SELECT bar_ts, status, target FROM signals ORDER BY bar_ts")
    if not sig:
        return dict(live=0, matched=0, mismatched=0, missed=0, halted=0)
    model = {int(t.timestamp() * 1000): float(v) for t, v in pos.items()}
    live = [(ts, tg) for ts, st, tg in sig if st == "LIVE"]
    bad = [ts for ts, tg in live if abs(model.get(ts, np.nan) - tg) > 1e-12]
    missed = sum(st == "MISSED" for _, st, _ in sig); halted = sum(st == "HALT" for _, st, _ in sig)
    eq_live = (store.q("SELECT SUM(pnl) FROM ledger")[0][0]) or 0.0
    led = {r[0] for r in store.q("SELECT bar_ts FROM ledger")}
    # backtest over the same bars: model position, same cost rule
    eq_bt, prev = 0.0, 0.0
    close = {int(t.timestamp() * 1000): c for t, c in bars.close.items()}
    for ts, _, _ in sig:
        m = model.get(ts, 0.0)
        if ts in led:
            eq_bt += m * float(np.log(close[ts + C.BAR_MS] / close[ts])) - abs(m - prev) * C.COST_BPS / 1e4
        prev = m
    rec = dict(live=len(live), matched=len(live) - len(bad), mismatched=len(bad), missed=int(missed), halted=int(halted),
               equity_live=eq_live, equity_backtest=eq_bt)
    store.x("INSERT INTO reconcile VALUES(?,?,?,?,?,?,?,?,?,?)",
            (now, len(sig), rec["live"], rec["matched"], rec["mismatched"], rec["missed"], rec["halted"], eq_live, eq_bt,
             json.dumps(bad[-20:])))
    return rec
