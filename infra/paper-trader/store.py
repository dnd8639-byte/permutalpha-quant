"""SQLite state. Everything the bot knows lives here, so a restart loses nothing."""
import os
import sqlite3

SCHEMA = """
CREATE TABLE IF NOT EXISTS bars(ts INTEGER PRIMARY KEY, open REAL, high REAL, low REAL, close REAL, volume REAL, first_seen INTEGER);
CREATE TABLE IF NOT EXISTS revisions(ts INTEGER, seen_at INTEGER, old_close REAL, new_close REAL);
CREATE TABLE IF NOT EXISTS signals(bar_ts INTEGER PRIMARY KEY, decided_at INTEGER, status TEXT, target REAL, model REAL,
                                   close REAL, reasons TEXT);
CREATE TABLE IF NOT EXISTS ledger(bar_ts INTEGER PRIMARY KEY, position REAL, bar_return REAL, cost REAL, pnl REAL, equity REAL);
CREATE TABLE IF NOT EXISTS reconcile(run_at INTEGER, bars_checked INTEGER, live INTEGER, matched INTEGER, mismatched INTEGER,
                                     missed INTEGER, halted INTEGER, equity_live REAL, equity_backtest REAL, detail TEXT);
CREATE TABLE IF NOT EXISTS heartbeat(at INTEGER, bar_ts INTEGER, status TEXT, note TEXT);
"""


class Store:
    def __init__(self, path):
        if path != ":memory:":
            os.makedirs(os.path.dirname(path), exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.executescript(SCHEMA)
        self.db.commit()

    def upsert_bars(self, bars, now_ms):
        """Insert new bars; if a stored bar's close changed, record the revision and keep the newest value."""
        new = 0
        for ts, o, h, l, c, v in bars:
            row = self.db.execute("SELECT close FROM bars WHERE ts=?", (ts,)).fetchone()
            if row is None:
                self.db.execute("INSERT INTO bars VALUES(?,?,?,?,?,?,?)", (ts, o, h, l, c, v, now_ms)); new += 1
            elif abs(row[0] - c) > 1e-9:
                self.db.execute("INSERT INTO revisions VALUES(?,?,?,?)", (ts, now_ms, row[0], c))
                self.db.execute("UPDATE bars SET open=?,high=?,low=?,close=?,volume=? WHERE ts=?", (o, h, l, c, v, ts))
        self.db.commit()
        return new

    def bars(self):
        import pandas as pd
        d = pd.read_sql_query("SELECT ts,open,high,low,close,volume FROM bars ORDER BY ts", self.db)
        d.index = pd.to_datetime(d.pop("ts"), unit="ms")
        return d

    def last_bar_ts(self):
        r = self.db.execute("SELECT MAX(ts) FROM bars").fetchone()[0]
        return r

    def q(self, sql, args=()):
        return self.db.execute(sql, args).fetchall()

    def x(self, sql, args=()):
        self.db.execute(sql, args); self.db.commit()
