"""What has the recorder collected?   python check.py"""
import glob, os, sqlite3, time

import config as C

files = sorted(glob.glob(os.path.join(C.DATA_DIR, "book_*.db")))
if not files:
    raise SystemExit("no data yet")
total = sum(os.path.getsize(f) + (os.path.getsize(f + "-wal") if os.path.exists(f + "-wal") else 0) for f in files)
print(f"{len(files)} monthly file(s), {total / 1e6:.1f} MB on disk\n")
db = sqlite3.connect(files[-1])
now = int(time.time() * 1000)
t = lambda ms: time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(ms / 1000)) if ms else "-"
print(f"{'venue':10s} {'snapshots':>10s} {'trades':>9s} {'first snapshot (UTC)':>21s} {'last snapshot (UTC)':>21s} {'ok last hour':>13s} {'median ms':>10s}")
for (venue,) in db.execute("SELECT DISTINCT venue FROM health ORDER BY venue"):
    n, first, last = db.execute("SELECT COUNT(*), MIN(ts), MAX(ts) FROM book WHERE venue=?", (venue,)).fetchone()
    ntr = db.execute("SELECT COUNT(*) FROM trades WHERE venue=?", (venue,)).fetchone()[0]
    h = db.execute("SELECT ok, ms FROM health WHERE venue=? AND ts>?", (venue, now - 3600_000)).fetchall()
    okr = f"{sum(x[0] for x in h) / len(h):.0%}" if h else "-"
    ms = sorted(x[1] for x in h)[len(h) // 2] if h else 0
    print(f"{venue:10s} {n:10d} {ntr:9d} {t(first):>21s} {t(last):>21s} {okr:>13s} {ms:10d}")
err = db.execute("SELECT venue, error, COUNT(*) FROM health WHERE ok=0 AND ts>? GROUP BY venue, error ORDER BY 3 DESC LIMIT 5", (now - 86400_000,)).fetchall()
if err:
    print("\nMost common errors in the last 24 hours:")
    for v, e, c in err:
        print(f"  {c:6d}x  {v}: {e}")
