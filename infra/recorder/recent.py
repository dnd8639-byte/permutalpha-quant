"""Last few minutes of recorder health per venue:  python recent.py [minutes]"""
import glob, os, sqlite3, sys, time
import config as C
mins = int(sys.argv[1]) if len(sys.argv) > 1 else 5
db = sqlite3.connect(sorted(glob.glob(os.path.join(C.DATA_DIR, "book_*.db")))[-1])
since = int(time.time() * 1000) - mins * 60_000
for v, ok, n in db.execute("SELECT venue, SUM(ok), COUNT(*) FROM health WHERE ts>? GROUP BY venue", (since,)):
    print(f"{v:10s} {ok}/{n} cycles OK in the last {mins} min")
for v, e, c in db.execute("SELECT venue, error, COUNT(*) FROM health WHERE ok=0 AND ts>? GROUP BY venue, error", (since,)):
    print(f"  {c}x {v}: {e}")
