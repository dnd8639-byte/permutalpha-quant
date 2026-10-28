"""How far from the mid-price do the recorded book levels reach?   python depthcheck.py
Looks at the last hour of snapshots for each venue. If the recorded levels only reach a few bps, then 'depth within
25 bps' is really 'depth of whatever was recorded', and the feature has to be redefined or the recorder made deeper."""
import glob, json, os, sqlite3, zlib

import numpy as np

SRC = os.environ.get("RECORDER_DATA", os.path.expanduser("~/recorder/data"))
f = sorted(glob.glob(os.path.join(SRC, "book_*.db")))[-1]
db = sqlite3.connect(f"file:{f}?mode=ro", uri=True)
print(f"{'venue':10s} {'levels':>7s} {'bid reach':>10s} {'ask reach':>10s}   USD depth within  2 bps    10 bps    25 bps")
for (v,) in db.execute("SELECT DISTINCT venue FROM book"):
    rows = db.execute("SELECT bid, ask, book FROM book WHERE venue=? ORDER BY ts DESC LIMIT 360", (v,)).fetchall()
    lv, br, ar, dep = [], [], [], {2: [], 10: [], 25: []}
    for bid, ask, blob in rows:
        b = json.loads(zlib.decompress(blob)); bids, asks = np.array(b["b"], float), np.array(b["a"], float)
        mid = (bid + ask) / 2
        lv.append(min(len(bids), len(asks)))
        br.append((mid - bids[-1, 0]) / mid * 1e4); ar.append((asks[-1, 0] - mid) / mid * 1e4)
        for B in dep:
            if min(br[-1], ar[-1]) >= B:
                dep[B].append((bids[bids[:, 0] >= mid * (1 - B / 1e4), 1].sum() + asks[asks[:, 0] <= mid * (1 + B / 1e4), 1].sum()) * mid)
    cov = lambda B: f"{np.median(dep[B]) / 1e6:6.2f}M" if len(dep[B]) > len(rows) / 2 else "  n/a  "
    print(f"{v:10s} {int(np.median(lv)):7d} {np.median(br):8.2f}bp {np.median(ar):8.2f}bp   {'':16s}{cov(2)}  {cov(10)}  {cov(25)}")
print("\n'reach' = median distance from mid to the deepest recorded level. 'n/a' = the recording does not reach that far.")
