"""Turn the database into something a person can read: a text summary and a small web page."""
import html
import os

import pandas as pd

import config as C


def summary(store):
    hb = store.q("SELECT at, bar_ts, status, note FROM heartbeat ORDER BY at DESC LIMIT 1")
    rec = store.q("SELECT live, matched, mismatched, missed, halted, equity_live, equity_backtest FROM reconcile ORDER BY run_at DESC LIMIT 1")
    sig = store.q("SELECT bar_ts, status, target, close, reasons FROM signals ORDER BY bar_ts DESC LIMIT 12")
    first = store.q("SELECT MIN(bar_ts) FROM signals")[0][0]
    t = lambda ms: pd.to_datetime(ms, unit="ms").strftime("%Y-%m-%d %H:%M") if ms else "-"
    d = dict(status=hb[0][2] if hb else "NO DATA", note=hb[0][3] if hb else "", last_run=t(hb[0][0]) if hb else "-",
             last_bar=t(hb[0][1]) if hb else "-", since=t(first), bars=store.q("SELECT COUNT(*) FROM bars")[0][0],
             revisions=store.q("SELECT COUNT(*) FROM revisions")[0][0],
             position={1.0: "LONG", -1.0: "SHORT", 0.0: "FLAT"}.get(sig[0][2], str(sig[0][2])) if sig else "-",
             price=sig[0][3] if sig else None, recent=[(t(a), b, c, e, f) for a, b, c, e, f in sig])
    if rec:
        d.update(live=rec[0][0], matched=rec[0][1], mismatched=rec[0][2], missed=rec[0][3], halted=rec[0][4],
                 eq_live=rec[0][5], eq_bt=rec[0][6])
    return d


def text(d):
    out = [f"PAPER TRADER  {d['status']}   paper position: {d['position']}   BTC {d['price']:,.0f}" if d["price"] else f"PAPER TRADER  {d['status']}",
           f"  last run {d['last_run']} UTC   last bar {d['last_bar']}   running since {d['since']}   {d['bars']} bars stored"]
    if d["note"]:
        out.append(f"  standing aside because: {d['note']}")
    if "live" in d:
        out.append(f"  decisions: {d['live']} live, {d['matched']} match the backtest, {d['mismatched']} differ, {d['missed']} missed, {d['halted']} halted")
        out.append(f"  paper return {d['eq_live']:+.4f}   backtest on the same bars {d['eq_bt']:+.4f}   (log return, after costs)   {d['revisions']} exchange revisions")
    return "\n".join(out)


def write_html(d, path=C.STATUS_HTML):
    ok = d["status"] == "OK"
    rows = "".join(f"<tr><td>{a}</td><td>{b}</td><td>{ {1.0:'LONG',-1.0:'SHORT',0.0:'FLAT'}.get(c, c) }</td><td>{e:,.0f}</td><td>{html.escape(f or '')[:60]}</td></tr>"
                   for a, b, c, e, f in d["recent"])
    stats = ""
    if "live" in d:
        stats = (f"<div class=g><div><b>{d['matched']}/{d['live']}</b><span>live decisions match the backtest</span></div>"
                 f"<div><b>{d['eq_live']:+.2%}</b><span>paper return after costs (backtest {d['eq_bt']:+.2%})</span></div>"
                 f"<div><b>{d['missed']} / {d['halted']}</b><span>hours missed / stood aside</span></div></div>")
    page = f"""<!doctype html><meta charset=utf-8><meta http-equiv=refresh content=60><meta name=viewport content="width=device-width,initial-scale=1">
<title>Paper trader</title><style>
body{{margin:0;padding:18px;background:#0F141B;color:#E6EBF1;font:16px/1.4 system-ui,sans-serif}}
h1{{font-size:15px;font-weight:500;color:#95A2B3;margin:0 0 6px;letter-spacing:.08em;text-transform:uppercase}}
.s{{font-size:54px;font-weight:700;color:{'#5CC08D' if ok else '#E57C7C'}}} .p{{font-size:26px;margin:2px 0 12px}}
.n{{color:#E3AE55;margin-bottom:10px}} .g{{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin:12px 0}}
.g div{{background:#161E29;border:1px solid #283341;border-radius:8px;padding:10px}} .g b{{display:block;font-size:24px}}
.g span,td,th,.f{{font-size:13px;color:#95A2B3}} table{{border-collapse:collapse;width:100%}} td{{padding:3px 8px 3px 0;border-top:1px solid #283341}}
</style><h1>Paper trader · no real money · {html.escape(C.SYMBOL)} hourly</h1>
<div class=s>{d['status']}</div><div class=p>{d['position']} &nbsp; BTC {d['price']:,.0f}</div>
{f"<div class=n>Standing aside: {html.escape(d['note'])}</div>" if d['note'] else ""}{stats}
<table>{rows}</table><p class=f>Last run {d['last_run']} UTC · running since {d['since']} · {d['bars']} bars · {d['revisions']} exchange revisions</p>"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "w") as f:
        f.write(page)
    os.replace(path + ".tmp", path)                             # never leave a half-written page
