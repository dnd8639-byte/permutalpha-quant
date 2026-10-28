"""edge-engine runner. Two stages, in this order, with the holdout locked until the end.

    python run.py dev BTC EURUSD GOLD ...      search on the first 70% of each market's data (walk-forward inside),
                                               then permutation-test the WHOLE search. Saves the finalist.
    python run.py holdout                      run the saved finalist ONCE on the last 30%. Refuses to run twice
                                               for the same finalist and market set.

Every strategy tried and every stage run is appended to runs/LEDGER.csv.
"""
import csv, json, os, sys, time
import numpy as np

import engine as E
from core import Market, load_hourly

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs"); os.makedirs(RUNS, exist_ok=True)
FINALIST = os.path.join(RUNS, "finalist.json")
EVENTS = os.path.join(HERE, "data", "events.csv")


def load_markets(names):
    cfg = json.load(open(os.path.join(HERE, "markets.json")))
    ev = []
    if os.path.exists(EVENTS):
        ev = [l.split(",")[0] for l in open(EVENTS).read().split()[1:]]
    out = []
    for n in names:
        c = cfg[n]; path = os.path.join(HERE, c["file"])
        if not os.path.exists(path):
            print(f"  {n}: no data file at {c['file']} (run  python data/download.py)"); continue
        out.append(Market(n, load_hourly(path), c["cost_bps"], c["delay_h"], ev))
        D = out[-1].D
        print(f"  {n:7s} {len(D):5d} days  {D.index[0]:%Y-%m-%d} .. {D.index[-1]:%Y-%m-%d}   dev ends {E.split_day(out[-1], S):%Y-%m-%d}")
    return out, ev


def ledger(row):
    path = os.path.join(RUNS, "LEDGER.csv"); new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row)); w.writeheader() if new else None; w.writerow(row)


S = E.Settings()

if __name__ == "__main__":
    stage, names = sys.argv[1], sys.argv[2:]
    n_perm = int(os.environ.get("N_PERM", "99"))
    if stage == "dev":
        print("DEVELOPMENT (first 70% of each market; the last 30% is not loaded into the search)")
        mk, ev = load_markets(names)
        P = E.prepare(mk, S, "dev")
        q = E.model_quality(P)
        for k, v in q.items():
            print(f"  ER model, {k}: out-of-sample corr {v['corr']:+.2f}; skill vs training mean {v['skill_vs_mean']:+.3f}, vs recent average {v['skill_vs_persistence']:+.3f}")
        log = []
        t0 = time.time(); res = E.search(P, S, log)
        dev_port, _, _ = E.strategy_returns(P, res["best"])
        dsr = E.deflated_sharpe(dev_port, res["n_trials"])
        print(f"  search: {res['n_trials']} strategies tried in {S.rounds} rounds ({time.time() - t0:.0f}s); best per round: "
              + " ".join(f"{x:.2f}" for x in res["history"]))
        print(f"  finalist {res['best']}  development Sharpe {res['best_score']:.2f}  deflated Sharpe {dsr:.2f}")
        print(f"  permutation test of the whole search ({n_perm} shuffled markets)...", flush=True)
        p, null = E.search_permutation_test(mk, S, res["best_score"], n=n_perm, events={m.name: ev for m in mk})
        print(f"  whole-search p = {p:.3f}   (shuffled best Sharpe: median {np.median(null):.2f}, 95th pct {np.percentile(null, 95):.2f})")
        for r in log:
            ledger(dict(time=time.strftime("%Y-%m-%d %H:%M"), stage="dev-trial", markets="+".join(m.name for m in mk), **r, p="", note=""))
        ledger(dict(time=time.strftime("%Y-%m-%d %H:%M"), stage="dev-search", markets="+".join(m.name for m in mk), round=S.rounds,
                    score=res["best_score"], genome=json.dumps(res["best"]), p=p, note=f"trials={res['n_trials']} dsr={dsr:.3f}"))
        json.dump(dict(genome=res["best"], markets=[m.name for m in mk], dev_score=res["best_score"], dev_p=p,
                       n_trials=res["n_trials"], holdout_opened=None), open(FINALIST, "w"), indent=1)
        print(f"  saved {FINALIST}. Decide whether it deserves the holdout BEFORE running:  python run.py holdout")
    elif stage == "holdout":
        fin = json.load(open(FINALIST))
        if fin.get("holdout_opened"):
            sys.exit(f"The holdout for this finalist was already opened on {fin['holdout_opened']}. Running it again would "
                     f"turn it into development data. Develop a new finalist instead.")
        print("HOLDOUT (last 30%), one run of the frozen finalist:", fin["genome"])
        mk, ev = load_markets(fin["markets"])
        fin["holdout_opened"] = time.strftime("%Y-%m-%d %H:%M"); json.dump(fin, open(FINALIST, "w"), indent=1)
        H = E.prepare(mk, S, "holdout")
        sb = E.side_and_beta(H, fin["genome"])
        p, null = E.holdout_permutation_test(mk, S, fin["genome"], sb["sharpe"], n=n_perm * 2 + 1, events={m.name: ev for m in mk})
        mc = E.cost_monte_carlo(H, fin["genome"])
        abl = E.gate_ablation(H, fin["genome"])
        print(f"  Sharpe {sb['sharpe']:.2f}  p = {p:.3f}  annual vol {sb['ann_vol']:.1%}  beta {sb['beta']:+.2f}  corr with market {sb['corr_market']:+.2f}")
        if sb["sharpe_long"] is not None:
            print(f"  long side Sharpe {sb['sharpe_long']:.2f}, short side {sb['sharpe_short']:.2f}")
        print(f"  cost Monte Carlo (costs x0.5-x3, delays 0-3h): Sharpe median {np.median(mc):.2f}, 5th pct {np.percentile(mc, 5):.2f}")
        print("  gate ablation: " + "  ".join(f"{k} {v:.2f}" for k, v in abl.items()))
        ledger(dict(time=time.strftime("%Y-%m-%d %H:%M"), stage="holdout", markets="+".join(fin["markets"]), round="",
                    score=sb["sharpe"], genome=json.dumps(fin["genome"]), p=p, note=f"beta={sb['beta']:.2f}"))
