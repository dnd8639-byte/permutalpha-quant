"""Does the daily engine find a planted edge, and reject pure noise?  python validate_daily.py [N_PERM]"""
import sys, time
import numpy as np
from daily_engine import Settings, prepare, search, model_report, gate_ablation, search_permutation_test, _events
from synth_daily import make_book


def main():
    n_perm = int(sys.argv[1]) if len(sys.argv) > 1 else 19
    ev = _events()
    for kind in ("regime", "noise"):
        t0 = time.time()
        book = make_book(kind, seed=7)
        prep = prepare(book, ev)
        mq = model_report(prep.quality)
        res = search(prep, Settings())
        g = res["best"]
        print(f"\n=== {kind.upper()} ===")
        print(f"ER model R2 {mq['er_r2']:+.4f} tiers {mq['er_by_tier']} | vol model R2 {mq['vol_r2']:+.3f} tiers {mq['range_by_vol_tier']}")
        print(f"best dev Sharpe {res['best_score']:.2f} after {res['n_trials']} trials: {g}")
        print("gate ablation:\n", gate_ablation(res["scorer"], g).round(2).to_string())
        p, null = search_permutation_test(book, ev, Settings(), res["best_score"], n=n_perm)
        print(f"whole-search permutation p = {p:.3f} (null median {np.median(null):.2f}, max {null.max():.2f})  [{(time.time()-t0)/60:.1f} min]")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
