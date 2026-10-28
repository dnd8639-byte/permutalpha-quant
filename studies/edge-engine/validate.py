"""Test the tester: on a market with a planted, gate-able edge the engine must find it and pass;
on pure noise it must not.   python validate.py   (a few minutes)"""
import time
import numpy as np
from core import Market
from engine import (Settings, cost_monte_carlo, holdout_permutation_test, model_quality, prepare, search,
                    search_permutation_test, side_and_beta, deflated_sharpe)
from synthetic import noise_market, regime_market


fmt = lambda x: '-' if x is None else f'{x:.2f}'


def run(label, hourly_list, s, n_search=19, n_hold=99):
    t0 = time.time()
    mk = [Market(f"{label}{i}", h, cost_bps=5, delay_h=1) for i, h in enumerate(hourly_list)]
    P = prepare(mk, s, "dev")
    res = search(P, s)
    q = model_quality(P)
    H = prepare(mk, s, "holdout")
    sb = side_and_beta(H, res["best"])
    print(f"  [{label}] search done in {time.time() - t0:.0f}s; permutation tests running...", flush=True)
    p_s, _ = search_permutation_test(mk, s, res["best_score"], n=n_search)
    p_h, _ = holdout_permutation_test(mk, s, res["best"], sb["sharpe"], n=n_hold)
    from engine import strategy_returns, gate_ablation
    dev_port, _, _ = strategy_returns(P, res["best"])
    dsr = deflated_sharpe(dev_port, res["n_trials"])   # selection bias lives in development
    abl = gate_ablation(H, res["best"])
    print(f"\n{label.upper()}: finalist {res['best']}")
    print(f"  ER model OOS: " + "; ".join(f"corr {v['corr']:.2f}, skill vs persistence {v['skill_vs_persistence']:+.2f}" for v in q.values()))
    print(f"  development best Sharpe {res['best_score']:.2f} after {res['n_trials']} trials; deflated Sharpe {dsr:.2f}; whole-search permutation p = {p_s:.3f}")
    print(f"  HOLDOUT Sharpe {sb['sharpe']:.2f}  p = {p_h:.3f}  beta {sb['beta']:+.2f}  "
          f"long/short Sharpe {fmt(sb['sharpe_long'])} / {fmt(sb['sharpe_short'])}   ({time.time() - t0:.0f}s)")
    print("  gate ablation on holdout (same rule, each gate): " + "  ".join(f"{k} {v:.2f}" for k, v in abl.items()))
    return p_s, p_h


if __name__ == "__main__":
    s = Settings(rounds=8, pop_init=16, children=12, fresh=4)
    run("regime", [regime_market(seed=1), regime_market(seed=2)], s)
    run("noise", [noise_market(seed=1), noise_market(seed=2)], s)
