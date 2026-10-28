"""Do famous published strategies lose their edge after publication?  (PREREGISTRATION_v6.md)

Ten published strategies, each with its published rule and no tuning, scored in three windows:
the authors' own sample (IS), after the sample but before publication (OOS), and after publication (POST).

    python data/download_french.py                 (once)
    python examples/06_publication_decay.py        (~1 minute)

Writes results/decay/*.csv, logs to results/LEDGER.csv and saves nulls to results/nulls/.
"""
import os
import sys

import _path  # noqa: F401
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
os.environ.setdefault("MCPT_LEDGER", os.path.join(RESULTS, "LEDGER.csv"))
os.environ.setdefault("MCPT_RUNS", os.path.join(RESULTS, "decay", "runs"))

from mcptlab import log_test, permutation_test, save_null  # noqa: E402
from mcptlab.decay import (Published, faber_timing, halloween, january_only, load_french,  # noqa: E402
                           max_drawdown, mean_stat, monday_effect, pooled_decay, signflip_series,
                           turn_of_month, window_stats, windows)

# ---- the preregistered list (section 3). Do not edit after results. -----------------------------
SPECS = [
    Published("size", "Size (SMB)", "Banz 1981", "1936-01", "1975-12", "1981-03"),
    Published("value", "Value (HML)", "Fama-French 1992", "1963-07", "1990-12", "1992-06"),
    Published("momentum", "Momentum (UMD)", "Jegadeesh-Titman 1993", "1965-01", "1989-12", "1993-03"),
    Published("reversal", "Short-term reversal", "Jegadeesh 1990", "1934-01", "1987-12", "1990-07"),
    Published("profitability", "Profitability (RMW)", "Novy-Marx 2013", "1963-07", "2010-12", "2013-04"),
    Published("january", "Small-firm January", "Keim 1983", "1963-01", "1979-12", "1983-06"),
    Published("monday", "Weekend / Monday", "French 1980", "1953-01", "1977-12", "1980-03"),
    Published("tom", "Turn-of-the-month", "Lakonishok-Smidt 1988", "1926-07", "1986-12", "1988-12"),
    Published("halloween", "Halloween / Sell in May", "Bouman-Jacobsen 2002", "1970-01", "1998-08", "2002-12"),
]
FABER = Published("faber", "10-month MA timing", "Faber 2007", "1973-01", "2005-12", "2007-03", pooled=False)
N_PERM = 10_000


class _Named:                       # what log_test needs from a "strategy"
    def __init__(self, name):
        self.name, self.grid = name, [None]


def build_series():
    ff3, daily = load_french("ff3_monthly"), load_french("ff3_daily")
    mom, strev, ff5 = load_french("mom_monthly"), load_french("strev_monthly"), load_french("ff5_monthly")
    end = min(x.index[-1] for x in (ff3, mom, strev, ff5))
    s = {"size": ff3["SMB"], "value": ff3["HML"], "momentum": mom["Mom"], "reversal": strev["ST_Rev"],
         "profitability": ff5["RMW"], "january": january_only(ff3["SMB"]),
         "monday": monday_effect(daily["Mkt-RF"]), "tom": turn_of_month(daily["Mkt-RF"]),
         "halloween": halloween(ff3["Mkt-RF"])}
    s = {k: v.loc[:end].dropna() for k, v in s.items()}
    return s, ff3.loc[:end], end


def main():
    series, ff3, end = build_series()
    out = os.path.join(RESULTS, "decay")
    os.makedirs(out, exist_ok=True)
    print(f"Data: Kenneth French Data Library, through {end}\n")

    # ---- per-strategy windows + replication check (section 4)
    rows, labels, included = [], {}, {}
    for sp in SPECS:
        r = series[sp.key]
        lab = windows(r, sp)
        labels[sp.key] = lab
        st = window_stats(r, lab)
        ok = st.loc["IS", "mean_pct"] > 0 and st.loc["IS", "t"] >= 1.5
        if ok:
            included[sp.key] = r
        for w in ("IS", "OOS", "POST"):
            rows.append(dict(strategy=sp.label, paper=sp.paper, window=w, **st.loc[w].to_dict(),
                             replicated_IS=ok))
    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(out, "window_stats.csv"), index=False)

    print(f"{'strategy':26s} {'paper':22s} {'IS %/mo':>8s} {'t':>5s} {'OOS':>7s} {'POST':>7s} {'t':>5s} "
          f"{'POST/IS':>8s}")
    for sp in SPECS:
        t = table[table.strategy == sp.label].set_index("window")
        ratio = t.loc["POST", "mean_pct"] / t.loc["IS", "mean_pct"]
        flag = "" if sp.key in included else "  <- did not replicate in-sample; excluded"
        print(f"{sp.label:26s} {sp.paper:22s} {t.loc['IS','mean_pct']:8.3f} {t.loc['IS','t']:5.1f} "
              f"{t.loc['OOS','mean_pct']:7.3f} {t.loc['POST','mean_pct']:7.3f} {t.loc['POST','t']:5.1f} "
              f"{ratio:8.0%}{flag}")

    # ---- primary test H1 + secondary H2 (section 5)
    if not included:
        print("\nNo strategy replicated in-sample; the pooled test cannot run.")
        return 1
    res = pooled_decay(included, labels)
    print(f"\nPOOLED ({res['n_strategies']} strategies, {res['n_obs']} strategy-months, "
          f"SE clustered by {res['n_months']} months)")
    print(f"  OOS  vs IS: {res['b_oos']:+.3f}  (se {res['se_oos']:.3f})  -> OOS returns are "
          f"{1 + res['b_oos']:.0%} of IS")
    print(f"  POST vs IS: {res['b_post']:+.3f}  (se {res['se_post']:.3f})  -> POST returns are "
          f"{1 + res['b_post']:.0%} of IS   one-sided p = {res['p_post']:.4f}")
    print(f"  H1 (b_POST < 0): {'PASS -- decay confirmed' if res['p_post'] < 0.05 else 'FAIL -- no significant decay'}")
    print(f"  H2 (POST below OOS, publication effect beyond statistical bias): diff "
          f"{res['b_post_minus_oos']:+.3f} (se {res['se_diff']:.3f}), p = {res['p_diff']:.4f}")
    pd.Series(res).to_csv(os.path.join(out, "pooled_primary.csv"), header=["value"])

    first = next(iter(included.values()))
    log_test(dict(test="pooled_decay_regression", real=res["b_post"], p=res["p_post"],
                  best_param=f"b_oos={res['b_oos']:.3f}"),
             _Named("publication_decay"), "FF_US_equity", first.to_timestamp(), 0.0, "b_post", 0,
             f"PREREG v6 H1 primary (trial 26); {res['n_strategies']} strategies; SE clustered by month")

    # ---- robustness (reported only)
    rob = {}
    no_cal = {k: v for k, v in included.items() if k not in ("january", "monday", "tom", "halloween")}
    if no_cal:
        rob["drop_calendar"] = pooled_decay(no_cal, labels)
    shifted = {k: windows(series[k], next(s for s in SPECS if s.key == k), post_shift_months=24) for k in included}
    rob["post_2y_earlier"] = pooled_decay(included, shifted)
    print("\nRobustness (reported, not gated):")
    for name, r in rob.items():
        print(f"  {name:16s} b_POST {r['b_post']:+.3f} (se {r['se_post']:.3f}), p = {r['p_post']:.4f}; "
              f"b_OOS {r['b_oos']:+.3f}")
    pd.DataFrame(rob).to_csv(os.path.join(out, "pooled_robustness.csv"))

    # ---- per-strategy sign-flip test on POST returns: is anything left? (logged)
    print(f"\nIs anything left after publication? Sign-flip test on POST months ({N_PERM:,} draws):")
    nulls_dir = os.path.join(RESULTS, "nulls")
    os.makedirs(nulls_dir, exist_ok=True)
    num = 13
    for sp in SPECS:
        post = series[sp.key][labels[sp.key] == "POST"]
        pt = permutation_test(mean_stat, post.to_numpy(), n_perm=N_PERM, null=signflip_series, procs=1)
        pt["real"] = float(pt["real"])
        print(f"  {sp.label:26s} POST mean {100 * post.mean():+.3f}%/mo   p = {pt['p']:.4f}")
        save_null(pt, os.path.join(nulls_dir, f"{num:02d}_decay_{sp.key}_post_signflip.csv"),
                  title=f"{sp.label} after publication ({sp.paper}) · sign-flip")
        log_test(dict(test="post_pub_signflip", real=pt["real"], p=pt["p"]), _Named(sp.key),
                 "FF_US_equity", post.to_timestamp(), 0.0, "mean", N_PERM,
                 f"PREREG v6 descriptive; {sp.paper}; POST from {post.index[0]}")
        num += 1

    # ---- Faber timing (section 3.2; not pooled)
    fb = faber_timing(ff3["Mkt-RF"] + ff3["RF"], ff3["RF"])
    lab = windows(fb["timed"], FABER)
    frows = []
    for w in ("IS", "OOS", "POST"):
        x = fb[lab == w]
        ex_t, ex_b = x["timed"] - x["rf"], x["bh"] - x["rf"]
        sh = lambda e: e.mean() / e.std() * np.sqrt(12)
        frows.append(dict(window=w, start=str(x.index[0]), end=str(x.index[-1]), months=len(x),
                          sharpe_timed=sh(ex_t), sharpe_buyhold=sh(ex_b), sharpe_diff=sh(ex_t) - sh(ex_b),
                          mdd_timed=max_drawdown(x["timed"]), mdd_buyhold=max_drawdown(x["bh"]),
                          timed_minus_bh_pct=100 * (x["timed"] - x["bh"]).mean(),
                          pct_months_invested=100 * x["pos"].mean()))
    ftab = pd.DataFrame(frows)
    ftab.to_csv(os.path.join(out, "faber.csv"), index=False)
    print("\nFaber 10-month moving-average timing (10 bps per switch) vs buy-and-hold:")
    for _, f in ftab.iterrows():
        print(f"  {f.window:4s} {f.start}..{f.end}  Sharpe {f.sharpe_timed:.2f} vs {f.sharpe_buyhold:.2f} "
              f"(diff {f.sharpe_diff:+.2f})   max DD {f.mdd_timed:.0%} vs {f.mdd_buyhold:.0%}   "
              f"return diff {f.timed_minus_bh_pct:+.2f}%/mo   invested {f.pct_months_invested:.0f}%")
    print(f"\nTables in {out}")


if __name__ == "__main__":
    sys.exit(main())
