"""Does the shape of the order book predict how efficiently the NEXT hour trades?   (see PREREGISTRATION.md)

    python study.py status     how much data there is, coverage and gaps. Shows NO results.
    python study.py pilot      exploratory analysis of hours before the lock date (free to look, free to change)
    python study.py freeze     record SHA-256 fingerprints of this file and PREREGISTRATION.md (do at the lock date)
    python study.py confirm    ONE-SHOT confirmatory test on hours after the lock date. Refuses to run before
                               1,000 confirmation hours exist, if the files changed since freeze, or if already run.

Input: data/hourly.csv from extract.py (copied from the Pi).
"""
import hashlib, json, os, sys, time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("MICRO_DATA", os.path.join(HERE, "data"))
RUNS = os.environ.get("MICRO_RUNS", os.path.join(HERE, "runs"))
LOCK = pd.Timestamp(os.environ.get("MICRO_LOCK", "2026-10-20 00:00"))     # pilot = before, confirmation = after
N_CONFIRM = 1000
MIN_TRAIN = 168                                                            # one week of hours before the first prediction
COST = 5e-4                                                                # 5 bps per side (taker fee + slippage)

PRIMARY = ["flow_agree", "book_agree", "abs_tfi", "abs_imb2", "log_depth2", "spread_bps", "er", "log_range"]
PRICE_ONLY = ["er", "log_range", "log_rv", "er_15m"]
ALL = PRIMARY + ["imb10_agree", "log_depth10", "imb25_agree", "log_depth25", "micro_agree", "abs_micro", "flow_agree_15m", "log_dep2_change", "venue_disp_bps",
                 "er_15m", "log_rv", "log_trades_rel", "log_volume_rel", "imb1_agree", "hour_sin", "hour_cos"]


# ---------------------------------------------------------------------------------------------- features
def load():
    H = pd.read_csv(os.path.join(DATA, "hourly.csv"), parse_dates=["hour"]).sort_values("hour").set_index("hour")
    return H[~H.index.duplicated()]


def build(H):
    F = pd.DataFrame(index=H.index)
    s = np.sign(H["ret_bps"])
    F["er"] = H["er"]
    F["log_range"] = np.log(H["range_bps"].clip(lower=0.1))
    F["log_rv"] = np.log(H["rv_bps2"].clip(lower=0.01))
    F["er_15m"] = H["er_15m"]
    F["flow_agree"] = s * H["tfi_1h"]                    # is order flow pushing the way the price already moved?
    F["book_agree"] = s * H["imb2_last5"]                # is resting depth leaning the same way?
    F["abs_tfi"] = H["tfi_1h"].abs()
    F["abs_imb2"] = H["imb2_last5"].abs()
    F["log_depth2"] = np.log(H["dep2_usd"].clip(lower=1))
    F["spread_bps"] = H["spread_bps"]
    F["imb10_agree"] = s * H["imb10_last5"]               # 10 and 25 bps: only once the recorder keeps 500 levels
    F["log_depth10"] = np.log(H["dep10_usd"].clip(lower=1))
    F["imb25_agree"] = s * H["imb25_last5"]
    F["log_depth25"] = np.log(H["dep25_usd"].clip(lower=1))
    F["micro_agree"] = s * H["micro_bps_last5"]
    F["abs_micro"] = H["micro_bps_last5"].abs()
    F["flow_agree_15m"] = np.sign(H["ret_15m_bps"]) * H["tfi_15m"]
    F["log_dep2_change"] = np.log(H["dep2_last5_ratio"].clip(lower=1e-3))
    F["venue_disp_bps"] = H["venue_disp_bps"]
    F["log_trades_rel"] = np.log((H["n_trades"] + 1) / (H["n_trades"].rolling(24, min_periods=12).mean().shift(1) + 1))
    F["log_volume_rel"] = np.log((H["btc_traded"] + 1e-6) / (H["btc_traded"].rolling(24, min_periods=12).mean().shift(1) + 1e-6))
    F["imb1_agree"] = s * H["imb1_last5"]
    hr = H.index.hour
    F["hour_sin"], F["hour_cos"] = np.sin(2 * np.pi * hr / 24), np.cos(2 * np.pi * hr / 24)
    nxt = H.index + pd.Timedelta(hours=1)
    ok = nxt.isin(H.index)                                # the next clock hour must exist (no gap)
    F["er_next"] = np.where(ok, H["er"].reindex(nxt).to_numpy(), np.nan)
    F["ret_next_bps"] = np.where(ok, H["ret_bps"].reindex(nxt).to_numpy(), np.nan)
    F["dir_flow"] = np.sign(H["tfi_1h"])                  # direction rule for the economic check
    return F.replace([np.inf, -np.inf], np.nan)


# ---------------------------------------------------------------------------------------------- models
def ridge():
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(), Ridge(alpha=10.0))


def gbm():
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(max_depth=2, learning_rate=0.05, max_iter=100, min_samples_leaf=20, random_state=0)


def walk_forward(F, cols, make, start, end, allow_nan=False):
    """Predict every hour in [start, end) with a model refit each UTC day on all rows strictly before that day.
    A training row at hour h has its target at h+1, which is complete when the day's first prediction is made."""
    d = F.dropna(subset=["er_next"] + ([] if allow_nan else cols))      # boosted trees can handle missing values
    out = []
    for day in pd.date_range(start.floor("D"), end, freq="D"):
        tr = d[d.index < day]
        te = d[(d.index >= max(day, start)) & (d.index < min(day + pd.Timedelta(days=1), end))]
        if len(te) == 0 or len(tr) < MIN_TRAIN:
            continue
        m = make().fit(tr[cols].to_numpy(float), tr["er_next"].to_numpy(float))
        lo, hi = np.quantile(m.predict(tr[cols].to_numpy(float)), [1 / 3, 2 / 3])
        p = m.predict(te[cols].to_numpy(float))
        out.append(pd.DataFrame({"pred": p, "tier": np.where(p < lo, 0, np.where(p < hi, 1, 2)), "y": te["er_next"],
                                 "ret_next_bps": te["ret_next_bps"], "dir": te["dir_flow"]}, index=te.index))
    return pd.concat(out) if out else pd.DataFrame(columns=["pred", "tier", "y", "ret_next_bps", "dir"])


# ---------------------------------------------------------------------------------------------- statistics
def spearman(a, b):
    return float(pd.Series(a).rank().corr(pd.Series(b).rank()))


def day_shuffle_p(O, n=2000, seed=0):
    """Null: pair each day's realized efficiencies with the predictions of a DIFFERENT random day, hour by hour.
    Keeps the time-of-day pattern and each day's internal structure; destroys only the link the test is about."""
    rng = np.random.default_rng(seed)
    P = O["pred"].groupby([O.index.floor("D"), O.index.hour]).first().unstack()
    Y = O["y"].groupby([O.index.floor("D"), O.index.hour]).first().unstack().reindex(P.index)
    real = spearman(O["pred"], O["y"])
    yv, pv = Y.to_numpy(), P.to_numpy()
    null = np.empty(n)
    for k in range(n):
        pp = pv[rng.permutation(len(pv))]
        m = ~np.isnan(pp) & ~np.isnan(yv)
        null[k] = spearman(pp[m], yv[m])
    return real, float((1 + (null >= real).sum()) / (n + 1)), null


def economics(O):
    """Follow the hour's order-flow direction for the next hour; compare all hours vs predicted-efficient hours."""
    pos_all = O["dir"].to_numpy()
    pos_gate = np.where(O["tier"] == 2, pos_all, 0.0)
    def run(pos):
        r = pos * O["ret_next_bps"].to_numpy() / 1e4
        turns = np.abs(np.diff(np.concatenate([[0.0], pos])))
        net = r - turns * COST
        return net
    res = {}
    for name, pos in (("every hour", pos_all), ("predicted-efficient hours only", pos_gate)):
        x = run(pos)
        res[name] = dict(mean_bps=float(np.nanmean(x) * 1e4), sharpe=float(np.nanmean(x) / np.nanstd(x) * np.sqrt(24 * 365)) if np.nanstd(x) > 0 else 0.0,
                         hours_in_market=int((pos != 0).sum()))
    return res


def needed_strength(F, start, end, seed=0):
    """How good would a forecast have to be for the gated flow-following rule to clear Sharpe 1 after costs?"""
    d = F[(F.index >= start) & (F.index < end)].dropna(subset=["er_next", "ret_next_bps", "dir_flow"])
    if len(d) < 100:
        return {}
    rng = np.random.default_rng(seed)
    z = (d["er_next"] - d["er_next"].mean()) / d["er_next"].std()
    out = {}
    for rho in (0.1, 0.2, 0.3, 0.5, 1.0):
        s = rho * z + np.sqrt(1 - rho ** 2) * rng.standard_normal(len(z))
        tier = pd.qcut(s, 3, labels=False)
        O = pd.DataFrame({"tier": tier, "dir": d["dir_flow"], "ret_next_bps": d["ret_next_bps"]}, index=d.index)
        out[rho] = economics(O)["predicted-efficient hours only"]["sharpe"]
    return out


def tiers(O):
    return O.groupby("tier")["y"].agg(["count", "mean"]).reindex([0, 1, 2]).rename(index={0: "low", 1: "mid", 2: "high"})


# ---------------------------------------------------------------------------------------------- commands
def status(H):
    gaps = (H.index.to_series().diff() > pd.Timedelta(hours=1)).sum()
    pilot, conf = (H.index < LOCK).sum(), (H.index >= LOCK).sum()
    print(f"{len(H)} hours: {H.index.min()} -> {H.index.max()} UTC, {gaps} gaps")
    print(f"venues present (median per hour): {H['venues'].median():.0f}   snapshots per hour (median): {H['n_snaps'].median():.0f}/360")
    print(f"pilot hours (before {LOCK}): {pilot}    confirmation hours: {conf} / {N_CONFIRM}")
    if conf:
        recent = H[H.index >= H.index.max() - pd.Timedelta(days=7)]
        rate = len(recent) / 7
        left = max(0, N_CONFIRM - conf)
        print(f"recent rate {rate:.1f} hours/day -> {N_CONFIRM} confirmation hours around "
              f"{(H.index.max() + pd.Timedelta(days=left / max(rate, 1e-9))).date()}")


def pilot(H):
    H = H[H.index < LOCK]                                  # the pilot never reads a confirmation hour, not even as a target
    F = build(H)
    start = F.index.min() + pd.Timedelta(hours=MIN_TRAIN)
    end = min(LOCK, F.index.max() + pd.Timedelta(hours=1))
    print(f"PILOT (exploratory): predictions for {start} -> {end} UTC, each day predicted by a model trained on earlier hours\n")
    rows = {}
    for name, cols, make in (("primary: order book, ridge", PRIMARY, ridge), ("all features, boosted trees", ALL, gbm),
                             ("price only, ridge", PRICE_ONLY, ridge)):
        O = walk_forward(F, cols, make, start, end, allow_nan=make is gbm)
        if len(O) < 48:
            print(f"{name}: only {len(O)} predicted hours so far; need more data"); continue
        rho, p, _ = day_shuffle_p(O, n=1000)
        t = tiers(O)
        rows[name] = O
        print(f"{name:30s} n={len(O):4d}  Spearman {rho:+.3f}  day-shuffle p {p:.3f}  "
              f"next-hour ER by tier {t['mean'].round(3).tolist()}")
    if "primary: order book, ridge" in rows:
        print("\nEconomic check (flow-following, 5 bps per side), primary model:")
        for k, v in economics(rows["primary: order book, ridge"]).items():
            print(f"  {k:32s} Sharpe {v['sharpe']:+.2f}   mean {v['mean_bps']:+.2f} bps/hour   hours in market {v['hours_in_market']}")
        ns = needed_strength(F, start, end)
        if ns:
            print("\nHow strong must the forecast be? gated Sharpe if the forecast had correlation rho with the truth:")
            print("  " + "   ".join(f"rho {k:.1f}: {v:+.2f}" for k, v in ns.items()))
    # simple one-variable looks: what moves with next-hour efficiency in the pilot?
    d = F[F.index < end].dropna(subset=["er_next"])
    print("\nOne feature at a time (pilot, Spearman with next-hour ER):")
    print("  " + "  ".join(f"{c} {spearman(d[c], d['er_next']):+.2f}" for c in ALL if d[c].notna().sum() > 30))


def fingerprints():
    h = {}
    for f in ("study.py", "PREREGISTRATION.md"):
        h[f] = hashlib.sha256(open(os.path.join(HERE, f), "rb").read()).hexdigest()
    return h


def freeze():
    os.makedirs(RUNS, exist_ok=True)
    path = os.path.join(RUNS, "freeze.json")
    if os.path.exists(path):
        sys.exit(f"Already frozen: {json.load(open(path))}")
    rec = dict(frozen_at_utc=time.strftime("%Y-%m-%d %H:%M", time.gmtime()), lock=str(LOCK), **fingerprints())
    json.dump(rec, open(path, "w"), indent=1)
    print(json.dumps(rec, indent=1))
    print("\nNow commit and push these files to GitHub: the commit time is public proof of when the rules were fixed.")


def confirm(H):
    path, fz = os.path.join(RUNS, "confirm.json"), os.path.join(RUNS, "freeze.json")
    if os.path.exists(path):
        sys.exit(f"The confirmation test has already been run: {json.load(open(path))['ran_at_utc']}. It is one-shot.")
    if not os.path.exists(fz):
        sys.exit("Run `python study.py freeze` first (at the lock date).")
    if fingerprints() != {k: v for k, v in json.load(open(fz)).items() if k in ("study.py", "PREREGISTRATION.md")}:
        sys.exit("study.py or PREREGISTRATION.md changed after the freeze. The confirmatory test must use the frozen rules.")
    if pd.Timestamp(json.load(open(fz))["lock"]) != LOCK:
        sys.exit("The lock date differs from the one recorded at the freeze.")
    F = build(H)
    n = int(((F.index >= LOCK) & F["er_next"].notna()).sum())
    if n < N_CONFIRM:
        sys.exit(f"Only {n} confirmation hours with a next hour; the test opens at {N_CONFIRM}. Nothing was computed.")
    rec = dict(ran_at_utc=time.strftime("%Y-%m-%d %H:%M", time.gmtime()))
    json.dump(rec, open(path, "w"))                                         # lock FIRST, then look
    O = walk_forward(F, PRIMARY, ridge, LOCK, F.index.max() + pd.Timedelta(hours=1))
    rho, p, null = day_shuffle_p(O, n=10000, seed=1)
    passed = bool(rho >= 0.10 and p < 0.01)
    t = tiers(O)
    sec = {}
    for name, cols, make in (("all features, boosted trees", ALL, gbm), ("price only, ridge", PRICE_ONLY, ridge)):
        Q = walk_forward(F, cols, make, LOCK, F.index.max() + pd.Timedelta(hours=1), allow_nan=make is gbm)
        sec[name] = spearman(Q["pred"], Q["y"])
    econ = economics(O)
    rec.update(n_hours=len(O), spearman=rho, p=p, passed=passed, tiers=t["mean"].round(4).tolist(), secondary=sec, economics=econ)
    json.dump(rec, open(path, "w"), indent=1, default=float)
    print(f"CONFIRMATORY TEST, {len(O)} hours after {LOCK}:")
    print(f"  primary Spearman {rho:+.3f}, day-shuffle p = {p:.4f}  ->  {'PASS' if passed else 'FAIL'} (needs >= 0.10 and p < 0.01)")
    print(f"  next-hour ER by predicted tier (low/mid/high): {t['mean'].round(3).tolist()}")
    print(f"  secondary: {json.dumps({k: round(v, 3) for k, v in sec.items()})}")
    for k, v in econ.items():
        print(f"  economics, {k}: Sharpe {v['sharpe']:+.2f}, {v['mean_bps']:+.2f} bps/hour")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    if cmd == "freeze":
        freeze()
    else:
        H = load()
        {"status": status, "pilot": pilot, "confirm": confirm}.get(cmd, lambda _: sys.exit(__doc__))(H)
