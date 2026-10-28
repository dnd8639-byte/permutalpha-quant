"""Project 10: information diffusion. How hard does scheduled news hit Bitcoin, and how fast is it absorbed?

Model    Two streams of timestamps: NEWS (scheduled US releases) and PRICE EVENTS (unusually large
         one-minute moves). The rate of price events is a Hawkes process excited by both:
            lambda_P(t) = mu + sum_news a_N * exp(-b_N (t - t_news)) + sum_price a_P * exp(-b_P (t - t_price))
         a_N = jump in activity when news lands; half-life ln2 / b_N = how fast it is absorbed.
         The second sum is the market reacting to itself ("echo").
Split    Every price event is then attributed to: background (mu), news, or echo.
Data     News: FOMC statements (2:00 pm New York) and CPI releases (8:30 am New York), dates from
         federalreserve.gov and bls.gov. Price: BTC/USDT one-minute bars, 12 hours either side.
         A price event = a minute whose |return| is over 4x the typical (median) size in the 12
         hours BEFORE the news, so the threshold never uses post-news data.
Checks   (1) simulate the model with known parameters and recover them;
         (2) placebo: pretend the news was 6 hours earlier. The news effect must vanish;
         (3) likelihood-ratio test of the model with news vs without.
Real     needs  python download_data.py news
Limits   Both releases land at the start of the busiest US trading hours, so the hours of raised
         activity AFTER the spike may be ordinary time-of-day seasonality (see project 2), not news.
         Only the jump at the exact release minute is cleanly attributable to the news. The proper
         control is the same clock times on days without a release. Binance.US is also a quiet
         venue: about 4 in 10 one-minute bars have no trades.
Run:  python p10_news_diffusion.py
"""
import os
import numpy as np
import pandas as pd
from common import A, B, C, FIG, HERE, INK, MUTED, nelder_mead, plt, style

FOMC = ["2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12", "2024-07-31", "2024-09-18", "2024-11-07", "2024-12-18",
        "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18", "2025-07-30", "2025-09-17", "2025-10-29", "2025-12-10",
        "2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17", "2026-07-29", "2026-09-16"]
CPI = ["2025-12-18", "2026-01-13", "2026-02-13", "2026-03-11", "2026-04-10", "2026-05-12", "2026-06-10", "2026-07-14",
       "2026-08-12", "2026-09-11"]
HALF = 12 * 3600.0                                             # seconds each side of the news


def events_utc():
    out = [(pd.Timestamp(d + " 14:00", tz="America/New_York").tz_convert("UTC"), "FOMC") for d in FOMC]
    out += [(pd.Timestamp(d + " 08:30", tz="America/New_York").tz_convert("UTC"), "CPI") for d in CPI]
    return sorted(out)


def window_ll(params, price, news, horizon, use_news=True):
    """Log-likelihood of one window's price-event times (seconds from window start)."""
    mu, aN, bN, aP, bP = params
    lam = np.full(len(price), mu)
    comp = mu * horizon
    if use_news:
        dt = price - news
        lam += np.where(dt > 0, aN * np.exp(-bN * np.clip(dt, 0, None)), 0.0)
        comp += aN / bN * (1 - np.exp(-bN * (horizon - news)))
    Aexc, decay = np.zeros(len(price)), np.exp(-bP * np.diff(price))
    for i in range(1, len(price)):
        Aexc[i] = decay[i - 1] * (1 + Aexc[i - 1])
    lam += aP * Aexc
    comp += aP / bP * np.sum(1 - np.exp(-bP * (horizon - price)))
    return np.sum(np.log(lam)) - comp


def fit(windows, use_news=True):
    """windows = list of (price_times, news_time). Pooled maximum likelihood over all windows."""
    rate = sum(len(w[0]) for w in windows) / (len(windows) * 2 * HALF)

    def nll(x):
        p = np.exp(x)
        if p[3] >= p[4]:
            return 1e12
        return -sum(window_ll(p, pr, nw, 2 * HALF, use_news) for pr, nw in windows)
    best = None
    for hl_news in (300.0, 1800.0):
        x0 = np.log([rate / 2, rate * 20, np.log(2) / hl_news, 0.005, 0.01])
        x, f = nelder_mead(nll, x0, iters=1500)
        if best is None or f < best[1]:
            best = (x, f)
    mu, aN, bN, aP, bP = np.exp(best[0])
    n_events = sum(len(w[0]) for w in windows)
    from_bg = mu * 2 * HALF * len(windows)
    from_news = aN / bN * len(windows) if use_news else 0.0
    return dict(mu=mu, aN=aN, bN=bN, aP=aP, bP=bP, ll=-best[1], half_life_min=np.log(2) / bN / 60,
                jump=aN / mu, echo=aP / bP, share_bg=from_bg / n_events, share_news=from_news / n_events,
                share_echo=1 - (from_bg + from_news) / n_events, n_events=n_events)


def simulate_window(mu, aN, bN, aP, bP, rng):
    news, t, out = HALF, 0.0, []
    while True:
        exc = sum(aP * np.exp(-bP * (t - s)) for s in out[-50:])
        lam_bar = mu + exc + aN                                # generous upper bound on the rate
        t += rng.exponential(1 / lam_bar)
        if t > 2 * HALF:
            return np.array(out), news
        lam = mu + (aN * np.exp(-bN * (t - news)) if t > news else 0.0) + sum(aP * np.exp(-bP * (t - s)) for s in out[-50:])
        if rng.random() * lam_bar <= lam:
            out.append(t)


def show(label, f):
    print(f"  {label:26s} news jump {f['jump']:6.1f}x background   half-life {f['half_life_min']:6.1f} min   echo n {f['echo']:.2f}   "
          f"events: background {f['share_bg']:.0%}, news {f['share_news']:.0%}, echo {f['share_echo']:.0%}")


def price_events(bars, news_ts, rng, mult=4.0):
    """bars: one-minute closes indexed by UTC time. Returns (event times in seconds, |returns| by minute offset)."""
    w = bars.loc[news_ts - pd.Timedelta(seconds=HALF):news_ts + pd.Timedelta(seconds=HALF)]
    if len(w) < 1200:
        return None
    r = np.log(w).diff().dropna()
    base = r.loc[:news_ts - pd.Timedelta(minutes=1)].abs()
    thr = mult * base[base > 0].median()
    ev = r[r.abs() > thr]
    secs = (ev.index - (news_ts - pd.Timedelta(seconds=HALF))).total_seconds().to_numpy() - rng.uniform(0, 60, len(ev))
    offs = ((r.index - news_ts).total_seconds() // 60).astype(int)
    return np.sort(np.clip(secs, 1e-3, 2 * HALF - 1e-3)), pd.Series(r.abs().to_numpy() / base.mean(), index=offs)


if __name__ == "__main__":
    rng = np.random.default_rng(5)
    true = dict(mu=0.0008, aN=0.02, bN=np.log(2) / 900, aP=0.004, bP=0.01)
    sim = [simulate_window(**true, rng=rng) for _ in range(30)]
    print("CHECK (1): 30 simulated news windows. True: jump 25.0x, half-life 15.0 min, echo n 0.40")
    show("fitted", fit(sim))
    print("CHECK (2): same data, news time moved 6 hours early (placebo)")
    show("placebo", fit([(p, n - 6 * 3600) for p, n in sim]))

    path = os.path.join(HERE, "data", "btc_1m_news_windows.csv")
    real = None
    if os.path.exists(path):
        bars = pd.read_csv(path); bars.index = pd.to_datetime(bars.pop("ts"), unit="ms", utc=True); bars = bars["close"].sort_index()
        bars = bars[~bars.index.duplicated()]
        wins, profile, kinds = [], [], []
        for ts, kind in events_utc():
            got = price_events(bars, ts, rng)
            if got is not None:
                wins.append((got[0], HALF)); profile.append(got[1]); kinds.append(kind)
        print(f"\nREAL: {len(wins)} news windows ({kinds.count('FOMC')} FOMC, {kinds.count('CPI')} CPI), {sum(len(w[0]) for w in wins)} price events")
        f1, f0 = fit(wins), fit(wins, use_news=False)
        show("all news", f1)
        lr = 2 * (f1["ll"] - f0["ll"])
        print(f"  CHECK (3): likelihood ratio, news model vs no-news model = {lr:.1f} (above 9.2 means p < 0.01 with 2 extra parameters)")
        show("placebo (6h early)", fit([(p, n - 6 * 3600) for p, n in wins]))
        for k in ("FOMC", "CPI"):
            sub = [w for w, kk in zip(wins, kinds) if kk == k]
            if len(sub) >= 5:
                show(k + " only", fit(sub))
        real = (pd.concat(profile, axis=1).mean(axis=1).sort_index(), f1)
    else:
        print("\nREAL data not found: run  python download_data.py news")

    p = plt()
    fig, axes = p.subplots(1, 2 if real else 1, figsize=(12 if real else 6.5, 3.8)); axes = np.atleast_1d(axes)
    mins = np.arange(-120, 241)
    counts = np.zeros(len(mins))
    for pr, nw in sim:
        idx = ((pr - nw) // 60).astype(int)
        for i in idx[(idx >= -120) & (idx <= 240)]:
            counts[i + 120] += 1
    fs = fit(sim)
    axes[0].bar(mins, counts / len(sim), width=1, color=B, label="simulated price events per minute")
    axes[0].plot(mins, 60 * (fs["mu"] / (1 - fs["echo"]) + np.where(mins >= 0, fs["aN"] * np.exp(-fs["bN"] * mins * 60), 0) / (1 - fs["echo"])), color=A, lw=1.5, label="fitted model")
    axes[0].axvline(0, color=C, lw=1); axes[0].set_xlabel("minutes from the news", fontsize=8, color=MUTED)
    axes[0].set_title("Check: known parameters are recovered", loc="left", fontsize=10, color=INK); axes[0].legend(frameon=False, fontsize=8); style(axes[0])
    if real:
        prof, f1 = real
        pr_ = prof.loc[-120:240]
        axes[1].bar(pr_.index, pr_.to_numpy(), width=1, color=B, label="average |1-minute move| (1 = normal)")
        axes[1].axvline(0, color=C, lw=1); axes[1].set_xlabel("minutes from the news", fontsize=8, color=MUTED)
        axes[1].set_title(f"Real BTC around FOMC and CPI: a spike (half-life {f1['half_life_min']:.0f} min), then hours of busier trading", loc="left", fontsize=10, color=INK)
        axes[1].legend(frameon=False, fontsize=8); style(axes[1])
    fig.tight_layout(); fig.savefig(f"{FIG}/p10_news_diffusion.png", dpi=130)
