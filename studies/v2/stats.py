"""v2 statistics: bootstrap Sharpe CI, Newey-West t, Deflated Sharpe Ratio.
Pure numpy/scipy so it runs anywhere (Mac, Pi, cloud)."""
import numpy as np
from scipy.stats import norm, skew as _skew, kurtosis as _kurt

ANN = 12
EULER_GAMMA = 0.5772156649015329


def sharpe(r, ann=ANN):
    r = np.asarray(r, float)
    r = r[np.isfinite(r)]
    sd = r.std(ddof=1)
    return np.nan if sd == 0 else r.mean() / sd * np.sqrt(ann)


def stationary_bootstrap_idx(T, n_boot, mean_block, rng):
    """Politis-Romano stationary bootstrap indices, shape (n_boot, T)."""
    p = 1.0 / mean_block
    idx = np.empty((n_boot, T), dtype=np.int64)
    idx[:, 0] = rng.integers(0, T, n_boot)
    new_block = rng.random((n_boot, T)) < p
    starts = rng.integers(0, T, (n_boot, T))
    for t in range(1, T):
        idx[:, t] = np.where(new_block[:, t], starts[:, t], (idx[:, t - 1] + 1) % T)
    return idx


def bootstrap_sharpe_ci(r, n_boot=10_000, mean_block=12, alpha=0.05, seed=42, ann=ANN):
    r = np.asarray(r, float)
    r = r[np.isfinite(r)]
    rng = np.random.default_rng(seed)
    idx = stationary_bootstrap_idx(len(r), n_boot, mean_block, rng)
    s = r[idx]
    sr = s.mean(1) / s.std(1, ddof=1) * np.sqrt(ann)
    lo, hi = np.percentile(sr, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return lo, hi


def newey_west_t(r, lags=12):
    r = np.asarray(r, float)
    r = r[np.isfinite(r)]
    T = len(r)
    e = r - r.mean()
    s = e @ e / T
    for L in range(1, lags + 1):
        w = 1 - L / (lags + 1)
        s += 2 * w * (e[L:] @ e[:-L]) / T
    return r.mean() / np.sqrt(s / T)


def psr(sr_hat, sr_star, T, sk, kurt_raw):
    """Probabilistic Sharpe Ratio. SRs are PER-PERIOD (not annualized); kurt is raw (normal=3)."""
    den = np.sqrt(1 - sk * sr_hat + (kurt_raw - 1) / 4 * sr_hat ** 2)
    return norm.cdf((sr_hat - sr_star) * np.sqrt(T - 1) / den)


def expected_max_sr(var_sr, n_trials):
    """Expected max of N per-period Sharpes under the null (Bailey & Lopez de Prado 2014)."""
    g = EULER_GAMMA
    return np.sqrt(var_sr) * ((1 - g) * norm.ppf(1 - 1 / n_trials)
                              + g * norm.ppf(1 - 1 / (n_trials * np.e)))


def deflated_sharpe(r, trial_sharpes_ann, ann=ANN):
    """DSR for return series r, given the annualized Sharpes of ALL trials (incl. this one)."""
    r = np.asarray(r, float)
    r = r[np.isfinite(r)]
    sr_hat = r.mean() / r.std(ddof=1)
    trials = np.asarray(trial_sharpes_ann, float) / np.sqrt(ann)
    sr0 = expected_max_sr(trials.var(ddof=1), len(trials))
    return psr(sr_hat, sr0, len(r), _skew(r), _kurt(r, fisher=False)), sr0 * np.sqrt(ann)


def full_report(r, trial_sharpes_ann, seed=42):
    lo, hi = bootstrap_sharpe_ci(r, seed=seed)
    dsr, sr0 = deflated_sharpe(r, trial_sharpes_ann)
    rr = np.asarray(r, float)
    eq = np.cumprod(1 + rr)
    return {
        "months": len(rr), "sharpe": sharpe(rr), "ci_lo": lo, "ci_hi": hi,
        "nw_t": newey_west_t(rr), "dsr": dsr, "sr0_ann": sr0,
        "ann_ret": rr.mean() * ANN, "ann_vol": rr.std(ddof=1) * np.sqrt(ANN),
        "max_dd": (eq / np.maximum.accumulate(eq) - 1).min(),
        "skew": _skew(rr), "G1": lo > 0, "G2": dsr > 0.95,
    }
