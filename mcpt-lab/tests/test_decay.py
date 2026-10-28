"""Harness for the publication-decay study (PREREGISTRATION_v6.md, H16-H19).
Synthetic data with a known answer: the pipeline must find planted decay, find none when none
is planted, recover calendar effects, and never use future data."""
import os, sys, unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
from mcptlab.decay import (NMV, Published, faber_timing, market_cost, monday_effect, monthly_cost,
                           monthly_units, pooled_decay, turn_of_month, windows)


def _synthetic_family(decay_oos, decay_post, seed=0, n_strat=9):
    """n_strat strategies, 1950-01..2025-12, IS 1960-1989, pub 1995-06; mean 0.5%/month in IS."""
    rng = np.random.default_rng(seed)
    idx = pd.period_range("1950-01", "2025-12", freq="M")
    spec = Published("x", "x", "x", "1960-01", "1989-12", "1995-06")
    lab = windows(pd.Series(0.0, index=idx), spec)
    mult = lab.map({"IS": 1.0, "OOS": decay_oos, "POST": decay_post}).fillna(1.0).to_numpy()
    common = rng.standard_normal(len(idx))                       # strategies share monthly shocks
    series, labels = {}, {}
    for i in range(n_strat):
        noise = 0.03 * (0.5 * common + np.sqrt(0.75) * rng.standard_normal(len(idx)))
        series[i] = pd.Series(0.005 * mult + noise, index=idx)
        labels[i] = lab
    return series, labels


class TestPooledDecay(unittest.TestCase):
    def test_H16_planted_decay_is_found(self):
        res = pooled_decay(*_synthetic_family(0.75, 0.40))
        self.assertLess(abs(res["b_post"] - (-0.60)), 2 * res["se_post"], res)
        self.assertLess(res["p_post"], 0.05, res)

    def test_H17_no_decay_is_not_found(self):
        hits = 0
        for s in range(5):
            res = pooled_decay(*_synthetic_family(1.0, 1.0, seed=s))
            self.assertLess(abs(res["b_post"]), 3 * res["se_post"], res)
            hits += res["p_post"] < 0.05
        self.assertLessEqual(hits, 1)


def _daily(seed, tom_bps=0.0, monday_bps=0.0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("1990-01-01", "2019-12-31")
    r = pd.Series(0.0003 + 0.01 * rng.standard_normal(len(idx)), index=idx)
    m = idx.to_period("M")
    first3 = pd.Series(idx).groupby(m).cumcount().to_numpy() < 3
    last = np.r_[m[1:] != m[:-1], True]
    r[first3 | last] += tom_bps / 1e4
    r[idx.dayofweek == 0] -= monday_bps / 1e4
    return r


class TestCalendar(unittest.TestCase):
    def test_H18_planted_tom_recovered(self):
        x = turn_of_month(_daily(1, tom_bps=10))
        self.assertLess(abs(x.mean() - 4 * 10 / 1e4), 2 * x.std() / np.sqrt(len(x)), x.mean())
        self.assertGreater(x.mean() / (x.std() / np.sqrt(len(x))), 2)

    def test_H18_planted_monday_recovered(self):
        x = monday_effect(_daily(2, monday_bps=15))
        # ~4.3 Mondays a month, each 15 bps worse than other days
        self.assertLess(abs(x.mean() - 4.3 * 15 / 1e4), 3 * x.std() / np.sqrt(len(x)), x.mean())
        self.assertGreater(x.mean() / (x.std() / np.sqrt(len(x))), 2)

    def test_H18_no_effect_is_zero(self):
        for f in (turn_of_month, monday_effect):
            x = f(_daily(3))
            self.assertLess(abs(x.mean()) / (x.std() / np.sqrt(len(x))), 2.5, f.__name__)


class TestFaber(unittest.TestCase):
    def test_H19_no_lookahead(self):
        rng = np.random.default_rng(0)
        idx = pd.period_range("1970-01", "2000-12", freq="M")
        mkt = pd.Series(0.007 + 0.045 * rng.standard_normal(len(idx)), index=idx)
        rf = pd.Series(0.003, index=idx)
        a = faber_timing(mkt, rf)
        t = pd.Period("1985-06", "M")
        mkt2 = mkt.copy(); mkt2[mkt2.index > t] = -0.2                # rewrite the future
        b = faber_timing(mkt2, rf)
        pd.testing.assert_series_equal(a.loc[:t, "signal"], b.loc[:t, "signal"])
        pd.testing.assert_series_equal(a.loc[:t + 1, "pos"], b.loc[:t + 1, "pos"])


class TestCosts(unittest.TestCase):
    """PREREGISTRATION_v7.md H20-H22."""
    idx = pd.period_range("1970-01", "2019-12", freq="M")

    def test_H20_zero_gross_nets_to_minus_cost(self):
        zero = pd.Series(0.0, index=self.idx)
        self.assertAlmostEqual((zero - monthly_cost("momentum", self.idx)).mean(), -0.0065, places=12)
        tom = zero - monthly_cost("tom", self.idx)                    # 2 units/month x era cost
        self.assertTrue(np.allclose(tom, -2 * market_cost(self.idx)))
        jan = zero - monthly_cost("january", self.idx)                # 4 units in January x 1.63%
        self.assertAlmostEqual(jan.groupby(self.idx.year).sum().iloc[0], -4 * 0.04 / 2.46, places=12)
        self.assertAlmostEqual((zero - monthly_cost("tom", self.idx, mult=2)).sum(), 2 * tom.sum(), places=12)

    def test_H21_unit_counts(self):
        daily = pd.bdate_range("1970-01-01", "2019-12-31")
        mondays = pd.Series(daily.dayofweek == 0, index=daily).groupby(daily.to_period("M")).sum()
        u = monthly_units("monday", self.idx, daily_index=daily)
        self.assertTrue((u == 2 * mondays.reindex(self.idx)).all())
        self.assertTrue((monthly_units("tom", self.idx) == 2).all())
        yearly = lambda k: monthly_units(k, self.idx).groupby(self.idx.year).sum()
        self.assertTrue((yearly("halloween") == 4).all())
        self.assertTrue((yearly("january") == 4).all())
        self.assertEqual(market_cost(pd.PeriodIndex(["1975-04", "1975-05", "1982-05", "2001-01"], freq="M")).tolist(),
                         [0.01, 0.005, 0.0005, 0.0002])

    def test_H22_breakeven(self):
        gross = pd.Series(0.004, index=self.idx)
        units = monthly_units("tom", self.idx)
        be = gross.mean() / units.mean()
        self.assertAlmostEqual((gross - units * be).mean(), 0.0, places=12)


if __name__ == "__main__":
    unittest.main()
