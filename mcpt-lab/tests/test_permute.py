"""Permutations must keep the distribution of price changes and destroy only their order."""
import os, sys, unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
from mcptlab import permute_close, permute_ohlc, signflip_ohlc, synthetic


def ohlc(n=600, seed=0):
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    o = np.r_[100, c[:-1]] * np.exp(rng.normal(0, 0.002, n))
    h = np.maximum(o, c) * np.exp(np.abs(rng.normal(0, 0.004, n)))
    l = np.minimum(o, c) * np.exp(-np.abs(rng.normal(0, 0.004, n)))
    return pd.DataFrame(dict(open=o, high=h, low=l, close=c), index=pd.date_range("2020", periods=n, freq="h"))


class TestPermute(unittest.TestCase):
    def test_close_same_returns_different_order(self):
        c = synthetic("noise", n=800, seed=1)
        p = permute_close(c, seed=5)
        r, rp = np.log(c).diff().dropna(), np.log(p).diff().dropna()
        np.testing.assert_allclose(np.sort(r), np.sort(rp), atol=1e-12)     # same moves...
        self.assertFalse(np.allclose(r.values, rp.values))                    # ...different order
        self.assertAlmostEqual(c.iloc[-1], p.iloc[-1], places=6)             # same total change

    def test_start_index_leaves_history_real(self):
        c = synthetic("noise", n=500, seed=2)
        p = permute_close(c, start_index=300, seed=1)
        np.testing.assert_allclose(c.iloc[:301].values, p.iloc[:301].values)

    def test_ohlc_bars_stay_valid(self):
        d = ohlc()
        p = permute_ohlc(d, seed=3)
        self.assertTrue((p.high >= p[["open", "close"]].max(axis=1) - 1e-9).all())
        self.assertTrue((p.low <= p[["open", "close"]].min(axis=1) + 1e-9).all())
        self.assertAlmostEqual(np.log(d.close.iloc[-1]), np.log(p.close.iloc[-1]), places=9)

    def test_ohlc_start_index(self):
        d = ohlc()
        p = permute_ohlc(d, start_index=400, seed=3)
        np.testing.assert_allclose(d.iloc[:401].values, p.iloc[:401].values)

    def test_seed_is_reproducible(self):
        d = ohlc()
        pd.testing.assert_frame_equal(permute_ohlc(d, seed=9), permute_ohlc(d, seed=9))

    def test_signflip_keeps_bar_sizes_in_place(self):
        d = ohlc()
        p = signflip_ohlc(d, seed=4)
        np.testing.assert_allclose(np.log(d.high / d.low).values, np.log(p.high / p.low).values, atol=1e-12)
        np.testing.assert_allclose(np.abs(np.log(d.close / d.open)).values,
                                   np.abs(np.log(p.close / p.open)).values, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
