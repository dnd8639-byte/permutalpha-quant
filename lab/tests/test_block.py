"""The block-bootstrap null.   python -m unittest tests.test_block"""
import unittest

import numpy as np


def abs_autocorr(close, lag=1):
    a = np.abs(np.diff(np.log(close)))
    return np.corrcoef(a[lag:], a[:-lag])[0, 1]


class TestBlockPermutation(unittest.TestCase):
    def test_is_a_permutation_of_whole_ordered_blocks(self):
        from oql.permute import block_permutation
        rng = np.random.default_rng(1)
        for n, L in [(10, 4), (1000, 20), (999, 7), (5, 20)]:
            p = block_permutation(n, L, rng)
            self.assertTrue(np.array_equal(np.sort(p), np.arange(n)))                   # each index exactly once
            runs = np.split(p, np.where(np.diff(p) != 1)[0] + 1)                       # maximal runs of +1 steps
            self.assertLessEqual(len(runs), n // L + 2)                                 # at most one run per block
        self.assertFalse(np.array_equal(block_permutation(1000, 20, rng), np.arange(1000)))

    def test_block_one_is_an_ordinary_shuffle(self):
        from oql.permute import block_permutation
        p = block_permutation(2000, 1, np.random.default_rng(2))
        self.assertGreater((np.diff(p) != 1).mean(), 0.99)

    def test_keeps_short_structure_breaks_long_trends(self):
        from oql.calibration import clustered_market
        from oql.permute import permute_book
        from oql.synthetic import trend_market
        df = clustered_market(4000, seed=3, persistence=0.99, shock=0.15)
        blk = permute_book({"x": df}, 5, method="block")["x"]
        bars = permute_book({"x": df}, 5, method="bars")["x"]
        self.assertGreater(abs_autocorr(blk.close.values), 0.6 * abs_autocorr(df.close.values))  # clustering kept
        self.assertLess(abs(abs_autocorr(bars.close.values)), 0.05)
        tr = trend_market(4000, seed=4, drift=0.004, stay=0.995)                       # long regimes (~200 bars)
        tb = permute_book({"x": tr}, 6, method="block")["x"]
        r, rb = np.diff(np.log(tr.close.values)), np.diff(np.log(tb.close.values))
        m, mb = np.convolve(r, np.ones(60), "valid"), np.convolve(rb, np.ones(60), "valid")
        self.assertGreater(abs(np.corrcoef(m[60:], m[:-60])[0, 1]), abs(np.corrcoef(mb[60:], mb[:-60])[0, 1]) + 0.2)
        ok = (blk.high >= blk[["open", "close"]].max(axis=1) - 1e-9) & (blk.low <= blk[["open", "close"]].min(axis=1) + 1e-9)
        self.assertTrue(ok.all())

    def test_joint_across_markets(self):
        from oql.calibration import book
        from oql.permute import permute_book
        b = book("noise", 1)
        p = permute_book(b, 9, method="block")
        r = {k: np.diff(np.log(v.close.values)) for k, v in p.items()}
        r0 = {k: np.diff(np.log(v.close.values)) for k, v in b.items()}
        c_real, c_perm = np.corrcoef(r0["m0"], r0["m1"])[0, 1], np.corrcoef(r["m0"], r["m1"])[0, 1]
        self.assertAlmostEqual(c_real, c_perm, delta=0.05)                             # same reordering for every market


if __name__ == "__main__":
    unittest.main()
