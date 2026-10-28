"""Core accounting: next-bar execution, costs, p-values, and a look-ahead canary."""
import os, sys, tempfile, unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
from mcptlab import (DonchianBreakout, Strategy, in_sample_mcpt, p_value, profit_factor,
                     strategy_returns, synthetic, walk_forward)


class Cheater(Strategy):
    """Peeks at tomorrow's price. The framework can't stop you writing this -- but the numbers
    it produces are so absurd that this test documents what a look-ahead bug looks like."""
    name = "cheater"
    grid = [1]

    def positions(self, close, param):
        return np.sign(close.shift(-1) - close).fillna(0.0)


class TestCore(unittest.TestCase):
    def test_position_applies_to_next_bar(self):
        close = pd.Series([100.0, 110.0, 99.0, 99.0])
        pos = pd.Series([1.0, 0.0, 0.0, 0.0])        # decided at bar 0's close
        r = strategy_returns(close, pos)
        self.assertAlmostEqual(r.iloc[0], np.log(110 / 100))
        self.assertEqual(r.iloc[1], 0.0)

    def test_costs_charged_per_unit_traded(self):
        close = pd.Series([100.0] * 5)
        pos = pd.Series([1.0, -1.0, -1.0, 0.0, 0.0])  # enter 1, flip 2, exit 1 = 4 units
        self.assertAlmostEqual(-strategy_returns(close, pos, cost_bps=10).sum(), 4 * 10 / 1e4)

    def test_p_value_convention(self):
        self.assertEqual(p_value(1.0, np.zeros(999)), 1 / 1000)       # never exactly 0
        self.assertEqual(p_value(1.0, np.ones(999) * 2), 1.0)

    def test_profit_factor(self):
        self.assertAlmostEqual(profit_factor(pd.Series([2.0, -1.0, 1.0, -1.0])), 1.5)

    def test_lookahead_canary(self):
        pf = profit_factor(strategy_returns(synthetic("noise", seed=1), Cheater().positions(synthetic("noise", seed=1), 1)))
        self.assertGreater(pf, 10)      # a backtest this good on noise = you are using the future

    def test_walk_forward_is_out_of_sample_only(self):
        c = synthetic("noise", n=900, seed=4)
        wf = walk_forward(DonchianBreakout(range(20, 101, 20)), c, train_bars=500, step_bars=100)
        self.assertEqual(wf["returns"].index[0], c.index[500])
        self.assertEqual(len(wf["params"]), 4)

    def test_ledger_and_autosave(self):
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["MCPT_LEDGER"] = os.path.join(tmp, "L.csv")
            os.environ["MCPT_RUNS"] = os.path.join(tmp, "runs")
            try:
                in_sample_mcpt(DonchianBreakout(range(20, 101, 40)), synthetic("noise", n=400), n_perm=5, procs=1)
                self.assertEqual(len(pd.read_csv(os.environ["MCPT_LEDGER"])), 1)
                self.assertEqual(len([f for f in os.listdir(os.environ["MCPT_RUNS"]) if f.endswith(".png")]), 1)
            finally:
                del os.environ["MCPT_LEDGER"], os.environ["MCPT_RUNS"]


if __name__ == "__main__":
    unittest.main()
