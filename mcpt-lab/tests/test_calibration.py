"""Small version of validation/validate.py (about 1 minute): noise should not pass, a planted
trend should. The full calibration (hundreds of runs) is in validation/validate.py."""
import os, sys, unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from mcptlab import DonchianBreakout, in_sample_mcpt, synthetic

STRAT = DonchianBreakout(range(10, 201, 20))


class TestCalibration(unittest.TestCase):
    def test_planted_trend_is_found(self):
        ps = [in_sample_mcpt(STRAT, synthetic("trend", n=1500, seed=s), n_perm=50, log=False)["p"] for s in range(3)]
        self.assertTrue(all(p < 0.05 for p in ps), ps)

    def test_noise_is_not_found(self):
        ps = [in_sample_mcpt(STRAT, synthetic("noise", n=1500, seed=s), n_perm=50, log=False)["p"] for s in range(6)]
        self.assertLessEqual(sum(p < 0.05 for p in ps), 1, ps)
        self.assertGreater(np.mean(ps), 0.2, ps)


if __name__ == "__main__":
    unittest.main()
