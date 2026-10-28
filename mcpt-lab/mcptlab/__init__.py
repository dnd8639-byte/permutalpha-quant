"""mcpt-lab: test trading strategies against shuffled markets before trusting them.

    from mcptlab import DonchianBreakout, in_sample_mcpt, synthetic
    res = in_sample_mcpt(DonchianBreakout(), synthetic("noise"), n_perm=200)
    print(res["p"])      # ~uniform on noise; small when the edge is real
"""
from .core import (METRICS, fixed_param_mcpt, in_sample_excellence, in_sample_mcpt, ledger_count,
                   load_null, log_test, optimize, p_value, permutation_test, permute, profit_factor,
                   save_null, sharpe, strategy_returns, walk_forward, walk_forward_mcpt)
from .data import load_ohlc, load_sample_btc, synthetic
from .permute import permute_close, permute_ohlc, signflip_ohlc
from .plotting import plot_grid, plot_null, plot_runs
from .strategies import DonchianBreakout, HawkesVolatility, MovingAverageCross, Strategy

__version__ = "1.0.0"
