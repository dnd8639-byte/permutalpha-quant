"""All settings in one place. PAPER TRADING ONLY: this program contains no code that can place an order."""
import os

HERE = os.path.dirname(os.path.abspath(__file__))

SYMBOL = "BTC/USDT"
EXCHANGE = "binanceus"            # public price data only; no account, no API key
BAR_MS = 3_600_000                # one-hour bars
PARAMS = (0.1, 96)                # VolatilityHawkes (kappa, lookback): frozen, chosen on 2018-2022 data
COST_BPS = 5.0                    # cost charged per unit of position change, as in every backtest
ANCHOR = "2026-01-01T00:00:00Z"   # history always starts here, so every run computes on the same bars
WARMUP_BARS = 600                 # bars needed before the first signal is trusted (ATR uses 336)

# RiskGate limits: if any check fails the bot goes FLAT and says why
MAX_BAR_MOVE = 0.25               # a one-hour move above 25% is treated as bad data
MAX_CLOCK_DRIFT_S = 30            # this computer's clock vs the exchange's clock
GAP_LOOKBACK_BARS = 24            # no missing bars allowed in the most recent day
RUN_AT_SECOND = 20                # run this many seconds after each hour closes

DB_PATH = os.environ.get("PAPER_DB", os.path.join(HERE, "state", "paper.db"))
KILL_FILE = os.path.join(HERE, "KILL")          # create this file to force the bot flat
STATUS_HTML = os.path.join(HERE, "state", "status.html")
