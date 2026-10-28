"""Settings for the order-book recorder. It only READS public market data: no account, no API key, no orders."""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("RECORDER_DATA", os.path.join(HERE, "data"))

# (ccxt exchange id, symbol). Each venue is independent: if one fails, the others keep recording.
VENUES = [
    ("binanceus", "BTC/USDT"),
    ("kraken", "BTC/USD"),
    ("coinbase", "BTC/USD"),
    # More markets: (label, exchange, symbol, depth). Labels must be unique; the BTC markets keep the plain names
    # above because the order-book study (~/micro) reads exactly those three.
    ("coinbase-eth", "coinbase", "ETH/USD", 100),
    ("kraken-eth", "kraken", "ETH/USD", 100),
]
INTERVAL_S = 10          # one snapshot of every venue's order book every 10 seconds
DEPTH = 500              # price levels kept on each side (was 50 until 2026-10-14: on Coinbase and Kraken 50 levels
                         # only reached ~3.5 bps from mid, too shallow to measure depth within 10-25 bps)
TIMEOUT_MS = 4000        # give up on a slow venue so it cannot delay the others
