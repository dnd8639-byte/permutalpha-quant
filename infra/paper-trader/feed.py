"""Where prices come from. LiveFeed talks to the exchange's public API; ReplayFeed replays history
(optionally with faults injected) so the exact same engine code can be tested offline."""
import time


class LiveFeed:
    def __init__(self, exchange, symbol):
        import ccxt
        self.ex = getattr(ccxt, exchange)({"enableRateLimit": True})
        self.symbol = symbol

    def now_ms(self):
        return int(time.time() * 1000)

    def exchange_time_ms(self):
        return self.ex.fetch_time()

    def closed_bars(self, since_ms, now_ms, bar_ms):
        """All fully closed hourly bars from since_ms up to now."""
        out = []
        while since_ms + bar_ms <= now_ms:
            batch = self.ex.fetch_ohlcv(self.symbol, "1h", since=since_ms, limit=1000)
            batch = [b for b in batch if b[0] + bar_ms <= now_ms]
            if not batch:
                break
            out += batch
            since_ms = batch[-1][0] + bar_ms
        return out


class ReplayFeed:
    """History as a list of [ts, o, h, l, c, v]. `clock` is set by the test. Faults:
    hide = set of bar timestamps the 'exchange' refuses to return; delay = {ts: ms late};
    spike = {ts: fake close}; revise = {ts: (first close shown, visible until ms)}; drift_ms = clock error."""
    def __init__(self, rows):
        self.rows = rows; self.clock = 0
        self.hide, self.delay, self.spike, self.revise, self.drift_ms = set(), {}, {}, {}, 0

    def now_ms(self):
        return self.clock

    def exchange_time_ms(self):
        return self.clock - self.drift_ms

    def closed_bars(self, since_ms, now_ms, bar_ms):
        out = []
        for r in self.rows:
            ts = r[0]
            if ts < since_ms or ts + bar_ms + self.delay.get(ts, 0) > now_ms or ts in self.hide:
                continue
            r = list(r)
            if ts in self.spike:
                r[4] = self.spike[ts]
            if ts in self.revise and now_ms < self.revise[ts][1]:
                r[4] = self.revise[ts][0]
            out.append(r)
        return out
