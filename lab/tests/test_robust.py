"""Robustness: the properties the lab's conclusions rest on, tested broadly rather than on one example.
python -m unittest tests.test_robust"""
import itertools
import json
import os
import tempfile
import unittest

import numpy as np
import pandas as pd


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        os.environ["OQL_CATALOG"] = os.path.join(self.tmp.name, "catalog")
        os.environ["OQL_LAB"] = self.tmp.name
        os.environ["OMP_NUM_THREADS"] = "1"

    def tearDown(self):
        self.tmp.cleanup()


def card(ids, **kw):
    from oql import board
    c = board.new_card("robust", "A mechanism long enough for the board to accept it here.")
    d = board.card_dir(c)
    with open(os.path.join(d, "PREREGISTRATION.md"), "w") as f:
        f.write("# done\n")
    with open(os.path.join(d, "experiment.py"), "w") as f:
        args = {"rules": ["momentum"], "n_perm": 39, **kw}
        f.write(f"from oql.experiment import Experiment\nEXPERIMENT = Experiment(datasets={ids!r}, "
                + ", ".join(f"{k}={v!r}" for k, v in args.items()) + ")\n")
    board.start(c["id"])
    return c, d


class TestNoLookAheadAnywhere(unittest.TestCase):
    """Every rule, side and gate: decisions up to bar t must not change when anything after t changes."""

    def test_all_rules_sides_and_gates(self):
        from oql import efficiency as E
        from oql.gates import GATES
        from oql.strategies import ALL_RULES, positions, valid
        bars = E.synthetic_bars(True, n_days=500, seed=21)
        later = bars.copy()
        t = 430 * 24 + 5
        later.iloc[t + 1:, :4] *= np.exp(np.cumsum(np.random.default_rng(3).normal(0, 0.01, len(bars) - t - 1)))[:, None]
        gates = [None] + [{"kind": k, **dict(zip(g, combo))} for k, g in GATES.items() for combo in itertools.product(*g.values())][::3]
        checked = 0
        for rule, (_, grid) in ALL_RULES.items():
            for combo in list(itertools.product(*grid.values()))[:3]:
                prm = dict(zip(grid, combo))
                if not valid(rule, prm):
                    continue
                for side in ("both", "long", "short"):
                    for gate in gates:
                        g = {"rule": rule, "params": prm, "side": side, **({"gate": gate} if gate else {})}
                        a = positions(ALL_RULES, bars, g, "x", {})
                        b = positions(ALL_RULES, later, g, "x", {})
                        self.assertTrue(np.array_equal(a[:t + 1], b[:t + 1]), g)
                        checked += 1
        self.assertGreater(checked, 300)

    def test_engine_pnl_up_to_t_ignores_the_future(self):
        from oql.engine import market_returns, prepare
        from oql.synthetic import noise_market
        df = noise_market(800, seed=4)
        later = df.copy(); later.iloc[601:, :4] *= 1.5
        sig = np.sign(np.sin(np.arange(800) / 7))
        a = market_returns(prepare({"x": df}, {"default": 2}).arrays["x"], sig)[0]
        b = market_returns(prepare({"x": later}, {"default": 2}).arrays["x"], sig)[0]
        self.assertTrue(np.allclose(a[:601], b[:601], equal_nan=True))


class TestEngineInvariants(unittest.TestCase):
    def test_messy_markets_never_produce_nan_or_explode(self):
        from oql.engine import MAX_LEVERAGE, prepare
        from oql.search import Scorer
        from oql.strategies import RULES
        rng = np.random.default_rng(5)
        for k in range(20):
            n = 600
            c = 100 * np.exp(np.cumsum(rng.standard_t(3, n) * 0.01))
            c[rng.integers(0, n, 30)] *= 1.0                                           # no-ops, keep
            flat = rng.integers(100, 500); c[flat:flat + 40] = c[flat]                   # a frozen feed
            idx = pd.bdate_range("2015-01-01", periods=n, tz="UTC").delete(rng.integers(0, n, 25))
            c = c[:len(idx)]
            df = pd.DataFrame({"open": c, "high": c * 1.01, "low": c * 0.99, "close": c, "volume": 1.0}, index=idx)
            prep = prepare({"x": df}, {"default": 3.0}, warmup=60)
            size = prep.arrays["x"]["size"]
            self.assertLessEqual(np.nanmax(size), MAX_LEVERAGE)
            self.assertEqual(np.nanmax(size[flat + 25:flat + 40]) if flat + 40 <= len(size) else 0.0, 0.0)  # stale: no position
            sc = Scorer(prep, RULES, 0.0)
            for g in ({"rule": "momentum", "params": {"lookback": 20}, "side": "both"},
                      {"rule": "breakout", "params": {"lookback": 50}, "side": "long"}):
                port = sc.returns(g)[0]
                self.assertFalse(port.isna().any(), (k, g))
                self.assertTrue(np.isfinite(port).all())

    def test_costs_only_ever_reduce_returns(self):
        from oql.engine import prepare
        from oql.search import Scorer
        from oql.strategies import RULES
        from oql.synthetic import noise_market
        prep = prepare({"x": noise_market(1000, seed=9)}, {"default": 2.0})
        sc = Scorer(prep, RULES, 0.0)
        g = {"rule": "momentum", "params": {"lookback": 5}, "side": "both"}
        totals = [sc.returns(g, cost_mult=m)[0].sum() for m in (0.0, 1.0, 2.0, 5.0)]
        self.assertTrue(all(a > b for a, b in zip(totals, totals[1:])))


class TestDeterminism(Base):
    def test_the_gate_gives_the_same_answer_twice(self):
        from oql import catalog, synthetic
        from oql.experiment import Experiment
        from oql.gate import run_gate
        ids = [catalog.write(f"n/m{i}/1d", synthetic.noise_market(1200, seed=i), license="own", origin="t").id for i in range(2)]
        exp = Experiment(datasets=ids, n_perm=19, rounds=3)
        a, b = run_gate(exp, procs=2, verbose=False), run_gate(exp, procs=1, verbose=False)
        self.assertEqual((a["p"], a["finalist"], a["null"]), (b["p"], b["finalist"], b["null"]))


class TestFrozenData(Base):
    def test_new_rows_are_ignored_and_edited_rows_refused(self):
        from oql import board, catalog, locks, synthetic
        df = synthetic.trend_market(1500, seed=1)
        catalog.write("t/m0/1d", df.iloc[:1200], license="own", origin="t")
        c, d = card(["t/m0/1d"])
        board.freeze(c["id"])
        catalog.write("t/m0/1d", df, license="own", origin="t")                       # the collector appended 300 days
        r = board.run_gate(c["id"], procs=2)
        self.assertLess(pd.Timestamp(r["split"]), df.index[1200])                     # split from the frozen 1,200 rows
        self.assertEqual(r["split"], str(df.index[int(1200 * 0.7)]))
        c2, d2 = card(["t/m0/1d"])
        board.freeze(c2["id"])
        edited = df.copy(); edited.iloc[100, 3] *= 1.01                                 # a revised old price
        catalog.write("t/m0/1d", edited, license="own", origin="t")
        with self.assertRaises(locks.LockError):
            board.run_gate(c2["id"], procs=2)

    def test_tests_that_cannot_pass_are_refused_at_the_freeze(self):
        from oql import board, catalog, synthetic
        catalog.write("t/m0/1d", synthetic.noise_market(400, seed=1), license="own", origin="t")
        for kw in ({"n_perm": 10}, {"alpha": 0.5}, {"cost_bps": {"default": -1}}, {"dev_frac": 0.99}, {"rules": ["nope"]}):
            c, d = card(["t/m0/1d"], **kw)
            with self.assertRaises((board.BoardError, KeyError)):
                board.freeze(c["id"])


def _append_many(args):
    lab, k = args
    import os
    os.environ["OQL_LAB"] = lab
    from oql import ledger
    for i in range(25):
        ledger.append(lab, card=f"p{k}", family="x", kind="explore", verdict="-", n_trials=1, note=str(i))
    return k


class TestLedgerChain(Base):
    def test_edits_removals_and_reorders_are_detected(self):
        import csv
        from oql import ledger
        lab = self.tmp.name
        for i in range(6):
            ledger.append(lab, card=f"c{i}", family="f", kind="explore", verdict="-", sharpe=0.1 * i, n_trials=1)
        self.assertTrue(ledger.verify(lab)["ok"])
        p = ledger.path(lab)
        with open(p) as f:
            original = f.read()
        for tamper in ("edit", "remove", "swap"):
            with open(p, newline="") as f:
                rows = list(csv.DictReader(f))
            if tamper == "edit":
                rows[2]["sharpe"] = "9.9"
            elif tamper == "remove":
                del rows[3]
            else:
                rows[1], rows[2] = rows[2], rows[1]
            ledger._write_all(p, rows)
            v = ledger.verify(lab)
            self.assertFalse(v["ok"], tamper)
            with open(p, "w") as f:
                f.write(original)
        self.assertTrue(ledger.verify(lab)["ok"])

    def test_an_old_ledger_is_sealed_then_chained(self):
        from oql import ledger
        lab = self.tmp.name
        with open(ledger.path(lab), "w") as f:
            f.write("utc,card,family,kind,verdict,p,sharpe,n_trials,finalist,data,note\n"
                    "2026-10-01 10:00,0001,f,gate,FAIL,0.26,0.22,25,x,d,old row\n")
        ledger.append(lab, card="0002", family="f", kind="gate", verdict="PASS", n_trials=3)
        rs = ledger.rows(lab)
        self.assertEqual([r["kind"] for r in rs], ["gate", "seal", "gate"])
        self.assertTrue(ledger.verify(lab)["ok"])

    def test_two_processes_appending_at_once_keep_one_chain(self):
        import multiprocessing as mp
        from oql import ledger
        lab = self.tmp.name
        with mp.get_context("spawn").Pool(2) as pool:
            pool.map(_append_many, [(lab, 0), (lab, 1)])
        v = ledger.verify(lab)
        self.assertTrue(v["ok"], v)
        self.assertEqual(v["rows"], 50)


class TestBoardConcurrency(Base):
    def test_a_long_gate_does_not_erase_changes_made_meanwhile(self):
        from oql import board, catalog, synthetic
        import oql.gate as G
        catalog.write("t/m0/1d", synthetic.noise_market(800, seed=2), license="own", origin="t")
        c, d = card(["t/m0/1d"])
        board.freeze(c["id"])
        real = G.run_gate

        def slow_gate(exp, **kw):                                    # someone adds a card while the gate runs
            board.new_card("made during the gate", "A mechanism long enough for the board to accept it here.")
            return real(exp, **kw)
        G.run_gate = slow_gate
        try:
            board.run_gate(c["id"], procs=1)
        finally:
            G.run_gate = real
        titles = [x["title"] for x in board.load()["cards"]]
        self.assertIn("made during the gate", titles)
        self.assertIn(board.get(board.load(), c["id"])["stage"], ("gate", "archive"))


if __name__ == "__main__":
    unittest.main()
