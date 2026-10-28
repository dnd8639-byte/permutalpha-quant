"""`oql demo`: a ready-to-run lab on synthetic markets, so a fresh install shows the whole pipeline in minutes.

It holds two cards already frozen and waiting for their gate (one on markets with a planted trend edge, one on pure
noise) plus two synthetic hourly markets for the Efficiency tab. Running both gates should give a PASS and a FAIL.
Nothing in it is real data.
"""
from __future__ import annotations

import os


def build(lab: str) -> str:
    lab = os.path.abspath(os.path.expanduser(lab))
    if os.path.exists(os.path.join(lab, "board.json")):
        raise FileExistsError(f"{lab} already has a board; choose another folder or delete it to start over")
    os.makedirs(lab, exist_ok=True)
    os.environ["OQL_LAB"], os.environ["OQL_CATALOG"] = lab, os.path.join(lab, "catalog")
    from . import board, catalog, efficiency, synthetic

    def card(title, mechanism, datasets):
        c = board.new_card(title, mechanism, lab=lab)
        d = board.card_dir(c, lab)
        for name, pairs in (("PREREGISTRATION.md", [("TODO", "Demo card on synthetic data.")]),
                            ("experiment.py", [("datasets=[],", f"datasets={datasets!r},")])):
            path = os.path.join(d, name)
            with open(path) as f:
                text = f.read()
            for old, new in pairs:
                text = text.replace(old, new)
            with open(path, "w") as f:
                f.write(text)
        board.start(c["id"], lab=lab)
        board.freeze(c["id"], lab=lab)

    trend = [catalog.write(f"synth/trend{i}/1d", synthetic.trend_market(seed=i), license="own", origin="synthetic").id
             for i in range(4)]
    noise = [catalog.write(f"synth/noise{i}/1d", synthetic.noise_market(seed=50 + i), license="own", origin="synthetic").id
             for i in range(4)]
    catalog.write("synth/planted-efficiency/1h", efficiency.synthetic_bars(True, n_days=1500, seed=3), license="own",
                  origin="synthetic")
    catalog.write("synth/random-walk/1h", efficiency.synthetic_bars(False, n_days=1500, seed=4), license="own",
                  origin="synthetic")
    card("Trend following on markets with a planted edge", "Regimes persist by construction, so trends continue.", trend)
    card("Trend following on pure noise", "None: these markets are random walks, so the gate must fail.", noise)
    return lab
