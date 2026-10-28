"""The ledger: one row per test ever run, pass or fail. Nothing is ever removed from it.
Judging a result means knowing how many things were tried before it; this file is that count.

Tamper evidence: every row carries `chain`, the SHA-256 of the previous row's chain plus this row's contents. Deleting,
reordering or editing any row breaks every chain value after it, which `oql ledger verify` reports. To make the
whole history checkable by others, publish the latest chain value somewhere dated (a git commit message, for example):
`oql ledger head` prints it. A ledger written before chaining existed is sealed as it stands the first time a row
is added, and the seal is recorded as a row.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import time

from .locks import hold

FIELDS = ["utc", "card", "family", "kind", "verdict", "p", "sharpe", "n_trials", "finalist", "data", "note", "chain"]
GENESIS = "0" * 64


def path(lab: str) -> str:
    return os.path.join(lab, "LEDGER.csv")


def _link(prev: str, row: dict) -> str:
    body = json.dumps({k: str(row.get(k, "")) for k in FIELDS if k != "chain"}, sort_keys=True)
    return hashlib.sha256((prev + body).encode()).hexdigest()


def _read(p: str) -> tuple[list[str], list[dict]]:
    with open(p, newline="") as f:
        r = csv.DictReader(f)
        return list(r.fieldnames or []), list(r)


def _write_all(p: str, rows: list[dict]) -> None:
    tmp = p + ".tmp"
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in FIELDS})
    os.replace(tmp, p)


def _seal_legacy(p: str) -> None:
    """A ledger from before chaining: chain its rows as they stand now and say so in a row of its own."""
    _, rows = _read(p)
    prev = GENESIS
    for r in rows:
        r["chain"] = prev = _link(prev, r)
    seal = {"utc": time.strftime("%Y-%m-%d %H:%M", time.gmtime()), "card": "-", "family": "-", "kind": "seal",
            "verdict": "-", "n_trials": 0, "note": f"{len(rows)} earlier rows chained as they stood at this time"}
    seal["chain"] = _link(prev, seal)
    _write_all(p, rows + [seal])


def append(lab: str, **row) -> None:
    p = path(lab)
    with hold(p + ".lock"):                               # two processes appending at once cannot fork the chain
        if os.path.exists(p) and "chain" not in _read(p)[0]:
            _seal_legacy(p)
        new = not os.path.exists(p)
        row = {k: row.get(k, "") for k in FIELDS} | {"utc": time.strftime("%Y-%m-%d %H:%M", time.gmtime())}
        if isinstance(row["finalist"], dict):
            row["finalist"] = json.dumps(row["finalist"], sort_keys=True)
        prev = GENESIS if new else (_read(p)[1] or [{"chain": GENESIS}])[-1].get("chain") or GENESIS
        row["chain"] = _link(prev, row)
        with open(p, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            if new:
                w.writeheader()
            w.writerow(row)


def rows(lab: str) -> list[dict]:
    p = path(lab)
    if not os.path.exists(p):
        return []
    return _read(p)[1]


def verify(lab: str) -> dict:
    """Recompute the chain. Returns {"ok", "rows", "first_bad" (1-based row number or None), "head"}."""
    rs = rows(lab)
    prev = GENESIS
    for i, r in enumerate(rs):
        if not r.get("chain"):
            return {"ok": False, "rows": len(rs), "first_bad": i + 1, "head": prev, "why": "row has no chain value"}
        if _link(prev, r) != r["chain"]:
            return {"ok": False, "rows": len(rs), "first_bad": i + 1, "head": prev,
                    "why": "row was edited, or a row before it was removed or reordered"}
        prev = r["chain"]
    return {"ok": True, "rows": len(rs), "first_bad": None, "head": prev}


def trials_in_family(lab: str, family: str) -> int:
    """Strategies searched across every card in one idea family (a card and all its revisions)."""
    return sum(int(r["n_trials"] or 0) for r in rows(lab) if r["family"] == family and r["kind"] == "gate")
