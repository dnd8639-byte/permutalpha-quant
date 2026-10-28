"""Locks that make the rules of a test hold in code, not in good intentions.

freeze(files)   SHA-256 fingerprints of the pre-registration and experiment at the moment they are fixed.
verify(frozen)  refuses if any frozen file changed since.
claim(path)     a one-shot lock: written BEFORE the result is computed, refuses forever after.
"""
from __future__ import annotations

import hashlib
import json
import os
import time


class LockError(RuntimeError):
    pass


def sha256(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def freeze(files: list[str]) -> dict:
    return {"frozen_utc": time.strftime("%Y-%m-%d %H:%M", time.gmtime()),
            "files": {os.path.basename(f): sha256(f) for f in files}}


def verify(frozen: dict, folder: str) -> None:
    for name, h in frozen["files"].items():
        p = os.path.join(folder, name)
        if not os.path.exists(p) or sha256(p) != h:
            raise LockError(f"{name} changed after it was frozen on {frozen['frozen_utc']} UTC. "
                            f"A changed test is a new test: start a new card (oql card revise).")


def claim(path: str, what: str) -> dict:
    if os.path.exists(path):
        with open(path) as f:
            rec = json.load(f)
        raise LockError(f"{what} was already run on {rec['claimed_utc']} UTC. It is one-shot.")
    rec = {"what": what, "claimed_utc": time.strftime("%Y-%m-%d %H:%M", time.gmtime())}
    with open(path, "x") as f:                        # 'x' fails if another process got there first
        json.dump(rec, f)
    return rec


try:
    import fcntl
except ImportError:                                       # Windows: no advisory locks; single-user use is still safe
    fcntl = None


_held: dict = {}                                          # path -> depth: re-entrant within one process


class hold:
    """An exclusive lock on `path` for the duration of a `with` block (released automatically if the process dies).
    blocking=False raises LockError at once if another process holds it. Re-entrant within a process."""

    def __init__(self, path: str, blocking: bool = True, what: str = "this"):
        self.path, self.blocking, self.what, self.f = os.path.abspath(path), blocking, what, None

    def __enter__(self):
        if _held.get(self.path):
            _held[self.path] += 1
            return self
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self.f = open(self.path, "a+")
        if fcntl is not None:
            try:
                fcntl.flock(self.f, fcntl.LOCK_EX | (0 if self.blocking else fcntl.LOCK_NB))
            except BlockingIOError:
                self.f.close()
                raise LockError(f"{self.what} is already running in another process")
        _held[self.path] = 1
        return self

    def __exit__(self, *exc):
        _held[self.path] -= 1
        if _held[self.path]:
            return False
        del _held[self.path]
        if fcntl is not None:
            fcntl.flock(self.f, fcntl.LOCK_UN)
        self.f.close()
        return False
