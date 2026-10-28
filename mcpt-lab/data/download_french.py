"""Download the Kenneth R. French Data Library files used by examples/06_publication_decay.py.
Free for research; not redistributed in this repo (git-ignored).

    python data/download_french.py        (a few seconds)
"""
import os
import ssl
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mcptlab.decay import FRENCH_DIR, FRENCH_FILES  # noqa: E402

BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
try:                                   # python.org macOS builds ship without root certificates
    import certifi
    CTX = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    CTX = ssl.create_default_context()
os.makedirs(FRENCH_DIR, exist_ok=True)
for name in FRENCH_FILES.values():
    out = os.path.join(FRENCH_DIR, name)
    with urllib.request.urlopen(BASE + name, context=CTX) as resp, open(out, "wb") as f:
        f.write(resp.read())
    print(f"  {name:45s} {os.path.getsize(out):>9,d} bytes")
print("Saved to", FRENCH_DIR)
