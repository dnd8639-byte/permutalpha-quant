# Lets the examples run straight from a cloned repo without `pip install -e .`
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
