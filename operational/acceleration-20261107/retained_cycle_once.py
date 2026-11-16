"""One canonical observation: bind the reviewed wrapper and receipt directory."""

import hashlib
import os
import sys
from pathlib import Path

assert len(sys.argv) == 1, "This launcher accepts no arguments"
root = Path(__file__).resolve().parents[3]
script = root / 'assembly-so101/operational/acceleration-20261107/retained_cycle.py'
out = root / 'robotics-assembly-3.0/outputs/chimera-20261106/benchmark-retained-monitor'
assert hashlib.sha256(script.read_bytes()).hexdigest() == 'bf3cd72b7394f1cad8c58064cca4854da6ee2ff9309317326efe8489c24c1832', "Reviewed wrapper changed"
os.execv(sys.executable, [sys.executable, str(script), "--out-dir", str(out)])
