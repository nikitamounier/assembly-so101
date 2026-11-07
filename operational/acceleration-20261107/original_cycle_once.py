"""One canonical observation: bind the reviewed wrapper and receipt directory."""

import hashlib
import os
import sys
from pathlib import Path

assert len(sys.argv) == 1, "This launcher accepts no arguments"
root = Path(__file__).resolve().parents[3]
script = root / 'robotics-assembly-3.0/operational/chimera-20261106/monitor_cycle.py'
out = root / 'robotics-assembly-3.0/outputs/chimera-20261106/monitor'
assert hashlib.sha256(script.read_bytes()).hexdigest() == 'a757f924d78d0eab736ba4946ca1a1f27bcf184ecf8aa66f9d5c786f02518b51', "Reviewed wrapper changed"
os.execv(sys.executable, [sys.executable, str(script), "--out-dir", str(out)])
