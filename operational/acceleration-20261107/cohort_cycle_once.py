"""One canonical observation: bind the reviewed wrapper and receipt directory."""

import hashlib
import os
import sys
from pathlib import Path

assert len(sys.argv) == 1, "This launcher accepts no arguments"
root = Path(__file__).resolve().parents[3]
script = root / 'assembly-so101/operational/acceleration-20261107/cohort_cycle.py'
out = root / 'robotics-assembly-3.0/outputs/chimera-20261106/benchmark-cohort-monitor'
assert hashlib.sha256(script.read_bytes()).hexdigest() == '429ae73e7eb9defb47317437940f64c976c06f8009ee3fd394fd849a127e4248', "Reviewed wrapper changed"
os.execv(sys.executable, [sys.executable, str(script), "--out-dir", str(out)])
