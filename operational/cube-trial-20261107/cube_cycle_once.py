"""Fixed cube monitor entry; no alternate directories or argv."""
import hashlib
import runpy
import sys
from pathlib import Path

if sys.argv[1:]:
    raise SystemExit("No arguments allowed")
p = Path(__file__).resolve().parent
for name, expected in {'cube_cycle.py': 'cadba7cb44a327aa167ca4be95bc4979e4347acf636fbb3d4e2c3b9865260cfb', 'monitor-pins.json': '2067b9304ef6dc71d6ddd8a8e1aa46a382f570841673c6a79f4f47010767e896'}.items():
    if hashlib.sha256((p / name).read_bytes()).hexdigest() != expected:
        raise SystemExit("Cube monitor local source pin mismatch")
runpy.run_path(str(p / "cube_cycle.py"), run_name="__main__")
