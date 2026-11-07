"""Serve exactly one benchmark learner; never reconnect to a completed phase."""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from assembly_policy.rl.remote import _nodelay, _serve  # noqa: E402
from assembly_policy.rl.transport import connect  # noqa: E402

p = Path(sys.argv[1]) / "server.json"
for _ in range(900):
    if p.exists():
        info = json.loads(p.read_text())
        try:
            c = _nodelay(connect(info))
            break
        except OSError:
            pass
    time.sleep(1)
else:
    raise RuntimeError("No benchmark learner within 15 minutes")
try:
    _serve(c, 2)
finally:
    c.close()
