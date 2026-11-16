"""Own the configured worker processes on one Sailbox."""
import json, subprocess, sys, time
from pathlib import Path

config = json.loads(Path('/run/assembly/config.json').read_text())
children = []
try:
    for _ in range(config['processes']):
        child = subprocess.Popen([sys.executable, '-u', '/opt/sail-live/worker.py'], stdin=subprocess.PIPE, text=True)
        children.append(child)
        child.stdin.write(json.dumps(config))
        child.stdin.close()
    while not Path('/run/assembly/stop').exists():
        if any(child.poll() is not None for child in children):
            raise RuntimeError('A SAIL rollout worker exited; refusing to leave a partially idle box')
        time.sleep(2)
finally:
    for child in children:
        child.terminate()
    for child in children:
        try: child.wait(timeout=20)
        except subprocess.TimeoutExpired: child.kill(); child.wait()
