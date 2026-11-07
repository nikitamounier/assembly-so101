"""Restart this controller only after relay preemption and verified learner absence."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path('/large_storage/goodarzilab/nikitamounier/assembly/source-85c006a')
RUN = ROOT / 'runs/chimera-v13-8192-20261106'
ART = ROOT / 'artifacts_local/chimera'
assert os.environ.get('SLURM_JOB_ID') == '3077705'
sys.path.insert(0, str(ROOT / 'operational/chimera-20261106'))
from controller import native

state = json.loads((ART / 'controller.json').read_text())
relay = native('3077705')
gpu = native('3598717')
assert relay['JobState'] == 'RUNNING' and int(relay['Restarts']) >= 45
assert gpu['JobState'] in ('RUNNING', 'PENDING', 'CONFIGURING')
assert int(gpu['Restarts']) > int(state['jobs']['3598717']['Restarts']), 'Old learner epoch may still be live'
assert time.time() - (ART / 'controller.json').stat().st_mtime > 120
lock = open(ART / 'controller.lock', 'a')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
result = subprocess.run(['squeue', '--steps', '--noheader', '--jobs=3598717', '--format=%i'], capture_output=True, text=True, timeout=30)
assert result.returncode == 0 and not result.stderr.strip(), result.stderr
steps = result.stdout.split()
assert all(step in ('3598717.batch', '3598717.extern') for step in steps), steps
manifest = json.loads((ART / 'source-hashes.json').read_text())
for name, expected in manifest.items():
    assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
import torch
checkpoint = torch.load(RUN / 'checkpoint.pt', map_location='cpu', weights_only=False)
last = json.loads((RUN / 'log.jsonl').read_text().splitlines()[-1])
assert checkpoint['it'] == last['it'] == 42
assert checkpoint['transitions'] == last['transitions'] == 9293226
receipt = dict(observed_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()), cause='relay_requeued_controller_step_lost', relay_restarts=relay['Restarts'], relay_host=relay['NodeList'], gpu_restarts=gpu['Restarts'], gpu_state=gpu['JobState'], checkpoint_iteration=checkpoint['it'], transitions=checkpoint['transitions'], source_valid=True, live_gpu_steps=steps)
(ART / 'relay-recovery-20261107.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt), flush=True)
lock.close()
os.execv(str(ROOT / '.venv/bin/python'), [str(ROOT / '.venv/bin/python'), str(ROOT / 'operational/chimera-20261106/pool_controller.py')])
