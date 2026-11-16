import json
import os
from pathlib import Path
import signal
import time

assert os.environ.get('SLURM_JOB_ID')=='3077705'
ROOT=Path('/large_storage/goodarzilab/nikitamounier/assembly/source-85c006a')
target=(ROOT/'operational/chimera-20261106/reuse_baseline.py').resolve()
out=ROOT.parent/'assembly-nist-eb5d15e/artifacts_local/chimera/ppo-profile-20261106'
state=json.loads((out/'controller.json').read_text())
assert state['phase']=='waiting_banks_and_so_capture',state['phase']
found=[]
for proc in Path('/proc').iterdir():
    if not proc.name.isdigit():continue
    try:
        if proc.stat().st_uid!=os.getuid():continue
        cwd=(proc/'cwd').resolve();args=(proc/'cmdline').read_bytes().decode().split('\0')
        if any((Path(arg) if Path(arg).is_absolute() else cwd/arg).resolve()==target for arg in args[1:] if arg):found.append(int(proc.name))
    except (OSError,UnicodeError):pass
assert len(found)==1,found
os.kill(found[0],signal.SIGTERM)
proc=Path('/proc')/str(found[0]);deadline=time.monotonic()+60
while proc.exists() and (proc/'stat').read_text().split()[2]!='Z':
    assert time.monotonic()<deadline
    time.sleep(.5)
print('refreshed waiting comparison coordinator',found[0],flush=True)
os.execv(str(ROOT/'.venv/bin/python'),[str(ROOT/'.venv/bin/python'),str(target)])
