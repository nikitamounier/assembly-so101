"""Checkpoint-safe one-time transition, executed inside the CPU allocation."""
import fcntl
import json
import os
from pathlib import Path
import sys
import time
from controller import ROOT, RUN, GPU, PYTHON, STATE, launch

assert os.environ.get('SLURM_JOB_ID') == '3077705'
out = ROOT/'artifacts_local/chimera/ppo-profile-20261106'
out.mkdir(exist_ok=True)
command = json.loads((RUN/'command.json').read_text())
assert command[1:3] == ['-m', 'assembly_policy.train']
backup = out/'original-command.json'
assert not backup.exists(), 'Refuse duplicate migration; reconcile existing capture first'
backup.write_text(json.dumps(command, indent=2)+'\n')
profile_command = [command[0], str(ROOT/'operational/chimera-20261106/profile_train.py'), 'assembly_policy.train', *command[3:]]
(RUN/'command.json').write_text(json.dumps(profile_command, indent=2)+'\n')
stop = launch(GPU, 1, [PYTHON, str(ROOT/'operational/chimera-20261106/learner.py'), '--stop'], out/'transition.log')
assert stop.wait(timeout=60) == 0
deadline = time.monotonic()+900
while time.monotonic()<deadline:
    state = json.loads(STATE.read_text())
    if state['phase']=='learner_failed' and state.get('learner_exit')==0:
        lock = open(STATE.with_suffix('.lock'),'a')
        try:
            fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            time.sleep(1)
            continue
        lock.close()
        sys.path.insert(0,str(ROOT))
        import torch
        ck = torch.load(RUN/'checkpoint.pt',map_location='cpu',weights_only=False)
        last = json.loads((RUN/'log.jsonl').read_text().splitlines()[-1])
        assert ck['it']==last['it'] and ck['transitions']==last['transitions']
        (out/'transition.json').write_text(json.dumps(dict(checkpoint_iteration=ck['it'],transitions=ck['transitions'],time=time.time()))+'\n')
        os.execv(PYTHON,[PYTHON,str(ROOT/'operational/chimera-20261106/pool_controller.py')])
    time.sleep(2)
raise RuntimeError('Checkpoint transition timed out; inspect before retry')
