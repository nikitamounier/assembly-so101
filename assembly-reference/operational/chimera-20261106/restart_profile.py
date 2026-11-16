"""Use all available workers, preserving all environments and a full batch."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from controller import ROOT,RUN,GPU,PYTHON,STATE,launch

assert os.environ.get('SLURM_JOB_ID')=='3077705'
out=ROOT/'artifacts_local/chimera/ppo-profile-20261106'
status=json.loads((out/'status.json').read_text())
assert status['phase']=='warmup','Do not interrupt an active capture'
state=json.loads(STATE.read_text())
manifest=ROOT/'artifacts_local/chimera/node-pool.json'
pool=json.loads(manifest.read_text())
active_jobs=[r['job_id'] for r in pool['jobs'] if state['jobs'][r['job_id']]['JobState']=='RUNNING']
assert len(active_jobs)>=38
workers=len(active_jobs)*5
pool['target_workers']=workers
pool['profile_native_jobs']=[GPU,*active_jobs]
pool['profile_note']='Temporary full-env capture on currently available CPU leases; configured CPU-job pool remains unchanged.'
manifest.write_text(json.dumps(pool,indent=2)+'\n')
cfg=json.loads((out/'worker-config.json').read_text())
cfg['jobs']=[j for j in cfg['jobs'] if j in active_jobs]
cfg['expected_workers']=len(cfg['jobs']);assert cfg['expected_workers']>=2
(out/'worker-config.json').write_text(json.dumps(cfg,indent=2)+'\n')
guardpath=ROOT/'artifacts_local/chimera/source-hashes.json'
guard=json.loads(guardpath.read_text())
name='operational/chimera-20261106/profile_train.py'
current=(ROOT/name).read_bytes()
assert hashlib.sha256(current).hexdigest()==guard[name]
candidate=(ROOT/'artifacts_local/chimera/profile-train-candidate.py').read_bytes()
compile(candidate,name,'exec')
(ROOT/name).write_bytes(candidate)
guard[name]=hashlib.sha256(candidate).hexdigest()
guardpath.write_text(json.dumps(guard,indent=2)+'\n')
stop=launch(GPU,1,[PYTHON,str(ROOT/'operational/chimera-20261106/learner.py'),'--stop'],out/'transition.log')
assert stop.wait(timeout=60)==0
deadline=time.monotonic()+900
while time.monotonic()<deadline:
    state=json.loads(STATE.read_text())
    if state['phase']=='learner_failed' and state.get('learner_exit')==0:
        lock=open(STATE.with_suffix('.lock'),'a')
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:time.sleep(1);continue
        lock.close()
        sys.path.insert(0,str(ROOT));import torch
        ck=torch.load(RUN/'checkpoint.pt',map_location='cpu',weights_only=False)
        last=json.loads((RUN/'log.jsonl').read_text().splitlines()[-1])
        assert ck['it']==last['it'] and ck['transitions']==last['transitions']
        (out/'transition-available.json').write_text(json.dumps(dict(it=ck['it'],workers=workers,active_jobs=active_jobs,time=time.time()))+'\n')
        os.execv(PYTHON,[PYTHON,str(ROOT/'operational/chimera-20261106/pool_controller.py')])
    time.sleep(2)
raise RuntimeError('Reconcile pending transition before retry')
