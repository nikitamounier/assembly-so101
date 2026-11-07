import fcntl,hashlib,json,os,sys,time
from pathlib import Path
ROOT=Path('/large_storage/goodarzilab/nikitamounier/assembly/source-85c006a');STATE=ROOT/'artifacts_local/chimera/controller.json';RUN=ROOT/'runs/chimera-v13-8192-20261106'
assert os.environ.get('SLURM_JOB_ID')=='3077705'
state=json.loads(STATE.read_text());assert state['phase']=='learner_failed'
lock=open(STATE.with_suffix('.lock'),'a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);lock.close()
sys.path.insert(0,str(ROOT));import torch
ck=torch.load(RUN/'checkpoint.pt',map_location='cpu',weights_only=False);last=json.loads((RUN/'log.jsonl').read_text().splitlines()[-1]);assert ck['it']==last['it'] and ck['transitions']==last['transitions']
manifest=ROOT/'artifacts_local/chimera/source-hashes.json';guard=json.loads(manifest.read_text());name='operational/chimera-20261106/pool_controller.py';old=(ROOT/name).read_bytes();assert hashlib.sha256(old).hexdigest()==guard[name]
backup=ROOT/'artifacts_local/chimera/pool-controller-before-gpu-requeue.py';backup.write_bytes(old)
data=(ROOT/'artifacts_local/chimera/pool-controller-gpu-requeue.py').read_bytes();compile(data,name,'exec');(ROOT/name).write_bytes(data);guard[name]=hashlib.sha256(data).hexdigest()
name='assembly_policy/rl/remote.py';assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==guard[name]
data=(ROOT/'artifacts_local/chimera/ppo-profile-20261106/original-remote.py').read_bytes();compile(data,name,'exec');(ROOT/name).write_bytes(data);guard[name]=hashlib.sha256(data).hexdigest()
manifest.write_text(json.dumps(guard,indent=2)+'\n')
(ROOT/'artifacts_local/chimera/gpu-requeue-repair.json').write_text(json.dumps(dict(checkpoint_iteration=ck['it'],transitions=ck['transitions'],time=time.time(),controller_sha256=guard['operational/chimera-20261106/pool_controller.py']))+'\n')
os.execv(str(ROOT/'.venv/bin/python'),[str(ROOT/'.venv/bin/python'),str(ROOT/'operational/chimera-20261106/pool_controller.py')])
