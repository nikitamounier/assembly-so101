"""Retire failed native unwinding and restore full-batch profiling safely."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from controller import ROOT,RUN,GPU,PYTHON,STATE,native

assert os.environ.get('SLURM_JOB_ID')=='3077705'
out=ROOT/'artifacts_local/chimera/ppo-profile-20261106'
state=json.loads(STATE.read_text());assert state['phase']=='learner_failed'
lock=open(STATE.with_suffix('.lock'),'a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);lock.close()
sys.path.insert(0,str(ROOT));import torch
ck=torch.load(RUN/'checkpoint.pt',map_location='cpu',weights_only=False)
last=json.loads((RUN/'log.jsonl').read_text().splitlines()[-1])
assert ck['it']==last['it']==28 and ck['transitions']==last['transitions']
old=json.loads((out/'status.json').read_text())
Path(old['path'],'rejection.json').write_text(json.dumps(dict(reason='Both asynchronously native-sampled workers stalled; closed-connection export failed after checkpoint. Invalid interruption window.',checkpoint_iteration=ck['it'],transitions=ck['transitions']))+'\n')
guardpath=ROOT/'artifacts_local/chimera/source-hashes.json';guard=json.loads(guardpath.read_text())
for name,candidate in [('profile_train.py','profile-train-safe.py'),('profile_worker.py','profile-worker-safe.py')]:
    relative='operational/chimera-20261106/'+name
    current=(ROOT/relative).read_bytes();assert hashlib.sha256(current).hexdigest()==guard[relative]
    (out/('failed-native-'+name)).write_bytes(current)
    data=(ROOT/'artifacts_local/chimera'/candidate).read_bytes();compile(data,relative,'exec')
    (ROOT/relative).write_bytes(data);guard[relative]=hashlib.sha256(data).hexdigest()
for name in ['profile_worker_entry.py','profiler_clock.c','repair_safe_profile.py']:
    relative='operational/chimera-20261106/'+name
    guard[relative]=hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()
guardpath.write_text(json.dumps(guard,indent=2)+'\n')
(out/'worker-config.json').write_text(json.dumps(dict(jobs=[GPU],expected_workers=2))+'\n')
for job in ['3602571','3602576']:
    subprocess.run(['scontrol','requeuehold',job],check=True,timeout=30)
subprocess.run(['scontrol','hold','3602572'],check=True,timeout=30)
poolpath=ROOT/'artifacts_local/chimera/node-pool.json';pool=json.loads(poolpath.read_text())
active_jobs=[r['job_id'] for r in pool['jobs'] if native(r['job_id'])['JobState']=='RUNNING']
assert len(active_jobs)==37,active_jobs
pool.update(target_workers=187,profile_native_jobs=[GPU,*active_jobs],profile_extra_gpu_workers=2,profile_held_jobs=['3602571','3602572','3602576'])
poolpath.write_text(json.dumps(pool,indent=2)+'\n')
env={k:v for k,v in os.environ.items() if not k.startswith(('SLURM_','SRUN_')) and k!='CUDA_VISIBLE_DEVICES'}
env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',MUJOCO_GL='egl',CUDA_VISIBLE_DEVICES='')
epoch=time.time()
log=open(out/'gpu-cpu-workers.log','a')
worker=subprocess.Popen(['srun',f'--jobid={GPU}','--overlap','--exact','--gpus=0','--ntasks=2','--cpus-per-task=2',PYTHON,str(ROOT/'operational/chimera-20261106/profile_worker_entry.py'),'--run-dir',str(RUN),'--newer-than',str(epoch)],env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
(out/'safe-worker-step.json').write_text(json.dumps(dict(srun_pid=worker.pid,job=GPU,ntasks=2,cpus_per_task=2,epoch=epoch))+'\n')
# Retain this CPU control step as normal learner coordinator; the diagnostic
# worker step is an independent child within the retained GPU allocation.
(out/'status.json').write_text(json.dumps(dict(phase='retry_ready',checkpoint_iteration=28,workers=187,time=time.time()))+'\n')
os.execv(PYTHON,[PYTHON,str(ROOT/'operational/chimera-20261106/pool_controller.py')])
