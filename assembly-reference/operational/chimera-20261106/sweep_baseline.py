"""Use already idle bank-builder leases for remaining original reset tasks."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

assert os.environ.get('SLURM_JOB_ID')=='3077705'
ROOT=Path('/large_storage/goodarzilab/nikitamounier/assembly/assembly-nist-eb5d15e')
sys.path.insert(0,str(ROOT))
from assembly_policy.sim.tasks import resolve_tasks
lock=open(ROOT/'artifacts_local/chimera/sweep.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
pool=json.loads((ROOT/'artifacts_local/chimera/node-pool.json').read_text())
env={k:v for k,v in os.environ.items() if not k.startswith(('SLURM_','SRUN_')) and k!='CUDA_VISIBLE_DEVICES'}
env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='')
children=[]
for row in pool['jobs']:
    index=row['index']
    if index>=5 or not (ROOT/'data/banks'/f'{resolve_tasks("all20")[index]}_train.npz').exists():continue
    job=row['job_id']
    p=subprocess.run(['scontrol','show','job',job,'-o'],capture_output=True,text=True,timeout=30)
    if 'JobState=RUNNING' not in p.stdout:continue
    log=open(ROOT/f'runs/profile/node-workers/{job}-sweep.log','a')
    child=subprocess.Popen(['srun',f'--jobid={job}','--overlap','--exact','--ntasks=1','--cpus-per-task=10','/large_storage/goodarzilab/nikitamounier/assembly/source-85c006a/.venv/bin/python',str(ROOT/'operational/baseline_banks.py'),'sweep'],env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    children.append((job,child))
print(json.dumps(dict(launched=[dict(job=j,srun_pid=p.pid) for j,p in children])),flush=True)
for job,child in children:print(json.dumps(dict(job=job,exit=child.wait())),flush=True)
