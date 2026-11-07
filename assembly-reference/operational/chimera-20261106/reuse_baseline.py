"""Checkpoint SO101, compare original NIST on the SAME leases, resume SO101.

Only original-NIST bank leases are cancelled. Shared SO101 allocations are
never cancelled. Original SO101 workers remain waiting while separate NIST
worker steps temporarily use the CPUs; only those temporary steps are stopped.
"""
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

assert os.environ.get('SLURM_JOB_ID')=='3077705'
SOURCE=Path('/large_storage/goodarzilab/nikitamounier/assembly/source-85c006a')
ROOT=SOURCE.parent/'assembly-nist-eb5d15e'
OUT=ROOT/'artifacts_local/chimera/ppo-profile-20261106'
SOOUT=SOURCE/'artifacts_local/chimera/ppo-profile-20261106'
PYTHON=str(SOURCE/'.venv/bin/python')
RUN=SOURCE/'runs/chimera-v13-8192-20261106'
SO_STATE=SOURCE/'artifacts_local/chimera/controller.json'
MANIFEST=ROOT/'artifacts_local/chimera/node-pool.json'
lock=open(OUT/'reuse.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
env={k:v for k,v in os.environ.items() if not k.startswith(('SLURM_','SRUN_','SBATCH_')) and k!='CUDA_VISIBLE_DEVICES'}
env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',MUJOCO_GL='egl')


def write(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)


def native(job):
    p=subprocess.run(['scontrol','show','job',job,'-o'],capture_output=True,text=True,timeout=30)
    if p.returncode:
        history=subprocess.run(['sacct','-X','-j',job,'--format=JobIDRaw,JobName%80,User,State','--parsable2','--noheader'],capture_output=True,text=True,timeout=30)
        rows=[line.split('|') for line in history.stdout.splitlines() if line.split('|')[0]==job]
        assert len(rows)==1,(job,p.stderr,history.stderr)
        _,name,user,state=rows[0]
        return dict(JobState=state.split()[0],JobName=name,UserId=user+'('+str(os.getuid())+')')
    return dict(x.split('=',1) for x in p.stdout.split() if '=' in x)


def processes(ending):
    found=[]
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():continue
        try:
            if proc.stat().st_uid!=os.getuid():continue
            args=(proc/'cmdline').read_bytes().decode().split('\0')
            cwd=(proc/'cwd').resolve()
            if any((Path(arg) if Path(arg).is_absolute() else cwd/arg).resolve()==Path(ending).resolve() for arg in args[1:] if arg):found.append(int(proc.name))
        except (OSError,UnicodeError):pass
    return found


def launch(job,cpus,command,log,ntasks=1,gpu_cpu=False):
    stream=open(log,'a')
    flags=['--gpus=0'] if gpu_cpu else []
    return subprocess.Popen(['srun',f'--jobid={job}','--overlap','--exact',f'--ntasks={ntasks}',f'--cpus-per-task={cpus}',*flags,*command],env={**env,'CUDA_VISIBLE_DEVICES':''} if gpu_cpu else env,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)


pool=json.loads(MANIFEST.read_text())
assert pool['gpu_job'] is None,'Reconcile already submitted baseline GPU first'
assert not (ROOT/'artifacts_local/chimera/ALL_BANKS_READY').exists()
bank_manifest=OUT/'bank-manifest.json'
if not bank_manifest.exists():write(bank_manifest,pool)
for pid in processes(str(ROOT/'operational/baseline_controller.py')):
    os.kill(pid,signal.SIGTERM)
deadline=time.monotonic()+30
while processes(str(ROOT/'operational/baseline_controller.py')):
    assert time.monotonic()<deadline
    time.sleep(.5)
bank_jobs=[r['job_id'] for r in pool['jobs']]
pending=[]
for job in bank_jobs:
    row=native(job)
    assert row['JobName'].startswith('nist-profile-cpu10-') and row['UserId'].startswith('nikitamounier(')
    if row['JobState']=='PENDING':pending.append(job)
if pending:subprocess.run(['scancel','--state=PENDING',','.join(pending)],check=True,timeout=60)

while True:
    banks=list((ROOT/'data/banks').glob('*_train.npz'))
    so=json.loads((SOOUT/'status.json').read_text())
    write(OUT/'controller.json',dict(phase='waiting_banks_and_so_capture',banks=len(banks),tiles=len(list((ROOT/'data/banks/tiles').glob('*.npz'))),so_capture=so,time=time.time(),bank_jobs=bank_jobs))
    if len(banks)==20 and so.get('qualified') is True:break
    time.sleep(10)

sys.path.insert(0,str(ROOT))
import numpy as np
from assembly_policy.sim.tasks import resolve_tasks
for task in resolve_tasks('all20'):
    z=np.load(ROOT/'data/banks'/f'{task}_train.npz')
    assert len(z['states'])==3200 and np.isfinite(z['states']).all()
    assert np.array_equal(np.bincount(z['family']),np.full(5,640))
for job in bank_jobs:
    row=native(job)
    if row['JobState'] not in ['CANCELLED','COMPLETED','FAILED','TIMEOUT','OUT_OF_MEMORY']:
        subprocess.run(['scancel',job],check=True,timeout=30)

# Obtain a normal durable SO101 boundary, leaving its CPU jobs and GPU intact.
so_state=json.loads(SO_STATE.read_text())
if so_state['phase'] not in ['waiting_allocations','learner_failed']:
    stop=launch('3598717',1,[PYTHON,str(SOURCE/'operational/chimera-20261106/learner.py'),'--stop'],OUT/'stop-so.log')
    assert stop.wait(timeout=60)==0
deadline=time.monotonic()+900
while True:
    state=json.loads(SO_STATE.read_text())
    if state['phase'] in ['waiting_allocations','learner_failed']:
        # The normal controller may be waiting for lease renewal; stop only
        # that coordinator after it has confirmed the learner already exited.
        for pid in processes(str(SOURCE/'operational/chimera-20261106/pool_controller.py')):
            os.kill(pid,signal.SIGTERM)
        slock=open(SO_STATE.with_suffix('.lock'),'a')
        try:fcntl.flock(slock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:slock.close();time.sleep(1);continue
        slock.close();break
    assert time.monotonic()<deadline,'SO101 checkpoint boundary timed out'
    time.sleep(2)

import torch
ck=torch.load(RUN/'checkpoint.pt',map_location='cpu',weights_only=False)
last=json.loads((RUN/'log.jsonl').read_text().splitlines()[-1])
assert ck['it']==last['it'] and ck['transitions']==last['transitions']
write(OUT/'so-checkpoint-handoff.json',dict(it=ck['it'],transitions=ck['transitions'],time=time.time()))
so_pool=json.loads((SOURCE/'artifacts_local/chimera/node-pool.json').read_text())
capture=json.loads((Path(so['path'])/'result.json').read_text())
cpu_jobs=[j for j in capture['jobs_after'] if j!='3598717']
extra=so_pool.get('profile_extra_gpu_workers',0)
pool=dict(generation='nist-same-hardware-profile',target_workers=len(cpu_jobs)*5+extra,target_cpus=len(cpu_jobs)*10+extra*2,gpu_job='3598717',profile_native_jobs=['3598717',*cpu_jobs],shared_allocations=True,jobs=[dict(job_id=j,cpus=10,workers=5) for j in cpu_jobs])
write(MANIFEST,pool)

# Preserve exact CPU allocations whenever possible; wait for their requeues.
while True:
    rows={j:native(j) for j in pool['profile_native_jobs']}
    gpu=rows['3598717']
    remaining=time.mktime(time.strptime(gpu['EndTime'],'%Y-%m-%dT%H:%M:%S'))-time.time() if gpu['JobState']=='RUNNING' else 0
    write(OUT/'controller.json',dict(phase='waiting_same_hardware',jobs=rows,time=time.time(),remaining_gpu_s=remaining,banks=20))
    if all(r['JobState']=='RUNNING' for r in rows.values()) and remaining>600:break
    time.sleep(10)
selected=['3598717'] if extra else cpu_jobs[:2]
write(OUT/'worker-config.json',dict(jobs=selected,expected_workers=extra if extra else 2,all_ranks=bool(extra)))
write(OUT/'hardware-match.json',dict(so_jobs=capture['jobs_after'],nist_jobs={j:{k:r.get(k) for k in ['JobState','NodeList','Restarts','StartTime','EndTime']} for j,r in rows.items()},gpu_job='3598717',so_checkpoint_iteration=ck['it']))
workers=[]
# SO101's diagnostic CPU workers are idle now; stop only their owned srun step.
safe_step=SOOUT/'safe-worker-step.json'
if safe_step.exists():
    pid=json.loads(safe_step.read_text())['srun_pid'];proc=Path('/proc')/str(pid)
    if proc.exists() and (proc/'stat').read_text().split()[2]!='Z':
        args=(proc/'cmdline').read_bytes().decode().split('\0')
        assert '--jobid=3598717' in args and str(SOURCE/'operational/chimera-20261106/profile_worker_entry.py') in args
        os.kill(pid,signal.SIGTERM)
        deadline=time.monotonic()+60
        while proc.exists() and (proc/'stat').read_text().split()[2]!='Z':
            assert time.monotonic()<deadline
            time.sleep(.5)
for job in cpu_jobs:
    workers.append(launch(job,2,[PYTHON,str(ROOT/'operational/worker_entry.py'),'--run-dir',str(ROOT/'runs/profile'),'--nthread','2','--wait','7200','--poll','1'],ROOT/f'runs/profile/node-workers/reused-{job}.log',5))
if extra:
    # Same spare GPU-host CPUs as SO101's two sampled workers. These are CPU
    # simulators and do not request or use another GPU.
    workers.append(launch('3598717',2,[PYTHON,str(ROOT/'operational/worker_entry.py'),'--run-dir',str(ROOT/'runs/profile'),'--nthread','2','--wait','7200','--poll','.01'],ROOT/'runs/profile/node-workers/reused-gpu-cpu.log',extra,gpu_cpu=True))
command=[PYTHON,str(ROOT/'operational/profile_train.py'),'assembly_policy.train','--run-dir',str(ROOT/'runs/profile'),'--tasks','all20','--bank-dir',str(ROOT/'data/banks'),'--device','cuda','--envs','8192','--steps','32','--iters','10000','--remote-workers',str(pool['target_workers']),'--accept-timeout','600','--min-conn-frac','1','--nthread','8','--save-every','1','--keep-every','1000000','--no-eval','--seed','20261105']
write(ROOT/'runs/profile/command.json',command)
write(OUT/'original-command.json',[PYTHON,'-m','assembly_policy.train',*command[3:]])
learner=launch('3598717',8,command,ROOT/'runs/profile/trainer.log')
while learner.poll() is None:
    ready=[json.loads(p.read_text()) for p in OUT.glob('ready-*.json')]
    if all(any(r['job']==j for r in ready) for j in selected):(OUT/'workers-ready').touch()
    status=json.loads((OUT/'status.json').read_text()) if (OUT/'status.json').exists() else {}
    write(OUT/'controller.json',dict(phase='running',capture=status,time=time.time(),banks=20,shared_allocations=True,worker_steps=len(workers)))
    time.sleep(5)
code=learner.returncode
status=json.loads((OUT/'status.json').read_text())
for child in workers:
    if child.poll() is None:child.terminate()
for child in workers:
    try:child.wait(timeout=30)
    except subprocess.TimeoutExpired:child.kill();child.wait(timeout=10)
write(OUT/'controller.json',dict(phase='complete' if code==0 and status.get('qualified') else 'needs_attention',capture=status,learner_exit=code,shared_allocations=True,bank_jobs_released=bank_jobs,time=time.time()))
# Always restore SO101 training, even if comparison profiling failed.
for job in so_pool.pop('profile_held_jobs',[]):
    subprocess.run(['scontrol','release',job],check=True,timeout=30)
so_pool['target_workers']=200
for field in ['profile_native_jobs','profile_note','profile_extra_gpu_workers']:so_pool.pop(field,None)
write(SOURCE/'artifacts_local/chimera/node-pool.json',so_pool)
os.chdir(SOURCE)
os.execv(PYTHON,[PYTHON,str(SOURCE/'operational/chimera-20261106/pool_controller.py')])
