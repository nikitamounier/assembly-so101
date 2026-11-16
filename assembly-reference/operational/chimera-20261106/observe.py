"""Compact allocated-only status for this assembly experiment; never reads server keys."""
import collections
import calendar
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time

ROOT=Path('/large_storage/goodarzilab/nikitamounier/assembly/source-85c006a')
RUN=ROOT/'runs/chimera-v13-8192-20261106'
assert os.environ.get('SLURM_JOB_ID'),'Monitoring must execute inside a Slurm allocation'

def read_json(path):
    try:return json.loads(path.read_text())
    except FileNotFoundError:return None

def tail(path,count):
    try:
        with path.open() as stream:return list(collections.deque(stream,maxlen=count))
    except FileNotFoundError:return []

def attempt_tail(path):
    lines=tail(path,200)
    starts=[i for i,line in enumerate(lines) if line.startswith(('[assembly-step]','[assembly-node-attempt]'))]
    return ''.join(lines[starts[-1]:][-8:]) if starts else ''.join(lines[-8:])

state=read_json(ROOT/'artifacts_local/chimera/controller.json')
checks={};manifest=read_json(ROOT/'artifacts_local/chimera/source-hashes.json')
if manifest:
    for name,expected in manifest.items():
        path=ROOT/name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:checks[name]='source_changed'
progress=tail(RUN/'log.jsonl',1)
last=json.loads(progress[0]) if progress else None
finite=last is None or all(math.isfinite(last[k]) for k in ('pi','vf','ent','kl','lr','std'))
pool=read_json(ROOT/'artifacts_local/chimera/node-pool.json')
job_ids=['3598717',*[r['job_id'] for r in pool['jobs']]] if pool else ['3598717','3598730']
jobs={}
for job in job_ids:
    row=subprocess.run(['scontrol','show','job',job,'-o'],text=True,capture_output=True,timeout=30)
    if row.returncode:jobs[job]={'state':'UNKNOWN','error':row.stderr.strip()}
    else:
        fields=dict(x.split('=',1) for x in row.stdout.split() if '=' in x)
        jobs[job]={k:fields.get(k) for k in ('JobState','NodeList','RunTime','EndTime','Restarts','NumCPUs','AllocTRES','Reason')}
checkpoint=RUN/'checkpoint.pt'
summary=dict(observed_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),allocated_monitor_job=os.environ['SLURM_JOB_ID'],source_valid=manifest is not None and not checks,source_errors=checks,controller={k:v for k,v in state.items() if k!='jobs'} if state else None,jobs=jobs,last_update=last,finite=finite,progress_age_s=time.time()-(RUN/'log.jsonl').stat().st_mtime if last else None,checkpoint=dict(bytes=checkpoint.stat().st_size,age_s=time.time()-checkpoint.stat().st_mtime) if checkpoint.exists() else None,disk_free_gib=shutil.disk_usage(ROOT).free/2**30,trainer_tail=attempt_tail(RUN/'trainer.log'),setup_tail=''.join(tail(ROOT/'setup-native.log',4)),worker_tail='' if pool else attempt_tail(RUN/'workers.log'))
errors=[]
if pool:
    summary['pool']=dict(generation=pool['generation'],target_cpus=pool['target_cpus'],target_workers=pool['target_workers'],jobs=len(pool['jobs']),nodes=len({jobs[r['job_id']].get('NodeList') for r in pool['jobs'] if jobs[r['job_id']].get('JobState')=='RUNNING'}),running_cpus=sum(r['cpus'] for r in pool['jobs'] if jobs[r['job_id']].get('JobState')=='RUNNING'),running_workers=sum(r['workers'] for r in pool['jobs'] if jobs[r['job_id']].get('JobState')=='RUNNING'))
    summary['node_worker_errors']={}
    for row in pool['jobs']:
        job=row['job_id'];text=attempt_tail(RUN/'node-workers'/f'{job}.log')
        if any(word in text.lower() for word in ('oom_kill','out of memory')):
            errors.append('worker_memory:'+job);summary['node_worker_errors'][job]=text
        if jobs[job].get('JobState') in ('FAILED','CANCELLED','TIMEOUT','COMPLETED','OUT_OF_MEMORY') and (state or {}).get('phase')!='complete':errors.append('cpu_job_terminal:'+job)
if any(word in summary['worker_tail'].lower() for word in ('oom_kill','out of memory')):errors.append('worker_memory')
if 'no reply in 60 s' in summary['trainer_tail']:errors.append('worker_reply_timeout')
if checks:errors.append('source_integrity')
if not finite:errors.append('nonfinite_update')
if state and state.get('phase') in ('learner_failed','allocation_terminal','checkpoint_request_failed'):errors.append(state['phase'])
if state and time.time()-calendar.timegm(time.strptime(state['observed_utc'],'%Y-%m-%dT%H:%M:%SZ'))>180:errors.append('controller_stale')
if state and state.get('worker_step_exit') is not None and state.get('phase')=='running':errors.append('worker_step_exited')
if summary['disk_free_gib']<5:errors.append('disk_headroom')
summary['errors']=errors
summary['action']='notify_parent' if errors else 'complete_delete_monitor' if state and state.get('phase')=='complete' else 'keep_monitor'
print(json.dumps(summary,sort_keys=True),flush=True)
