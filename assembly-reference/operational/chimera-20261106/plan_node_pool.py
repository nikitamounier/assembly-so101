"""Allocated scheduler-only per-node screening and explicit single-node submission."""
import datetime
import json
import os
from pathlib import Path
import re
import subprocess
import time

ROOT=Path('/large_storage/goodarzilab/nikitamounier/assembly/source-85c006a')
OUT=ROOT/'artifacts_local/chimera'
RUN=ROOT/'runs/chimera-v13-8192-20261106'
assert os.environ.get('SLURM_JOB_ID'),'Run scheduler orchestration inside an allocation'
SUBMIT_ENV={k:v for k,v in os.environ.items() if not k.startswith(('SLURM_','SRUN_','SBATCH_'))}
now=datetime.datetime.now();deadline=now+datetime.timedelta(minutes=5)
inventory=subprocess.check_output(['sinfo','-p','cpu_preemptible','-N','-h','-o','%N|%C|%t'],text=True)
raw=[];plan=[]
for line in inventory.splitlines():
    node,counts,state=line.split('|')
    if state not in ('mix','idle','alloc'):continue
    for cpus in range(140,0,-10):
        command=['sbatch','--test-only','-p','cpu_preemptible','--qos=preempt','--nodes=1','--nodelist='+node,'--ntasks='+str(cpus//2),'--cpus-per-task=2','--mem-per-cpu=2G','--time=02:00:00','--job-name=so101-node-'+node,'--wrap=true']
        r=subprocess.run(command,capture_output=True,text=True,timeout=30,env=SUBMIT_ENV);body=r.stdout+r.stderr;raw.append(dict(node=node,cpus=cpus,returncode=r.returncode,output=body))
        match=re.search(r'to start at (\S+)',body)
        if r.returncode==0 and match and datetime.datetime.fromisoformat(match[1])<=deadline:
            plan.append(dict(node=node,cpus=cpus,workers=cpus//2,start_projection=match[1]));print(json.dumps(plan[-1]),flush=True);break
OUT.mkdir(parents=True,exist_ok=True)
(OUT/'per-node-projections.json').write_text(json.dumps(dict(checked_local=now.isoformat(),deadline=deadline.isoformat(),inventory=inventory,checks=raw,candidates=plan),indent=2)+'\n')
selected=[];total=0
for row in sorted(plan,key=lambda r:(-r['cpus'],r['node'])):
    count=min(row['cpus'],1500-total)//10*10
    if count<10:continue
    selected.append({**row,'cpus':count,'workers':count//2});total+=count
if not selected:raise RuntimeError('No node request projects within five minutes; no jobs submitted')
assert len({r['node'] for r in selected})==len(selected)
manifest=dict(generation=time.strftime('%Y%m%dT%H%M%SZ',time.gmtime()),gpu_job='3598717',retired_cpu_jobs=['3598730'],target_cpus=0,target_workers=0,jobs=[],profile='cpu_preemptible/two-hour/per-node/2threads/2GiB-perCPU',projected_cpus=total)
path=OUT/'node-pool.json';(RUN/'node-workers').mkdir(parents=True,exist_ok=True)
for row in selected:
    command=['sbatch','--parsable','--nodelist='+row['node'],'--ntasks='+str(row['workers']),'--job-name=so101-node-'+row['node'],'--output='+str(RUN/'node-workers/%j.log'),str(ROOT/'operational/chimera-20261106/node_workers.sbatch')]
    r=subprocess.run(command,capture_output=True,text=True,timeout=30,env=SUBMIT_ENV)
    if r.returncode:print(json.dumps(dict(submit_error=row,stderr=r.stderr)),flush=True);continue
    job=r.stdout.strip().split(';')[0];assert job.isdigit(),r.stdout
    manifest['jobs'].append({**row,'job_id':job});manifest['target_cpus']+=row['cpus'];manifest['target_workers']+=row['workers']
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(manifest,indent=2)+'\n');temp.replace(path)
    print(json.dumps(dict(submitted=job,**row)),flush=True)
assert manifest['jobs'],'All submissions failed'
print(json.dumps(dict(manifest=str(path),target_cpus=manifest['target_cpus'],workers=manifest['target_workers'],nodes=len(manifest['jobs']))),flush=True)
