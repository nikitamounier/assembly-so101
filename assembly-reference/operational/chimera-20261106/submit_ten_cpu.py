"""Submit a fixed pool of portable ten-CPU jobs from an allocated controller."""
import json
import os
from pathlib import Path
import subprocess
import time

assert os.environ.get('SLURM_JOB_ID'),'Requires an allocated step'
ROOT=Path('/large_storage/goodarzilab/nikitamounier/assembly/source-85c006a')
OUT=ROOT/'artifacts_local/chimera';RUN=ROOT/'runs/chimera-v13-8192-20261106'
ENV={k:v for k,v in os.environ.items() if not k.startswith(('SLURM_','SRUN_','SBATCH_'))}
path=OUT/'node-pool.json';old=json.loads(path.read_text())
path.with_name('node-pool-before-ten-cpu.json').write_bytes(path.read_bytes())
manifest=dict(generation=time.strftime('%Y%m%dT%H%M%SZ',time.gmtime()),gpu_job='3598717',retired_cpu_jobs=old.get('retired_cpu_jobs',[])+[r['job_id'] for r in old['jobs']],target_cpus=0,target_workers=0,jobs=[],profile='cpu_preemptible/two-hour/ten-CPU-unpinned-jobs/2threads/2GiB-perCPU',requested_jobs=40,cpus_per_job=10,min_start_workers=5)
for index in range(40):
    command=['sbatch','--parsable','--job-name=so101-cpu10-'+str(index),'--output='+str(RUN/'node-workers/%j.log'),str(ROOT/'operational/chimera-20261106/node_workers.sbatch')]
    result=subprocess.run(command,capture_output=True,text=True,env=ENV,timeout=30)
    if result.returncode:print(json.dumps(dict(index=index,error=result.stderr)),flush=True);continue
    job=result.stdout.strip().split(';')[0];assert job.isdigit(),result.stdout
    manifest['jobs'].append(dict(job_id=job,node='scheduler-selected',cpus=10,workers=5,index=index));manifest['target_cpus']+=10;manifest['target_workers']+=5
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(manifest,indent=2)+'\n');temp.replace(path)
    print(json.dumps(dict(index=index,job_id=job,cpus=10,workers=5)),flush=True)
assert manifest['jobs'],'No jobs submitted'
print(json.dumps(dict(jobs=len(manifest['jobs']),cpus=manifest['target_cpus'],workers=manifest['target_workers'],generation=manifest['generation'])),flush=True)
