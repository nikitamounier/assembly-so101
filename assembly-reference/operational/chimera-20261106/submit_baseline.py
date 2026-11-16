"""Submit independent ten-CPU leases for original-NIST bank/profile work."""
import json
import os
from pathlib import Path
import subprocess

assert os.environ.get('SLURM_JOB_ID')=='3077705'
ROOT=Path('/large_storage/goodarzilab/nikitamounier/assembly/assembly-nist-eb5d15e')
manifest=ROOT/'artifacts_local/chimera/node-pool.json'
assert not manifest.exists(),'Refuse duplicate submission; reconcile native jobs first'
env={k:v for k,v in os.environ.items() if not k.startswith(('SLURM_','SRUN_','SBATCH_')) and k!='CUDA_VISIBLE_DEVICES'}
pool=dict(generation='nist-profile-eb5d15e-20261107',target_cpus=400,target_workers=200,gpu_job=None,jobs=[])
manifest.write_text(json.dumps(pool,indent=2)+'\n')
for i in range(40):
    result=subprocess.run(['sbatch','--parsable',f'--job-name=nist-profile-cpu10-{i}',f'--output={ROOT}/runs/profile/node-workers/%j.log',str(ROOT/'operational/baseline_workers.sbatch'),str(i)],env=env,cwd=ROOT,capture_output=True,text=True,timeout=30)
    assert result.returncode==0,result.stderr
    job=result.stdout.strip().split(';')[0];assert job.isdigit(),result.stdout
    pool['jobs'].append(dict(job_id=job,cpus=10,workers=5,index=i))
    temp=manifest.with_suffix('.tmp');temp.write_text(json.dumps(pool,indent=2)+'\n');temp.replace(manifest)
print(json.dumps(pool))
