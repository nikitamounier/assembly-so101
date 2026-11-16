"""Prepare diagnostic wrappers around unchanged original NIST source."""
import hashlib
import json
import os
from pathlib import Path
import shutil

assert os.environ.get('SLURM_JOB_ID')=='3077705'
SOURCE=Path('/large_storage/goodarzilab/nikitamounier/assembly/source-85c006a')
ROOT=SOURCE.parent/'assembly-nist-eb5d15e'
OPS=ROOT/'operational';OPS.mkdir(exist_ok=True)
OUT=ROOT/'artifacts_local/chimera/ppo-profile-20261106';OUT.mkdir(parents=True,exist_ok=True)
(ROOT/'runs/profile/node-workers').mkdir(parents=True,exist_ok=True)
for name in ['baseline_banks.py','baseline_workers.sbatch','baseline_gpu.sbatch']:
    shutil.copyfile(SOURCE/'operational/chimera-20261106'/name,OPS/name)
worker=(SOURCE/'operational/chimera-20261106/profile_worker.py').read_text()
worker=worker.replace('ROOT=Path(__file__).resolve().parents[2]',f'ROOT=Path({str(ROOT)!r})')
(OPS/'profile_worker.py').write_text(worker)
entry='''import json,os,sys
from pathlib import Path
ROOT=Path(%r)
sys.path.insert(0,str(ROOT))
cfg=ROOT/'artifacts_local/chimera/ppo-profile-20261106/worker-config.json'
if cfg.exists() and os.environ.get('SLURM_JOB_ID') in json.loads(cfg.read_text())['jobs'] and (json.loads(cfg.read_text()).get('all_ranks') or os.environ.get('SLURM_PROCID')=='0'):
    import profile_worker
import argparse
from assembly_policy.rl.remote import worker_main
p=argparse.ArgumentParser();p.add_argument('--run-dir');p.add_argument('--nthread',type=int);p.add_argument('--wait',type=float);p.add_argument('--poll',type=float)
a=p.parse_args();worker_main(a.run_dir,a.nthread,a.wait,a.poll)
''' % str(ROOT)
(OPS/'worker_entry.py').write_text(entry)
learner=(SOURCE/'operational/chimera-20261106/profile_train.py').read_text()
learner=learner.replace('ROOT = Path(__file__).resolve().parents[2]',f'ROOT = Path({str(ROOT)!r})')
learner=learner.replace("assert os.environ.get('SLURM_JOB_ID') == '3598717'", "assert os.environ.get('SLURM_JOB_ID') == json.loads((ROOT/'artifacts_local/chimera/node-pool.json').read_text())['gpu_job']")
learner=learner.replace("['3598717', *[str(r['job_id']) for r in pool['jobs']]]", "[pool['gpu_job'], *[str(r['job_id']) for r in pool['jobs']]]")
learner=learner.replace("write(self.run / 'command.json', json.loads(original.read_text()))", "write(self.run / 'command.json', json.loads(original.read_text()))\n                self.a.iters = self.it\n                self.env.close()")
(OPS/'profile_train.py').write_text(learner)
manifest={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for parent in ['assembly_policy','plugin','assets'] for p in (ROOT/parent).rglob('*') if p.is_file() and '__pycache__' not in str(p) and 'build/' not in str(p)}
(ROOT/'artifacts_local/chimera/source-hashes.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(dict(root=str(ROOT),source='eb5d15e',guard_files=len(manifest))))
