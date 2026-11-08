"""Configure the validated 4,092-core continuation and retain all cluster leases."""
from pathlib import Path
import json,hashlib
ROOT=Path(__file__).resolve().parents[4];OPS=Path(__file__).resolve().parent.parent
assert json.loads((ROOT/'reset-validation-v4.json').read_text())['status']=='PASS'
cfg=json.loads((OPS/'cluster.json').read_text());cfg.update(envs=5090,worker_slots=509)
(OPS/'cluster.json').write_text(json.dumps(cfg,indent=2)+'\n')
for name in ['cluster.py','monitor.py','failover.py']:
 p=OPS/name;s=p.read_text();assert "RUN=ROOT/'run-v3'" in s;s=s.replace("RUN=ROOT/'run-v3'","RUN=ROOT/'run-v4'")
 s=s.replace('assets_local/wowrobo/training-minimum-24-v1','assets_local/wowrobo/training-accepted-v4')
 if name=='cluster.py':
  s=s.replace("'--steps','32'","'--steps','32','--micro-batch','1024'")
  # Keep failed or closed clients from consuming one entire native interval per retry.
 p.write_text(s)
# A closed handshake must not prevent draining the next pending worker this step.
p=ROOT/'assembly-so101/assembly_policy/rl/remote.py';s=p.read_text();a='except (OSError, EOFError, socket.timeout, AuthenticationError):\n                return\n            busy =';b='except (OSError, EOFError, AuthenticationError):\n                continue\n            except socket.timeout:\n                return\n            busy =';assert a in s;s=s.replace(a,b);p.write_text(s)
files={}
for base in [ROOT/'assembly-so101/assembly_policy',ROOT/'assembly-so101/configs',ROOT/'robotics-assembly-3.0/assembly_policy/so101',ROOT/'robotics-assembly-3.0/outputs/equilibrium-model-20261107/profile-centered-v6',ROOT/'robotics-assembly-3.0/assets_local/wowrobo/training-accepted-v4']:
 for p in base.rglob('*'):
  if p.is_file() and p.suffix in ('.py','.cc','.json','.xml','.npz') and '__pycache__' not in p.parts:files[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
(ROOT/'active-inputs.json').write_text(json.dumps(files,indent=2)+'\n')
(ROOT/'run-v4').mkdir(exist_ok=True)
state=ROOT/'autoscale-v4';state.mkdir(exist_ok=True)
config=dict(enabled=True,run_dir=str(ROOT/'run-v4'),root=str(ROOT/'assembly-so101'),python='/large_storage/goodarzilab/nikitamounier/assembly/cube-trial-20261107-v6/.venv/bin/python',worker_script=str(OPS/'worker.sbatch'),worker_script_sha256=hashlib.sha256((OPS/'worker.sbatch').read_bytes()).hexdigest(),base_job_ids=cfg['cpu_jobs'],max_chimera_cpus=1000,worker_slots=509,cloud_workers=409,cpus_per_worker=10,max_jobs_per_cycle=10,progress_fresh_s=1200,scope='DIAGNOSTIC_ACCEPTED_RESET_HYBRID')
# Worker submissions point at this new run rather than the copied v3 path.
p=OPS/'worker.sbatch';s=p.read_text().replace('run-v3','run-v4');p.write_text(s);config['worker_script_sha256']=hashlib.sha256(p.read_bytes()).hexdigest()
(state/'active-run.json').write_text(json.dumps(config,indent=2)+'\n')
result=dict(root=str(ROOT),run=str(ROOT/'run-v4'),frozen_inputs=len(files),config=cfg,pins={n:hashlib.sha256((OPS/n).read_bytes()).hexdigest() for n in ['cluster.py','monitor.py','failover.py','autoscale.py','worker.sbatch']},modal_rollout_physical_cores=4092,modal_worker_processes=409,maximum_chimera_rollout_cpus=1000,target_batch=5090*32,micro_batch=1024,ppo_epochs_and_minibatches_unchanged=True)
(ROOT/'ops-v4.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
