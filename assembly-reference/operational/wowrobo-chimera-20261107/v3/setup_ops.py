from pathlib import Path
import json,hashlib
ROOT=Path(__file__).resolve().parents[4];HERE=Path(__file__).parent;OPS=HERE.parent
parent=json.loads((ROOT/'parent-run.json').read_text())['run'];cfg=json.loads((ROOT/'handoff-cluster.json').read_text())
script=OPS/'cluster.py';s=script.read_text().replace("RUN=ROOT/'run-v2'","RUN=ROOT/'run-v3'").replace("assets_local/wowrobo/training-debug-v1","assets_local/wowrobo/training-minimum-24-v1");script.write_text(s)
(OPS/'cluster.json').write_text(json.dumps(cfg,indent=2)+'\n')
for name in ['monitor.py','failover.py']:
 p=OPS/name;p.write_text(p.read_text().replace("RUN=ROOT/'run-v2'","RUN=ROOT/'run-v3'"))
# Freeze all runtime inputs and this model/bank; avoid rescanning large grids every heartbeat.
files={}
for base in [ROOT/'assembly-so101/assembly_policy',ROOT/'assembly-so101/configs',ROOT/'robotics-assembly-3.0/assembly_policy/so101',ROOT/'robotics-assembly-3.0/outputs/equilibrium-model-20261107/profile-centered-v6',ROOT/'robotics-assembly-3.0/assets_local/wowrobo/training-minimum-24-v1']:
 for p in base.rglob('*'):
  if p.is_file() and p.suffix in ('.py','.json','.xml','.npz') and '__pycache__' not in p.parts:files[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
(ROOT/'active-inputs.json').write_text(json.dumps(files,indent=2)+'\n')
(RUN:=ROOT/'run-v3').mkdir(exist_ok=True)
config=dict(enabled=True,run_dir=str(RUN),root=str(ROOT/'assembly-so101'),python=cfg['python'] if 'python' in cfg else '/large_storage/goodarzilab/nikitamounier/assembly/cube-trial-20261107-v6/.venv/bin/python',worker_script=str(OPS/'worker.sbatch'),worker_script_sha256=hashlib.sha256((OPS/'worker.sbatch').read_bytes()).hexdigest(),base_job_ids=cfg['cpu_jobs'],max_chimera_cpus=1000,worker_slots=100,cloud_workers=0,cpus_per_worker=10,max_jobs_per_cycle=10,progress_fresh_s=1200,scope='DIAGNOSTIC_MINIMUM_VALID_RESETS')
state=ROOT/'autoscale-v3';state.mkdir(exist_ok=True);(state/'active-run.json').write_text(json.dumps(config,indent=2)+'\n')
result=dict(root=str(ROOT),run=str(RUN),ops=str(OPS),pins={name:hashlib.sha256((OPS/name).read_bytes()).hexdigest() for name in ['monitor.py','autoscale.py','failover.py','cluster.py','worker.sbatch']},frozen_inputs=len(files),baseline_cpus=len(cfg['cpu_jobs'])*10,gpu_job=cfg['gpu_job'],parent=parent)
(ROOT/'ops-v3.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
