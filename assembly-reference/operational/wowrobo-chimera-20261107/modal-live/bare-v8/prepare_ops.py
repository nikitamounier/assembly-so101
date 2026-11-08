from pathlib import Path
import json,hashlib
ROOT=Path(__file__).resolve().parent.parent;REL=ROOT/'bare-v8';HERE=ROOT/'ops-v8';OLD=ROOT/'ops-v7';HERE.mkdir(exist_ok=True)
assert json.loads((REL/'distributed-smoke.json').read_text())['status']=='PASS'
for n in ['cluster.py','monitor.py','failover.py','autoscale.py','worker.sbatch','learner_bootstrap.py']:
 s=(OLD/n).read_text().replace("ROOT/'run-v6'","ROOT/'run-v8'").replace("ROOT/'active-inputs-v7.json'","ROOT/'active-inputs-v8.json'")
 (HERE/n).write_text(s)
cfg=json.loads((OLD/'cluster.json').read_text());(HERE/'cluster.json').write_text(json.dumps(cfg,indent=2)+'\n')
manifest=json.loads((ROOT/'active-inputs-v7.json').read_text());manifest.pop('ops-v7/learner_bootstrap.py',None);manifest['ops-v8/learner_bootstrap.py']=hashlib.sha256((HERE/'learner_bootstrap.py').read_bytes()).hexdigest();release=json.loads((REL/'release.json').read_text());manifest.update(release['files']);manifest['bare-v8/release.json']=hashlib.sha256((REL/'release.json').read_bytes()).hexdigest();(ROOT/'active-inputs-v8.json').write_text(json.dumps(manifest,indent=2)+'\n')
scale=ROOT/'autoscale-v8';scale.mkdir(exist_ok=True);x=json.loads((ROOT/'autoscale-v7/active-run.json').read_text());x.update(enabled=False,run_dir=str(ROOT/'run-v8'),worker_script=str(HERE/'worker.sbatch'),worker_script_sha256=hashlib.sha256((HERE/'worker.sbatch').read_bytes()).hexdigest(),capacity_note='Approximate bareMuJoCo first; no CPU scaling until parent validates new run.');(scale/'active-run.json').write_text(json.dumps(x,indent=2)+'\n')
pins={n:hashlib.sha256((HERE/n).read_bytes()).hexdigest() for n in ['cluster.py','monitor.py','failover.py','autoscale.py','worker.sbatch','learner_bootstrap.py']};result=dict(root=str(ROOT),run=str(ROOT/'run-v8'),config=cfg,pins=pins,physics_backend='bare-mujoco-boxes-v8',physics_signature=release['physics_signature'],envs=9600,worker_slots=960,baseline_chimera_cpus=400,cloud_migration_pending=True)
(ROOT/'ops-v8.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
