from pathlib import Path
import json,hashlib
R=Path(__file__).resolve().parent.parent;N=R/'native-v15';O=R/'ops-v14';D=R/'ops-v15';D.mkdir(exist_ok=True)
qa=json.loads((N/'qualification.json').read_text());assert qa['status']=='PASS' and qa['byte_identical']
prep=json.loads((N/'prepared.json').read_text());assert prep['physics_signature']==qa['physics_signature']
p=N/'training-accepted-v15/manifest.json';meta=json.loads(p.read_text());meta['collision_qualification']='Hoisted immutable grid lookup/shared distance-gradient interpolation, retaining general separation/four-way cache: independent exhaustive oracle/PPO parity, all1143 current resets exact, 10world/10thread currentpolicy byte/RSS parity, small distributed PPO finite PASS';p.write_text(json.dumps(meta,indent=2)+'\n')
for p in (N/'robotics-assembly-3.0/outputs').rglob('*'):
 if p.is_file() and (p.suffix=='.so' or p.name=='build-receipt.json'):prep['files'][str(p.relative_to(R))]=hashlib.sha256(p.read_bytes()).hexdigest()
prep['files']['native-v15/training-accepted-v15/manifest.json']=hashlib.sha256((N/'training-accepted-v15/manifest.json').read_bytes()).hexdigest();prep['status']='QUALIFIED_READY_NOT_ACTIVE';(N/'prepared.json').write_text(json.dumps(prep,indent=2)+'\n')
names=['cluster.py','monitor.py','failover.py','autoscale.py','worker.sbatch','learner_bootstrap.py']
for n in names:(D/n).write_text((O/n).read_text().replace('run-v14','run-v15').replace('ops-v14','ops-v15').replace('active-inputs-v14','active-inputs-v15').replace('native-v14','native-v15').replace('training-accepted-v14','training-accepted-v15'))
cfg=json.loads((O/'cluster.json').read_text());assert set(str(i) for i in range(3602571,3602611))<=set(cfg['cpu_jobs']);cfg['scope']='QUALIFIED_HOISTED_GRID_CONTINUATION';(D/'cluster.json').write_text(json.dumps(cfg,indent=2)+'\n')
manifest=json.loads((R/'active-inputs-v14.json').read_text());manifest.update(prep['files']);manifest['native-v15/prepared.json']=hashlib.sha256((N/'prepared.json').read_bytes()).hexdigest();manifest['ops-v15/learner_bootstrap.py']=hashlib.sha256((D/'learner_bootstrap.py').read_bytes()).hexdigest();(R/'active-inputs-v15.json').write_text(json.dumps(manifest,indent=2)+'\n')
# Keep every existing submission fence and active/pending job. No fresh ledger.
import shutil
scale=R/'autoscale-v15';shutil.copytree(R/'autoscale-v14',scale,dirs_exist_ok=True)
p=scale/'active-run.json';x=json.loads(p.read_text());x.update(enabled=False,run_dir=str(R/'run-v15'),root=str(N/'assembly-so101'),worker_script=str(D/'worker.sbatch'),worker_script_sha256=hashlib.sha256((D/'worker.sbatch').read_bytes()).hexdigest(),capacity_note='Arm only after runtime15 first genuinely new finite durable PPO; preserve inherited submission fences.');p.write_text(json.dumps(x,indent=2)+'\n')
x=dict(status='READY_NOT_ACTIVE',root=str(R),run=str(R/'run-v15'),config=cfg,pins={n:hashlib.sha256((D/n).read_bytes()).hexdigest() for n in names},physics_signature=prep['physics_signature'],physics_backend='wowrobo-native-source-sdf',timestep=.0005,substeps=80,control_dt=.04)
(R/'ops-v15.json').write_text(json.dumps(x,indent=2)+'\n');print(json.dumps(x),flush=True)
