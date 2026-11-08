"""Prepare a qualified source-pinned successor; never start or reclaim resources."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parent.parent
REL=ROOT/'native-v12';OLD=ROOT/'ops-v10';HERE=ROOT/'ops-v12'
HERE.mkdir(exist_ok=True)
qa=json.loads((REL/'production-worker-proof.json').read_text())
smoke=json.loads((REL/'distributed-smoke.json').read_text())
assert qa['status']==smoke['status']=='PASS' and qa['byte_identical']
assert qa['worker_envs']==qa['worker_threads']==10 and qa['rss_pass']
prepared=json.loads((REL/'prepared.json').read_text())
assert prepared['physics_signature']==qa['new_physics_signature']==smoke['physics_signature']
meta_path=REL/'training-accepted-v12/manifest.json'
meta=json.loads(meta_path.read_text())
meta['collision_qualification']='247-pose exhaustive oracle, exact policy/Adam/SGS PPO parity, 10-world/10-thread worker61-array byte parity, RSS and distributed PPO PASS; full fleet speed not yet measured'
meta_path.write_text(json.dumps(meta,indent=2)+'\n')
prepared['files'][str(meta_path.relative_to(ROOT))]=hashlib.sha256(meta_path.read_bytes()).hexdigest()
prepared['status']='QUALIFIED_AWAITING_STATE_BUILD_RESOURCE_RETURN'
(REL/'prepared.json').write_text(json.dumps(prepared,indent=2)+'\n')
for name in ['cluster.py','monitor.py','failover.py','autoscale.py','worker.sbatch','learner_bootstrap.py']:
    s=(OLD/name).read_text().replace('run-v10','run-v12').replace('active-inputs-v10','active-inputs-v12').replace('native-v10','native-v12').replace('training-accepted-v10','training-accepted-v12')
    (HERE/name).write_text(s)
cfg=json.loads((OLD/'cluster.json').read_text())
priority=json.loads((ROOT/'run-v10/state-build-cpu-priority.json').read_text())
if not cfg['cpu_jobs']:
    saved=Path(priority['state_root'])/'borrowed-training/training-config-before.json'
    cfg['cpu_jobs']=json.loads(saved.read_text())['cpu_jobs']
assert len(cfg['cpu_jobs'])==40
cfg.pop('state_build_priority',None)
cfg['scope']='QUALIFIED_NATIVE_SWEPT_CONTACT_CONTINUATION'
(HERE/'cluster.json').write_text(json.dumps(cfg,indent=2)+'\n')
manifest=json.loads((ROOT/'active-inputs-v10.json').read_text())
manifest.update(prepared['files'])
manifest['native-v12/prepared.json']=hashlib.sha256((REL/'prepared.json').read_bytes()).hexdigest()
manifest['ops-v12/learner_bootstrap.py']=hashlib.sha256((HERE/'learner_bootstrap.py').read_bytes()).hexdigest()
(ROOT/'active-inputs-v12.json').write_text(json.dumps(manifest,indent=2)+'\n')
scale=ROOT/'autoscale-v12';scale.mkdir(exist_ok=True)
x=json.loads((ROOT/'autoscale-v10/active-run.json').read_text())
x.update(enabled=False,run_dir=str(ROOT/'run-v12'),root=str(REL/'assembly-so101'),worker_script=str(HERE/'worker.sbatch'),worker_script_sha256=hashlib.sha256((HERE/'worker.sbatch').read_bytes()).hexdigest(),capacity_note='Await explicit state-build resource return and versioned continuation; arm only after first actual new finite PPO checkpoint.')
(scale/'active-run.json').write_text(json.dumps(x,indent=2)+'\n')
result=dict(status='READY_NOT_ACTIVE',root=str(ROOT),run=str(ROOT/'run-v12'),config=cfg,pins={n:hashlib.sha256((HERE/n).read_bytes()).hexdigest() for n in ['cluster.py','monitor.py','failover.py','autoscale.py','worker.sbatch','learner_bootstrap.py']},physics_signature=prepared['physics_signature'],physics_backend='wowrobo-native-source-sdf',timestep=.0005,substeps=80,control_dt=.04,resource_gate='run-v10/state-build-cpu-priority.json status complete_training_resources_restored')
(ROOT/'ops-v12.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result),flush=True)
