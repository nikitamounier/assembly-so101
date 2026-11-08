"""Capacity-only successor; exact same frozen physical source and reset bank."""
from pathlib import Path
import hashlib,json,shutil
ROOT=Path(__file__).resolve().parent.parent;HERE=Path(__file__).parent;OLD=ROOT/'ops-v6'
assert json.loads((ROOT/'gpu-shape-v7.json').read_text())['status']=='PASS'
assert json.loads((ROOT/'reset-validation-v6.json').read_text())['status']=='PASS'
for name in ('cluster.py','monitor.py','failover.py','autoscale.py','worker.sbatch','learner_bootstrap.py'):
    s=(OLD/name).read_text().replace("ROOT/'active-inputs-v6.json'","ROOT/'active-inputs-v7.json'")
    if name=='cluster.py':s=s.replace("'--min-conn-frac','.05'","'--min-conn-frac','.025'")
    (HERE/name).write_text(s)
cfg=json.loads((OLD/'cluster.json').read_text());cfg.update(envs=9600,worker_slots=960)
(HERE/'cluster.json').write_text(json.dumps(cfg,indent=2)+'\n')
run=ROOT/'run-v6';run.mkdir(exist_ok=True)
assert (run/'worker-auth.key').exists()  # Existing private lineage stays unchanged
files=json.loads((ROOT/'active-inputs-v6.json').read_text());files.pop('ops-v6/learner_bootstrap.py',None)
files['ops-v7/learner_bootstrap.py']=hashlib.sha256((HERE/'learner_bootstrap.py').read_bytes()).hexdigest()
(ROOT/'active-inputs-v7.json').write_text(json.dumps(files,indent=2)+'\n')
scale=ROOT/'autoscale-v7';scale.mkdir(exist_ok=True)
x=json.loads((ROOT/'autoscale-v6/active-run.json').read_text());x.update(enabled=False,run_dir=str(run),worker_script=str(HERE/'worker.sbatch'),worker_script_sha256=hashlib.sha256((HERE/'worker.sbatch').read_bytes()).hexdigest(),cloud_workers=858,modal_reserved_worker_processes=409,sail_reserved_worker_processes=449,max_chimera_cpus=1000,worker_slots=960)
(scale/'active-run.json').write_text(json.dumps(x,indent=2)+'\n')
alias=ROOT/'v7'
if not alias.exists():alias.symlink_to(HERE,target_is_directory=True)
result=dict(root=str(ROOT),run=str(run),config=cfg,accepted=json.loads((ROOT/'reset-update-v6.json').read_text()),capacity_only=True,capacity_phase=7,same_run_continuation=True,old_envs=5090,new_envs=9600,steps=32,target_batch=307200,ppo_epochs=5,ppo_minibatches=4,micro_batch=1024,modal_workers=409,sail_workers_reserved=449,chimera_workers_max=100,pins={n:hashlib.sha256((HERE/n).read_bytes()).hexdigest() for n in ('cluster.py','monitor.py','failover.py','autoscale.py','worker.sbatch','learner_bootstrap.py')})
(ROOT/'ops-v7.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
