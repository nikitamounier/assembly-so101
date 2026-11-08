"""Create isolated run-v6 operators, leaving the active v5 source frozen."""
from pathlib import Path
import hashlib,json,shutil
ROOT=Path(__file__).resolve().parent.parent;HERE=Path(__file__).parent
OLD=ROOT/'robotics-assembly-3.0/operational/wowrobo-chimera-20261107'
assert json.loads((ROOT/'reset-validation-v6.json').read_text())['status']=='PASS'
cfg=json.loads((OLD/'cluster.json').read_text())
for name in ('cluster.py','monitor.py','failover.py','autoscale.py','worker.sbatch'):
 s=(OLD/name).read_text().replace('Path(__file__).resolve().parents[3]','Path(__file__).resolve().parent.parent').replace("RUN=ROOT/'run-v5'","RUN=ROOT/'run-v6'").replace('training-accepted-v5','training-accepted-v6').replace("ROOT/'active-inputs.json'","ROOT/'active-inputs-v6.json'")
 if name=='cluster.py':
  s=s.replace("[PY,'-m','assembly_policy.train','--suite'","[PY,str(HERE/'learner_bootstrap.py'),'--suite'")
  s=s.replace("'--worker-init-wait','300'","'--worker-init-wait','60'")
  s=s.replace("'--wowrobo-debug-resets'","'--wowrobo-debug-resets','--bank-per-family','0'")
  # Bound initial collection even under steady reconnect traffic. The original
  # listener checks timeout only after an empty accept, which can take20min.
  bootstrap="HERE/'learner_bootstrap.py'"
  assert bootstrap in s
 (HERE/name).write_text(s)
(HERE/'cluster.json').write_text(json.dumps(cfg,indent=2)+'\n')
run=ROOT/'run-v6';run.mkdir(exist_ok=True)
# Preserve the same worker authentication lineage; secret remains mode0600.
shutil.copy2(ROOT/'run-v5/worker-auth.key',run/'worker-auth.key');(run/'worker-auth.key').chmod(0o600)
assert hashlib.sha256((ROOT/'replay-observer/record_training.py').read_bytes()).hexdigest()=='56dbfc0961f706798749721f9e23ca0952e995e8f31b15b05abe7fb7809a0cd8'
assert hashlib.sha256((ROOT/'replay-observer/telemetry.py').read_bytes()).hexdigest()=='9c10de333ab202e31e5e8c7d0e00bf3a7e70e397c18e71a29305e9b6ce6d8411'
files=json.loads((ROOT/'active-inputs.json').read_text())
files={k:v for k,v in files.items() if '/training-accepted-v5/' not in k}
for base in (ROOT/'robotics-assembly-3.0/assets_local/wowrobo/training-accepted-v6',ROOT/'replay-observer'):
 for p in base.rglob('*'):
  if p.is_file() and p.suffix in ('.py','.json','.npz'):files[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
files[str((HERE/'learner_bootstrap.py').relative_to(ROOT))]=hashlib.sha256((HERE/'learner_bootstrap.py').read_bytes()).hexdigest()
(ROOT/'active-inputs-v6.json').write_text(json.dumps(files,indent=2)+'\n')
scale=ROOT/'autoscale-v6';scale.mkdir(exist_ok=True)
x=json.loads((ROOT/'autoscale-v5/active-run.json').read_text());x.update(enabled=True,run_dir=str(run),worker_script=str(HERE/'worker.sbatch'),worker_script_sha256=hashlib.sha256((HERE/'worker.sbatch').read_bytes()).hexdigest())
(scale/'active-run.json').write_text(json.dumps(x,indent=2)+'\n')
result=dict(root=str(ROOT),run=str(run),config=cfg,pins={n:hashlib.sha256((HERE/n).read_bytes()).hexdigest() for n in ('cluster.py','monitor.py','failover.py','autoscale.py','worker.sbatch','learner_bootstrap.py')},accepted=json.loads((ROOT/'reset-update-v6.json').read_text()),frozen_inputs=len(files))
(ROOT/'ops-v6.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
