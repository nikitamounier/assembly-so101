"""Freeze the final prelaunch bank refresh and its operators, before any learner starts."""
from pathlib import Path
import json,hashlib
ROOT=Path(__file__).resolve().parents[4];C=ROOT/'robotics-assembly-3.0';OPS=Path(__file__).parent.parent
assert json.loads((ROOT/'reset-validation-v5.json').read_text())['status']=='PASS'
for name in ['cluster.py','monitor.py','failover.py']:
 p=OPS/name;s=p.read_text();assert "RUN=ROOT/'run-v4'" in s;s=s.replace("RUN=ROOT/'run-v4'","RUN=ROOT/'run-v5'").replace('training-accepted-v4','training-accepted-v5');p.write_text(s)
files={}
for base in [ROOT/'assembly-so101/assembly_policy',ROOT/'assembly-so101/configs',C/'assembly_policy/so101',C/'outputs/equilibrium-model-20261107/profile-centered-v6',C/'assets_local/wowrobo/training-accepted-v5']:
 for p in base.rglob('*'):
  if p.is_file() and p.suffix in ('.py','.cc','.json','.xml','.npz') and '__pycache__' not in p.parts:files[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
(ROOT/'active-inputs.json').write_text(json.dumps(files,indent=2)+'\n')
state=ROOT/'autoscale-v5';state.mkdir(exist_ok=True);old=ROOT/'autoscale-v4/active-run.json';cfg=json.loads(old.read_text());cfg['run_dir']=str(ROOT/'run-v5');(state/'active-run.json').write_text(json.dumps(cfg,indent=2)+'\n');cfg4=json.loads(old.read_text());cfg4['enabled']=False;old.write_text(json.dumps(cfg4,indent=2)+'\n')
(ROOT/'run-v5').mkdir(exist_ok=True)
result=json.loads((ROOT/'ops-v4.json').read_text());result.update(run=str(ROOT/'run-v5'),frozen_inputs=len(files),bank='training-accepted-v5',pins={n:hashlib.sha256((OPS/n).read_bytes()).hexdigest() for n in ['cluster.py','monitor.py','failover.py','autoscale.py','worker.sbatch']},accepted=json.loads((ROOT/'reset-update-v5.json').read_text()))
(ROOT/'ops-v5.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
