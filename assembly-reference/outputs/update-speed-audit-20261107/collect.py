from pathlib import Path
import json,shlex,subprocess,datetime
HERE=Path(__file__).resolve().parent;C=HERE.parents[1]
target=json.loads((C/'operational/wowrobo-chimera-20261107/modal-live/target.json').read_text());R=target['remote']
code=r'''
from pathlib import Path
import json,time,collections
root=Path(ROOT);run=root/'run-v5'
result=dict(observed_unix=time.time(),root=str(root))
for name in ['command.json','controller.json']:
 p=run/name;result[name]=json.loads(p.read_text()) if p.exists() else None
p=run/'log.jsonl';result['log_rows']=[json.loads(l) for l in p.read_text().splitlines()[-20:]] if p.exists() else []
result['log_age_s']=time.time()-p.stat().st_mtime if p.exists() else None
result['learner_logs']={p.name:p.read_text().splitlines()[-45:] for p in run.glob('learner-*.log')}
result['worker_errors']={p.name:p.read_text().splitlines()[-4:] for p in run.glob('worker-*.log') if any(s in p.read_text()[-2000:] for s in ['Error','Traceback','Killed','warning'])}
files=['assembly-so101/assembly_policy/rl/remote.py','assembly-so101/assembly_policy/train.py','assembly-so101/assembly_policy/wowrobo/env.py','robotics-assembly-3.0/assembly_policy/so101/wowrobo_rollout_friction.cc','robotics-assembly-3.0/assembly_policy/so101/wowrobo_training_runtime.py','robotics-assembly-3.0/assembly_policy/so101/wowrobo_contact_backend.py']
result['sources']={f:(root/f).read_text() for f in files}
print(json.dumps(result))
'''
cmd='srun --jobid=3077705 --overlap --exact --ntasks=1 --cpus-per-task=1 '+target['python']+' -c '+shlex.quote('ROOT='+repr(R)+'\n'+code)
p=subprocess.run(['ssh','-o','ConnectTimeout=15','chimera-login',cmd],text=True,capture_output=True,timeout=90)
if p.returncode:print(p.stderr);p.check_returncode()
r=json.loads(p.stdout);(HERE/'live-snapshot.json').write_text(json.dumps(r,indent=2)+'\n')
for name,source in r.pop('sources').items():
 dest=HERE/'snapshot'/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(source)
controller=r.pop('controller.json');r['controller_summary']=dict(time=controller['time'],gpu_job=controller['gpu_job'],states=dict(__import__('collections').Counter(str(x.get('scheduler',[None])[0]) for x in controller['workers'].values())),launcher_failures={k:v for k,v in controller['workers'].items() if v.get('exit') not in (None,0)}) if controller else None
print(json.dumps(r,indent=2))
