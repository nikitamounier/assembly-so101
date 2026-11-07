"""Deploy diagnostic hook after verifying source guard; renew three owned jobs."""
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT=Path('/large_storage/goodarzilab/nikitamounier/assembly/source-85c006a')
assert os.environ.get('SLURM_JOB_ID')=='3077705'
out=ROOT/'artifacts_local/chimera/ppo-profile-20261106'
state=json.loads((ROOT/'artifacts_local/chimera/controller.json').read_text())
selected=[]; hosts=set()
rows=sorted(state['jobs'].items(),key=lambda kv:(not kv[1].get('NodeList','').startswith('CPU'),kv[0]))
for job,row in rows:
    host=row.get('NodeList')
    if job!='3598717' and row.get('JobState')=='RUNNING' and host not in hosts:
        selected.append(job); hosts.add(host)
    if len(selected)==3:break
assert len(selected)==3
assert not (out/'worker-config.json').exists(), 'Reconcile existing activation first'
(out/'worker-config.json').write_text(json.dumps(dict(jobs=selected,expected_workers=3,initial_hosts=sorted(hosts)),indent=2)+'\n')
lib=ROOT/'artifacts_local/chimera/vmprof-native';lib.mkdir(exist_ok=True)
for name in ['unwind','unwind-x86_64']:
    target=lib/f'lib{name}.so'
    if not target.exists():target.symlink_to(f'/usr/lib/x86_64-linux-gnu/lib{name}.so.8')
manifest=ROOT/'artifacts_local/chimera/source-hashes.json'
guard=json.loads(manifest.read_text())
name='assembly_policy/rl/remote.py'
original=(ROOT/name).read_bytes()
assert hashlib.sha256(original).hexdigest()==guard[name]
(out/'original-remote.py').write_bytes(original)
candidate=(ROOT/'artifacts_local/chimera/remote-profile-candidate.py').read_bytes()
compile(candidate,str(ROOT/name),'exec')
temp=(ROOT/name).with_suffix('.py.tmp');temp.write_bytes(candidate);temp.replace(ROOT/name)
guard[name]=hashlib.sha256(candidate).hexdigest()
for name in ['profile_train.py','profile_worker.py','start_profile.py','activate_worker_profile.py']:
    relative='operational/chimera-20261106/'+name
    guard[relative]=hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()
temp=manifest.with_suffix('.tmp');temp.write_text(json.dumps(guard,indent=2)+'\n');temp.replace(manifest)
for job in selected:
    p=subprocess.run(['scontrol','requeue',job],capture_output=True,text=True,timeout=30)
    assert p.returncode==0,(job,p.stderr)
print(json.dumps(dict(requeued=selected,hosts=sorted(hosts))))
