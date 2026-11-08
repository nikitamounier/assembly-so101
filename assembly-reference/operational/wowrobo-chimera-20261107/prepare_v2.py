from pathlib import Path
import json,subprocess,shlex
h=Path(__file__).parent.resolve();root=h.parents[2];c=json.loads((h/'target.json').read_text());r=c['remote'];cfg=json.loads((h/'cluster.json').read_text());assert cfg['gpu_job']=='3606264'
files=[(root/'assembly-so101/assembly_policy/wowrobo/env.py',r+'/assembly-so101/assembly_policy/wowrobo/env.py'),(root/'robotics-assembly-3.0/assembly_policy/so101/wowrobo_training_runtime.py',r+'/robotics-assembly-3.0/assembly_policy/so101/wowrobo_training_runtime.py'),*[(h/n,c['script']+n) for n in ['cluster.py','cluster.json','monitor.py']]]
for local,remote in files:subprocess.run(['scp','-o','ConnectTimeout=15',str(local),'chimera-login:'+remote],check=True)
code=f'''from pathlib import Path
import json,hashlib
from assembly_policy.wowrobo.runtime import load_runtime
root=Path({r!r});run=root/'run-v2';run.mkdir(exist_ok=True)
m,p,sig=load_runtime();bank=root/'robotics-assembly-3.0/assets_local/wowrobo/training-debug-v1/manifest.json';x=json.loads(bank.read_text());assert x['training_qualified'] is False;x['model_signature']=sig;bank.write_text(json.dumps(x,indent=2)+'\\n')
manifest=json.loads((root/'active-inputs.json').read_text());(root/'active-inputs.invalid-startup.json').write_text(json.dumps(manifest,indent=2)+'\\n');manifest={{f:hashlib.sha256((root/f).read_bytes()).hexdigest() for f in manifest}};(root/'active-inputs.json').write_text(json.dumps(manifest,indent=2)+'\\n')
receipt=dict(model_signature=sig,nq=m.nq,nv=m.nv,neq=m.neq,arena_bytes=m.narena,status='DIAGNOSTIC_UNQUALIFIED_RESETS',scope='Fresh single-learner run on same GPU3606264; old startup invalidated; no old policy/optimizer/sampler reused');(root/'deployment-v2.json').write_text(json.dumps(receipt,indent=2)+'\\n');print(json.dumps(receipt))
'''
subprocess.run(['ssh','-o','ConnectTimeout=15','chimera-login','srun --jobid=3077705 --overlap --exact --ntasks=1 --cpus-per-task=1 '+c['env']+c['python']+' -c '+shlex.quote(code)],check=True)
