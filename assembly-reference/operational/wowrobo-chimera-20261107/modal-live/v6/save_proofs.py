from pathlib import Path
import json,subprocess,shlex
H=Path(__file__).resolve().parent.parent;t=json.loads((H/'target-v6.json').read_text());r=t['remote']
code=f'''from pathlib import Path
import json,hashlib,time
r=Path({r!r});paths=['training-proof-v6.json','replay-proof-v6.json','reset-update-v6.json','reset-validation-v6.json','reset-finite-v6.json','migration-v6.json','ops-v6.json','ops-v6/linux-physics-parity.json','run-v5/handoff-closed-v6.json','run-v5/worker-closure-v6.json','run-v6/config.json','run-v6/controller.json','run-v6/training-replays/manifest.json','run-v6/training-replays/index.json']
result={{'time_unix':time.time(),'files':{{}}}}
for name in paths:
 p=r/name
 if p.exists():result['files'][name]={{'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'data':json.loads(p.read_text())}}
print(json.dumps(result))
'''
p=subprocess.run(['ssh','-o','ConnectTimeout=15','chimera-login','srun --jobid=3077705 --overlap --exact -n1 -c1 --mem=1G '+t['python']+' -c '+shlex.quote(code)],capture_output=True,text=True,timeout=60);p.check_returncode();result=json.loads(p.stdout);(H/'v6/live-proof.json').write_text(json.dumps(result,indent=2)+'\n');print('Saved',len(result['files']),'proof records')
