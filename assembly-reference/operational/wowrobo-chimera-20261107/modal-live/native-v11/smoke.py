from pathlib import Path
import json,subprocess,sys,threading,time,torch,numpy as np
from assembly_policy import train
ROOT=Path(__file__).resolve().parents[1];REL=ROOT/'native-v11';RUN=REL/'distributed-smoke';BANK=REL/'training-accepted-v11'
torch.set_num_threads(1)
assert not (RUN/'checkpoint.pt').exists()
workers=[]
def launch():
 deadline=time.time()+180
 while not (RUN/'server.json').exists():
  assert time.time()<deadline;time.sleep(.2)
 for i in range(2):
  f=(REL/f'smoke-worker-{i}.log').open('w');p=subprocess.Popen([sys.executable,'-m','assembly_policy.rl.remote','--run-dir',str(RUN),'--nthread','1','--wait','180','--poll','.2'],stdout=f,stderr=subprocess.STDOUT);workers.append(p);f.close()
threading.Thread(target=launch,daemon=True).start()
a=train.parse(['--suite','wowrobo','--wowrobo-debug-resets','--bank-per-family','0','--bank-dir',str(BANK),'--run-dir',str(RUN),'--envs','4','--steps','2','--iters','2','--epochs','1','--minibatches','1','--micro-batch','1024','--device','cpu','--remote-workers','2','--nthread','1','--accept-timeout','10','--min-conn-frac','1','--worker-init-wait','20','--save-every','1','--no-eval','--max-hours','.1','--seed','20261107','--cost-coef','.00002','--action-rate-coef','0','--eps-q','.01'])
class SmokeTrainer(train.Trainer):
 def _maybe_resume(self):
  import io,hashlib
  raw=(ROOT/'run-v10/checkpoint.pt').read_bytes();d=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=False)
  assert d['args']['physics_backend']=='wowrobo-native-source-sdf' and d['args']['physics_signature']=='34d224d4edff266369b3f1322baf80013c0daf09599a9eeb8751c6270a88ff92'
  assert d['args']['observation_signature']==self.args['observation_signature']
  self.model.load_state_dict(d['model'],strict=True);self.args['smoke_parent_policy']=dict(it=d['it'],sha256=hashlib.sha256(raw).hexdigest(),weights_only=True,run='run-v10')
t=time.perf_counter();tr=SmokeTrainer(a)
try:tr.train()
finally:
 tr.env.close()
 for p in workers:
  p.terminate()
  try:p.wait(timeout=10)
  except subprocess.TimeoutExpired:p.kill();p.wait()
rows=[json.loads(x) for x in (RUN/'log.jsonl').read_text().splitlines()];assert len(rows)==2 and rows[-1]['transitions']>0 and all(r['unstable']==0 for r in rows)
ck=torch.load(RUN/'checkpoint.pt',map_location='cpu',weights_only=False)
assert ck['args']['physics_backend']=='wowrobo-native-source-sdf'
assert ck['args']['model_signature']==json.loads((REL/'prepared.json').read_text())['physics_signature']
assert all(torch.isfinite(x).all() for x in ck['model'].values())
result=dict(status='PASS',distributed_workers=2,envs=4,iterations=ck['it'],transitions=ck['transitions'],wall_s=time.perf_counter()-t,latest=rows[-1],physics_signature=ck['args']['model_signature'],checkpoint_durable=True)
(REL/'distributed-smoke.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
