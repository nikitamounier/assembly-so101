from pathlib import Path
import json,subprocess,sys,threading,time,torch,numpy as np
from assembly_policy import train
ROOT=Path(__file__).resolve().parents[1];REL=ROOT/'bare-v8';RUN=REL/'distributed-smoke';BANK=ROOT/'robotics-assembly-3.0/assets_local/wowrobo/training-accepted-v6'
torch.set_num_threads(2)
assert not (RUN/'checkpoint.pt').exists()
workers=[]
def launch():
 deadline=time.time()+180
 while not (RUN/'server.json').exists():
  assert time.time()<deadline;time.sleep(.2)
 for i in range(2):
  f=(REL/f'smoke-worker-{i}.log').open('w');p=subprocess.Popen([sys.executable,'-m','assembly_policy.rl.remote','--run-dir',str(RUN),'--nthread','2','--wait','180','--poll','.2'],stdout=f,stderr=subprocess.STDOUT);workers.append(p);f.close()
threading.Thread(target=launch,daemon=True).start()
a=train.parse(['--suite','wowrobo','--wowrobo-debug-resets','--bank-per-family','0','--bank-dir',str(BANK),'--run-dir',str(RUN),'--envs','20','--steps','32','--iters','3','--epochs','5','--minibatches','4','--micro-batch','1024','--device','cuda','--remote-workers','2','--nthread','2','--accept-timeout','10','--min-conn-frac','1','--worker-init-wait','20','--save-every','1','--no-eval','--max-hours','.1','--seed','20261107','--cost-coef','.00002','--action-rate-coef','0','--eps-q','.01'])
t=time.perf_counter();tr=train.Trainer(a)
try:tr.train()
finally:
 tr.env.close()
 for p in workers:
  p.terminate()
  try:p.wait(timeout=10)
  except subprocess.TimeoutExpired:p.kill();p.wait()
rows=[json.loads(x) for x in (RUN/'log.jsonl').read_text().splitlines()];assert len(rows)==3 and rows[-1]['transitions']>0
ck=torch.load(RUN/'checkpoint.pt',map_location='cpu',weights_only=False)
assert ck['args']['physics_backend']=='bare-mujoco-boxes-v8'
assert ck['args']['model_signature']==json.loads((REL/'scene.json').read_text())['scene_sha256']
assert all(torch.isfinite(x).all() for x in ck['model'].values())
result=dict(status='PASS',distributed_workers=2,envs=20,iterations=ck['it'],transitions=ck['transitions'],wall_s=time.perf_counter()-t,latest=rows[-1],physics_signature=ck['args']['model_signature'],checkpoint_durable=True)
(REL/'distributed-smoke.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
