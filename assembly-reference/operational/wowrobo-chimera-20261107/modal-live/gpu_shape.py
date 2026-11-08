"""Check policy memory at scaled inference and accumulated-update chunk sizes."""
from pathlib import Path
import json,torch,time
from assembly_policy.rl.nets import ActorCritic
ROOT=Path(__file__).resolve().parents[4]
torch.set_num_threads(2);device=torch.device('cuda')
model=ActorCritic(448,proprio_dim=250,act_dim=12).to(device)
torch.cuda.reset_peak_memory_stats();t=time.perf_counter()
with torch.no_grad():
 points=torch.zeros(5090,448*3,device=device);prop=torch.zeros(5090,250,device=device)
 dist=model.dist(points,prop);value=model.value(points,prop)
 assert torch.isfinite(dist.mean).all() and torch.isfinite(value).all()
del points,prop,dist,value
points=torch.zeros(1024,448*3,device=device);prop=torch.zeros(1024,250,device=device)
loss=model.dist(points,prop).mean.square().mean()+model.value(points,prop).square().mean();loss.backward();torch.cuda.synchronize()
result=dict(status='PASS',gpu=torch.cuda.get_device_name(),inference_envs=5090,gradient_chunk=1024,peak_allocated_bytes=torch.cuda.max_memory_allocated(),elapsed_s=time.perf_counter()-t,optimizer_step_performed=False,training_run_modified=False)
(ROOT/'gpu-shape-v4.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
