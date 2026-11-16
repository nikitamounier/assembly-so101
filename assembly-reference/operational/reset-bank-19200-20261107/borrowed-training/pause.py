"""Apply the human's assembly-CPU priority; retain leases/checkpoint/source."""
import argparse,hashlib,json,os,signal,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('mode',choices=['configure','learner','checkpoint','worker']);p.add_argument('--training-root',type=Path,required=True);p.add_argument('--state-root',type=Path,required=True);a=p.parse_args();r=a.training_root.resolve();state=a.state_root.resolve();run=r/'run-v10';folder=state/'borrowed-training';folder.mkdir(exist_ok=True)
def load(p):return json.loads(p.read_text())
def write(p,v):
 t=p.with_name(p.name+'.'+str(os.getpid())+'.tmp');t.write_text(json.dumps(v,indent=2)+'\n');t.replace(p)
def info(pid):
 p=Path('/proc')/str(pid);stat=(p/'stat').read_text().rsplit(')',1)[1].split();return {'pid':pid,'start':int(stat[19]),'state':stat[0],'args':(p/'cmdline').read_bytes().split(b'\0'),'uid':p.stat().st_uid}
if a.mode=='configure':
 cfgpath=r/'ops-v10/cluster.json';cfg=load(cfgpath);backup=folder/'training-config-before.json'
 if backup.exists():assert not cfg['cpu_jobs'] and cfg.get('state_build_priority',{}).get('state_root')==str(state)
 else:
  assert len(cfg['cpu_jobs'])==40;write(backup,cfg)
  cfg['cpu_jobs']=[];cfg['state_build_priority']={'state_root':str(state),'controller_job':3625780,'reason':'Human correction: use all assembly CPUs for state build','preserve_training_checkpoint':True,'restore_after_bank':True};write(cfgpath,cfg)
  write(run/'state-build-cpu-priority.json',cfg['state_build_priority'])
 write(folder/'priority.json',dict(status='training_cpu_launches_disabled',run=str(run),cpu_jobs=load(backup)['cpu_jobs'],gpu_job=cfg['gpu_job'],unix=time.time()))
 print(json.dumps({'mode':a.mode,'leases_retained':40,'training_cpu_launches_disabled':True}))
elif a.mode=='learner':
 row=load(run/'learner-process.json');assert str(row['job'])==os.environ['SLURM_JOB_ID'];x=info(int(row['pid']));assert x['uid']==os.getuid();assert str(r/'ops-v10/learner_bootstrap.py').encode() in x['args'] and str(run).encode() in x['args'];os.kill(x['pid'],signal.SIGSTOP);write(folder/'suspended-learner.json',dict(pid=x['pid'],start=x['start'],job=row['job'],run=str(run),paused=True,unix=time.time()));print(json.dumps({'learner_paused':x['pid']}))
elif a.mode=='checkpoint':
 import io,torch
 raw=(run/'checkpoint.pt').read_bytes();ck=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=False)
 for v in ck['model'].values():assert torch.isfinite(v).all()
 for s in ck['opt']['state'].values():
  for v in s.values():
   if isinstance(v,torch.Tensor):assert torch.isfinite(v).all()
 dst=run/'checkpoint.before-state-build-priority.pt'
 if dst.exists():assert dst.read_bytes()==raw
 else:dst.write_bytes(raw)
 proof=dict(status='PASS',it=ck['it'],transitions=ck['transitions'],checkpoint_sha256=hashlib.sha256(raw).hexdigest(),archive=str(dst),unix=time.time());write(folder/'checkpoint-proof.json',proof);print(json.dumps(proof))
else:
 assert (folder/'checkpoint-proof.json').exists();assert not load(r/'ops-v10/cluster.json')['cpu_jobs'];cfg=load(folder/'training-config-before.json');job=os.environ['SLURM_JOB_ID'];assert job in cfg['cpu_jobs'];closed=[]
 for p in Path('/proc').iterdir():
  if not p.name.isdigit():continue
  try:
   x=info(int(p.name));args=x['args']
   native = args[1:3]==[b'-m',b'assembly_policy.rl.remote'] and str(run).encode() in args
   launcher = str(r/'ops-v10/cluster.py').encode() in args and b'node' in args
   if x['uid']!=os.getuid() or not (native or launcher):continue
   if ('SLURM_JOB_ID='+job).encode() not in (p/'environ').read_bytes().split(b'\0'):continue
   os.kill(x['pid'],signal.SIGTERM);closed.append(dict(pid=x['pid'],start=x['start']))
  except (FileNotFoundError,ProcessLookupError,PermissionError):pass
 deadline=time.monotonic()+5
 while time.monotonic()<deadline:
  alive=[]
  for x in closed:
   try:
    y=info(x['pid'])
    if y['start']==x['start'] and y['state']!='Z':alive.append(x)
   except (FileNotFoundError,ProcessLookupError):pass
  if not alive:break
  time.sleep(.1)
 for x in alive:
  try:os.kill(x['pid'],signal.SIGKILL)
  except ProcessLookupError:pass
 write(folder/f'closed-worker-{job}.json',dict(job=job,closed=closed,leases_retained=True,unix=time.time()));print(json.dumps(dict(job=job,closed=len(closed),leases_retained=True)))
