"""Verify and seal all19200 native resets; never changes geometry/admission."""
from __future__ import annotations
import argparse,concurrent.futures,fcntl,hashlib,importlib.util,json,os,shutil,subprocess,sys,time
from contextlib import ExitStack
from pathlib import Path

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
 t=p.with_name(p.name+'.'+str(os.getpid())+'.tmp');t.write_text(json.dumps(v,indent=2)+'\n');t.replace(p)
def load(p):return json.loads(p.read_text())
def copy_verified(src,dst):
 if not src.is_file():raise ValueError(f'Missing native artifact:{src}')
 h=sha(src)
 if dst.exists():
  if sha(dst)!=h:raise ValueError(f'Conflicting stage artifact:{dst}')
  return
 dst.parent.mkdir(parents=True,exist_ok=True);t=dst.with_name(dst.name+'.copy.tmp');shutil.copyfile(src,t)
 if sha(t)!=h:raise ValueError('Copy digest mismatch')
 t.replace(dst)
def complete(plan,stage):
 p=plan/'shards'/stage/'qualified/manifest.json'
 if not p.exists():return False
 m=load(p);return bool(m.get('complete')) and m.get('progress',{}).get('accepted_train')==640 and m.get('progress',{}).get('accepted_eval')==128

def publish_stage(root,settings,stage,tasks):
 plan=Path(settings['plan']);master=plan/'shards'/stage;master.mkdir(exist_ok=True);identities=set();accepted={}
 with ExitStack() as leases:
  fds=[]
  for task in tasks:
   folder=plan/'partitions'/stage/str(task['part']);f=leases.enter_context((folder/'.lease').open('a'))
   try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
   except BlockingIOError:return False
   fds.append(f.fileno());proof=load(root/'done'/f"{task['index']}.json")
   if proof['task']!=task or (proof['accepted_train'],proof['accepted_eval'])!=(10,2):raise ValueError('Bad completion quota/ownership')
  for task in tasks:
   folder=plan/'partitions'/stage/str(task['part'])
   for p in sorted((folder/'attempts').glob('*.json')):
    row=load(p)
    if row['stage']!=stage or row['slot']%64!=task['part']:raise ValueError('Foreign partition receipt')
    identities.add(row['identity'])
    if row.get('training_eligible'):
     key=(row['split'],row['slot'])
     if key in accepted:raise ValueError('Duplicate accepted slot')
     if not row['physical'].get('physical_admission_pass'):raise ValueError('Missing native admission')
     if row['metadata'].get('rng_namespace')!=settings['rng_namespace']:raise ValueError('Wrong RNG namespace')
     accepted[key]=row
  wanted={(split,i) for split,n in [('train',640),('eval',128)] for i in range(n)}
  if set(accepted)!=wanted or len(identities)!=1:raise ValueError('Stage union quota or identity mismatch')
  for task in tasks:
   folder=plan/'partitions'/stage/str(task['part'])
   for p in sorted((folder/'attempts').glob('*.json')):
    row=load(p)
    for key in ('archive','physical_input_archive','qualified_archive'):
     if row.get(key):
      rel=Path(row[key])
      if rel.is_absolute() or '..' in rel.parts:raise ValueError('Unsafe native artifact path')
      copy_verified(folder/rel,master/rel)
    copy_verified(p,master/'attempts'/p.name)
  env=dict(os.environ,PYTHONPATH=settings['project'])
  with (root/'logs'/f'publish-{stage}.log').open('a') as log:
   subprocess.run([sys.executable,str(root/'launch/worker.py'),'--root',str(root),'--publish-only','--stage',stage],env=env,stdout=log,stderr=subprocess.STDOUT,pass_fds=tuple(fds),check=True)
  if not complete(plan,stage):raise ValueError('Canonical stage publication incomplete')
 return True

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=a.root.resolve();s=load(r/'settings.json');plan=Path(s['plan']);spec=load(plan/'plan.json');assert os.environ.get('SLURM_JOB_ID');(r/'logs').mkdir(exist_ok=True)
 with (r/'.coordinator.lock').open('a') as owner:
  fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
  if not (r/'VALIDATED').exists():
   # Seal launcher/source identities before any native generation starts.
   cfg=load(plan/'config.json');expected={name:sha(r/'launch'/name) for name in ('worker.py','pool.py','coordinator.py')};cfg['execution_sha256']=expected;write(plan/'config.json',cfg);spec['config_sha256']=sha(plan/'config.json');write(plan/'plan.json',spec);s['execution_sha256']=expected;write(r/'settings.json',s)
   source=Path(s['source_root']);profile=Path(s['physical'])/'profile.json';reference=Path(s['project'])/'operational/reset-bank-chimera-20261107/source-model-arrays.npz'
   subprocess.run([sys.executable,str(source/'launch/parity-fixed.py'),'verify','--profile',str(profile),'--reference',str(reference),'--output',str(r/'native-array-parity.json')],env=dict(os.environ,PYTHONPATH=s['project']),check=True)
   if not load(r/'native-array-parity.json')['pass_']:raise ValueError('Native model parity failed')
   write(r/'environment.json',dict(python=sys.version,executable=sys.executable,job=os.environ['SLURM_JOB_ID'],source_root=s['source_root']))
   (r/'VALIDATED').write_text('Frozen native source arrays, CAD/grid bytes and launch identities verified\n')
  tasks=load(r/'assignment.json')['tasks'];stage_tasks={stage:[t for t in tasks if t['stage']==stage] for stage in spec['stage_ids']};futures={};last_report=0
  with concurrent.futures.ThreadPoolExecutor(max_workers=2) as publishers:
   while True:
    if (r/'STOP').exists():raise RuntimeError('STOP marker requests coordinator exit')
    done_ids={int(p.stem) for p in (r/'done').glob('*.json')}
    for stage,future in list(futures.items()):
     if future.done():
      result=future.result();del futures[stage]
      if result:print(json.dumps(dict(stage=stage,published=True)),flush=True)
    completed=[stage for stage in spec['stage_ids'] if complete(plan,stage)]
    for stage in spec['stage_ids']:
     if len(futures)>=2:break
     if stage in completed or stage in futures:continue
     if all(t['index'] in done_ids for t in stage_tasks[stage]):futures[stage]=publishers.submit(publish_stage,r,s,stage,stage_tasks[stage])
    if time.monotonic()-last_report>15:
     dispatch=load(r/'dispatch.json');proof=dict(status='generating' if len(completed)<25 else 'sealing',accepted_at_least=len(done_ids)*12,target=19200,completed_groups=len(done_ids),total_groups=len(tasks),completed_stages=completed,active_groups=len(dispatch.get('active',{})),failed_groups=dispatch.get('failures',{}),publishing=list(futures),unix=time.time());write(r/'progress.json',proof);last_report=time.monotonic()
    if len(completed)==25 and not futures:break
    time.sleep(5)
  sys.path.insert(0,s['project']);module_spec=importlib.util.spec_from_file_location('frozen_framework',Path(s['project'])/'scripts/build_wowrobo_reset_framework.py');framework=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(framework);framework.merge(plan,spec)
  command=[sys.executable,str(r/'launch/finalize.py'),'--root',str(r),'--project',s['project'],'--plan',str(plan),'--source-root',s['source_root']]
  subprocess.run(command,env=dict(os.environ,PYTHONPATH=s['project']),check=True)
  write(r/'result.json',dict(status='complete',accepted=19200,train=16000,eval=3200,job=os.environ['SLURM_JOB_ID'],unix=time.time()));print('COMPLETE19200',flush=True)
if __name__=='__main__':
 try:main()
 except Exception as error:
  if '--root' in sys.argv:
   r=Path(sys.argv[sys.argv.index('--root')+1]);write(r/'coordinator-error.json',dict(error=repr(error),unix=time.time()));(r/'STOP').write_text(repr(error)+'\n')
  raise
