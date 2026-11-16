"""Checkpoint-safe capacity phase6→7 handoff; no Slurm allocation is cancelled."""
from pathlib import Path
import json,subprocess,shlex,time,sys
HERE=Path(__file__).resolve().parent;H=HERE.parent;t=json.loads((H/'target-v7.json').read_text());R=t['remote'];PY=t['python']
old_ops=json.loads((H/'ops-v7.json').read_text());script=R+'/ops-v7/cluster.py';digest=old_ops['pins']['cluster.py']
expected='import hashlib,runpy,sys; p='+repr(script)+'; assert hashlib.sha256(open(p,"rb").read()).hexdigest()=='+repr(digest)+';sys.argv=[p,"control"];runpy.run_path(p,run_name="__main__")'
import hashlib
controller_code_sha=hashlib.sha256(expected.encode()).hexdigest()
def relay(code,job='3077705'):
 p=subprocess.run(['ssh','-o','ConnectTimeout=15','chimera-login',f'srun --jobid={job} --overlap --exact -n1 -c1 --mem=1G --cpu-bind=none '+PY+' -c '+shlex.quote(code)],capture_output=True,text=True,timeout=100)
 if p.returncode:print(p.stdout,p.stderr);p.check_returncode()
 return p.stdout
mode=sys.argv[1]
if mode=='begin':
 print(relay(f"import json;from pathlib import Path;r=Path({R!r});assert json.loads((r/'reset-validation-v6.json').read_text())['status']=='PASS';assert (r/'ops-v8.json').exists();assert json.loads((r/'bare-v8/distributed-smoke.json').read_text())['status']=='PASS';print('successor gated')"),flush=True)
 (H/'handoff-active-v8.json').write_text(json.dumps(dict(status='intentional_handoff',time_unix=time.time(),old_run=R+'/run-v6',new_run=R+'/run-v8',new_ops=R+'/ops-v8'))+'\n')
 code=f'''from pathlib import Path
import json,os,signal,time,hashlib
r=Path({R!r});ops=r/'ops-v7';run=r/'run-v6';controllers=[]
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  if p.stat().st_uid!=os.getuid():continue
  a=(p/'cmdline').read_bytes().split(b'\\0')
  if (str(ops/'cluster.py').encode() in a and b'control' in a) or (len(a)>=3 and a[1]==b'-c' and hashlib.sha256(a[2]).hexdigest()=={controller_code_sha!r}):
   start=(p/'stat').read_text().split()[21];os.kill(int(p.name),signal.SIGSTOP);controllers.append(dict(pid=int(p.name),start=start))
 except (FileNotFoundError,ProcessLookupError,PermissionError):pass
assert len(controllers)==1,controllers
cfg=json.loads((ops/'cluster.json').read_text());scale=r/'autoscale-v7/active-run.json';x=json.loads(scale.read_text());x['enabled']=False;scale.write_text(json.dumps(x)+'\\n')
assert not json.loads((r/'autoscale-v7/submissions.json').read_text()).get('jobs',[]) if (r/'autoscale-v7/submissions.json').exists() else True
newcfgpath=r/'ops-v8/cluster.json';newcfg=json.loads(newcfgpath.read_text());newcfg['gpu_job']=cfg['gpu_job'];newcfgpath.write_text(json.dumps(newcfg)+'\\n');state=dict(controllers=controllers,config=cfg,time=time.time());(run/'handoff-controller-v8.json').write_text(json.dumps(state)+'\\n');print(json.dumps(state))
'''
 state=json.loads(relay(code));(HERE/'handoff-state.json').write_text(json.dumps(state)+'\n')
 code=f'''from pathlib import Path
import json,os,signal,time
run=Path({R+'/run-v6'!r});row=json.loads((run/'learner-process.json').read_text());p=Path('/proc')/str(row['pid']);a=(p/'cmdline').read_bytes().split(b'\\0');assert str(run.parent/'ops-v7/learner_bootstrap.py').encode() in a and str(run).encode() in a;row.update(start=(p/'stat').read_text().split()[21],time=time.time());os.kill(row['pid'],signal.SIGTERM);(run/'handoff-request-v8.json').write_text(json.dumps(row)+'\\n');print(json.dumps(row))
'''
 print(relay(code,state['config']['gpu_job']),flush=True)
elif mode=='check':
 state=json.loads((HERE/'handoff-state.json').read_text())
 code=f'''from pathlib import Path
import json,time
run=Path({R+'/run-v6'!r});row=json.loads((run/'handoff-request-v8.json').read_text());p=Path('/proc')/str(row['pid']);alive=p.exists() and (p/'stat').read_text().split()[21]==row['start'] and (p/'stat').read_text().split()[2]!='Z';lines=(run/'log.jsonl').read_text().splitlines();result=dict(learner_alive=alive,latest=json.loads(lines[-1]),checkpoint_bytes=(run/'checkpoint.pt').stat().st_size,time=time.time())
if not alive:(run/'handoff-closed-v8.json').write_text(json.dumps(result)+'\\n')
print(json.dumps(result))
'''
 print(relay(code,state['config']['gpu_job']),flush=True)
elif mode=='close-workers':
 state=json.loads((HERE/'handoff-state.json').read_text());worker=(H/'close_old_worker.py').read_text().replace('RUN_TOKEN',repr(R+'/run-v6'))
 code=f'''from pathlib import Path
import json,os,signal,subprocess
from concurrent.futures import ThreadPoolExecutor
r=Path({R!r});run=r/'run-v6';assert (run/'handoff-closed-v8.json').exists()
for row in {state['controllers']!r}:
 p=Path('/proc')/str(row['pid'])
 if p.exists() and (p/'stat').read_text().split()[21]==row['start']:
  assert (p/'stat').read_text().split()[2] in ('T','t');os.kill(row['pid'],signal.SIGKILL)
env={{k:v for k,v in os.environ.items() if not k.startswith(('SLURM_','SRUN_','SBATCH_'))}}
def close(job):
 q=subprocess.run(['squeue','-j',job,'-h','-o','%T'],capture_output=True,text=True,check=True).stdout.strip()
 if q!='RUNNING':return dict(job=job,state=q,no_running_old_step=True)
 p=subprocess.run(['srun','--jobid='+job,'--overlap','--exact','-n1','-c1','--mem=1G','--cpu-bind=none',{PY!r},'-c',{worker!r}],env=env,capture_output=True,text=True,timeout=65);assert p.returncode==0,(job,p.stderr);return json.loads(p.stdout)
with ThreadPoolExecutor(max_workers=4) as pool:rows=list(pool.map(close,{state['config']['cpu_jobs']!r}))
(run/'worker-closure-v8.json').write_text(json.dumps(rows,indent=2)+'\\n');print(json.dumps(dict(closed_jobs=len(rows),all_leases_retained=True)))
'''
 print(relay(code),flush=True)
