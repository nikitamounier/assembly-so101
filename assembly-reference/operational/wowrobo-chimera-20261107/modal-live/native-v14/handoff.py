"""Parent runtime switch: archive committed PPO; terminate only identified old processes; keep all leases."""
from pathlib import Path
import ast,json,subprocess,shlex,time,sys,hashlib
H=Path(__file__).resolve().parent.parent;N=H/'native-v14';t=json.loads((H/'target-v13.json').read_text());R=t['remote'];PY=t['python']
def relay(code,job='3077705',cpus=1,mem='1G'):
 cmd=f'srun --jobid={job} --overlap --exact -n1 -c{cpus} --mem={mem} --cpu-bind=none '+PY+' -c '+shlex.quote(code)
 p=subprocess.run(['ssh','-o','ConnectTimeout=15','chimera-login',cmd],capture_output=True,text=True,timeout=180)
 if p.returncode:print(p.stdout,p.stderr,flush=True);p.check_returncode()
 return p.stdout
mode=sys.argv[1]
if mode=='begin':
 # Evaluate exactly the historical launch wrapper without executing its subprocess.run.
 module=ast.parse((H/'start_v13.py').read_text());module.body=[v for v in module.body if not (isinstance(v,ast.Expr) and isinstance(v.value,ast.Call) and isinstance(v.value.func,ast.Attribute) and v.value.func.attr=='run')];scope={'__file__':str(H/'start_v13.py')};exec(compile(module,'historical-controller-wrapper','exec'),scope);sha=hashlib.sha256(scope['code'].encode()).hexdigest()
 code=f'''from pathlib import Path
import json,os,signal,time,hashlib
r=Path({R!r});run=r/'run-v13';ops=r/'ops-v13'
assert json.loads((r/'native-v14/qualification.json').read_text())['status']=='PASS'
manifest=json.loads((r/'active-inputs-v14.json').read_text());assert all(hashlib.sha256((r/n).read_bytes()).hexdigest()==h for n,h in manifest.items())
controllers=[]
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  if p.stat().st_uid!=os.getuid():continue
  a=(p/'cmdline').read_bytes().split(b'\\0')
  if (str(ops/'cluster.py').encode() in a and b'control' in a) or (len(a)>=3 and a[1]==b'-c' and hashlib.sha256(a[2]).hexdigest()=={sha!r}):controllers.append(dict(pid=int(p.name),start=(p/'stat').read_text().split()[21]))
 except (FileNotFoundError,ProcessLookupError,PermissionError):pass
assert len(controllers)==1,controllers
for row in controllers:
 p=Path('/proc')/str(row['pid']);assert (p/'stat').read_text().split()[21]==row['start'];os.kill(row['pid'],signal.SIGSTOP)
scale=r/'autoscale-v13/active-run.json';x=json.loads(scale.read_text());x['enabled']=False;scale.write_text(json.dumps(x)+'\\n')
cfg=json.loads((ops/'cluster.json').read_text());assert set(str(i) for i in range(3602571,3602611))<=set(cfg['cpu_jobs'])
# Adopt any dynamic additions made while qualification ran, preserving every ledger/fence.
import shutil
shutil.copytree(r/'autoscale-v13',r/'autoscale-v14',dirs_exist_ok=True)
p=r/'autoscale-v14/active-run.json';sx=json.loads(p.read_text());sx.update(enabled=False,run_dir=str(r/'run-v14'),root=str(r/'native-v14/assembly-so101'),worker_script=str(r/'ops-v14/worker.sbatch'),worker_script_sha256=hashlib.sha256((r/'ops-v14/worker.sbatch').read_bytes()).hexdigest());p.write_text(json.dumps(sx)+'\\n')
p=r/'ops-v14/cluster.json';nc=json.loads(p.read_text());nc['cpu_jobs']=cfg['cpu_jobs'];nc['gpu_job']=cfg['gpu_job'];p.write_text(json.dumps(nc)+'\\n');state=dict(controllers=controllers,config=cfg,time=time.time());(run/'handoff-controller-v14.json').write_text(json.dumps(state)+'\\n');print(json.dumps(state))
'''
 x=json.loads(relay(code));(N/'handoff-state.json').write_text(json.dumps(x,indent=2)+'\n');print(json.dumps(x),flush=True)
elif mode=='close-workers':
 state=json.loads((N/'handoff-state.json').read_text());worker=(H/'close_old_worker.py').read_text().replace('RUN_TOKEN',repr(R+'/run-v13'))
 code=f'''from pathlib import Path
import json,os,signal,subprocess
from concurrent.futures import ThreadPoolExecutor
r=Path({R!r});run=r/'run-v13';assert json.loads((run/'handoff-closed-v14.json').read_text())['status']=='PASS'
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
(run/'worker-closure-v14.json').write_text(json.dumps(rows,indent=2)+'\\n');print(json.dumps(dict(closed_jobs=len(rows),all_leases_retained=True)))
'''
 print(relay(code),flush=True)
