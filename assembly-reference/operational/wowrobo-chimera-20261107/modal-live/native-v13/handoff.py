"""Checkpoint-safe capacity phase6→7 handoff; no Slurm allocation is cancelled."""
from pathlib import Path
import json,subprocess,shlex,time,sys
HERE=Path(__file__).resolve().parent;H=HERE.parent;t=json.loads((H/'target-v10.json').read_text());R=t['remote'];PY=t['python']
old_ops=json.loads((H/'ops-v10.json').read_text());script=R+'/ops-v10/cluster.py';digest=old_ops['pins']['cluster.py']
expected='import hashlib,runpy,sys; p='+repr(script)+'; assert hashlib.sha256(open(p,"rb").read()).hexdigest()=='+repr(digest)+';sys.argv=[p,"control"];runpy.run_path(p,run_name="__main__")'
import hashlib
controller_code_sha=hashlib.sha256(expected.encode()).hexdigest()
def relay(code,job='3077705'):
 p=subprocess.run(['ssh','-o','ConnectTimeout=15','chimera-login',f'srun --jobid={job} --overlap --exact -n1 -c1 --mem=1G --cpu-bind=none '+PY+' -c '+shlex.quote(code)],capture_output=True,text=True,timeout=100)
 if p.returncode:print(p.stdout,p.stderr);p.check_returncode()
 return p.stdout
mode=sys.argv[1]
gate_call=subprocess.run([sys.executable,str(H/'priority_status.py')],text=True,capture_output=True,check=True)
gate=json.loads(json.loads(gate_call.stdout)['stdout']);assert gate['ready_for_parent_handoff'], 'State build has not closed all borrowed workers'
(HERE/'resource-gate.json').write_text(json.dumps(gate,indent=2)+'\n')
if mode=='begin':
 print(relay(f"import json;from pathlib import Path;r=Path({R!r});assert json.loads((r/'reset-validation-v6.json').read_text())['status']=='PASS';assert (r/'ops-v13.json').exists();assert json.loads((r/'native-v13/production-worker-proof.json').read_text())['status']=='PASS';assert json.loads((r/'native-v13/distributed-smoke.json').read_text())['status']=='PASS';print('successor gated')"),flush=True)
 (H/'handoff-active-v13.json').write_text(json.dumps(dict(status='intentional_handoff',time_unix=time.time(),old_run=R+'/run-v10',new_run=R+'/run-v13',new_ops=R+'/ops-v13'))+'\n')
 code=f'''from pathlib import Path
import json,os,signal,time,hashlib
r=Path({R!r});ops=r/'ops-v10';run=r/'run-v10';controllers=[]
priority=json.loads((run/'state-build-cpu-priority.json').read_text());assert priority['preserve_training_checkpoint'];ready=json.loads((Path(priority['state_root'])/'borrowed-training/restoration-ready.json').read_text());assert ready['ready_for_handoff'] and ready['automatic_resume'] is False
returned=json.loads((Path(priority['state_root'])/'exports/return-receipt.json').read_text());assert returned['status']=='complete' and returned['accepted']==19200
cfg=json.loads((ops/'cluster.json').read_text());assert not cfg['cpu_jobs'] and cfg['state_build_priority']['state_root']==priority['state_root']
manifest=json.loads((r/'active-inputs-v13.json').read_text());assert all(hashlib.sha256((r/n).read_bytes()).hexdigest()==h for n,h in manifest.items())
ledger=r/'autoscale-v10/submissions.json';assert not json.loads(ledger.read_text()).get('jobs',[]) if ledger.exists() else True
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  if p.stat().st_uid!=os.getuid():continue
  a=(p/'cmdline').read_bytes().split(b'\\0')
  if (str(ops/'cluster.py').encode() in a and b'control' in a) or (len(a)>=3 and a[1]==b'-c' and hashlib.sha256(a[2]).hexdigest()=={controller_code_sha!r}):
   start=(p/'stat').read_text().split()[21];controllers.append(dict(pid=int(p.name),start=start))
 except (FileNotFoundError,ProcessLookupError,PermissionError):pass
assert len(controllers)==1,controllers
for row in controllers:
 p=Path('/proc')/str(row['pid']);assert (p/'stat').read_text().split()[21]==row['start'];os.kill(row['pid'],signal.SIGSTOP)
cfg=json.loads((ops/'cluster.json').read_text());scale=r/'autoscale-v10/active-run.json';x=json.loads(scale.read_text());x['enabled']=False;scale.write_text(json.dumps(x)+'\\n')
assert not json.loads((r/'autoscale-v10/submissions.json').read_text()).get('jobs',[]) if (r/'autoscale-v10/submissions.json').exists() else True
newcfgpath=r/'ops-v13/cluster.json';newcfg=json.loads(newcfgpath.read_text());newcfg['gpu_job']=cfg['gpu_job'];newcfgpath.write_text(json.dumps(newcfg)+'\\n');closure_cfg=dict(cfg);closure_cfg['cpu_jobs']=newcfg['cpu_jobs'];assert len(closure_cfg['cpu_jobs'])==40;state=dict(controllers=controllers,config=closure_cfg,old_cpu_jobs=[],time=time.time());(run/'handoff-controller-v13.json').write_text(json.dumps(state)+'\\n');print(json.dumps(state))
'''
 state=json.loads(relay(code));(HERE/'handoff-state.json').write_text(json.dumps(state)+'\n')
 code=f"from pathlib import Path;import json;r=Path({R!r});row=json.loads((r/'resource-repair/native-pause-proof.json').read_text())['learner'];(r/'run-v10/handoff-request-v13.json').write_text(json.dumps(row)+'\\n');print(json.dumps(row))"
 print(relay(code),flush=True)
elif mode=='check':
 state=json.loads((HERE/'handoff-state.json').read_text())
 code=f'''from pathlib import Path
import json,time
run=Path({R+'/run-v10'!r});row=json.loads((run/'handoff-request-v13.json').read_text());p=Path('/proc')/str(row['pid']);alive=p.exists() and (p/'stat').read_text().split()[21]==row['start'] and (p/'stat').read_text().split()[2]!='Z';lines=(run/'log.jsonl').read_text().splitlines();result=dict(learner_alive=alive,latest=json.loads(lines[-1]),checkpoint_bytes=(run/'checkpoint.pt').stat().st_size,time=time.time())
if not alive:(run/'handoff-closed-v13.json').write_text(json.dumps(result)+'\\n')
print(json.dumps(result))
'''
 print(relay(code,state['config']['gpu_job']),flush=True)
elif mode=='close-workers':
 state=json.loads((HERE/'handoff-state.json').read_text());worker=(H/'close_old_worker.py').read_text().replace('RUN_TOKEN',repr(R+'/run-v10'))
 code=f'''from pathlib import Path
import json,os,signal,subprocess
from concurrent.futures import ThreadPoolExecutor
r=Path({R!r});run=r/'run-v10';assert (run/'handoff-closed-v13.json').exists()
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
(run/'worker-closure-v13.json').write_text(json.dumps(rows,indent=2)+'\\n');print(json.dumps(dict(closed_jobs=len(rows),all_leases_retained=True)))
'''
 print(relay(code),flush=True)
