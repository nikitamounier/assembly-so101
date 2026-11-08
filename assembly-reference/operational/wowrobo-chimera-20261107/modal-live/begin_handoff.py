"""Request a checkpoint-safe old learner stop after all successor gates pass."""
from pathlib import Path
import json,subprocess,shlex,time
HERE=Path(__file__).resolve().parent;old=json.loads((HERE.parent/'v3/target.json').read_text());new=json.loads((HERE/'release-target.json').read_text());R=old['remote'];PY=old['python']
def relay(code,job='3077705'):
 p=subprocess.run(['ssh','-o','ConnectTimeout=15','chimera-login','srun --jobid='+job+' --overlap --exact --ntasks=1 --cpus-per-task=1 --cpu-bind=none '+PY+' -c '+shlex.quote(code)],capture_output=True,text=True,timeout=90)
 if p.returncode:print(p.stdout,p.stderr);p.check_returncode()
 return p.stdout
gate=f"from pathlib import Path; import json; r=Path({new['remote']!r}); assert json.loads((r/'reset-validation-v4.json').read_text())['status']=='PASS'; assert json.loads((r/'gpu-shape-v4.json').read_text())['status']=='PASS'; print('PASS')"
print(relay(gate),flush=True)
(HERE/'handoff-active.json').write_text(json.dumps(dict(status='intentional_handoff',time_unix=time.time(),old_run=R+'/run-v3',new_run=new['remote']+'/run-v4'),indent=2)+'\n')
code=f'''from pathlib import Path
import json,os,signal,time
root=Path({R!r});ops=root/'robotics-assembly-3.0/operational/wowrobo-chimera-20261107';run=root/'run-v3';stopped=[]
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  if p.stat().st_uid!=os.getuid():continue
  a=(p/'cmdline').read_bytes().split(b'\\0')
  if str(ops/'cluster.py').encode() in a and b'control' in a:
   start=(p/'stat').read_text().split()[21];os.kill(int(p.name),signal.SIGSTOP);stopped.append(dict(pid=int(p.name),start=start))
 except (FileNotFoundError,ProcessLookupError,PermissionError):pass
config=root/'autoscale-v3/active-run.json';x=json.loads(config.read_text());x['enabled']=False;config.write_text(json.dumps(x,indent=2)+'\\n')
cfg=json.loads((ops/'cluster.json').read_text());ledger=root/'autoscale-v3/submissions.json';extra=json.loads(ledger.read_text()) if ledger.exists() else {{'jobs':[]}}
result=dict(stopped_controllers=stopped,config=cfg,additional=extra,time=time.time());(run/'handoff-controller-v4.json').write_text(json.dumps(result,indent=2)+'\\n');print(json.dumps(result))
'''
state=json.loads(relay(code));(HERE/'handoff-cluster.json').write_text(json.dumps(state,indent=2)+'\n');cfg=state['config']
code=f'''from pathlib import Path
import json,os,signal,time
run=Path({R+'/run-v3'!r});row=json.loads((run/'learner-process.json').read_text());p=Path('/proc')/str(row['pid']);a=(p/'cmdline').read_bytes().split(b'\\0');assert b'assembly_policy.train' in a and str(run).encode() in a;start=(p/'stat').read_text().split()[21];os.kill(row['pid'],signal.SIGTERM);proof=dict(pid=row['pid'],start=start,job=os.environ['SLURM_JOB_ID'],time=time.time(),reason='Authorized versioned accepted states plus Modal scale');(run/'handoff-request-v4.json').write_text(json.dumps(proof,indent=2)+'\\n');print(json.dumps(proof))
'''
print(relay(code,cfg['gpu_job']),flush=True)
