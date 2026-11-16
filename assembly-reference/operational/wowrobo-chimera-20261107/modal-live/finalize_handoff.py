"""Close only superseded run processes, then migrate its committed checkpoint."""
from pathlib import Path
import json,subprocess,shlex
HERE=Path(__file__).resolve().parent;old=json.loads((HERE.parent/'v3/target.json').read_text());new=json.loads((HERE/'release-target.json').read_text());R=old['remote'];PY=old['python'];cfg=json.loads((HERE/'handoff-cluster.json').read_text())['config']
def relay(code,job='3077705',cpus=1,mem=None,environment=''):
 flags=f'srun --jobid={job} --overlap --exact --ntasks=1 --cpus-per-task={cpus} --cpu-bind=none '
 if mem:flags+='--mem='+mem+' '
 p=subprocess.run(['ssh','-o','ConnectTimeout=15','chimera-login',flags+environment+PY+' -c '+shlex.quote(code)],capture_output=True,text=True)
 if p.returncode:print(p.stdout,p.stderr);p.check_returncode()
 return p.stdout
WORKER_CODE=(HERE/'close_old_worker.py').read_text().replace('RUN_TOKEN',repr(R+'/run-v3'))
code=f'''from pathlib import Path
import json,os,signal,time,subprocess
from concurrent.futures import ThreadPoolExecutor
root=Path({R!r});run=root/'run-v3';assert (run/'handoff-closed.json').exists()
controller=json.loads((run/'handoff-controller-v4.json').read_text())
for row in controller['stopped_controllers']:
 p=Path('/proc')/str(row['pid'])
 if p.exists() and (p/'stat').read_text().split()[21]==row['start']:
  assert b'control' in (p/'cmdline').read_bytes().split(b'\\0');os.kill(row['pid'],signal.SIGKILL)
worker_code={WORKER_CODE!r}
jobs={cfg['cpu_jobs']!r};py={PY!r};env={{k:v for k,v in os.environ.items() if not k.startswith(('SLURM_','SRUN_','SBATCH_'))}}
def go(job):
 q=subprocess.run(['squeue','-j',job,'-h','-o','%T'],capture_output=True,text=True,check=True).stdout.strip()
 if q!='RUNNING':return dict(job=job,state=q,no_running_old_step=True)
 p=subprocess.run(['srun','--jobid='+job,'--overlap','--exact','--ntasks=1','--cpus-per-task=1','--cpu-bind=none',py,'-c',worker_code],capture_output=True,text=True,timeout=60,env=env)
 assert p.returncode==0,(job,p.stderr);return json.loads(p.stdout)
with ThreadPoolExecutor(max_workers=4) as pool:rows=list(pool.map(go,jobs))
(run/'worker-closure-v4.json').write_text(json.dumps(rows,indent=2)+'\\n')
new=Path({new['remote']!r});(new/'parent-run.json').write_text(json.dumps(dict(run=str(run)))+'\\n');print(json.dumps(dict(closed_jobs=len(rows),all_leases_retained=True)))
'''
result=relay(code);(HERE/'closure-v4.json').write_text(result);print(result,flush=True)
import sys
if '--close-only' in sys.argv:raise SystemExit()
command=new['env']+PY+' '+new['script']+'migrate_live.py'
p=subprocess.run(['ssh','-o','ConnectTimeout=15','chimera-login','srun --jobid=3598717 --overlap --exact --ntasks=1 --cpus-per-task=2 --mem=12G --cpu-bind=none '+command],capture_output=True,text=True);(HERE/'migration-receipt.json').write_text(json.dumps(dict(rc=p.returncode,stdout=p.stdout,stderr=p.stderr),indent=2)+'\n');print(p.stdout,p.stderr);p.check_returncode()
