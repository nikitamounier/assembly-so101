"""Run one exclusively assigned prepared stage without changing admission code."""
import argparse,fcntl,json,os,subprocess,sys,time
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--index',type=int,required=True);a=p.parse_args()
r=a.root.resolve(); project=r/'robotics-assembly-3.0'; plan=project/'outputs/reset-bank-1800-20261107-v4'; assignment=json.loads((r/'launch/farm-assignment.json').read_text()); stage=assignment['assigned_stages'][a.index]
assert os.environ.get('SLURM_JOB_ID') and stage not in assignment['active_stages']
# The original coordinator is paused until every farm stage is finished.
pid=assignment['coordinator_pid']; proc=Path('/proc')/str(pid)
# Parent lives on another node, so its preparation witness is authoritative;
# only a farm controller may resume it after stage completion.
assert assignment['coordinator_paused'] is True
out=plan/'shards'/stage;out.mkdir(exist_ok=True)
def write(v):
    p=out/'farm-worker.json'; t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2)+'\n');t.replace(p)
with (out/'.farm-worker.lock').open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    os.set_inheritable(lock.fileno(), True)
    base={'stage':stage,'job':os.environ['SLURM_JOB_ID'],'array_job':os.environ.get('SLURM_ARRAY_JOB_ID'),'node':os.uname().nodename,'pid':os.getpid()}
    write(dict(base,status='running',unix=time.time()))
    env=dict(os.environ,PYTHONPATH=str(project),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
    for attempts in (20,40,80,160):
        cmd=[sys.executable,str(project/'scripts/build_wowrobo_randomized_bank.py'),'--config',str(plan/'config.json'),'--output',str(out),'--stages',stage,'--validation','physical','--max-attempts',str(attempts),'--physical-profile',str(plan/'physical'),'--settle-seconds','.08','--hold-seconds','.2','--full']
        with (out/'farm-generation.log').open('a') as log:
            code=subprocess.run(cmd,cwd=project,env=env,stdout=log,stderr=subprocess.STDOUT,pass_fds=(lock.fileno(),)).returncode
        q=out/'qualified/manifest.json'
        if code==0 and q.exists() and json.loads(q.read_text()).get('complete'):
            write(dict(base,status='complete',attempt_limit=attempts,unix=time.time()));break
        write(dict(base,status='retrying',last_exit=code,attempt_limit=attempts,unix=time.time()))
    else:
        write(dict(base,status='incomplete',last_exit=code,unix=time.time()));sys.exit(2)
