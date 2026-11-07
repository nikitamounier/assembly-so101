"""Checkpoint-pause the original learner; retain every allocation for handoff."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import threading
import time

ROOT=Path('/large_storage/goodarzilab/nikitamounier/assembly/source-85c006a')
RUN=ROOT/'runs/chimera-v13-8192-20261106'
ART=ROOT/'artifacts_local/chimera';STATE=ART/'controller.json'
assert os.environ.get('SLURM_JOB_ID')=='3077705'
assert not (RUN/'user-pause.json').exists(),'Reconcile an existing pause instead of repeating it'
manifest_path=ART/'source-hashes.json';manifest=json.loads(manifest_path.read_text())
for name,expected in manifest.items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expected,name
pool=json.loads((ART/'node-pool.json').read_text())
state=json.loads(STATE.read_text());before=json.loads((RUN/'log.jsonl').read_text().splitlines()[-1])
controller=str(ROOT/'operational/chimera-20261106/pool_controller.py')
pids=[]
for path in Path('/proc').iterdir():
    if not path.name.isdigit():continue
    try:
        if path.stat().st_uid!=os.getuid():continue
        argv=(path/'cmdline').read_bytes().decode().split('\0')
        if controller in argv:pids.append(int(path.name))
    except (FileNotFoundError,ProcessLookupError,PermissionError):continue
assert len(pids)<=1,pids
if pids:
    os.kill(pids[0],signal.SIGSTOP)
    time.sleep(.1)
    assert '\nState:\tT' in (Path('/proc')/str(pids[0])/'status').read_text()

phase='user_pause_checkpoint_requested';stop_writer=threading.Event()
def publish():
    row=dict(state,phase=phase,observed_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),pause_reason='Human reserves original GPU/CPU allocations for cube trial chat; no automatic resume',pause_controller_pid=pids[0] if pids else None)
    temp=STATE.with_suffix('.pause.tmp');temp.write_text(json.dumps(row,indent=2)+'\n');temp.replace(STATE)
def heartbeat():
    while not stop_writer.wait(10):publish()
publish();thread=threading.Thread(target=heartbeat,daemon=True);thread.start()
ENV={k:v for k,v in os.environ.items() if not k.startswith(('SLURM_','SRUN_')) and k!='CUDA_VISIBLE_DEVICES'}
ENV.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')

def steps():
    result=subprocess.run(['squeue','--steps','--noheader','--jobs=3598717','--format=%i'],capture_output=True,text=True,timeout=30)
    assert result.returncode==0 and not result.stderr.strip(),result.stderr
    return [x for x in result.stdout.split() if x not in ('3598717.batch','3598717.extern')]
try:
    if steps():
        command=['srun','--jobid=3598717','--overlap','--exact','--ntasks=1','--cpus-per-task=1',str(ROOT/'.venv/bin/python'),str(ROOT/'operational/chimera-20261106/learner.py'),'--stop']
        result=subprocess.run(command,env=ENV,capture_output=True,text=True,timeout=90)
        (ART/'user-pause-stop-receipt.json').write_text(json.dumps(dict(command=command,returncode=result.returncode,stdout=result.stdout,stderr=result.stderr),indent=2)+'\n')
        assert result.returncode==0,result.stderr
    deadline=time.time()+900
    while steps():
        assert time.time()<deadline,'Learner did not finish a checkpoint within15min'
        time.sleep(5)
    import torch
    checkpoint=RUN/'checkpoint.pt'
    ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
    last=json.loads((RUN/'log.jsonl').read_text().splitlines()[-1])
    assert ck['it']==last['it']>=before['it']
    assert ck['transitions']==last['transitions']>=before['transitions']
    assert all(torch.isfinite(value).all() for value in ck['model'].values() if isinstance(value,torch.Tensor))
    if pids:
        os.kill(pids[0],signal.SIGTERM);os.kill(pids[0],signal.SIGCONT)
        for _ in range(100):
            if not (Path('/proc')/str(pids[0])).exists():break
            time.sleep(.1)
    lock=open(STATE.with_suffix('.lock'),'a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);lock.close()
    stat=checkpoint.stat()
    receipt=dict(paused=True,user_requested=True,run=str(RUN),gpu_job='3598717',retained_cpu_jobs=[r['job_id'] for r in pool['jobs']],released_jobs=[],reserved_for_thread='01a1b546-facb-7af3-8c19-411d620dd1e7',automatic_resume=False,controller_pid=pids[0] if pids else None,host=socket.gethostname(),paused_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),checkpoint=dict(it=ck['it'],transitions=ck['transitions'],bytes=stat.st_size,mtime_ns=stat.st_mtime_ns,sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest()))
    (RUN/'user-pause.json').write_text(json.dumps(receipt,indent=2)+'\n')
    # A durable intentional pause needs no live controller heartbeat. Keep stale
    # detection for every other phase and validate the sealed checkpoint record.
    observer=ROOT/'operational/chimera-20261106/observe.py';old=observer.read_text()
    needle="if state and time.time()-calendar.timegm(time.strptime(state['observed_utc'],'%Y-%m-%dT%H:%M:%SZ'))>180:errors.append('controller_stale')"
    assert old.count(needle)==1
    replacement="""pause_record=read_json(RUN/'user-pause.json')
pause_valid=bool(state and state.get('phase')=='paused_by_user' and pause_record and pause_record.get('paused') and last and pause_record['checkpoint']['it']==last['it'] and pause_record['checkpoint']['transitions']==last['transitions'] and checkpoint.exists() and pause_record['checkpoint']['bytes']==checkpoint.stat().st_size and pause_record['checkpoint']['mtime_ns']==checkpoint.stat().st_mtime_ns)
summary['intentional_pause']=pause_valid
if state and state.get('phase')=='paused_by_user' and not pause_valid:errors.append('pause_record_invalid')
if state and not pause_valid and time.time()-calendar.timegm(time.strptime(state['observed_utc'],'%Y-%m-%dT%H:%M:%SZ'))>180:errors.append('controller_stale')"""
    (ART/'observe-before-user-pause.py').write_text(old)
    observer.write_text(old.replace(needle,replacement))
    manifest['operational/chimera-20261106/observe.py']=hashlib.sha256(observer.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    phase='paused_by_user';stop_writer.set();thread.join();publish()
    print(json.dumps(receipt),flush=True)
except BaseException:
    phase='checkpoint_request_failed';stop_writer.set();thread.join();publish();raise
