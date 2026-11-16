"""One continuous4092-core pool; provider leases rotate without a pool timer."""
from pathlib import Path
import datetime,fcntl,hashlib,json,os,shlex,subprocess,time
from queue import Empty
import modal
from modal_continuous import broker,worker
from recovery_ticket import rotate_completed

HERE=Path(__file__).resolve().parent
TARGET=json.loads((HERE/'target-v6.json').read_text());R=TARGET['remote'];PY=TARGET['python'];RUN='run-v6'
LEASE=int(os.environ.get('ASSEMBLY_MODAL_LEASE_SECONDS','43200'))
assert 600<=LEASE<=82800
STAMP=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
OUT=HERE/STAMP
def allocated(code):
    return ['ssh','-o','ConnectTimeout=15','-o','ServerAliveInterval=30','chimera-login','srun --jobid=3077705 --overlap --exact -n1 -c1 --mem=1G '+PY+' -c '+shlex.quote(code)]
def read_remote(code):
    p=subprocess.run(allocated(code),capture_output=True,text=True,timeout=60)
    p.check_returncode();return json.loads(p.stdout)

def main():
    owner=(HERE/'continuous-owner.lock').open('a');fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
    OUT.mkdir(mode=0o700)
    receipt=dict(status='starting',mode='continuous',run=R+'/'+RUN,hosts=69,cpu_per_container=60,requested_physical_cores=4092,lease_seconds=LEASE,seconds=None,automatic_lease_renewal=True,elapsed_time_cutoff=False,started_unix=time.time(),stop_request_path=str(OUT/'stop-request.json'),credit_snapshot=dict(remaining_usd=243.27,usage_usd=286.73,usage_limit_usd=500,charges_usd=0,checked_unix=time.time()),estimated_full_pool_usd_per_hour=[219,245])
    def save():
        for path,value in [(OUT/'receipt.json',receipt),(HERE/'active.json',dict(receipt=str(OUT/'receipt.json'),**receipt))]:
            tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)
    save()
    image=(modal.Image.debian_slim(python_version='3.12').apt_install('g++','libgl1','openssl')
        .add_local_file(str(HERE/'runtime.tar.gz'),'/tmp/runtime.tar.gz',copy=True)
        .run_commands('tar -xzf /tmp/runtime.tar.gz -C /')
        .add_local_file(str(HERE.parents[3]/'assembly-so101/requirements-worker.txt'),'/tmp/requirements-worker.txt',copy=True)
        .run_commands('python -m pip install -r /tmp/requirements-worker.txt')
        .pip_install('torch==2.11.0',index_url='https://download.pytorch.org/whl/cpu'))
    for name in ('gateway.py','recovery_ticket.py','modal_continuous.py','worker_entry_continuous.py'):
        image=image.add_local_file(str(HERE/name),'/opt/modal-live/'+name,copy=True)
    image=image.env(dict(PYTHONPATH='/opt/modal-live:'+R+'/assembly-so101',WOWROBO_CANONICAL_ROOT=R+'/robotics-assembly-3.0',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',SO101_SURFACE_MODE='1'))
    app=modal.App('wowrobo-continuous-rollouts')
    # Only the tiny broker uses nonpreemptible CPU; bulk simulation stays at
    # normal rates. Epoch refresh also covers its normal lease replacement.
    relay=app.function(image=image,cpu=(2,2),memory=(2048,2048),timeout=LEASE+60,max_containers=1,nonpreemptible=True,serialized=True,include_source=False)(broker)
    cpu=app.function(name='worker',image=image,cpu=(60,60),memory=(49152,98304),timeout=LEASE+60,max_containers=68,serialized=True,include_source=False)(worker)
    small=app.function(name='worker_remainder',image=image,cpu=(12,12),memory=(8192,16384),timeout=LEASE+60,max_containers=1,serialized=True,include_source=False)(worker)
    calls=[];agent_proc=None;agent_log=None;agent_epoch=None
    remote_record=R+'/'+RUN+'/modal-continuous-agent-'+STAMP+'.json'
    def close_agent():
        nonlocal agent_proc,agent_log,agent_epoch
        if agent_epoch is not None:
            code=f'''from pathlib import Path
import json,os,signal,time
record=Path({remote_record!r});closed=False
if record.exists():
 row=json.loads(record.read_text());assert row['pool']=={STAMP!r}
 p=Path('/proc')/str(row['pid'])
 if p.exists() and (p/'stat').read_text().split()[21]==row['start']:
  assert p.stat().st_uid==os.getuid() and {STAMP.encode()!r} in (p/'cmdline').read_bytes()
  os.kill(row['pid'],signal.SIGTERM);closed=True
print(json.dumps(dict(closed=closed,pool={STAMP!r})))
'''
            result=read_remote(code);receipt['last_agent_closure']=result
        if agent_proc is not None:
            try:agent_proc.wait(timeout=15)
            except subprocess.TimeoutExpired:agent_proc.terminate();agent_proc.wait(timeout=15)
        if agent_log:agent_log.close()
        agent_proc=agent_log=agent_epoch=None
    def start_agent(endpoint):
        nonlocal agent_proc,agent_log,agent_epoch
        if agent_epoch is not None:close_agent()
        source=(HERE/'gateway.py').read_text().replace("if __name__=='__main__':","if False:")
        bootstrap=f'''import sys,json,asyncio,os,socket
from pathlib import Path
pool={STAMP!r}
p=Path({remote_record!r});row=dict(pool=pool,pid=os.getpid(),start=Path('/proc/self/stat').read_text().split()[21],host=socket.gethostname(),job=os.environ['SLURM_JOB_ID']);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(row));tmp.replace(p)
exec({source!r})
c=json.loads(sys.stdin.read());asyncio.run(agent(c['reverse'],c['target'],c['connections'],c['seconds']))
'''
        agent_log=(OUT/'chimera-agent.log').open('a')
        agent_proc=subprocess.Popen(allocated(bootstrap),stdin=subprocess.PIPE,stdout=agent_log,stderr=subprocess.STDOUT,text=True)
        agent_proc.stdin.write(json.dumps(dict(reverse=endpoint['reverse'],target=dict(ticket_path=R+'/'+RUN+'/server.json',framing='multiprocessing'),connections=409,seconds=LEASE+300)));agent_proc.stdin.close()
        agent_epoch=endpoint['epoch'];receipt.update(agent_launcher_pid=agent_proc.pid,broker_epoch=agent_epoch,remote_agent_record=remote_record);save()
    try:
        with modal.enable_output(),app.run(),modal.Queue.ephemeral() as queue,modal.Dict.ephemeral() as directory:
            receipt['app_id']=app.app_id;save();print(json.dumps(dict(event='continuous_app_started',app_id=app.app_id)),flush=True)
            calls.append(dict(role='broker',call=relay.spawn(queue,directory,LEASE)))
            first=queue.get(timeout=300);assert first['event']=='broker_ready'
            endpoint=directory.get('broker');start_agent(endpoint)
            original=read_remote("from pathlib import Path;import json;print(Path("+repr(R+'/'+RUN+'/server.json')+").read_text())")
            initialized=set();stepped=set()
            def event(row):
                if row.get('event')=='modal_worker_initialized':initialized.add(row['worker_id'])
                if row.get('event')=='modal_first_control_step':stepped.add(row['worker_id'])
                receipt.update(initialized_workers=len(initialized),first_control_workers=len(stepped))
            def spawn(role):
                if role=='broker':return relay.spawn(queue,directory,LEASE)
                return (cpu if role=='worker' else small).spawn(original,10 if role=='worker' else 12,LEASE,6 if role=='worker' else 1,directory,queue)
            for _ in range(2):calls.append(dict(role='worker',call=spawn('worker')))
            receipt['status']='packed_smoke_in_progress';save();deadline=time.monotonic()+600
            while len(stepped)<12:
                if time.monotonic()>deadline:raise TimeoutError('Continuous packed smoke did not pass12finite actual workers')
                current=directory.get('broker')
                if current and current['epoch']!=agent_epoch:start_agent(current)
                try:event(queue.get(timeout=5))
                except (Empty,TimeoutError,modal.exception.TimeoutError):pass
                if agent_proc.poll() is not None:raise RuntimeError('Outbound bridge exited during smoke')
                receipt['controller_heartbeat_unix']=time.time();save()
            receipt['packed_smoke_passed']=True;save()
            for _ in range(66):calls.append(dict(role='worker',call=spawn('worker')))
            calls.append(dict(role='worker_remainder',call=spawn('worker_remainder')))
            cpu.update_autoscaler(min_containers=68,max_containers=68);small.update_autoscaler(min_containers=1,max_containers=1)
            receipt.update(status='workers_dispatched',function_call_ids=[row['call'].object_id for row in calls],lease_rotations=0);save()
            print(json.dumps(dict(event='continuous_workers_dispatched',cores=4092,automatic_lease_renewal=True)),flush=True)
            last_health=last_completion=0
            while True:
                if (OUT/'stop-request.json').exists() and json.loads((OUT/'stop-request.json').read_text()).get('stop'):
                    receipt['status']='user_stopped';break
                current=directory.get('broker')
                if current and current['epoch']!=agent_epoch:start_agent(current)
                if agent_proc.poll() is not None:raise RuntimeError('Chimera bridge exited; inspect before recovery')
                try:event(queue.get(timeout=5))
                except (Empty,TimeoutError,modal.exception.TimeoutError):pass
                if time.monotonic()-last_health>30:
                    receipt['lease_rotations']+=rotate_completed(calls,spawn,timeouts=(TimeoutError,modal.exception.TimeoutError))
                    receipt['function_call_ids']=[row['call'].object_id for row in calls];last_health=time.monotonic()
                if time.monotonic()-last_completion>60:
                    state=read_remote("from pathlib import Path;import json;r=Path("+repr(R+'/'+RUN)+");c=json.loads((r/'controller.json').read_text());print(json.dumps(dict(complete=c.get('phase')=='complete' and c.get('latest',{}).get('it',0)>=10000 and (r/'checkpoint.pt').stat().st_size>0,it=c.get('latest',{}).get('it'))))")
                    receipt['learner_iteration']=state['it'];last_completion=time.monotonic()
                    if state['complete']:receipt['status']='training_complete';break
                receipt['controller_heartbeat_unix']=time.time();save()
    except BaseException as exc:
        receipt.update(status='failed',error=repr(exc));raise
    finally:
        for row in calls:
            try:row['call'].cancel()
            except Exception:pass
        try:close_agent()
        except Exception as exc:receipt['agent_cleanup_error']=repr(exc)
        receipt['finished_unix']=time.time();save()

if __name__=='__main__':main()
