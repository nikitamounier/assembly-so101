"""Attached, bounded live Modal CPU ramp; all handles and resources recorded."""
from pathlib import Path
import datetime,hashlib,json,os,shlex,subprocess,sys,time
from queue import Empty
import modal
from modal_runtime import broker,worker

HERE=Path(__file__).resolve().parent
TARGET=json.loads((Path(os.environ['ASSEMBLY_MODAL_TARGET']) if 'ASSEMBLY_MODAL_TARGET' in os.environ else HERE.parent/'v3/target.json').read_text())
R=TARGET['remote'];PY=TARGET['python']
RUN_NAME=os.environ.get('ASSEMBLY_MODAL_RUN','run-v3')
FULL=os.environ.get('ASSEMBLY_MODAL_FULL')=='1'
PROCESSES=6 if FULL else 1
SECONDS=int(os.environ.get('ASSEMBLY_MODAL_SECONDS','3600'))
HOSTS=68 if FULL else int(os.environ.get('ASSEMBLY_MODAL_HOSTS','6'))
CPUS=60 if FULL else int(os.environ.get('ASSEMBLY_MODAL_CPUS','10'))
assert 1<=HOSTS<=68 and 1<=CPUS<=64 and 60<=SECONDS<=7200
STAMP=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
OUT=HERE/STAMP;OUT.mkdir(mode=0o700)
def allocated(code):
 return ['ssh','-o','ConnectTimeout=15','-o','ServerAliveInterval=30','chimera-login','srun --jobid=3077705 --overlap --exact --ntasks=1 --cpus-per-task=1 '+PY+' -c '+shlex.quote(code)]

def main():
 receipt=dict(status='starting',run=R+'/'+RUN_NAME,hosts=HOSTS+(1 if FULL else 0),cpu_per_container=CPUS,requested_physical_cores=4092 if FULL else HOSTS*CPUS,seconds=SECONDS,memory_mib_per_container=98304 if FULL else 16384,scope='bounded hybrid continuation' if FULL else 'bounded live-worker ramp; existing PPO steps/batch configuration unchanged')
 def save():
  (OUT/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
  (HERE/'active.json').write_text(json.dumps(dict(receipt=str(OUT/'receipt.json'),**receipt),indent=2)+'\n')
 save()
 manifest=json.loads((HERE/'runtime-manifest.json').read_text())
 # Extract under the original absolute root: no geometry/XML/profile bytes are rewritten.
 image=(modal.Image.debian_slim(python_version='3.12').apt_install('g++','libgl1','openssl')
        .add_local_file(str(HERE/'runtime.tar.gz'),'/tmp/runtime.tar.gz',copy=True)
        .run_commands('tar -xzf /tmp/runtime.tar.gz -C /')
        .add_local_file(str(HERE.parents[3]/'assembly-so101/requirements-worker.txt'),'/tmp/requirements-worker.txt',copy=True)
        .run_commands('python -m pip install -r /tmp/requirements-worker.txt')
        .pip_install('torch==2.11.0',index_url='https://download.pytorch.org/whl/cpu')
        .add_local_file(str(HERE/'gateway.py'),'/opt/modal-live/gateway.py',copy=True)
        .add_local_file(str(HERE/'modal_runtime.py'),'/opt/modal-live/modal_runtime.py',copy=True)
        .add_local_file(str(HERE/'worker_entry.py'),'/opt/modal-live/worker_entry.py',copy=True)
        .env(dict(PYTHONPATH='/opt/modal-live:'+R+'/assembly-so101',WOWROBO_CANONICAL_ROOT=R+'/robotics-assembly-3.0',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',SO101_SURFACE_MODE='1')))
 app=modal.App('wowrobo-live-rollouts')
 relay=app.function(image=image,cpu=(2,2),memory=(2048,2048),timeout=SECONDS+900,max_containers=1,serialized=True,include_source=False)(broker)
 memory=98304 if FULL else 16384
 memory_request=49152 if FULL else 8192
 cpu=app.function(name='worker',image=image,cpu=(CPUS,CPUS),memory=(memory_request,memory),timeout=SECONDS+60,max_containers=HOSTS,serialized=True,include_source=False)(worker)
 if FULL:
  small=app.function(name='worker_remainder',image=image,cpu=(12,12),memory=(8192,16384),timeout=SECONDS+60,max_containers=1,serialized=True,include_source=False)(worker)
 calls=[];agent_proc=None;agent_log=None
 try:
  with modal.enable_output(),app.run(),modal.Queue.ephemeral() as queue:
   receipt['app_id']=app.app_id;save();print(json.dumps(dict(event='app_started',app_id=app.app_id)),flush=True)
   if os.environ.get('ASSEMBLY_MODAL_PREPARE_ONLY')=='1':
    receipt['status']='image_ready';return
   calls.append(relay.spawn(queue,SECONDS+600));endpoints=queue.get(timeout=300)
   # Only private stdin carries worker authentication and reverse-channel secrets.
   setup="import json,pathlib,sys; p=pathlib.Path("+repr(R+'/'+RUN_NAME+'/server.json')+");sys.stdout.write(p.read_text())"
   original=json.loads(subprocess.run(allocated(setup),check=True,capture_output=True,text=True,timeout=45).stdout)
   ticket={**original,**endpoints['worker'],'trusted_network':False}
   agent_source=(HERE/'gateway.py').read_text()
   bootstrap='import sys,json,asyncio; exec('+repr(agent_source.replace("if __name__=='__main__':","if False:"))+'); c=json.loads(sys.stdin.read()); asyncio.run(agent(c["reverse"],c["target"],c["connections"],c["seconds"]))'
   agent_log=(OUT/'chimera-agent.log').open('w')
   agent_proc=subprocess.Popen(allocated(bootstrap),stdin=subprocess.PIPE,stdout=agent_log,stderr=subprocess.STDOUT,text=True)
   agent_proc.stdin.write(json.dumps(dict(reverse=endpoints['reverse'],target=dict(ticket_path=R+'/'+RUN_NAME+'/server.json',framing='multiprocessing'),connections=409 if FULL else HOSTS,seconds=SECONDS+600)));agent_proc.stdin.close()
   initialized=set();stepped=set()
   def event(row):
    if row.get('event')=='modal_worker_initialized':initialized.add(row['worker_id'])
    if row.get('event')=='modal_first_control_step':stepped.add(row['worker_id'])
    receipt.update(initialized_workers=len(initialized),first_control_workers=len(stepped));save()
   initial=2 if FULL else HOSTS
   for _ in range(initial):calls.append(cpu.spawn(ticket,10 if FULL else CPUS,SECONDS,PROCESSES,queue))
   if FULL:
    receipt['status']='packed_smoke_in_progress';save();deadline=time.monotonic()+480
    while len(stepped)<12:
     if time.monotonic()>deadline:raise TimeoutError('Packed Modal workers did not all complete a finite control step; full fleet not launched')
     try:event(queue.get(timeout=20))
     except (Empty,TimeoutError,modal.exception.TimeoutError):continue
    receipt['packed_smoke_passed']=True;save();print(json.dumps(dict(event='packed_smoke_passed',workers=len(stepped))),flush=True)
    for _ in range(HOSTS-initial):calls.append(cpu.spawn(ticket,10,SECONDS,PROCESSES,queue))
    calls.append(small.spawn(ticket,12,SECONDS,1,queue))
    cpu.update_autoscaler(min_containers=HOSTS,max_containers=HOSTS)
    small.update_autoscaler(min_containers=1,max_containers=1)
   receipt.update(status='workers_dispatched',function_call_ids=[c.object_id for c in calls],agent_launcher_pid=agent_proc.pid,worker_transport='pinned TLS on both public legs; learner mutual HMAC end to end',runtime_bundle_sha256=hashlib.sha256((HERE/'runtime.tar.gz').read_bytes()).hexdigest());save()
   print(json.dumps(dict(event='workers_dispatched',cores=receipt['requested_physical_cores'],receipt=str(OUT/'receipt.json'))),flush=True)
   started=time.monotonic();last_health=0
   while time.monotonic()-started<SECONDS:
    if agent_proc.poll() is not None:raise RuntimeError('Chimera outbound agent exited; see private local log')
    try:event(queue.get(timeout=10))
    except (Empty,TimeoutError,modal.exception.TimeoutError):pass
    if time.monotonic()-last_health>30:
     for call in calls:
      try:
       outcome=call.get(timeout=0)
      except (TimeoutError,modal.exception.TimeoutError):continue
      raise RuntimeError('A bounded pool function terminated early: '+str(outcome))
     last_health=time.monotonic()
    receipt['controller_heartbeat_unix']=time.time();save()
   receipt['status']='bounded_ramp_complete'
 except BaseException as exc:
  receipt.update(status='failed',error=repr(exc));raise
 finally:
  for c in calls:
   try:c.cancel()
   except Exception:pass
  if agent_proc is not None and agent_proc.poll() is None:
   agent_proc.terminate()
   try:agent_proc.wait(timeout=15)
   except subprocess.TimeoutExpired:agent_proc.kill();agent_proc.wait()
  if agent_log:agent_log.close()
  receipt['finished_unix']=time.time();save()

if __name__=='__main__':main()
