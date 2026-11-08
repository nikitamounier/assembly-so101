"""Additive SAIL pool: frozen native physics, private credentials, owned cleanup."""
import datetime, fcntl, hashlib, json, os, secrets, shlex, signal, subprocess, time
from pathlib import Path
import sail

HERE = Path(__file__).resolve().parent
MODAL = HERE.parent/'modal-live'
WORKSPACE = HERE.parents[3]
TARGET = json.loads((HERE/'target.json' if (HERE/'target.json').exists() else MODAL/'target-v6.json').read_text())
ROOT, PYTHON = TARGET['remote'], TARGET['python']
RUN = ROOT+'/'+TARGET.get('run','run-v6')
# 63*64 +7*8 rollout vCPUs +4 relay vCPUs = exactly4092 total.
# A4-vCPU physics worker would serialize its ten-env slice and create a straggler.
PROFILES = ([dict(size='ultra_experimental',cpus=64,processes=7,threads=9,memory=96,disk=64)]*63
            + [dict(size='l',cpus=8,processes=1,threads=8,memory=32,disk=64)]*7)
HOSTS = len(PROFILES)
PROCESSES = sum(p['processes'] for p in PROFILES)
STAMP = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
OUT = HERE/STAMP

def allocated(code):
    return ['ssh','-o','ConnectTimeout=15','-o','ServerAliveInterval=30','chimera-login',
            'srun --jobid=3077705 --overlap --exact -n1 -c1 --mem=1G '+PYTHON+' -c '+shlex.quote(code)]

def remote_ticket():
    code = 'from pathlib import Path; print(Path('+repr(RUN+'/server.json')+').read_text())'
    result = subprocess.run(allocated(code), capture_output=True, text=True, check=True, timeout=60)
    return json.loads(result.stdout)

def refresh_prelaunch_route():
    """The main training owner publishes the versioned route, never physics."""
    global RUN
    path=HERE/'target.json'
    if not path.exists():return
    try:current=json.loads(path.read_text())
    except json.JSONDecodeError:return
    assert current['remote']==ROOT and current['python']==PYTHON, 'New frozen root/runtime needs explicit image qualification'
    name=current.get('run','run-v6')
    assert '/' not in name and name.startswith('run-v')
    RUN=ROOT+'/'+name

def exec_ok(box, command, **kwargs):
    result = box.exec(command, **kwargs).wait()
    if result.exit_code:
        raise RuntimeError('SAIL command failed: '+str(result.exit_code)+' '+result.stderr[-1000:])
    return result.stdout

def transient_error(error):
    return getattr(error,'retryable',False) or 'placement changed' in str(error).lower()

def retry_io(operation, *, attempts=8, sleep=time.sleep):
    """Retry idempotent I/O across provider route/placement changes only."""
    for attempt in range(attempts):
        try:return operation()
        except sail.SailError as error:
            if not transient_error(error) or attempt+1==attempts:raise
            sleep(min(2**attempt,15))

def build_image():
    image = (sail.Image.from_registry('python:3.12-bookworm', architecture='amd64')
             .apt_install('g++','libgl1','openssl')
             .add_local_file(MODAL/'runtime.tar.gz','/tmp/runtime.tar.gz')
             .run_commands('tar -xzf /tmp/runtime.tar.gz -C / && rm /tmp/runtime.tar.gz')
             .add_local_file(WORKSPACE/'assembly-so101/requirements-worker.txt','/tmp/requirements-worker.txt')
             .run_commands('python -m pip install -r /tmp/requirements-worker.txt',
                           'python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu'))
    for name in ['worker.py','supervise.py','broker.py']:
        image = image.add_local_file(HERE/name, '/opt/sail-live/'+name)
    for name in ['gateway.py','recovery_ticket.py']:
        image = image.add_local_file(MODAL/name, '/opt/sail-live/'+name)
    return image.env(dict(PYTHONPATH='/opt/sail-live:'+ROOT+'/assembly-so101',
                          WOWROBO_CANONICAL_ROOT=ROOT+'/robotics-assembly-3.0',
                          OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',SO101_SURFACE_MODE='1')).build(timeout=1800)

def main():
    request=HERE/'stop-request.json'
    if request.exists() and json.loads(request.read_text()).get('stop'):
        print(json.dumps(dict(status='stopped_by_owner',reason='Existing runtime is superseded; no new resources created')))
        return
    owner = (HERE/'owner.lock').open('a')
    fcntl.flock(owner, fcntl.LOCK_EX|fcntl.LOCK_NB)
    for line in (WORKSPACE/'.env').read_text().splitlines():
        if line.startswith('SAIL_API_KEY='): os.environ['SAIL_API_KEY'] = line.split('=',1)[1].strip().strip('"').strip("'")
    assert os.environ.get('SAIL_API_KEY')
    OUT.mkdir(mode=0o700)
    boxes, workers = [], []
    agent = agent_log = remote_agent_record = None
    receipt = dict(status='building', pool=STAMP, run=RUN, target_worker_boxes=HOSTS,
                   target_reserved_worker_vcpus=4088,target_reserved_total_vcpus=4092,target_broker_vcpus=4,target_worker_processes=PROCESSES,
                   target_rollout_threads=sum(p['processes']*p['threads'] for p in PROFILES),
                   launcher_pid=os.getpid(), source='existing frozen run-v6 runtime',
                   started_unix=time.time(), boxes=[], worker_boxes=[], no_elapsed_time_cutoff=True)
    def save():
        receipt['heartbeat_unix'] = time.time()
        for path in [OUT/'receipt.json', HERE/'active.json']:
            temporary = path.with_suffix('.tmp')
            temporary.write_text(json.dumps(receipt,indent=2)+'\n'); temporary.replace(path)
    def stop(signum, frame): raise KeyboardInterrupt('SAIL pool stop signal')
    signal.signal(signal.SIGTERM, stop)
    save()
    try:
        app = sail.App.find(name='wowrobo-run-v6-rollouts', mint_if_missing=True)
        receipt['app_id'] = app.id; save()
        image = build_image()
        receipt.update(status='image_ready', image_id=image._image_id,runtime_sha256=hashlib.sha256((MODAL/'runtime.tar.gz').read_bytes()).hexdigest()); save()
        # No paid fleet until the owning main thread has expanded the learner.
        required_slots=409+100+PROCESSES
        receipt.update(status='waiting_for_learner_capacity',required_worker_slots=required_slots,observed_worker_slots=0);save()
        while True:
            refresh_prelaunch_route();receipt['run']=RUN
            try:
                original=remote_ticket();receipt['observed_worker_slots']=original['n_conns']
                receipt.pop('capacity_observation_error',None)
            except (subprocess.TimeoutExpired,subprocess.CalledProcessError) as error:
                receipt['capacity_observation_error']=type(error).__name__;save();time.sleep(15);continue
            save()
            if (HERE/'stop-request.json').exists() and json.loads((HERE/'stop-request.json').read_text()).get('stop'):
                receipt['status']='user_stopped';return
            if original['n_conns']>=required_slots:break
            time.sleep(15)
        broker = sail.Sailbox.create(app=app, name='wowrobo-broker-'+STAMP, image=image, size='m',
                    memory_limit_gib=8,disk_limit_gib=32,visibility='private',auto_sleep=sail.AutoSleep.never(),
                    ingress_ports=[sail.IngressPort(8741,'tcp'),sail.IngressPort(8742,'tcp')])
        boxes.append(broker); receipt['boxes'].append(broker.sailbox_id); receipt['broker_box']=broker.sailbox_id; save()
        token = secrets.token_hex(32)
        retry_io(lambda: broker.fs.write('/run/assembly/broker.json',json.dumps(dict(token=token)),mode=0o600))
        exec_ok(broker, 'python -u /opt/sail-live/broker.py > /run/assembly/broker.log 2>&1', background=True)
        deadline = time.monotonic()+180
        while not retry_io(lambda: broker.fs.exists('/run/assembly/ready')):
            if time.monotonic()>deadline: raise TimeoutError('SAIL TLS broker startup failed')
            time.sleep(2)
        cert = retry_io(lambda: broker.fs.read('/run/assembly/cert.pem')).decode()
        a = broker.wait_for_listener(8741,timeout=180).endpoint
        b = broker.wait_for_listener(8742,timeout=180).endpoint
        endpoint = dict(host=a.host,port=a.port,tls_ca=cert,trusted_network=False)
        reverse = dict(host=b.host,port=b.port,tls_ca=cert,token=token)
        original = remote_ticket()
        assert original['n_conns'] >= required_slots, 'Insufficient learner worker slots'
        ticket = {**original, **endpoint}
        source = (MODAL/'gateway.py').read_text().replace("if __name__=='__main__':", 'if False:')
        remote_agent_record=RUN+'/sail-agent-'+STAMP+'.json'
        code = ('import sys,json,asyncio,os;from pathlib import Path;pool='+repr(STAMP)+';'
                'record=Path('+repr(remote_agent_record)+');'
                'row=dict(provider="sail",pool=pool,pid=os.getpid(),start=Path("/proc/self/stat").read_text().split()[21]);'
                'record.write_text(json.dumps(row));exec('+repr(source)+');'
                'c=json.loads(sys.stdin.read());asyncio.run(agent(c["reverse"],c["target"],c["connections"],c["seconds"]))')
        agent_log = (OUT/'chimera-agent.log').open('w')
        agent = subprocess.Popen(allocated(code),stdin=subprocess.PIPE,stdout=agent_log,stderr=subprocess.STDOUT,text=True)
        agent.stdin.write(json.dumps(dict(reverse=reverse,target=dict(ticket_path=RUN+'/server.json',framing='multiprocessing'),connections=PROCESSES,seconds=7*86400))); agent.stdin.close()
        receipt.update(status='first_box_starting', agent_launcher_pid=agent.pid,remote_agent_record=remote_agent_record); save()
        def start_worker():
            index = len(workers)
            profile=PROFILES[index]
            box = sail.Sailbox.create(app=app,name='wowrobo-worker-'+STAMP+'-'+str(index), image=image,
                      size=profile['size'],memory_limit_gib=profile['memory'],disk_limit_gib=profile['disk'],visibility='private',auto_sleep=sail.AutoSleep.never())
            boxes.append(box); workers.append(box)
            receipt['boxes'].append(box.sailbox_id);receipt['worker_boxes'].append(box.sailbox_id);save()
            retry_io(lambda: box.fs.write('/run/assembly/ticket.json',json.dumps(ticket),mode=0o600))
            retry_io(lambda: box.fs.write('/run/assembly/config.json',json.dumps(dict(ticket=ticket,ticket_path='/run/assembly/ticket.json',group=box.sailbox_id,processes=profile['processes'],threads=profile['threads'])),mode=0o600))
            exec_ok(box,'python -u /opt/sail-live/supervise.py > /run/assembly/workers.log 2>&1',background=True)
            return box
        first = start_worker()
        deadline = time.monotonic()+900
        while True:
            lines = retry_io(lambda: first.fs.read('/run/assembly/workers.log')).decode(errors='replace').splitlines()
            events=[]
            for line in lines:
                try: events.append(json.loads(line))
                except json.JSONDecodeError: pass
            initialized = {e['pid'] for e in events if e.get('event')=='initialized'}
            finite = {e['pid'] for e in events if e.get('event')=='first_control' and e.get('finite')}
            receipt.update(initialized_first_box=len(initialized),finite_first_control_first_box=len(finite));save()
            if len(finite)==PROFILES[0]['processes']:
                (OUT/'first-control-proof.json').write_text(json.dumps([e for e in events if e.get('event') in ('initialized','first_control')],indent=2)+'\n')
                break
            if agent.poll() is not None: raise RuntimeError('SAIL outbound agent exited')
            if time.monotonic()>deadline:
                (OUT/'first-box-failure.log').write_text('\n'.join(lines)[-16000:])
                raise TimeoutError('First SAIL box did not return all finite control results')
            time.sleep(10)
        receipt['status']='expanding';save()
        for _ in range(HOSTS-1):start_worker()
        reserved=sum(p['cpus'] for p in PROFILES[:len(workers)])
        threads=sum(p['processes']*p['threads'] for p in PROFILES[:len(workers)])
        receipt.update(status='workers_dispatched',reserved_worker_vcpus=reserved,reserved_total_vcpus=reserved+4,rollout_threads=threads);save()
        print(json.dumps(dict(event='sail_workers_dispatched',receipt=str(OUT/'receipt.json'),reserved_vcpus=reserved,rollout_threads=threads)),flush=True)
        last_refresh=last_progress=0
        while True:
            if (HERE/'stop-request.json').exists() and json.loads((HERE/'stop-request.json').read_text()).get('stop'):
                receipt['status']='user_stopped';break
            if agent.poll() is not None:raise RuntimeError('SAIL outbound agent exited')
            if time.monotonic()-last_refresh>30:
                try:
                    updated={**remote_ticket(),**endpoint}
                    if updated!=ticket:
                        for box in workers:
                            retry_io(lambda: box.fs.write('/run/assembly/ticket.next',json.dumps(updated),mode=0o600))
                            # Atomic replacement is idempotent even if the first response was lost.
                            retry_io(lambda: exec_ok(box,['python','-c',"from pathlib import Path;p=Path('/run/assembly/ticket.next');p.exists() and p.replace('/run/assembly/ticket.json')"]))
                        ticket=updated
                    receipt.pop('ticket_refresh_error',None)
                except (subprocess.TimeoutExpired,subprocess.CalledProcessError) as error:
                    receipt['ticket_refresh_error']=type(error).__name__
                except sail.SailError as error:
                    if not transient_error(error):raise
                    receipt['ticket_refresh_error']=str(error)
                last_refresh=time.monotonic()
            if time.monotonic()-last_progress>60:
                code='from pathlib import Path;import json;r=Path('+repr(RUN)+');c=json.loads((r/"controller.json").read_text());print(json.dumps(dict(it=c.get("latest",{}).get("it",0))))'
                try:
                    state=json.loads(subprocess.run(allocated(code),capture_output=True,text=True,check=True,timeout=60).stdout)
                    receipt['learner_iteration']=state['it']
                    if state['it']>=10000:receipt['status']='training_complete';break
                    receipt.pop('learner_observation_error',None)
                except (subprocess.TimeoutExpired,subprocess.CalledProcessError) as error:
                    receipt['learner_observation_error']=type(error).__name__
                counts=[]
                for box,profile in zip(workers,PROFILES):
                    try:
                        text=retry_io(lambda: box.fs.read('/run/assembly/workers.log')).decode(errors='replace')
                        probe="import json,psutil;print(json.dumps([p.pid for p in psutil.process_iter(['cmdline']) if '/opt/sail-live/worker.py' in (p.info['cmdline'] or [])]))"
                        live=set(json.loads(retry_io(lambda: exec_ok(box,['python','-c',probe],timeout=30))))
                    except sail.SailError as error:
                        if not transient_error(error):raise
                        counts.append(dict(box=box.sailbox_id,initialized=0,first_controls=0,observation_error=str(error)));continue
                    if len(live)!=profile['processes']:
                        raise RuntimeError('SAIL worker process exited on '+box.sailbox_id+'; see native worker log before replacement')
                    events=[]
                    for line in text.splitlines():
                        try:events.append(json.loads(line))
                        except json.JSONDecodeError:pass
                    initialized={e['pid'] for e in events if e.get('event')=='initialized'}
                    finite={e['pid'] for e in events if e.get('event')=='first_control' and e.get('finite')}
                    counts.append(dict(box=box.sailbox_id,initialized=len(initialized&live),first_controls=len(finite&live),live_worker_processes=len(live),observed_unix=time.time()))
                receipt['worker_progress']=counts
                receipt['unique_first_control_workers']=sum(row['first_controls'] for row in counts)
                if all(row['first_controls']>=profile['processes'] for row,profile in zip(counts,PROFILES)):
                    receipt['status']='running'
                last_progress=time.monotonic()
            save();time.sleep(5)
    except BaseException as error:
        receipt.update(status='failed', error=type(error).__name__+': '+str(error));save();raise
    finally:
        if remote_agent_record is not None:
            code=f'''from pathlib import Path
import os,json,signal
p=Path({remote_agent_record!r});closed=False
if p.exists():
 r=json.loads(p.read_text());assert r['provider']=='sail' and r['pool']=={STAMP!r}
 q=Path('/proc')/str(r['pid'])
 if q.exists() and (q/'stat').read_text().split()[21]==r['start']:
  assert q.stat().st_uid==os.getuid() and {STAMP.encode()!r} in (q/'cmdline').read_bytes()
  os.kill(r['pid'],signal.SIGTERM);closed=True
print(json.dumps(dict(closed=closed)))
'''
            try:
                result=subprocess.run(allocated(code),capture_output=True,text=True,check=True,timeout=60)
                receipt['remote_agent_cleanup']=json.loads(result.stdout)
            except Exception as error:receipt['remote_agent_cleanup_error']=type(error).__name__
        if agent is not None:
            agent.terminate()
            try:agent.wait(timeout=20)
            except subprocess.TimeoutExpired:agent.kill();agent.wait()
        if agent_log:agent_log.close()
        cleanup_errors=[]
        for box in reversed(boxes):
            try:retry_io(box.terminate)
            except Exception as error:cleanup_errors.append(dict(box=box.sailbox_id,error=str(error)))
        receipt.update(finished_unix=time.time(),cleanup_errors=cleanup_errors);save()

if __name__=='__main__':main()
