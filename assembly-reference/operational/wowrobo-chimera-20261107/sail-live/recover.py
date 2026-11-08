"""Adopt live SAIL models; observational failures never tear down the fleet."""
import concurrent.futures, datetime, fcntl, json, os, secrets, signal, subprocess, threading, time
from pathlib import Path
import sail
import launch as L

HERE=L.HERE
STAMP=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
OUT=HERE/STAMP

def main():
    request=HERE/'stop-request.json'
    if request.exists() and json.loads(request.read_text()).get('stop'):
        print(json.dumps(dict(status='stopped_by_owner',reason='Existing runtime is superseded; no new resources created')))
        return
    owner=(HERE/'owner.lock').open('a');fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for line in (L.WORKSPACE/'.env').read_text().splitlines():
        if line.startswith('SAIL_API_KEY='):os.environ['SAIL_API_KEY']=line.split('=',1)[1].strip().strip('"').strip("'")
    old=json.loads((HERE/'recover-from.json').read_text());OUT.mkdir(mode=0o700)
    stop=threading.Event();lock=threading.RLock();workers={};owned=set(old['boxes']);agent=None;agent_log=None;agent_record=None
    row=dict(pool=STAMP,adopted_from=old['pool'],app_id=old['app_id'],run=L.RUN,status='recovering',launcher_pid=os.getpid(),started_unix=time.time(),boxes=list(owned),worker_boxes=[],target_reserved_total_vcpus=4092,target_worker_processes=448,observation_errors={},no_elapsed_time_cutoff=True)
    def save():
        with lock:
            row['heartbeat_unix']=time.time()
            for path in [HERE/'active.json',OUT/'receipt.json']:
                temp=path.with_suffix('.tmp');temp.write_text(json.dumps(row,indent=2)+'\n');temp.replace(path)
    def heartbeat():
        while not stop.wait(5):save()
    signal.signal(signal.SIGTERM,lambda *args:stop.set())
    signal.signal(signal.SIGINT,lambda *args:stop.set())
    save();threading.Thread(target=heartbeat,daemon=True).start()
    image=L.build_image()
    app=old['app_id']
    inventory={b.sailbox_id:b for b in sail.Sailbox.list(app_id=app,limit=300)}
    broker=inventory.get(old['broker_box'])
    if broker is None or broker.status!='running':
        broker=sail.Sailbox.create(app=app,name='wowrobo-broker-'+STAMP,image=image,size='m',memory_limit_gib=8,disk_limit_gib=32,visibility='private',auto_sleep=sail.AutoSleep.never(),ingress_ports=[sail.IngressPort(8741,'tcp'),sail.IngressPort(8742,'tcp')])
        owned.add(broker.sailbox_id);row['boxes']=list(owned);save()
        L.retry_io(lambda:broker.fs.write('/run/assembly/broker.json',json.dumps(dict(token=secrets.token_hex(32))),mode=0o600))
        result=broker.exec('python -u /opt/sail-live/broker.py > /run/assembly/broker.log 2>&1',background=True,idempotency_key='broker-'+STAMP,timeout=120).wait()
        if result.exit_code:raise RuntimeError('New SAIL broker start failed')
        while not L.retry_io(lambda:broker.fs.exists('/run/assembly/ready')):time.sleep(2)
    row['broker_box']=broker.sailbox_id;save()
    cert=L.retry_io(lambda:broker.fs.read('/run/assembly/cert.pem')).decode()
    token=json.loads(L.retry_io(lambda:broker.fs.read('/run/assembly/broker.json')))['token']
    a=broker.wait_for_listener(8741,timeout=180).endpoint;b=broker.wait_for_listener(8742,timeout=180).endpoint
    endpoint=dict(host=a.host,port=a.port,tls_ca=cert,trusted_network=False)
    reverse=dict(host=b.host,port=b.port,tls_ca=cert,token=token)
    L.refresh_prelaunch_route();row['run']=L.RUN
    def new_ticket():return {**L.remote_ticket(),**endpoint}
    ticket=new_ticket();assert ticket['n_conns']>=957

    def close_agent():
        nonlocal agent,agent_log
        if agent_record:
            code=f'''from pathlib import Path
import json,os,signal
p=Path({agent_record!r})
if p.exists():
 r=json.loads(p.read_text());q=Path('/proc')/str(r['pid'])
 assert r['provider']=='sail' and r['pool']=={STAMP!r}
 if q.exists() and (q/'stat').read_text().split()[21]==r['start']:
  assert q.stat().st_uid==os.getuid() and {STAMP.encode()!r} in (q/'cmdline').read_bytes()
  os.kill(r['pid'],signal.SIGTERM)
'''
            try:subprocess.run(L.allocated(code),capture_output=True,text=True,timeout=60)
            except Exception:pass
        if agent is not None and agent.poll() is None:
            agent.terminate()
            try:agent.wait(timeout=15)
            except subprocess.TimeoutExpired:agent.kill();agent.wait()
        if agent_log:agent_log.close()
        agent=agent_log=None

    def start_agent():
        nonlocal agent,agent_log,agent_record
        close_agent();agent_record=L.RUN+'/sail-agent-'+STAMP+'.json'
        source=(L.MODAL/'gateway.py').read_text().replace("if __name__=='__main__':",'if False:')
        code=('import sys,json,asyncio,os;from pathlib import Path;pool='+repr(STAMP)+';'
              'p=Path('+repr(agent_record)+');p.write_text(json.dumps(dict(provider="sail",pool=pool,pid=os.getpid(),start=Path("/proc/self/stat").read_text().split()[21])));'
              'exec('+repr(source)+');c=json.loads(sys.stdin.read());asyncio.run(agent(c["reverse"],c["target"],c["connections"],c["seconds"]))')
        agent_log=(OUT/'chimera-agent.log').open('a')
        agent=subprocess.Popen(L.allocated(code),stdin=subprocess.PIPE,stdout=agent_log,stderr=subprocess.STDOUT,text=True)
        agent.stdin.write(json.dumps(dict(reverse=reverse,target=dict(ticket_path=L.RUN+'/server.json',framing='multiprocessing'),connections=448,seconds=7*86400)));agent.stdin.close()
        row.update(agent_launcher_pid=agent.pid,remote_agent_record=agent_record);save()

    slots=old.get('worker_slots') or {str(i):box_id for i,box_id in enumerate(old['worker_boxes'])}
    for index,box_id in ((int(i),box_id) for i,box_id in slots.items()):
        box=inventory.get(box_id)
        if box is not None and box.status=='running':workers[index]=box
    started=set(workers)
    row['adopted_worker_boxes']=len(workers);row['worker_slots']={str(i):box.sailbox_id for i,box in workers.items()};save();start_agent()

    def post_ticket(box,value):
        L.retry_io(lambda:box.fs.write('/run/assembly/ticket.json',json.dumps(value),mode=0o600),attempts=3)
    # Reuse loaded native models: update their connection ticket, never restart them.
    for index,box in workers.items():
        try:post_ticket(box,ticket)
        except Exception as error:row['observation_errors'][str(index)]='ticket: '+str(error)

    def create_worker(index):
        profile=L.PROFILES[index];name='wowrobo-recovered-'+STAMP+'-'+str(index)
        # Stable per-slot name lets a later reconciliation adopt an uncertain create.
        matches=sail.Sailbox.list(app_id=app,name=name,exclude_statuses=['terminated','failed'],limit=10)
        box=matches[0] if matches else sail.Sailbox.create(app=app,name=name,image=image,size=profile['size'],memory_limit_gib=profile['memory'],disk_limit_gib=profile['disk'],visibility='private',auto_sleep=sail.AutoSleep.never())
        with lock:
            owned.add(box.sailbox_id);workers[index]=box;row['boxes']=list(owned)
            row['worker_slots']={str(i):b.sailbox_id for i,b in workers.items()};save()
        post_ticket(box,ticket)
        config=dict(ticket=ticket,ticket_path='/run/assembly/ticket.json',group=box.sailbox_id,processes=profile['processes'],threads=profile['threads'])
        L.retry_io(lambda:box.fs.write('/run/assembly/config.json',json.dumps(config),mode=0o600),attempts=3)
        result=box.exec('python -u /opt/sail-live/supervise.py > /run/assembly/workers.log 2>&1',background=True,idempotency_key='assembly-supervise-'+box.sailbox_id,timeout=180).wait()
        if result.exit_code:raise RuntimeError('Worker startup returned '+str(result.exit_code))
        with lock:started.add(index)
        return box

    def observe(item):
        index,box=item
        try:
            text=L.retry_io(lambda:box.fs.read('/run/assembly/workers.log'),attempts=3).decode(errors='replace')
            events=[]
            for line in text.splitlines():
                try:events.append(json.loads(line))
                except json.JSONDecodeError:pass
            initialized={e['pid'] for e in events if e.get('event')=='initialized'}
            finite={e['pid'] for e in events if e.get('event')=='first_control' and e.get('finite')}
            return dict(index=index,box=box.sailbox_id,initialized=len(initialized),first_controls=len(finite),observed_unix=time.time())
        except Exception as error:
            return dict(index=index,box=box.sailbox_id,initialized=0,first_controls=0,observation_error=type(error).__name__+': '+str(error),observed_unix=time.time())

    last_refresh=last_learner_check=0
    try:
        while not stop.is_set():
            if (HERE/'stop-request.json').exists() and json.loads((HERE/'stop-request.json').read_text()).get('stop'):break
            try:
                if agent is None or agent.poll() is not None:start_agent()
                live=sail.Sailbox.list(app_id=app,limit=300);by_id={b.sailbox_id:b for b in live}
                # Provider losses affect that slot only; never tear down healthy peers.
                for index,box in list(workers.items()):
                    current=by_id.get(box.sailbox_id)
                    if current is not None and current.status in ('failed','terminated'):
                        del workers[index];started.discard(index)
                missing=[i for i in range(len(L.PROFILES)) if i not in workers or i not in started]
                if missing:
                    row['status']='restoring_capacity';save()
                    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                        futures={executor.submit(create_worker,index):index for index in missing}
                        for future in concurrent.futures.as_completed(futures):
                            index=futures[future]
                            try:future.result();row['observation_errors'].pop(str(index),None)
                            except Exception as error:row['observation_errors'][str(index)]='startup: '+str(error)
                            row['worker_boxes']=[workers[i].sailbox_id for i in sorted(workers)];save()
                if time.monotonic()-last_refresh>30:
                    updated=new_ticket()
                    for index,box in list(workers.items()):
                        try:post_ticket(box,updated)
                        except Exception as error:row['observation_errors'][str(index)]='ticket: '+str(error)
                    ticket=updated;last_refresh=time.monotonic()
                with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
                    counts=list(executor.map(observe,list(workers.items())))
                live=sail.Sailbox.list(app_id=app,limit=300)
                current_ids={b.sailbox_id for b in workers.values()}|{broker.sailbox_id}
                row['provider_boxes']=[dict(box=b.sailbox_id,status=b.status,vcpu_count=b.vcpu_count,used_vcpus=b.cpu_used_vcpu) for b in live if b.sailbox_id in current_ids]
                row['running_total_reserved_vcpus']=sum((b['vcpu_count'] or 0) for b in row['provider_boxes'] if b['status']=='running')
                row['worker_progress']=counts;row['worker_boxes']=[workers[i].sailbox_id for i in sorted(workers)]
                row['worker_slots']={str(i):b.sailbox_id for i,b in workers.items()}
                row['unique_first_control_workers']=sum(min(c['first_controls'],L.PROFILES[c['index']]['processes']) for c in counts)
                row['initialized_workers']=sum(min(c['initialized'],L.PROFILES[c['index']]['processes']) for c in counts)
                row['provider_observed_unix']=time.time()
                row['status']='running' if row['running_total_reserved_vcpus']==4092 and row['unique_first_control_workers']==448 else 'initializing'
                if time.monotonic()-last_learner_check>60:
                    try:
                        code='from pathlib import Path;import json;c=json.loads(Path('+repr(L.RUN+'/controller.json')+').read_text());print(json.dumps(dict(it=c.get("latest",{}).get("it",0))))'
                        result=subprocess.run(L.allocated(code),capture_output=True,text=True,check=True,timeout=45)
                        row['learner_iteration']=json.loads(result.stdout)['it']
                        if row['learner_iteration']>=10000:row['training_complete']=True;stop.set()
                    except Exception as error:row['learner_observation_error']=type(error).__name__
                    last_learner_check=time.monotonic()
                row.pop('loop_error',None);save()
            except Exception as error:
                # Even a failed observation leaves all healthy simulation workers intact.
                row.update(status='observation_degraded',loop_error=type(error).__name__+': '+str(error));save()
            stop.wait(10)
    finally:
        # Cleanup happens only for an explicit stop, never for an observation error.
        row['status']='stopping';save();close_agent()
        cleanup=[]
        for box_id in owned:
            try:L.retry_io(lambda:sail.Sailbox.from_id(box_id).terminate(),attempts=3)
            except Exception as error:cleanup.append(dict(box=box_id,error=str(error)))
        row.update(status='stopped',finished_unix=time.time(),cleanup_errors=cleanup);save()

if __name__=='__main__':main()
