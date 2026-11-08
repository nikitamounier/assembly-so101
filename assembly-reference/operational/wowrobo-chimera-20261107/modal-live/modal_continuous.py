"""Automatically renewed transport and worker leases for one continuous pool."""
def broker(queue,directory,lease_seconds):
    import asyncio,modal,secrets,subprocess,tempfile,time,uuid
    from pathlib import Path
    from gateway import broker_servers
    with tempfile.TemporaryDirectory() as tmp:
        cert=Path(tmp)/'cert.pem';key=Path(tmp)/'key.pem'
        subprocess.run(['openssl','req','-x509','-newkey','rsa:2048','-nodes','-days','2','-subj','/CN=assembly-so101','-keyout',str(key),'-out',str(cert)],check=True,capture_output=True)
        key.chmod(0o600);token=secrets.token_hex(32);epoch=uuid.uuid4().hex
        with modal.forward(8741,unencrypted=True) as a,modal.forward(8742,unencrypted=True) as b:
            def ready():
                row=dict(epoch=epoch,time_unix=time.time(),worker=dict(host=a.tcp_socket[0],port=a.tcp_socket[1],tls_ca=cert.read_text()),reverse=dict(host=b.tcp_socket[0],port=b.tcp_socket[1],tls_ca=cert.read_text(),token=token))
                directory.put('broker',row)
                queue.put(dict(event='broker_ready',epoch=epoch))
            asyncio.run(broker_servers(cert,key,token,ready,lease_seconds))
    return dict(status='lease_complete')

def worker(base_ticket,threads,lease_seconds,processes,directory,progress):
    import json,subprocess,sys,tempfile,time,threading,uuid
    from pathlib import Path
    from recovery_ticket import atomic_ticket
    group=uuid.uuid4().hex;started=time.monotonic()
    with tempfile.TemporaryDirectory() as folder:
        path=Path(folder)/'ticket.json';epoch=None;children=[];drainers=[]
        def refresh():
            nonlocal epoch
            current=directory.get('broker')
            if not current:return False
            if current['epoch']!=epoch:
                atomic_ticket(path,{**base_ticket,**current['worker'],'trusted_network':False})
                epoch=current['epoch']
            return True
        while not refresh():time.sleep(1)
        def drain(proc):
            for line in proc.stdout:
                print(line,end='',flush=True)
                try:event=json.loads(line)
                except json.JSONDecodeError:continue
                if event.get('event','').startswith('modal_'):
                    event['worker_id']=group+':'+str(proc.pid);progress.put(event)
        for _ in range(processes):
            proc=subprocess.Popen([sys.executable,'/opt/modal-live/worker_entry_continuous.py'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
            children.append(proc)
            proc.stdin.write(json.dumps(dict(ticket=base_ticket,ticket_path=str(path),threads=threads,lease_seconds=lease_seconds,worker_group=group)));proc.stdin.close()
            thread=threading.Thread(target=drain,args=(proc,),daemon=True);thread.start();drainers.append(thread)
        try:
            while time.monotonic()-started<lease_seconds:
                if any(p.poll() is not None for p in children):raise RuntimeError('A continuous native worker exited; inspect log before retrying')
                refresh();time.sleep(5)
        finally:
            for proc in children:proc.terminate()
            for proc in children:
                try:proc.wait(timeout=15)
                except subprocess.TimeoutExpired:proc.kill();proc.wait()
            for thread in drainers:thread.join(timeout=2)
    return dict(status='lease_complete',threads=threads,processes=processes)
