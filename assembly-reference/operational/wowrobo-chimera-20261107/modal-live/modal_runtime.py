"""Functions for the bounded current-model cloud ramp."""
def broker(queue,seconds):
    import asyncio,modal,secrets,subprocess,tempfile
    from pathlib import Path
    from gateway import broker_servers
    with tempfile.TemporaryDirectory() as tmp:
        cert=Path(tmp)/'cert.pem';key=Path(tmp)/'key.pem'
        subprocess.run(['openssl','req','-x509','-newkey','rsa:2048','-nodes','-days','2','-subj','/CN=assembly-so101','-keyout',str(key),'-out',str(cert)],check=True,capture_output=True)
        key.chmod(0o600);token=secrets.token_hex(32)
        with modal.forward(8741,unencrypted=True) as a,modal.forward(8742,unencrypted=True) as b:
            def ready():
                queue.put(dict(worker=dict(host=a.tcp_socket[0],port=a.tcp_socket[1],tls_ca=cert.read_text()),reverse=dict(host=b.tcp_socket[0],port=b.tcp_socket[1],tls_ca=cert.read_text(),token=token)))
            asyncio.run(broker_servers(cert,key,token,ready,seconds))
    return {'status':'expired'}

def worker(ticket,threads,seconds,processes=1,progress=None):
    import json,os,subprocess,sys,tempfile,time,threading,socket,uuid
    group=uuid.uuid4().hex
    started=time.monotonic()
    with tempfile.TemporaryDirectory() as directory:
        from pathlib import Path
        path=Path(directory)/'ticket.json';path.write_text(json.dumps(ticket));path.chmod(0o600)
        # The elastic learner admits workers between long native control intervals.
        # Allow its current interval to complete before timing out authentication.
        # Never put a ticket in argv: code and ticket are supplied over private stdin.
        children=[];drainers=[]
        def drain(proc):
            for line in proc.stdout:
                print(line,end='',flush=True)
                if progress is not None:
                    try:event=json.loads(line)
                    except json.JSONDecodeError:continue
                    if event.get('event','').startswith('modal_'):
                        event['worker_id']=group+':'+str(proc.pid)
                        progress.put(event)
        for _ in range(processes):
            proc=subprocess.Popen([sys.executable,'/opt/modal-live/worker_entry.py'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
            children.append(proc)
            proc.stdin.write(json.dumps(dict(ticket=ticket,threads=threads,seconds=seconds,worker_group=group)));proc.stdin.close()
            thread=threading.Thread(target=drain,args=(proc,),daemon=True);thread.start();drainers.append(thread)
        try:
            while time.monotonic()-started<seconds:
                if any(p.poll() is not None for p in children):raise RuntimeError('a packed worker exited')
                time.sleep(1)
        finally:
            for proc in children:proc.terminate()
            for proc in children:
                try:proc.wait(timeout=15)
                except subprocess.TimeoutExpired:proc.kill();proc.wait()
            for thread in drainers:thread.join(timeout=2)
    return dict(elapsed_s=time.monotonic()-started,threads=threads)
