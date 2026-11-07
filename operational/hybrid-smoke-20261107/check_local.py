"""Exercise the exact smoke assertions over two local TLS worker connections."""
import contextlib
import importlib.util
from pathlib import Path
import queue
import sys
import tarfile
import tempfile
import threading

sys.path.append('/tmp/assembly-provider-sdk/lib/python3.12/site-packages')
import modal

bundle=Path(sys.argv[1]).resolve()
with tempfile.TemporaryDirectory(prefix='hybrid-smoke-check-') as tmp:
    root=Path(tmp)
    with tarfile.open(bundle) as archive:archive.extractall(root,filter='data')
    sys.path.insert(0,str(root))
    spec=importlib.util.spec_from_file_location('smoke_runtime',root/'smoke_runtime.py')
    runtime=importlib.util.module_from_spec(spec)
    code=(root/'smoke_runtime.py').read_text().replace("Path('/tmp/hybrid-smoke')",repr(root/'run').replace('PosixPath','Path'))
    exec(compile(code,str(root/'smoke_runtime.py'),'exec'),runtime.__dict__)
    # Local-only test identities exercise two slots; live test uses real hostnames.
    runtime.socket.gethostname=lambda:threading.current_thread().name
    @contextlib.contextmanager
    def forward(*args,**kwargs):
        yield type('Tunnel',(),dict(tcp_socket=('127.0.0.1',8701)))()
    modal.forward=forward
    q=queue.Queue();result={};errors=[]
    def serve():
        try:result.update(runtime.learner(q))
        except BaseException as e:errors.append(e)
    learner=threading.Thread(target=serve,daemon=True);learner.start()
    ticket=q.get(timeout=60)
    threads=[]
    for tag in ('local-a','local-b'):
        t=threading.Thread(target=lambda tag=tag:runtime.worker(ticket,tag),daemon=True);t.start();threads.append(t)
    learner.join(90)
    if learner.is_alive():raise TimeoutError('Local smoke did not finish')
    if errors:raise errors[0]
    for thread in threads:thread.join(5)
    print(result['status'],result.get('ppo_updates'),flush=True)
