"""Read the explicit state-build resource gate without changing either experiment."""
from pathlib import Path
import json,subprocess,shlex,datetime
HERE=Path(__file__).resolve().parent
t=json.loads((HERE/'target-v10.json').read_text())
code='''from pathlib import Path
import json,hashlib
r=Path(ROOT_TOKEN)
p=r/'run-v10/state-build-cpu-priority.json'
priority=json.loads(p.read_text()) if p.exists() else {}
cfg=json.loads((r/'ops-v10/cluster.json').read_text())
state=Path(priority['state_root']) if priority.get('state_root') else None
returned=state/'exports/return-receipt.json' if state else None
receipt=json.loads(returned.read_text()) if returned and returned.exists() else {}
ready=r/'native-v12/deployment-ready.json'
release=json.loads(ready.read_text()) if ready.exists() else {}
ready_path=state/'borrowed-training/restoration-ready.json' if state else None
resource_ready=json.loads(ready_path.read_text()) if ready_path and ready_path.exists() else {}
closure_path=state/'borrowed-training/worker-closure-proof.json' if state else None
closure=json.loads(closure_path.read_text()) if closure_path and closure_path.exists() else {}
leases=closure.get('leases',{})
expected={str(j) for j in range(3602571,3602611)}
closure_valid=closure.get('status')=='PASS' and set(leases)==expected and all(v.get('status') in ('no_live_borrowed_workers','allocation_not_running') for v in leases.values())
handoff_ready=bool(resource_ready.get('ready_for_handoff') is True
    and resource_ready.get('status')=='resources_ready_for_science_owner_handoff'
    and resource_ready.get('automatic_resume') is False
    and resource_ready.get('accepted')==receipt.get('accepted')==19200
    and receipt.get('status')=='complete' and closure_valid
    and resource_ready.get('worker_closure_sha256')==hashlib.sha256(closure_path.read_bytes()).hexdigest()
    and resource_ready.get('return_receipt_sha256')==hashlib.sha256(returned.read_bytes()).hexdigest()
    and resource_ready.get('checkpoint_preserved')=='64a8d77312dfc5aaea84896d4e5cebeb7143d4cd175c937ccf4dd74ea303a237')
result=dict(priority=priority,cpu_jobs=len(cfg['cpu_jobs']),bank_return=receipt,
            native_v12=release,resources_ready=resource_ready,closure_valid=closure_valid,
            ready_for_parent_handoff=handoff_ready,
            resource_return_mode='science-owner handoff only; CPU config and processes remain paused')
print(json.dumps(result))
'''.replace('ROOT_TOKEN',repr(t['remote']))
cmd=shlex.join(['srun','--jobid=3077705','--overlap','--exact','-n1','-c1','--mem=1G',t['python'],'-c',code])
p=subprocess.run(['ssh','-o','ConnectTimeout=15','chimera-login',cmd],text=True,capture_output=True,timeout=90)
folder=HERE/'monitor-receipts';folder.mkdir(exist_ok=True)
stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
record=dict(returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)
path=folder/(stamp+'-state-build-priority.json');path.write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(dict(receipt=str(path),**record)),flush=True)
p.check_returncode()
