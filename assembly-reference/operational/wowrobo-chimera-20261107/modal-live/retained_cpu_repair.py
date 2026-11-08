"""Pinned, allocated terminal-only repair; preserves the state-build resource role."""
from pathlib import Path
import datetime,hashlib,json,shlex,subprocess,sys
HERE=Path(__file__).resolve().parent;t=json.loads((HERE/'target-v10.json').read_text())
script=t['remote']+'/resource-repair/retained_cpu_repair_remote.py'
sha=hashlib.sha256((HERE/'retained_cpu_repair_remote.py').read_bytes()).hexdigest()
args=[script,*sys.argv[1:]]
assert sys.argv[1:] in ([],['--apply'],['--active-run','run-v13'],['--apply','--active-run','run-v13'])
code=f'import hashlib,runpy,sys;p={script!r};assert hashlib.sha256(open(p,"rb").read()).hexdigest()=={sha!r};sys.argv={args!r};runpy.run_path(p,run_name="__main__")'
cmd=shlex.join(['srun','--jobid=3077705','--overlap','--exact','-n1','-c1','--mem=1G',t['python'],'-c',code])
p=subprocess.run(['ssh','-o','ConnectTimeout=15','chimera-login',cmd],text=True,capture_output=True,timeout=120)
record=dict(returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)
stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
folder=HERE/'monitor-receipts';folder.mkdir(exist_ok=True)
receipt=folder/(stamp+'-retained-cpu-repair.json');receipt.write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(dict(receipt=str(receipt),**record)),flush=True);p.check_returncode()
