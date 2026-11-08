"""Reattach one pinned controller to retained leases; its inherited locks prevent duplicates."""
from pathlib import Path
import json,shlex,subprocess
HERE=Path(__file__).resolve().parent
target=json.loads((HERE/'target-v14.json').read_text());ops=json.loads((HERE/'ops-v14.json').read_text())
script=target['script']+'cluster.py';digest=ops['pins']['cluster.py']
code="from pathlib import Path;import json,hashlib,runpy,sys;r=Path("+repr(target['remote'])+");assert json.loads((r/'native-v14/migration.json').read_text())['status']=='PASS';assert json.loads((r/'run-v13/handoff-closed-v14.json').read_text())['status']=='PASS';assert len(json.loads((r/'run-v13/worker-closure-v14.json').read_text()))>=40;manifest=json.loads((r/'active-inputs-v13.json').read_text());assert all(hashlib.sha256((r/n).read_bytes()).hexdigest()==h for n,h in manifest.items());p="+repr(script)+'; assert hashlib.sha256(open(p,"rb").read()).hexdigest()=='+repr(digest)+';sys.argv=[p,"control"];runpy.run_path(p,run_name="__main__")'
command='srun --jobid=3077705 --overlap --exact --ntasks=1 --cpus-per-task=1 '+target['env']+target['python']+' -c '+shlex.quote(code)
subprocess.run(['ssh','-o','ConnectTimeout=15','-o','ServerAliveInterval=30','-o','ServerAliveCountMax=6','chimera-login',command],check=True)
