"""Reattach one pinned controller to retained leases; its inherited locks prevent duplicates."""
from pathlib import Path
import json,shlex,subprocess
HERE=Path(__file__).resolve().parent
target=json.loads((HERE/'target-v9.json').read_text());ops=json.loads((HERE/'ops-v9.json').read_text())
script=target['script']+'cluster.py';digest=ops['pins']['cluster.py']
code='import hashlib,runpy,sys; p='+repr(script)+'; assert hashlib.sha256(open(p,"rb").read()).hexdigest()=='+repr(digest)+';sys.argv=[p,"control"];runpy.run_path(p,run_name="__main__")'
command='srun --jobid=3077705 --overlap --exact --ntasks=1 --cpus-per-task=1 '+target['env']+target['python']+' -c '+shlex.quote(code)
subprocess.run(['ssh','-o','ConnectTimeout=15','-o','ServerAliveInterval=30','-o','ServerAliveCountMax=6','chimera-login',command],check=True)
