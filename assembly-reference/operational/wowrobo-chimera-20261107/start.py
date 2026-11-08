import json,subprocess
from pathlib import Path
h=Path(__file__).parent;c=json.loads((h/'target.json').read_text())
cmd='srun --jobid=3077705 --overlap --exact --ntasks=1 --cpus-per-task=1 '+c['env']+c['python']+' '+c['script']+'cluster.py control'
print('Starting retained-lease diagnostic controller: '+c['remote'],flush=True)
subprocess.run(['ssh','-o','ServerAliveInterval=30','-o','ServerAliveCountMax=6','chimera-login',cmd],check=True)
