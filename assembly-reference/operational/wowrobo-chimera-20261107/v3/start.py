from pathlib import Path
import json,subprocess
c=json.loads((Path(__file__).parent/'target.json').read_text())
subprocess.run(['ssh','-o','ConnectTimeout=15','-o','ServerAliveInterval=30','-o','ServerAliveCountMax=6','chimera-login','srun --jobid=3077705 --overlap --exact --ntasks=1 --cpus-per-task=1 '+c['env']+c['python']+' '+c['script']+'cluster.py control'],check=True)
