import json,subprocess
from pathlib import Path
h=Path(__file__).parent;c=json.loads((h/'target.json').read_text())
subprocess.run(['ssh','-o','ConnectTimeout=15','chimera-login','srun --jobid=3077705 --overlap --exact --ntasks=1 --cpus-per-task=1 '+c['env']+c['python']+' '+c['script']+'autoscale.py --state-dir '+c['remote']+'/autoscale-v2'],check=True)
