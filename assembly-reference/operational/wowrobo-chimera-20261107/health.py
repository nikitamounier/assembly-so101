import json,subprocess
from pathlib import Path
h=Path(__file__).parent;c=json.loads((h/'target.json').read_text())
subprocess.run(['ssh','-o','ConnectTimeout=15','-o','ConnectionAttempts=1','chimera-login','srun --jobid=3077705 --overlap --exact --ntasks=1 --cpus-per-task=1 '+c['python']+' '+c['script']+'monitor.py'],check=True)
