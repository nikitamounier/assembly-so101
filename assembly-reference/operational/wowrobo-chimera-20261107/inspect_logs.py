from pathlib import Path
import json,subprocess,shlex
h=Path(__file__).parent;c=json.loads((h/'target.json').read_text());r=c['remote']
code=f'''from pathlib import Path
import json
p=Path({r!r})/'run-v2'
print('LATEST LEARNER');print((p/'learner-3606264.log').read_text()[-9000:])
print('WORKER ERRORS')
for f in sorted(p.glob('worker-*.log')):
 s=f.read_text()[-3000:]
 if any(x in s for x in ['Traceback','ValueError','Error:','Out Of Memory','Aborted']):print(f.name,s[-1300:])
'''
subprocess.run(['ssh','-o','ConnectTimeout=15','chimera-login','srun --jobid=3077705 --overlap --exact --ntasks=1 --cpus-per-task=1 '+c['python']+' -c '+shlex.quote(code)],check=True)
