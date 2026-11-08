import json,subprocess,shlex
from pathlib import Path
h=Path(__file__).parent;c=json.loads((h/'target.json').read_text());r=c['remote']
for name in ['cluster.py','cluster.json']:subprocess.run(['scp',str(h/name),'chimera-login:'+c['script']+name],check=True)
code=f'''from pathlib import Path
import json,os,signal
run=Path({r!r})/'run';t=run/'server.json'
if t.exists() and json.loads(t.read_text())['n_conns']!=512:
 row=json.loads((run/'learner-process.json').read_text());p=Path('/proc')/str(row['pid']);argv=(p/'cmdline').read_bytes().split(b'\\0');assert str(run).encode() in argv and b'assembly_policy.train' in argv;os.kill(row['pid'],signal.SIGTERM);print('Requested checkpoint-safe restart for512worker slots')
else:print('512slot configuration ready')
'''
# Process identity must be checked on the learner host, not the CPU relay.
subprocess.run(['ssh','chimera-login','srun --jobid=3606264 --overlap --exact --ntasks=1 --cpus-per-task=1 '+c['python']+' -c '+shlex.quote(code)],check=True)
