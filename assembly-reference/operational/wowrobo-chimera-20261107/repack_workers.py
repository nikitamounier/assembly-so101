import json,subprocess,shlex
from pathlib import Path
h=Path(__file__).parent;c=json.loads((h/'target.json').read_text());r=c['remote']
for n in ['cluster.py','cluster.json']:subprocess.run(['scp',str(h/n),'chimera-login:'+c['script']+n],check=True)
code=f'''from pathlib import Path
import os,signal,json
root=Path({r!r});run=root/'run';state=json.loads((run/'controller.json').read_text());killed=[]
# Kill only this new controller and its exact srun child launchers. Never batch jobs or old holds.
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  if p.stat().st_uid!=os.getuid():continue
  argv=(p/'cmdline').read_bytes().split(b'\\0')
  if len(argv)>2 and str(root/'robotics-assembly-3.0/operational/wowrobo-chimera-20261107/cluster.py').encode() in argv and argv[-2:-1]==[b'control']:
   os.kill(int(p.name),signal.SIGTERM);killed.append(int(p.name))
 except (FileNotFoundError,ProcessLookupError,PermissionError):pass
for k,v in state['workers'].items():
 pid=v.get('launcher_pid')
 if not pid:continue
 p=Path('/proc')/str(pid)
 try:
  argv=(p/'cmdline').read_bytes().split(b'\\0')
  assert b'srun' in Path(argv[0].decode()).name.encode() and str(root/'robotics-assembly-3.0/operational/wowrobo-chimera-20261107/cluster.py').encode() in argv
  os.kill(pid,signal.SIGTERM);killed.append(pid)
 except (FileNotFoundError,ProcessLookupError):pass
print(json.dumps(dict(stopped_owned_launchers=killed)))
'''
# Controller / srun launchers live on relay host.
subprocess.run(['ssh','chimera-login','srun --jobid=3077705 --overlap --exact --ntasks=1 --cpus-per-task=1 '+c['python']+' -c '+shlex.quote(code)],check=True)
