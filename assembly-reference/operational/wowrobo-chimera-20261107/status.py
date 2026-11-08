import json,subprocess,shlex
from pathlib import Path
h=Path(__file__).parent;c=json.loads((h/'target.json').read_text());r=c['remote']
code=f"from pathlib import Path; import json; p=Path({r!r})/'run-v2'; f=p/'controller.json'; x=json.loads(f.read_text()) if f.exists() else {{}}; print(json.dumps(dict(time=x.get('time'), latest=x.get('latest'),running_steps=sum(v.get('exit') is None and v.get('launcher_pid') is not None for v in x.get('workers',{{}}).values()),step_errors={{k:v for k,v in x.get('workers',{{}}).items() if v.get('exit') is not None}}))); print('LEARNER LOG'); log=p/'learner-3606264.log'; print(log.read_text()[-6000:] if log.exists() else 'NO LOG'); print('WORKER LOG'); log=next(p.glob('worker-*.log'),None); print(log.read_text()[-2500:] if log else 'NO LOG')"
subprocess.run(['ssh','-o','ConnectTimeout=15','-o','ConnectionAttempts=1','chimera-login','srun --jobid=3077705 --overlap --exact --ntasks=1 --cpus-per-task=1 '+c['python']+' -c '+shlex.quote(code)],check=True)
