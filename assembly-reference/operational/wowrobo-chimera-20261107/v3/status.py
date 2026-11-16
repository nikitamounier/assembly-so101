from pathlib import Path
import json,subprocess,shlex
c=json.loads((Path(__file__).parent/'target.json').read_text());r=c['remote'];code=f"from pathlib import Path; import json; r=Path({r!r}); p=r/'run-v3'; c=json.loads((p/'controller.json').read_text()); print(json.dumps(dict(running_steps=sum(v.get('launcher_pid') is not None and v.get('exit') is None for v in c['workers'].values()),errors={{k:v for k,v in c['workers'].items() if v.get('exit') is not None}}))); f=p/('learner-'+c['gpu_job']+'.log'); print(f.read_text()[-6500:] if f.exists() else 'No learner log')"
subprocess.run(['ssh','-o','ConnectTimeout=15','chimera-login','srun --jobid=3077705 --overlap --exact --ntasks=1 --cpus-per-task=1 '+c['python']+' -c '+shlex.quote(code)],check=True)
