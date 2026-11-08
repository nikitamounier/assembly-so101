"""Explicit parent-operated release steps; never run automatically during bank build."""
from pathlib import Path
import hashlib,json,shlex,subprocess,sys
HERE=Path(__file__).resolve().parent;H=HERE.parent
t=json.loads((H/'target-v13.json').read_text());ops=json.loads((H/'ops-v13.json').read_text())
mode=sys.argv[1]
name={'close-checkpoint':'close_checkpoint.py','migrate':'migrate.py','prove':'prove.py'}[mode]
remote=t['remote']+'/native-v13/'+name
sha=hashlib.sha256((HERE/name).read_bytes()).hexdigest()
code=f'''from pathlib import Path
import hashlib,json,runpy
r=Path({t['remote']!r});p=Path({remote!r})
assert hashlib.sha256(p.read_bytes()).hexdigest()=={sha!r}
priority=json.loads((r/'run-v10/state-build-cpu-priority.json').read_text());ready=json.loads((Path(priority['state_root'])/'borrowed-training/restoration-ready.json').read_text());assert ready['ready_for_handoff'] and ready['automatic_resume'] is False
manifest=json.loads((r/'active-inputs-v13.json').read_text())
assert all(hashlib.sha256((r/n).read_bytes()).hexdigest()==h for n,h in manifest.items())
runpy.run_path(str(p),run_name='__main__')
'''
command='srun --jobid='+ops['config']['gpu_job']+' --overlap --exact -n1 -c2 --mem=16G --cpu-bind=none '+t['env']+t['python']+' -c '+shlex.quote(code)
p=subprocess.run(['ssh','-o','ConnectTimeout=15','chimera-login',command],text=True,capture_output=True)
record=dict(mode=mode,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)
(HERE/(mode+'-receipt.json')).write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record),flush=True);p.check_returncode()
