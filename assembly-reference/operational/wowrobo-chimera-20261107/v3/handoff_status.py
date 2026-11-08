from pathlib import Path
import json,subprocess,shlex
H=Path(__file__).parent;old=json.loads((H.parent/'target.json').read_text());r=old['remote'];cfg=json.loads((H/'handoff-target.json').read_text())
code=f'''from pathlib import Path
import json,os,time
root=Path({r!r});run=root/'run-v2';row=json.loads((run/'handoff-request.json').read_text());p=Path('/proc')/str(row['pid']);alive=False
if p.exists():
 try:alive=(p/'stat').read_text().split()[21]==row['start'] and (p/'stat').read_text().split()[2]!='Z'
 except FileNotFoundError:pass
lines=(run/'log.jsonl').read_text().splitlines();latest=json.loads(lines[-1]) if lines else None
if not alive:
 proof=dict(learner_closed=True,pid=row['pid'],start=row['start'],time=time.time(),latest=latest,checkpoint_bytes=(run/'checkpoint.pt').stat().st_size);(run/'handoff-closed.json').write_text(json.dumps(proof,indent=2)+'\\n')
print(json.dumps(dict(learner_alive=alive,stop_age_s=time.time()-row['time'],latest=latest,checkpoint_bytes=(run/'checkpoint.pt').stat().st_size)))
'''
subprocess.run(['ssh','-o','ConnectTimeout=15','chimera-login','srun --jobid='+cfg['gpu_job']+' --overlap --exact --ntasks=1 --cpus-per-task=1 --cpu-bind=none '+old['python']+' -c '+shlex.quote(code)],check=True)
code=f"from pathlib import Path; import json; r=Path({r!r}); p=r/'autoscale-v2/submissions.json'; print(p.read_text() if p.exists() else 'NO_ADDITIONAL_JOBS')"
subprocess.run(['ssh','-o','ConnectTimeout=15','chimera-login','srun --jobid=3077705 --overlap --exact --ntasks=1 --cpus-per-task=1 '+old['python']+' -c '+shlex.quote(code)],check=True)
