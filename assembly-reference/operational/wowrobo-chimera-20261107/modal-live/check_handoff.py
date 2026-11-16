from pathlib import Path
import json,subprocess,shlex
HERE=Path(__file__).resolve().parent;old=json.loads((HERE.parent/'v3/target.json').read_text());cfg=json.loads((HERE/'handoff-cluster.json').read_text())['config'];R=old['remote']
code=f'''from pathlib import Path
import json,time
run=Path({R+'/run-v3'!r});row=json.loads((run/'handoff-request-v4.json').read_text());p=Path('/proc')/str(row['pid']);alive=False
if p.exists():
 try:alive=(p/'stat').read_text().split()[21]==row['start'] and (p/'stat').read_text().split()[2]!='Z'
 except FileNotFoundError:pass
lines=(run/'log.jsonl').read_text().splitlines();latest=json.loads(lines[-1]) if lines else None
proof=dict(learner_alive=alive,stop_age_s=time.time()-row['time'],latest=latest,checkpoint_bytes=(run/'checkpoint.pt').stat().st_size,pid=row['pid'],start=row['start'],time=time.time())
if not alive:(run/'handoff-closed.json').write_text(json.dumps(proof,indent=2)+'\\n')
print(json.dumps(proof))
'''
p=subprocess.run(['ssh','-o','ConnectTimeout=15','chimera-login','srun --jobid='+cfg['gpu_job']+' --overlap --exact --ntasks=1 --cpus-per-task=1 --cpu-bind=none '+old['python']+' -c '+shlex.quote(code)],capture_output=True,text=True);(HERE/'handoff-status.json').write_text(p.stdout);print(p.stdout,p.stderr);p.check_returncode()
