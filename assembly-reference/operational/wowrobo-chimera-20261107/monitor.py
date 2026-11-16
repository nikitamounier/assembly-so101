"""Compact read-only health for the frozen diagnostic run."""
from pathlib import Path
import hashlib,json,math,time,os
ROOT=Path(__file__).resolve().parents[3];RUN=ROOT/'run-v2'
manifest=json.loads((ROOT/'active-inputs.json').read_text())
bad=[f for f,h in manifest.items() if hashlib.sha256((ROOT/f).read_bytes()).hexdigest()!=h]
c=json.loads((RUN/'controller.json').read_text());rows=[]
p=RUN/'log.jsonl'
if p.exists():
 for line in p.read_text().splitlines()[-5:]:rows.append(json.loads(line))
r=rows[-1] if rows else None
finite=all(math.isfinite(float(v[k])) for v in rows for k in ['pi','vf','ent','kl','lr','std','fps','safety','success'])
ck=RUN/'checkpoint.pt'
state='keep_monitor'
if bad or not finite:state='ACTION source_or_nonfinite_failure'
elif r and r['it']>=10000 and ck.exists() and ck.stat().st_size>0:state='complete'
elif (time.time()-p.stat().st_mtime if p.exists() else time.time()-(RUN/'command.json').stat().st_mtime)>1200:state='ACTION no_progress_20min'
print(json.dumps(dict(action=state,source_valid=not bad,changed=bad,finite=finite,run=str(RUN),status='DIAGNOSTIC_UNQUALIFIED_RESETS',latest=r,progress_age_s=time.time()-p.stat().st_mtime if p.exists() else None,checkpoint_bytes=ck.stat().st_size if ck.exists() else 0,controller_age_s=time.time()-c['time'],running_steps=sum(v.get('launcher_pid') is not None and v.get('exit') is None for v in c['workers'].values()),scheduler_pending=sum(v.get('scheduler',[None])[0]=='PENDING' for v in c['workers'].values()))))
