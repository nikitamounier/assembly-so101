from pathlib import Path
import json
REL=Path(__file__).resolve().parent;b=json.loads((REL/'worker-qa2-baseline/result.json').read_text());c=json.loads((REL/'worker-qa2-candidate/result.json').read_text());different=[k for k in b['arrays'] if b['arrays'][k]!=c['arrays'].get(k)];assert b['arrays'].keys()==c['arrays'].keys()
# Retained10CPUjobs request2GiB perCPU=20GiB; leave2GiB extra headroom.
limit_kib=18*1024*1024;rss_ok=c['worker']['maxrss_kib']<limit_kib
x=dict(status='PASS' if not different and rss_ok else 'FAIL',arrays_compared=len(b['arrays']),byte_identical=not different,different=different,baseline_peak_rss_mib=b['worker']['maxrss_kib']/1024,candidate_peak_rss_mib=c['worker']['maxrss_kib']/1024,worker_envs=10,worker_threads=10,actual_controls=3,friction_variation=True,rss_gate_mib=limit_kib/1024,rss_pass=rss_ok,old_physics_signature=b['model_signature'],new_physics_signature=c['model_signature'],baseline_step_wall_s=b['step_wall_s'],candidate_step_wall_s=c['step_wall_s'],step_speedup=b['step_wall_s']/c['step_wall_s'],scope='Real production10thread/10env RPCworker, deterministic states/actions, physicalpayloadparity andpeakRSS; no fleet-speedclaim')
(REL/'production-worker-proof.json').write_text(json.dumps(x,indent=2)+'\n');print(json.dumps(x),flush=True);assert not different and rss_ok,x
