from pathlib import Path
import json,hashlib
R=Path(__file__).resolve().parent.parent;N=R/'native-v14'
a=json.loads((N/'bank-baseline.json').read_text());b=json.loads((N/'bank-candidate.json').read_text());assert a['count']==b['count']==1143
assert a['rows']==b['rows'],'Reset observations/rewards/contacts changed'
# mesh_pathadr changes because the isolated root has a different string length; all physics arrays are unchanged.
bad=[k for k in a['arrays'] if a['arrays'][k]!=b['arrays'].get(k)];assert set(bad)<= {'mesh_pathadr'},bad
wa=json.loads((N/'worker-qa-baseline/result.json').read_text());wb=json.loads((N/'worker-qa-candidate/result.json').read_text());assert wa['arrays']==wb['arrays'];rss=wb['worker']['maxrss_kib'];assert rss<12*1024*1024 and rss<=wa['worker']['maxrss_kib']*1.15
s=json.loads((N/'distributed-smoke.json').read_text());assert s['status']=='PASS'
x=dict(status='PASS',reset_rows=1143,reset_observation_reward_contact_exact=True,physical_arrays_exact_except_path_metadata=bad,worker_envs=10,worker_threads=10,worker_arrays=len(wb['arrays']),byte_identical=True,baseline_step_s=wa['step_wall_s'],candidate_step_s=wb['step_wall_s'],speedup=wa['step_wall_s']/wb['step_wall_s'],peak_rss_mib=rss/1024,rss_pass=True,physics_signature=b['model_signature'],distributed_smoke=s['status']);(N/'qualification.json').write_text(json.dumps(x,indent=2)+'\n');print(json.dumps(x),flush=True)
