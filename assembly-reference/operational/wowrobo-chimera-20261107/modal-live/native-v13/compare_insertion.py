from pathlib import Path
import json,numpy as np
r=Path(__file__).resolve().parent;b=json.loads((r/'insertion-baseline.json').read_text());c=json.loads((r/'insertion-candidate.json').read_text());assert b['commands']==c['commands'] and b['stages']==c['stages'];rows=[];failures=[]
for i,stage in enumerate(b['stages']):
 common=min(b['first_guard'][i],c['first_guard'][i],6);delta=float(np.linalg.norm(np.array(b['poses'])[:common,i]-np.array(c['poses'])[:common,i],axis=1).max()) if common else 0.;newguard=b['first_guard'][i]==999 and c['first_guard'][i]!=999
 row=dict(stage=stage,baseline_guard=b['first_guard'][i],candidate_guard=c['first_guard'][i],max_response_delta_m=delta,baseline_initial_error_m=b['errors_m'][0][i],baseline_final_error_m=b['errors_m'][-1][i],candidate_final_error_m=c['errors_m'][-1][i]);rows.append(row)
 if newguard or delta>.0005:failures.append(row)
x=dict(status='PASS' if not failures else 'FAIL',scope=b['scope'],rows=rows,failures=failures);(r/'insertion-dynamics.json').write_text(json.dumps(x,indent=2)+'\n');print(json.dumps(x),flush=True);assert not failures,failures
