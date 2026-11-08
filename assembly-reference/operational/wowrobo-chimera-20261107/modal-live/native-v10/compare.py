from pathlib import Path
import json,numpy as np
REL=Path(__file__).resolve().parent;b=json.loads((REL/'baseline-qualification.json').read_text());c=json.loads((REL/'candidate-qualification.json').read_text());assert b['arrays']==c['arrays'];assert b['initial']==c['initial'];assert b['insertion']==c['insertion'];bo=b['options'].copy();co=c['options'].copy();assert bo.pop('timestep')==.00025 and co.pop('timestep')==.0005;assert bo==co
failures=[];rows=[]
for group in ('held','near'):
 bg,cg=b['dynamic'][group],c['dynamic'][group];assert bg['stages']==cg['stages']
 for i,stage in enumerate(bg['stages']):
  common=min(bg['first_guard'][i],cg['first_guard'][i],bg['first_success'][i],cg['first_success'][i],12);delta=float(np.max(np.linalg.norm(np.array(bg['poses'])[:common,i]-np.array(cg['poses'])[:common,i],axis=1))) if common else 0.0
  new_guard=bg['first_guard'][i]==999 and cg['first_guard'][i]!=999;lost_success=bg['first_success'][i]!=999 and cg['first_success'][i]==999
  lost=sorted(set(map(tuple,bg['final_grasp'][i]))-set(map(tuple,cg['final_grasp'][i]))) if bg['first_guard'][i]==cg['first_guard'][i]==bg['first_success'][i]==cg['first_success'][i]==999 else []
  row=dict(group=group,stage=stage,baseline_guard=bg['first_guard'][i],candidate_guard=cg['first_guard'][i],max_position_delta_m=delta,lost_grasp_pairs=lost,lost_success=lost_success)
  rows.append(row)
  if new_guard or lost_success or delta>.0005 or lost:failures.append(row)
x=dict(status='PASS' if not failures else 'FAIL',old_signature=b['identity'],new_signature=c['identity'],model_arrays_exact=len(b['arrays']),initial_obs_reward_contacts_exact=len(b['initial']),insertion_controls_exact=len(b['insertion']),changed_model_option_only='timestep .00025 to .0005',old_substeps=160,new_substeps=80,control_dt=.04,paired_dynamic_controls=240,scope='10original held/near starts, paired0.48s command/retention dynamics,125fixed-pose insertion controls; boundedqualificationnotfullassemblycertification',rows=rows,failures=failures)
(REL/'qualification.json').write_text(json.dumps(x,indent=2)+'\n');print(json.dumps(x),flush=True);assert not failures,failures
