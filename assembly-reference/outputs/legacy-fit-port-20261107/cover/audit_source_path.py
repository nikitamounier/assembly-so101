"""Independent finite source-triangle audit of actual visited cover slider ranges."""
from pathlib import Path
import hashlib,json
import mujoco,numpy as np
from assembly_policy.so101.wowrobo_boundaries import NativeBoundaryAudit
ROOT=Path.cwd();OUT=ROOT/'outputs/legacy-fit-port-20261107/cover';SOURCE=ROOT/'outputs/linear-insertion-20261107/source-snapshot/scene.xml';ALIGN=ROOT/'outputs/linear-insertion-aligned-20261107/cover/alignment.json'
a=json.loads(ALIGN.read_text());matrix=json.loads((OUT/'validation-matrix.json').read_text());m=mujoco.MjModel.from_xml_path(str(SOURCE));d=mujoco.MjData(m);audit=NativeBoundaryAudit(m,max_samples=1024);active=[g for g in audit.targets if audit.info[g]['piece']=='base_side_cover'];receiver=[g for g in audit.targets if audit.info[g]['piece']=='base_servo'];bounds={};ranges={};allq=[]
for case in matrix['cases']:
 label=case['case'];trace=np.load(OUT/f'{label}-trace.npz')['trace'];lo=float(trace[:,1].min());hi=float(trace[:,1].max());ranges[label]=(lo,hi);steps=max(1,int(np.ceil((hi-lo)/.0005)));sample=np.linspace(lo,hi,steps+1);allq.extend(sample.tolist());allq.extend([float(trace[0,1]),float(trace[-1,1])])
allq=sorted(set(round(q,12)for q in allq));records=[]
for i,q in enumerate(allq):
 d.qpos[:]=a['start_qpos'];qa=a['qpos_address'];d.qpos[qa:qa+3]+=q*np.array(a['axis_world']);mujoco.mj_forward(m,d);bounds=audit._bounds(d,active+receiver);pairs=audit._pairs(d,[(x,y)for x in active for y in receiver],bounds);overlap=max([r['penetration_m']for r in pairs],default=0);records.append({'axial_m':q,'source_max_sampled_overlap_m':overlap,'pairs':pairs})
 if i%10==0:print(i,len(allq),q,overlap,flush=True)
case_summary=[]
for case in matrix['cases']:
 label=case['case'];lo,hi=ranges[label];hits=[r for r in records if lo-1e-12<=r['axial_m']<=hi+1e-12];worst=max(hits,key=lambda r:r['source_max_sampled_overlap_m']);case_summary.append({'case':label,'visited_axial_min_m':lo,'visited_axial_max_m':hi,'audited_poses':len(hits),'source_sampled_max_overlap_m':worst['source_max_sampled_overlap_m'],'worst_axial_m':worst['axial_m'],'sampled_material_gate_pass':worst['source_max_sampled_overlap_m']<=.0005})
report={'source_scene_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),'alignment_sha256':hashlib.sha256(ALIGN.read_bytes()).hexdigest(),'scope':'Actual visited range of every1Dtrace,includingrelaxationandextrema, fixedactualfeaturealignedshape. <=.5mm axialspacing;1024sourcevertices+centroidssampledperdirectionafterAABB. NoSDF/convexhulldistances. Finitegeometricaudit,notcontinuouscollisionorfree-graspqualification.','case_summary':case_summary,'records':records};(OUT/'source-path-audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(case_summary,indent=2),flush=True)
