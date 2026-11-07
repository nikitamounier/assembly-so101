from pathlib import Path
from argparse import Namespace
import json,mujoco,numpy as np
from assembly_policy.so101.wowrobo_contact_backend import load_backend,set_surface_mode
from assembly_policy.so101.wowrobo_insertion_guide import build_guide
from assembly_policy.so101.wowrobo_boundaries import NativeBoundaryAudit
from scripts.validate_wowrobo_fit_profile import run_case
ROOT=Path.cwd();out=ROOT/'outputs/legacy-fit-port-20261107/final/negative-wall';source=out/'source';source.mkdir(parents=True,exist_ok=True);load_backend(ROOT/'outputs/linear-insertion-updated-20261107/grids/manifest.json');set_surface_mode(1);alignment=json.loads((out.parent/'negative-wall-alignment.json').read_text());(source/'alignment.json').write_text(json.dumps(alignment,indent=2)+'\n');build_guide(ROOT/'outputs/linear-insertion-updated-20261107/scene.xml',source/'alignment.json',source/'guide-scene.xml','forearm_module');args=Namespace(step='final',seconds=2.5,release_seconds=.5);saved={'qpos':np.array([0.]),'qvel':np.array([0.])};results=[]
# Independent exactsourceaudit of eachnegativecase maximumtravel andrelaxedendpoint.
m=mujoco.MjModel.from_xml_path(str(ROOT/'outputs/linear-insertion-20261107/source-snapshot/scene.xml'));d=mujoco.MjData(m);audit=NativeBoundaryAudit(m,max_samples=4096);active=[g for g in audit.targets if audit.info[g]['piece']=='forearm_module'];other=[g for g in audit.targets if audit.info[g]['piece']!='forearm_module'];qa=int(m.joint('forearm_module_free').qposadr[0]);qstart=np.array(alignment['start_qpos']);axis=np.array(alignment['axis_world'])
for profile in ['baseline','legacy_mapped']:
 r=run_case(args,source,out,alignment,saved,profile,20.);checks=[]
 for label,mm in [('maximum',r['all_substep_maximum_axial_mm']),('relaxed',r['final_relaxed_axial_mm'])]:
  d.qpos[:]=qstart;d.qpos[qa:qa+3]+=mm*.001*axis;mujoco.mj_forward(m,d);bounds=audit._bounds(d,audit.info);pairs=audit._pairs(d,[(a,b) for a in active for b in other],bounds);peak=max([x['penetration_m'] for x in pairs],default=0);checks.append({'phase':label,'axial_mm':mm,'max_source_penetration_mm':1000*peak,'below_0p5mm':peak<=.0005})
 result={'case':r['case'],'drive_mm':r['final_drive_axial_mm'],'relaxed_mm':r['final_relaxed_axial_mm'],'reference_mm':25,'native_guard_failures':r['physical_case_guard_failures'],'source_checks':checks,'negative_control_pass':r['all_substep_maximum_axial_mm']<25 and all(x['below_0p5mm'] for x in checks)};results.append(result);print('NEGATIVEINDEPENDENT',json.dumps(result),flush=True)
(out/'independent-negative-wall-review.json').write_text(json.dumps({'sourceproof':'../negative-wall-source-proof.json','fixture':'Clearapproach;reference+1mmreceiverwallnormaloffset has1.122mmsourceinterference andmustnotseat','cases':results},indent=2)+'\n')
