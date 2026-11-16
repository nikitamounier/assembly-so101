from pathlib import Path
import json,numpy as np,mujoco
from types import SimpleNamespace
from assembly_policy.so101.wowrobo_assembly_fit import coherent_step_alignment
from assembly_policy.so101.wowrobo_contact_backend import load_backend,set_surface_mode
from assembly_policy.so101.wowrobo_insertion_guide import build_guide
from scripts.validate_wowrobo_fit_profile import run_case
root=Path.cwd();common=root/'outputs/insertion-fix-20261107/final-common-v1';out=root/'outputs/insertion-fix-20261107/upper/canonical-common-v1/negative-wall';out.mkdir(exist_ok=True,parents=True);load_backend(common/'grids/manifest.json');set_surface_mode(1);m=mujoco.MjModel.from_xml_path(str(common/'scene.xml'));old=json.loads((root/'outputs/linear-insertion-aligned-20261107/upper/alignment.json').read_text());reg=json.loads((root/'outputs/insertion-fix-20261107/pin/consistent-v3/registration.json').read_text());a=coherent_step_alignment(m,old,'upper',pin_registration=reg);normal=np.asarray(old['feature_frame']['socket_frame_world'])[:,0];qa=int(m.joint('upper_arm_module_free').qposadr[0]);offset=.001*normal
for key in ['start_qpos','reference_qpos']:
 q=np.array(a[key]);q[qa:qa+3]+=offset;a[key]=q.tolist()
a['negative_wall_offset_m']=offset.tolist();(out/'alignment.json').write_text(json.dumps(a,indent=2)+'\n');build_guide(common/'scene.xml',out/'alignment.json',out/'guide-scene.xml','upper_arm_module');r=run_case(SimpleNamespace(step='upper',seconds=3.,release_seconds=.2),out,out,a,{'qpos':np.zeros(1),'qvel':np.zeros(1)},'legacy_mapped',20.);print('NEGATIVEblocked',r['final_drive_axial_mm']<25,r['physical_case_guard_failures'])
