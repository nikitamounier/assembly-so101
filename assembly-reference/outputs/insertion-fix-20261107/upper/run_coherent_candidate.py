from pathlib import Path
import argparse,json
import mujoco,numpy as np
from assembly_policy.so101.wowrobo_contact_backend import load_backend
from assembly_policy.so101.wowrobo_assembly_fit import coherent_step_alignment
from assembly_policy.so101.wowrobo_insertion_guide import build_guide
from scripts.validate_wowrobo_fit_profile import run_case
ROOT=Path(__file__).resolve().parents[3];candidate=ROOT/'outputs/insertion-fix-20261107/combined-v1';out=Path(__file__).resolve().parent/'coherent-combined-v1';out.mkdir(exist_ok=True);load_backend(candidate/'grids/manifest.json');m=mujoco.MjModel.from_xml_path(str(candidate/'scene.xml'));raw=json.load(open(ROOT/'outputs/linear-insertion-aligned-20261107/upper/alignment.json'));reg=json.load(open(ROOT/'outputs/insertion-fix-20261107/pin/consistent-v3/registration.json'));a=coherent_step_alignment(m,raw,'upper',pin_registration=reg);(out/'alignment.json').write_text(json.dumps(a,indent=2)+'\n');build_guide(candidate/'scene.xml',out/'alignment.json',out/'guide-scene.xml','upper_arm_module');args=argparse.Namespace(step='upper',seconds=2.5,release_seconds=.5);r=run_case(args,out,out,a,{'qpos':np.array([0.]),'qvel':np.array([0.])},'legacy_mapped',20.);print(json.dumps({k:v for k,v in r.items() if k in ['final_drive_axial_mm','drive_stop','final_relaxed_axial_mm','all_substep_peak_native_penetration_mm','physical_case_guard_failures']},indent=2))
