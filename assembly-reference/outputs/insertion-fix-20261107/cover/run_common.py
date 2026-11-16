from pathlib import Path
from types import SimpleNamespace
import argparse,json,numpy as np
from assembly_policy.so101.wowrobo_cover_fit import write_alignment
from assembly_policy.so101.wowrobo_contact_backend import load_backend,set_surface_mode
from assembly_policy.so101.wowrobo_insertion_guide import build_guide
from scripts.validate_wowrobo_fit_profile import run_case
p=argparse.ArgumentParser();p.add_argument('--negative',action='store_true');args=p.parse_args();root=Path.cwd();common=root/'outputs/insertion-fix-20261107/final-common-v1';out=root/'outputs/insertion-fix-20261107/cover/canonical-common-v1';out=out/'negative-wall'if args.negative else out;out.mkdir(parents=True,exist_ok=True)
a=write_alignment(root/'outputs/linear-insertion-aligned-20261107/cover/alignment.json',common/'source-scene.xml',out/'alignment.json')
if args.negative:
 for key in ['start_qpos','reference_qpos']:a[key][14]+=.001
 a['negative_wall_offset_m']=[0,.001,0];(out/'alignment.json').write_text(json.dumps(a,indent=2)+'\n')
load_backend(common/'grids/manifest.json');set_surface_mode(1);build_guide(common/'scene.xml',out/'alignment.json',out/'guide-scene.xml','base_side_cover');r=run_case(SimpleNamespace(step='cover',seconds=8.,release_seconds=.5),out,out,a,{'qpos':np.zeros(1),'qvel':np.zeros(1)},'legacy_mapped',20.);(out/'receipt.json').write_text(json.dumps({'candidate':str(common),'negative':args.negative,'drive':r},indent=2)+'\n')
if not args.negative:
 s={'model':str(out/'legacy_mapped_20N.xml'),'grid_manifest':str(common/'grids/manifest.json'),'trace':str(out/'legacy_mapped_20N-trace.npz'),'trace_key':'trace','qpos_column':1,'title':'COVER | corrected fit','method':'Small pose correction | original cover CAD | unchanged contact law | 20 N','result':f"{r['final_drive_axial_mm']:.2f} / 55 mm insertion; drive off included",'limitation':'Fixed receiver; virtual axial guide; helper arms hidden','lookat':[.005,0,.08],'distance':.32,'azimuth':125,'elevation':-25,'hide_helpers':True,'force':{'piece':'base_side_cover','axis_world':a['axis_world'],'maximum_N':20,'ramp_s':0,'off_s':r['drive_simulated_seconds']},'output':str(out/'insertion.mp4')};(out/'render-spec.json').write_text(json.dumps(s,indent=2)+'\n')
