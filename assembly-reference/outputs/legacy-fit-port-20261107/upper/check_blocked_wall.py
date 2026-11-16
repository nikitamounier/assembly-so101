"""Unchanged socket-wall negative control, with the same shared validator."""
from pathlib import Path
import argparse,json
import numpy as np
from assembly_policy.so101.wowrobo_contact_backend import load_backend
from assembly_policy.so101.wowrobo_insertion_guide import build_guide
from scripts.validate_wowrobo_fit_profile import run_case
root=Path(__file__).resolve().parents[3];out=Path(__file__).resolve().parent;load_backend(root/'outputs/linear-insertion-updated-20261107/grids/manifest.json');alignment=json.load(open(root/'outputs/linear-insertion-aligned-20261107/upper/alignment.json'));socket=np.array(alignment['feature_frame']['socket_frame_world']);axis=socket[:,1]
negative=dict(alignment,start_qpos=alignment['reference_qpos'],axis_world=axis.tolist(),authored_axial_travel_m=.005,qualification='Negativecontrol: attempt5mmlateralpenetrationofactualsocketwall;notinsertionsuccess.')
src=out/'blockedwall-source';src.mkdir(exist_ok=True);config=src/'alignment.json';config.write_text(json.dumps(negative,indent=2)+'\n');build_guide(root/'outputs/linear-insertion-updated-20261107/scene.xml',config,src/'guide-scene.xml','upper_arm_module')
output=out/'blockedwall';output.mkdir(exist_ok=True);args=argparse.Namespace(step='upper',seconds=.25,release_seconds=.05);saved={'qpos':np.array([0.]),'qvel':np.array([0.])}
results=[run_case(args,src,output,negative,saved,p,20.) for p in ['baseline','legacy_mapped']]
(output/'blockedwall-summary.json').write_text(json.dumps({'axis_world':axis.tolist(),'attempted_travel_mm':5.,'drive_N':20.,'cases':results,'scope':'Sameactualcase/socketsourcegeometry,currentseatedreference,rigidlateralguide;noaddedpatches. Doesnotvalidatefreebodyrelease.'},indent=2)+'\n')
