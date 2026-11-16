import argparse,json
from pathlib import Path
import mujoco,numpy as np
from assembly_policy.so101.wowrobo_boundaries import NativeBoundaryAudit
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).parent
p=argparse.ArgumentParser();p.add_argument('--name',required=True);a=p.parse_args();m=mujoco.MjModel.from_xml_path(str(ROOT/'assets_local/wowrobo/photo-resets-repaired-cover-support-v2/scene.xml'));audit=NativeBoundaryAudit(m);result={}
with np.load(OUT/(a.name+'.npz'))as z:
 for phase in ['initial_qpos','qpos']:
  d=mujoco.MjData(m);d.qpos[:]=z[phase];mujoco.mj_forward(m,d);r=audit.audit(d);result[phase]=r;print(phase,r['candidate_rejected'],r['candidate_rejection_reasons'],r['max_sampled_penetration_m'],flush=True)
(OUT/(a.name+'-boundaries.json')).write_text(json.dumps(result,indent=2)+'\n')
