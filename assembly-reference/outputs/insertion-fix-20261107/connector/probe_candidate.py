"""Explicit candidate-only force drive plus free-root gravity release."""
from pathlib import Path
import argparse,json,numpy as np,mujoco,xml.etree.ElementTree as ET
from types import SimpleNamespace
from assembly_policy.so101.wowrobo_contact_backend import load_backend,set_surface_mode
from assembly_policy.so101.wowrobo_insertion_guide import build_guide
from scripts.validate_wowrobo_fit_profile import run_case
from scripts.diagnose_wowrobo_insertion import configure
ROOT=Path(__file__).resolve().parents[3]
def run(args):
 args.output.mkdir(exist_ok=True,parents=True);load_backend(args.grids);set_surface_mode(1)
 alignment_path=ROOT/'outputs/linear-insertion-aligned-20261107/connector/alignment.json';alignment=json.loads(alignment_path.read_text());from assembly_policy.so101.wowrobo_assembly_fit import coherent_step_alignment;raw_model=mujoco.MjModel.from_xml_path(str(args.scene));alignment=coherent_step_alignment(raw_model,alignment,'connector');extra=args.extra_travel_mm/1000;alignment['authored_axial_travel_m']+=extra;qref=np.array(alignment['reference_qpos']);qref[20:23]+=extra*np.array(alignment['axis_world']);alignment['reference_qpos']=qref.tolist();alignment['diagnostic_extra_travel_mm']=args.extra_travel_mm;alignment_path=args.output/'coherent-alignment.json';alignment_path.write_text(json.dumps(alignment,indent=2)+'\n');build_guide(args.scene,alignment_path,args.output/'guide-scene.xml','shoulder_connector')
 a=SimpleNamespace(step='connector',seconds=4.,release_seconds=.5);folder=args.output/'drive';folder.mkdir(exist_ok=True);case=run_case(a,args.output,folder,alignment,{'qpos':np.array([0.]),'qvel':np.array([0.])},'legacy_mapped',args.force)
 label=f'legacy_mapped_{args.force:g}N';original=mujoco.MjModel.from_xml_path(str(folder/(label+'.xml')));od=mujoco.MjData(original);state=np.load(folder/(label+'-drive-state.npz'));od.qpos[:]=state['qpos'];od.qvel[:]=state['qvel'];mujoco.mj_forward(original,od)
 tree=ET.parse(folder/(label+'.xml'));root=tree.getroot();body=root.find(".//body[@name='shoulder_connector']");body.set('pos',' '.join(format(float(x),'.17g') for x in od.body('shoulder_connector').xpos));body.set('quat',' '.join(format(float(x),'.17g') for x in od.body('shoulder_connector').xquat))
 for child in list(body):
  if child.tag in ['joint','freejoint']:body.remove(child)
 ET.SubElement(body,'freejoint',name='candidate_release_free');root.find('option').set('gravity','0 0 -9.81');release=args.output/'free-release';release.mkdir(exist_ok=True);scene=release/'scene.xml';tree.write(scene,encoding='unicode');m=mujoco.MjModel.from_xml_path(str(scene));configure(m,'shoulder_connector',['base_servo','base_side_cover','assembly_pin','upper_arm_module','forearm_module'],True)
 for g in range(m.ngeom):
  if m.geom_type[g]==mujoco.mjtGeom.mjGEOM_PLANE:
   m.geom_contype[g]=2;m.geom_conaffinity[g]=1;b=int(m.geom_bodyid[g]);m.body_contype[b]|=2;m.body_conaffinity[b]|=1
 d=mujoco.MjData(m);d.qvel[:3]=np.array(alignment['axis_world'])*float(state['qvel'][0]);mujoco.mj_forward(m,d);initial=d.body('shoulder_connector').xpos.copy();registration=max(float(np.max(np.abs(d.geom(m.geom(g).name).xpos-od.geom_xpos[g]))) for g in range(m.ngeom) if m.geom(g).name);assert registration<1e-12
 trace=[];peak=0.;reason='duration elapsed'
 for i in range(2000):
  mujoco.mj_step(m,d);mujoco.mj_forward(m,d);depth=max(0.,-min((float(c.dist) for c in d.contact),default=0));peak=max(peak,depth)
  if i%10==0:trace.append(np.r_[d.time,depth,d.qpos,d.qvel])
  if depth>.0005 or not np.isfinite(d.qpos).all():reason='native0.5mm/nonfinite guard';break
 od.qpos[0]=alignment['authored_axial_travel_m'];mujoco.mj_forward(original,od);error=float(np.linalg.norm(d.geom(alignment['mating_geom']).xpos-od.geom(alignment['mating_geom']).xpos));velocity=float(np.linalg.norm(d.qvel));drift=float(np.linalg.norm(d.body('shoulder_connector').xpos-initial));r={'release_seconds':d.time,'stop':reason,'root_drift_mm':drift*1000,'mating_reference_error_mm':error*1000,'final_velocity_norm':velocity,'peak_native_penetration_mm':peak*1000,'endpoint_registration_error_m':registration,'warnings':int(sum(x.number for x in d.warning)),'drive_reference_reached':case['final_drive_axial_mm']>=alignment['authored_axial_travel_m']*1000,'retained_reference_within_1mm':error<.001,'candidate_geometry_only':True,'qualification':'Receiver-supported free incoming root with gravity; receiver and helper fixtures remain; internal shape locked. Source/wall guards separate.'};np.savez_compressed(release/'trace.npz',trace=np.asarray(trace),qpos=d.qpos,qvel=d.qvel);(release/'release.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2));(args.output/'summary.json').write_text(json.dumps({'drive':case,'release':r},indent=2)+'\n')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--scene',type=Path,required=True);p.add_argument('--grids',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--force',type=float,default=20);p.add_argument('--extra-travel-mm',type=float,default=0.);run(p.parse_args())
