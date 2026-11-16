"""Native current-fit insertion, unforced gravity release and old-pin negative."""
from pathlib import Path
import argparse,hashlib,json,time,xml.etree.ElementTree as ET
import mujoco,numpy as np
from assembly_policy.so101.wowrobo_contact_backend import prepare_scene,load_backend,set_surface_mode
from assembly_policy.so101.wowrobo_insertion_guide import build_guide
from assembly_policy.so101.wowrobo_pin_registration import measure_pin_registration
from assembly_policy.so101.wowrobo_assembly_fit import completed_frames
from assembly_policy.so101.wowrobo_balanced_push import BalancedPush
from assembly_policy.so101.wowrobo_mating_profile import write_profile_scene
from scripts.diagnose_wowrobo_insertion import configure,reactions
ROOT=Path('/Users/nikitamounier/Documents/ChatGPT/robot assembly 2/robotics-assembly-3.0');OUT=Path(__file__).resolve().parent

def hashfile(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def nums(a):return' '.join(format(float(v),'.17g')for v in np.asarray(a).ravel())

def roi_valid(model,data):
 """Reject pin/case material interactions outside the fine field's declared ROI."""
 gid=model.geom('pin').id;mid=model.geom_dataid[gid];a=model.mesh_vertadr[mid];n=model.mesh_vertnum[mid];world=model.mesh_vert[a:a+n]@data.geom_xmat[gid].reshape(3,3).T+data.geom_xpos[gid]
 case=model.geom('servo_1_case').id;cm=model.geom_dataid[case];native=(world-data.geom_xpos[case])@data.geom_xmat[case].reshape(3,3);r=np.empty(9);mujoco.mju_quat2Mat(r,model.mesh_quat[cm]);local=native@r.reshape(3,3).T+model.mesh_pos[cm]
 field=json.loads((OUT/'fields/case-roi.json').read_text());bounds=np.array(field['bounds']);near=np.all((local>=bounds[0]-.0005)&(local<=bounds[1]+.0005),axis=1);selected=local[near]
 if not len(selected):return True
 low,high=np.array(field['valid_contact_roi_m']);return bool(np.all((selected>=low+.00005)&(selected<=high-.00005)))

def main(kind,resume_drive=False,full_case=False):
 out=OUT/kind;out.mkdir(parents=True,exist_ok=True);source=ROOT/'outputs/linear-insertion-20261107/source-snapshot'
 if kind=='fit':grid=OUT/'grids/manifest.json';mesh=OUT/'assets/pin.obj'
 elif kind=='length-negative':grid=OUT.parent/'fit-grids/manifest.json';mesh=ROOT/'assets_local/wowrobo/fit-pin-v2/pin.obj'
 else:grid=OUT.parent/'negative-grids/manifest.json';mesh=ROOT/'assets_local/wowrobo/components-v2/meshes/pin.obj'
 dim=json.loads((OUT/'assets/manifest.json').read_text())['pin_mm']
 tree=ET.parse(source/'scene.xml');tree.find("asset/mesh[@name='pin']").set('file',str(mesh));full=out/'source-scene.xml';tree.write(full,encoding='unicode');m=mujoco.MjModel.from_xml_path(str(full));d=mujoco.MjData(m);mujoco.mj_setState(m,d,np.load(source/'stages/13_pin_aligned_train.npz')['states'][0],mujoco.mjtState.mjSTATE_FULLPHYSICS);mujoco.mj_forward(m,d)
 coherent_qpos,fixture_report=completed_frames(m,d.qpos.copy(),'pin');d.qpos[:]=coherent_qpos;mujoco.mj_forward(m,d)
 registration=measure_pin_registration(m,d,dim,head_gap_m=.0001);up=np.array(registration['bore_axis_world']);axis=-up;rotation=np.array(registration['rotation_world']).reshape(3,3);qa=m.joint('assembly_pin_free').qposadr[0];root=np.array(registration['root_world_m']);d.qpos[qa:qa+3]=root+up*.032;mujoco.mju_mat2Quat(d.qpos[qa+3:qa+7],rotation.ravel());start=d.qpos.copy();reference=start.copy();reference[qa:qa+3]=root-up*.0001
 alignment={'start_qpos':start.tolist(),'reference_qpos':reference.tolist(),'axis_world':axis.tolist(),'authored_axial_travel_m':.0321,'mating_geom':'pin','mating_body':'assembly_pin','receiver_body':'shoulder_connector','registration':registration,'kind':kind,'coherent_prerequisites':fixture_report,'inertia':'Frozen .1kg explicit display mass and tensor retained; not calibrated polymer mass'};align=out/'alignment.json';align.write_text(json.dumps(alignment,indent=2)+'\n')
 backend=out/'sdf-scene.xml';info=prepare_scene(full,grid,backend);set_surface_mode(1);profile=out/'profile-scene.xml';write_profile_scene(backend,profile,'pin',profile='legacy_mapped');guided=out/'insertion-scene.xml';guide=build_guide(profile,align,guided,'assembly_pin');m=mujoco.MjModel.from_xml_path(str(guided));active=configure(m,'assembly_pin',['base_servo','base_side_cover','shoulder_connector'],True);m.dof_damping[:]=200;d=mujoco.MjData(m);mujoco.mj_forward(m,d);push=BalancedPush(m,d,active,axis,damping=0);initial=d.body('assembly_pin').xpos.copy();trace=[];peak=0;stop='duration elapsed';started=time.monotonic()
 if resume_drive:
  saved=np.load(out/'insertion-trace.npz');trace=np.column_stack([saved['time'],saved['qpos']]).tolist();last_velocity=saved['qvel'].copy()
  for row in trace:
   d.qpos[:]=row[1:];mujoco.mj_forward(m,d);peak=max(peak,max(0,-min((float(c.dist)for c in d.contact),default=0)))
   if not full_case and not roi_valid(m,d):raise RuntimeError('Saved native drive trace exits finefield ROI')
  d.time=trace[-1][0];d.qvel[:]=last_velocity;mujoco.mj_forward(m,d)
 else:
  for i in range(16000):
   force=1*min(1,float(d.time)/.5);push.apply(d,force);mujoco.mj_step(m,d);mujoco.mj_forward(m,d);depth=max(0,-min((float(c.dist)for c in d.contact),default=0));peak=max(peak,depth);trace.append([float(d.time),*d.qpos.tolist()])
   if not full_case and not roi_valid(m,d):stop='finefield ROI guard';break
   if not np.isfinite(d.qpos).all():stop='nonfinite';break
   if depth>.0005:stop='0.5mm native penetration';break
   if d.qpos[0]>.0326:stop='head stop overtravel';break
   if i%2000==0:print(kind,'t',d.time,'qmm',d.qpos[0]*1000,'contacts',d.ncon,flush=True)
 push.apply(d,1);mujoco.mj_forward(m,d);drive=float(d.qpos[0]);contact=reactions(m,d,active,axis);q=np.array(trace);np.savez_compressed(out/'insertion-trace.npz',time=q[:,0],qpos=q[:,1:],qvel=d.qvel)
 report={'kind':kind,'insertion_model':str(guided),'grid_manifest':str(grid),'source_pin_mesh':str(mesh),'source_pin_mesh_sha256':hashfile(mesh),'receiving_case_field_sha256':json.loads(grid.read_text())['meshes']['servo_1_case']['grid_sha256'],'registration':registration,'force_N':1,'native_implicit_damping_Ns_m':200,'gravity_during_insertion':[0,0,0],'drive_s':float(d.time),'final_axial_mm':drive*1000,'head_stop_reference_mm':32.1,'head_gap_mm':(0.0321-drive)*1000,'max_native_penetration_mm':peak*1000,'drive_stop':stop,'final_reaction':contact,'guide':guide,'full_qpos_time_trace':'insertion-trace.npz','wall_seconds':time.monotonic()-started,'training_eligible':False,'coherent_prerequisites':fixture_report,'snapshot_contract':'Registered cover plus feature-centered connector engineering target; not a settled connector assertion'}
 (out/'render-spec.json').write_text(json.dumps({'model':str(guided),'grid_manifest':str(grid),'trace':str(out/'insertion-trace.npz'),'time_key':'time','qpos_key':'qpos','title':f'{kind.upper()} PIN | native insertion','method':'Current source mesh, fine case SDF, balanced 1N axial push; guide explicit','result':f'Axial{drive*1000:.3f}mm; head gap{(.0321-drive)*1000:.3f}mm','limitation':'Fixture insertion; mass/inertias provisional. Release tested separately.','lookat':registration['head_stop_surface_world_m'],'distance':.36,'azimuth':125,'elevation':-25},indent=2)+'\n')
 (out/'receipt.json').write_text(json.dumps(report,indent=2)+'\n')
 if kind=='fit'and stop=='duration elapsed':
  # Change joint representation while preserving every world-space surface exactly.
  free_tree=ET.parse(guided);body=free_tree.find(".//body[@name='assembly_pin']");body.remove(body.find('joint'));position=d.body('assembly_pin').xpos.copy();quat=d.body('assembly_pin').xquat.copy();body.set('pos',nums(position));body.set('quat',nums(quat));ET.SubElement(body,'freejoint',name='released_pin_free');free_tree.find('option').set('gravity','0 0 -9.81');free=out/'release-scene.xml';free_tree.write(free,encoding='unicode');fm=mujoco.MjModel.from_xml_path(str(free));fa=configure(fm,'assembly_pin',['base_servo','base_side_cover','shoulder_connector'],True);fd=mujoco.MjData(fm);mujoco.mj_forward(fm,fd);geom_errors=[]
  for gid in range(m.ngeom):
   name=m.geom(gid).name;new=fm.geom(name).id if name else gid;geom_errors.append(float(np.max(np.abs(d.geom_xpos[gid]-fd.geom_xpos[new]))))
  assert max(geom_errors)<1e-12
  before=fd.body('assembly_pin').xpos.copy();rtrace=[[0,*fd.qpos.tolist()]];rpeak=0;release_stop='duration elapsed'
  for i in range(4000):
   fd.qfrc_applied[:]=0;mujoco.mj_step(fm,fd);mujoco.mj_forward(fm,fd);depth=max(0,-min((float(c.dist)for c in fd.contact),default=0));rpeak=max(rpeak,depth);rtrace.append([float(fd.time),*fd.qpos.tolist()])
   if not full_case and not roi_valid(fm,fd):release_stop='finefield ROI guard';break
   if depth>.0005:release_stop='0.5mm native penetration';break
   if np.linalg.norm(fd.body('assembly_pin').xpos-before)>.01:release_stop='10mm release drift';break
  rt=np.array(rtrace);np.savez_compressed(out/'release-trace.npz',time=rt[:,0],qpos=rt[:,1:],qvel=fd.qvel);report['release']={'model':str(free),'duration_s':float(fd.time),'external_forces':'zero','gravity':[0,0,-9.81],'pin_free_DOFs':int(fm.nv),'geometry_transfer_max_error_m':max(geom_errors),'root_drift_mm':float(np.linalg.norm(fd.body('assembly_pin').xpos-before)*1000),'max_native_penetration_mm':rpeak*1000,'stop':release_stop,'mass_kg':float(fm.body('assembly_pin').mass[0]),'trace':'release-trace.npz'}
 (out/'receipt.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--kind',choices=['fit','negative','length-negative'],required=True);p.add_argument('--resume-drive',action='store_true');p.add_argument('--full-case',action='store_true');a=p.parse_args();main(a.kind,a.resume_drive,a.full_case)
