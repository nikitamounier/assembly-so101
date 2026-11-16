from pathlib import Path
exec(Path('outputs/linear-insertion-aligned-20261107/final/solve_alignment.py').read_text().split('results=[]')[0])
from scipy.spatial.transform import Rotation
import trimesh
assets=Path('assets_local/wowrobo/components-v2/meshes');upper=trimesh.load_mesh(assets/'upper_arm_module_printed_0.obj',process=False)
c=upper.triangles_center;n=upper.face_normals;ar=upper.area_faces
mask=(abs(n[:,0])>.999)&(c[:,0]>.018)&(c[:,0]<.021)&(c[:,1]>.061)&(c[:,1]<.086)&(c[:,2]>.114)&(c[:,2]<.131)
px=np.sum(n[mask]*np.sign(n[mask,0])[:,None]*ar[mask,None],axis=0);px/=np.linalg.norm(px)
mask=(n[:,2]>.999)&(c[:,1]>.06)&(c[:,1]<.09)&(c[:,2]>.129)&(c[:,2]<.134)
pz=np.sum(n[mask]*ar[mask,None],axis=0);pz-=px*(pz@px);pz/=np.linalg.norm(pz)
cx=-local_axis;Q=Rotation.align_vectors(px[None],cx[None])[0].as_matrix();axis=-rr@pz
state(0);gid=m.geom('servo_3_case').id;cg=rr.T@(d.geom_xpos[gid]-rp);audit=NativeBoundaryAudit(m,max_samples=4096);active=[g for g in audit.targets if audit.info[g]['piece']=='forearm_module'];other=[g for g in audit.targets if audit.info[g]['piece']!='forearm_module']
def state_new(dx,stand=0):
 d.qpos[:]=qoriginal;r=rr@Q;p=rp+rr@(cg+dx*px)-r@cg-stand*axis;q=Rotation.from_matrix(r).as_quat();d.qpos[qa:qa+3]=p;d.qpos[qa+3:qa+7]=q[[3,0,1,2]];d.qpos[hinge]=0;mujoco.mj_forward(m,d);return d.qpos.copy()
rows=[]
for dx in np.arange(-.0020,-.002701,-.00005):
 state_new(dx);bounds=audit._bounds(d,audit.info);pairs=audit._pairs(d,[(gid,m.geom('upper_arm_module_printed_0').id)],bounds);maxm=max([x['penetration_m'] for x in pairs],default=0);row={'dx':float(dx),'max_m':maxm};rows.append(row);print(row,flush=True)
best=min(rows,key=lambda x:x['max_m']);dx=best['dx'];qend=state_new(dx);qstart=state_new(dx,.025);audit=NativeBoundaryAudit(m,max_samples=1024);sweep=[]
for f in np.linspace(0,1,26):
 state_new(dx,.025*(1-f));res=check();row={'fraction':float(f),'remaining_standoff_m':float(.025*(1-f)),'max_sampled_penetration_m':res['max_m'],'pairs':res['pairs']};sweep.append(row);print('sweep',row,flush=True)
r=json.loads((out/'alignment.json').read_text());r.update(initial_qpos=qstart.tolist(),start_qpos=qstart.tolist(),qpos_initial=qstart.tolist(),end_qpos=qend.tolist(),reference_qpos=qend.tolist(),qpos_end=qend.tolist(),axis_world=axis.tolist(),swept_sample_overlap=sweep,max_swept_sample_overlap_m=max(x['max_sampled_penetration_m'] for x in sweep));r['feature_frame'].update(receiver_local_center_correction_m=(dx*px).tolist(),receiver_local_pocket_wall_normal=px.tolist(),receiver_local_pocket_up_normal=pz.tolist(),source_case_side_normal=cx.tolist(),case_rigid_correction_rotation=Q.tolist(),case_rigid_correction_angle_deg=float(np.rad2deg(Rotation.from_matrix(Q).magnitude())),registered_wall_normal_error_rad=float(np.arccos(np.clip((Q@cx)@px,-1,1))),receiver_local_center_correction_scalar_m=dx,fit_residual_seat_m=best['max_m']);r['feature_frame']['reference']='Opposed native receiver pocketwall triangle plane normals and upper lip plane; sourcecase dominant sideplane normal rotated intoexact alignment. Center fit minimizes sampled wallinterference.';r['fit_limitation']='Sourcecase is nonwatertight with mutually nonorthogonal shellplanes. Exact dominant sideplane+slotaxis alignment does not establish allfaces perfect or zerooverlap.';(out/'feature-pocket-fit.json').write_text(json.dumps(rows,indent=2)+'\n');(out/'alignment.json').write_text(json.dumps(r,indent=2)+'\n');print('FINAL',best,'axis',axis,'wallerror',r['feature_frame']['registered_wall_normal_error_rad'],'maxsweep',r['max_swept_sample_overlap_m'])
