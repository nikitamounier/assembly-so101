exec(open(__file__.replace('gaps.py','scan.py')).read().split('results=[]')[0])
for g in range(m.ngeom):
 n=m.geom(g).name or ''
 source=v.grid_records.get(n.removesuffix('__contact'),{}).get('source_mesh_name','')
 if n=='base_side_cover_printed_0__contact' or source.startswith('left')and any(x in source for x in ['wrist_roll_follower','moving_jaw']):m.geom_margin[g]=.003;print('margin',g,n,source)
results=[]
for stage in ['01_cover_pickup','03_cover_aligned','04_cover_entry']:
 d=mujoco.MjData(m)
 with np.load(ROOT/f'assets_local/wowrobo/photo-resets-repaired-cover-support-v2/stages/{stage}_train.npz')as z:mujoco.mj_setState(m,d,z['states'][0],mujoco.mjtState.mjSTATE_FULLPHYSICS);d.ctrl[:]=z['controls'][0]
 mujoco.mj_forward(m,d);q0=d.qpos.copy();p0=d.site_xpos[sid].copy();r0=d.site_xmat[sid].reshape(3,3).copy();scratch=mujoco.MjData(vis);scratch.qpos[:]=q0
 for shift in [0.,.0002,.0004,.0006]:
  target=p0+r0[:,2]*shift
  def fun(x):
   scratch.qpos[qs]=x;mujoco.mj_forward(vis,scratch);r=scratch.site_xmat[sid].reshape(3,3)
   return np.r_[1000*(scratch.site_xpos[sid]-target),2*(r[:,2]-r0[:,2]),r[:,0]-r0[:,0]]
  fit=least_squares(fun,q0[qs],bounds=(np.maximum(m.jnt_range[jids,0]+1e-7,q0[qs]-.05),np.minimum(m.jnt_range[jids,1]-1e-7,q0[qs]+.05)),max_nfev=50)
  for delta in [0.,-.015,-.02,-.022,-.024,-.026,-.028]:
   d.qpos[:]=q0;d.qpos[qs]=fit.x;d.qpos[jaw]=q0[jaw]+delta;mujoco.mj_forward(m,d)
   gap={}
   for i in range(d.ncon):
    a,b=map(int,d.contact.geom[i]);names=[m.geom(g).name for g in [a,b]]
    if 'base_side_cover_printed_0__contact'not in names:continue
    other=b if names[0]=='base_side_cover_printed_0__contact'else a
    mesh=v.grid_records.get(m.geom(other).name.removesuffix('__contact'),{}).get('source_mesh_name','')
    pad='fixed'if'moving_jaw'not in mesh else'moving'
    if not mesh.startswith('left') or not any(x in mesh for x in ['wrist_roll_follower','moving_jaw']):continue
    point=(d.contact.pos[i]-d.site_xpos[sid])@d.site_xmat[sid].reshape(3,3)
    if not(-.035<point[0]<.004 and abs(point[1])<.01):continue
    gap[pad]=min(gap.get(pad,1),float(d.contact.dist[i]))
   r={'stage':stage,'shift_m':shift,'jaw_delta':delta,'gaps':gap,'left_q':np.r_[fit.x,d.qpos[jaw]].tolist()};results.append(r);print(r,flush=True)
(OUT/'gaps.json').write_text(json.dumps(results,indent=2)+'\n')
