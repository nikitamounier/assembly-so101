exec(open('outputs/equilibrium-active-mating-grasps-20261107/jaw21.py').read().split('results=[]')[0])
z=np.load(OUT/'0.005.npz');d=mujoco.MjData(m);d.qpos[:]=z['final_qpos'];mujoco.mj_forward(m,d);R=d.site_xmat[sid].reshape(3,3)
for con in d.contact:
 ga,gb=map(int,con.geom);ba,bb=map(int,m.geom_bodyid[[ga,gb]])
 if bid not in [ba,bb]:continue
 other=gb if ba==bid else ga;name=m.geom(other).name or ''
 if 'moving_jaw'not in name:continue
 pt=(con.pos-d.site_xpos[sid])@R;n=con.frame[:3]*(1 if other==ga else -1)
 print(name,con.dist,pt.tolist(),float(n@R[:,2]),flush=True)
print('jaw',d.qpos[jaw],z['qpos_candidate'][jaw],flush=True)
