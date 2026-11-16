from pathlib import Path
import json,time
import numpy as np,trimesh
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.transform import Rotation
ROOT=Path.cwd();OUT=ROOT/'outputs/insertion-fix-20261107/cover';manifest=json.loads((ROOT/'outputs/linear-insertion-updated-20261107/grids/manifest.json').read_text());a=json.loads((ROOT/'outputs/linear-insertion-aligned-20261107/cover/alignment.json').read_text());R=Rotation.from_quat(np.roll(a['final_root_qpos'][3:],-1)).as_matrix();p=np.array(a['final_root_qpos'][:3]);receiverp=np.array([-.00607785,0,-.0003943]);names=['base_side_cover_printed_0','base_servo_printed_0','servo_1_case','servo_1_horns'];data={}
for n in names:
 r=manifest['meshes'][n];mesh=trimesh.load_mesh(r['source_file'],process=False);pts=np.vstack((mesh.vertices,mesh.triangles_center));pts=pts[np.linspace(0,len(pts)-1,min(len(pts),6000),dtype=int)];grid=np.memmap(r['grid_file'],dtype='<f2',mode='r',offset=44,shape=tuple(r['shape'][::-1]));data[n]={'mesh':mesh,'points':pts,'grid':grid,'origin':np.array(r['origin']),'shape':np.array(r['shape']),'h':r['spacing']}
cover=data[names[0]];centerlocal=cover['mesh'].bounds.mean(0);centerworld=p+R@centerlocal
# Bidirectional SDF source query; signs and precision remain grid approximations.
def evaluate(n,q):
 d=data[n];c=(q-d['origin'])/d['h'];cl=np.clip(c,0,d['shape']-1-1e-8);base=np.floor(cl).astype(int);f=cl-base;v=np.zeros(len(q))
 for z in (0,1):
  for y in (0,1):
   for x in (0,1):
    v+=np.prod(np.where(np.array([x,y,z]),f,1-f),axis=1)*d['grid'][base[:,2]+z,base[:,1]+y,base[:,0]+x]
 return v+np.linalg.norm((c-cl)*d['h'],axis=1)
def transforms(x):
 # x=(dx,dy,roll,pitch,yaw), mm/degrees; Z seatingdatumfixedatroot.
 dr=Rotation.from_euler('xyz',x[2:],degrees=True).as_matrix();rn=dr@R;pn=centerworld+np.array([x[0]*.001,x[1]*.001,0])-rn@centerlocal
 # Preserve bottom/front root insertion depth; no upward cheat.
 pn[2]=p[2]
 return rn,pn
count=0;best=[1e9,None]
def objective(x,record=False):
 global count
 rn,pn=transforms(x);cp=cover['points']@rn.T+pn;parts=[]
 for n in names[1:]:
  q=(cp-receiverp)@R;parts.append(evaluate(n,q));rp=data[n]['points']@R.T+receiverp;parts.append(evaluate(names[0],(rp-pn)@rn))
 dist=np.concatenate(parts);pen=np.maximum(-dist,0);top=np.sort(pen)[-20:];score=float(top[-1]+.15*top.mean()+1e-5*np.linalg.norm(x[:2])+1e-6*np.linalg.norm(x[2:]));count+=1
 if score<best[0]:
  best[:]=[score,x.copy()];print(count,'best',score,'rawmax',pen.max(),'x',x,flush=True)
 return (score,float(pen.max()))if record else score
bounds=[(-.3,.3),(-.1,.1),(-.7,.7),(-.7,.7),(-.15,.15)]
start=time.monotonic();initial=objective(np.zeros(5),True);res=differential_evolution(objective,bounds,popsize=5,maxiter=15,tol=.02,polish=False,seed=1181,workers=1);local=minimize(objective,res.x,method='Powell',bounds=bounds,options={'maxfev':250,'xtol':1e-5,'ftol':1e-3});x=best[1];rn,pn=transforms(x);q=np.roll(Rotation.from_matrix(rn).as_quat(),1);out={'method':'BidirsourceSDFapproxsmallrigidfitsearch;rootZfixed, noCADalteration','initial_grid_objective':initial,'best_grid_objective':objective(x,True),'parameters_mm_deg':x.tolist(),'root_position_world_m':pn.tolist(),'root_quaternion_wxyz':q.tolist(),'initial_root_position_world_m':p.tolist(),'elapsed_s':time.monotonic()-start,'evaluations':count};(OUT/'rigid-fit-search.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2),flush=True)
