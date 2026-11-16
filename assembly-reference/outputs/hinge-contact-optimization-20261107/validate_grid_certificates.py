"""Independent trilinear oracle for the compiled conservative certificates."""
import ctypes,os,struct,tempfile
from pathlib import Path
import mujoco,numpy as np
h=Path(__file__).resolve().parent
rng=np.random.default_rng(19341)
with tempfile.TemporaryDirectory() as tmp:
 p=Path(tmp);n=17;origin=-1.;step=.125
 z,y,x=np.meshgrid(np.arange(n),np.arange(n),np.arange(n),indexing='ij')
 grid=(np.sqrt((origin+x*step)**2+(origin+y*step)**2+(origin+z*step)**2)-.3).astype('<f2')
 (p/'grid_90001.bin').write_bytes(struct.pack('<3i4d',n,n,n,origin,origin,origin,step)+grid.tobytes())
 os.environ['ASSEMBLENIST_SDF_DIR']=tmp
 libpath=h/'local-native-v3/build/libassemblenist_sdf.so'
 mujoco.mj_loadPluginLibrary(str(libpath));lib=ctypes.CDLL(str(libpath))
 m=mujoco.MjModel.from_xml_string('''<mujoco><extension><plugin plugin="assemblenist.sdf.grid"><instance name="a"><config key="id" value="90001"/></instance><instance name="b"><config key="id" value="90001"/></instance></plugin></extension><asset><mesh name="a"><plugin instance="a"/></mesh><mesh name="b"><plugin instance="b"/></mesh></asset><worldbody><body><freejoint/><geom name="a" type="sdf" mesh="a" mass="1"><plugin instance="a"/></geom></body><body pos="2 0 0"><freejoint/><geom name="b" type="sdf" mesh="b" mass="1"><plugin instance="b"/></geom></body></worldbody></mujoco>''')
 d=mujoco.MjData(m);mujoco.mj_forward(m,d)
 ptr=ctypes.POINTER(ctypes.c_double)
 query=lib.assemblenist_grid_box_separated_fine;query.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ptr,ptr,ctypes.c_double];query.restype=ctypes.c_int
 def arr(x):return np.ascontiguousarray(x,dtype=np.float64)
 def ref(points):
  f=(points-origin)/step;c=np.clip(f,0,n-1.000001);i=np.floor(c).astype(int);t=c-i
  def at(x,y,z):return grid[i[:,2]+z,i[:,1]+y,i[:,0]+x].astype(float)
  a=at(0,0,0)+(at(1,0,0)-at(0,0,0))*t[:,0];b=at(0,1,0)+(at(1,1,0)-at(0,1,0))*t[:,0]
  c0=a+(b-a)*t[:,1];a=at(0,0,1)+(at(1,0,1)-at(0,0,1))*t[:,0];b=at(0,1,1)+(at(1,1,1)-at(0,1,1))*t[:,0]
  c1=a+(b-a)*t[:,1]
  return c0+(c1-c0)*t[:,2]+np.linalg.norm((f-c)*step,axis=1)
 rejected=0
 for _ in range(1200):
  center=rng.uniform(-1.3,1.3,3);radius=rng.uniform(0,.22,3);lo,hi=arr(center-radius),arr(center+radius);threshold=float(rng.choice([-.02,0,.02]))
  if query(m._address,d._address,0,lo.ctypes.data_as(ptr),hi.ctypes.data_as(ptr),threshold):
   rejected+=1;points=np.vstack([lo,hi,rng.uniform(lo,hi,(128,3))]);assert np.all(ref(points)>threshold)
 assert rejected>500,rejected
 pair=lib.assemblenist_grid_pair_separated;pair.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_int,ptr,ptr,ctypes.c_double];pair.restype=ctypes.c_int
 rot=arr(np.eye(3).ravel());far=arr([2,0,0]);overlap=arr([0,0,0])
 assert pair(m._address,d._address,0,1,rot.ctypes.data_as(ptr),far.ctypes.data_as(ptr),.00005)==1
 assert pair(m._address,d._address,0,1,rot.ctypes.data_as(ptr),overlap.ctypes.data_as(ptr),.00005)==0
 print('PASS:1200actualcompiled fine-box queries,',rejected,'certified clear boxes; independent128point trilinear checks; separated volume accepted/overlap refused.')
