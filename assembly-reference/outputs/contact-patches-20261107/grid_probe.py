from pathlib import Path
import sys,os,json,time,importlib.util,subprocess,hashlib,shutil
import numpy as np
import mujoco
from patch import apply
root,out=map(Path,sys.argv[1:3]);tag=sys.argv[3];variant=tag;out=out/tag;out.mkdir(parents=True,exist_ok=True)
rel=root/'native-v13';sys.path.insert(0,str(rel/'assembly-so101'));os.environ['WOWROBO_CANONICAL_ROOT']=str(root/'robotics-assembly-3.0')
s=importlib.util.spec_from_file_location('_wowrobo_training_native_backend',rel/'robotics-assembly-3.0/assembly_policy/so101/wowrobo_contact_backend.py');b=importlib.util.module_from_spec(s);sys.modules[s.name]=b;s.loader.exec_module(b)
native=out/'native'
if len(sys.argv)<5 or sys.argv[4] not in ['ppo','reuse']:
 shutil.copytree(root/'native-contact-optimization-20261107/contact-separation-20261107/both/native',native,dirs_exist_ok=True)
 if variant!='baseline':
  if variant=="hoisted":
   from hoist import apply as apply_hoisted
   apply_hoisted(native)
  elif variant=="patch":
   from patch_winners import apply as apply_winners
   apply_winners(native)
  else:apply(native,variant)
  for script in (['build.sh','build_contact_search.sh'] if variant in ['fused','winner','combined','hoisted'] else ['build_contact_search.sh']):
   subprocess.run(['bash',str(native/script)],env=dict(os.environ,PYTHON=sys.executable),check=True)
original=b.load_backend;b.build_native=lambda directory:Path(directory);b.load_backend=lambda manifest:original(manifest,native)
from assembly_policy.wowrobo.env import BatchedAssemblyEnv,EnvConfig
bank=rel/'training-accepted-v13';meta=json.loads((bank/'manifest.json').read_text());e=BatchedAssemblyEnv(meta['task_names'],np.zeros(2,np.int64),EnvConfig(profile_path=meta['profile_path'],review_model=meta['review_model'],nthread=2))
import ctypes
package=Path(mujoco.__file__).parent;library=next(package.glob('libmujoco.so*'));probe=out/'grid-probe.so'
subprocess.run(['c++','-std=c++17','-shared','-fPIC','-O2','-Wall','-Werror','-I'+str(package/'include'),str(out.parent/'grid_probe.cc'),str(native/'build/libassemblenist_sdf.so'),str(library),'-Wl,-rpath,'+str(package),'-Wl,-rpath,'+str(native/'build'),'-o',str(probe)],check=True)
lib=ctypes.CDLL(str(probe));ptr=ctypes.POINTER(ctypes.c_double);lib.compare_grid_samples.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ptr,ctypes.c_int,ctypes.c_double]
rng=np.random.default_rng(20261107);queries=0;grids=0
try:
 for geom in range(e.m.ngeom):
  instance=int(e.m.geom_plugin[geom])
  if instance<0:continue
  box=e.m.geom_aabb[geom];points=np.ascontiguousarray(box[:3]+rng.uniform(-1.3,1.3,(1000,3))*box[3:])
  for threshold in [-.0001,0.,.00005]:
   status=lib.compare_grid_samples(e.m._address,e.data[0]._address,instance,points.ctypes.data_as(ptr),len(points),threshold)
   if status==-1:break
   assert status==0,(geom,threshold,status)
   queries+=len(points)
  if status!=-1:grids+=1
 (out/'grid-probe.json').write_text(json.dumps(dict(status='PASS',grids=grids,queries=queries,thresholds=[-.0001,0.,.00005],scope='Bitwise distances and all distance-qualified gradients vs separate native plugin callbacks, including finite outside-grid queries'))+'\n');print('grid-probe PASS',grids,queries)
finally:e.close()
