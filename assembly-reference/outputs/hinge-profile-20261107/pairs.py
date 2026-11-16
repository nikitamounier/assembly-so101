"""Private pair-attribution wrapper; instrumented output checked against plain."""
from pathlib import Path
import ctypes,hashlib,importlib.util,json,os,subprocess,sys,time
import mujoco,numpy as np
root,out=map(Path,sys.argv[1:3]);plain='--plain' in sys.argv;tag='plain' if plain else 'instrumented';out=out/('pairs-'+tag);out.mkdir(exist_ok=True)
rel=root/'native-v13';sys.path.insert(0,str(rel/'assembly-so101'));os.environ['WOWROBO_CANONICAL_ROOT']=str(root/'robotics-assembly-3.0')
s=importlib.util.spec_from_file_location('_wowrobo_training_native_backend',rel/'robotics-assembly-3.0/assembly_policy/so101/wowrobo_contact_backend.py');b=importlib.util.module_from_spec(s);sys.modules[s.name]=b;s.loader.exec_module(b)
p=b.build_native(out/'native')
if not plain:
 c=p/'so101_contact_search.c';s=c.read_text();anchor='static int contact_search(const mjModel* m, mjData* d, mjPreContact* con,'
 decl='''#include <mutex>
static std::mutex prof_mutex;
static double prof_pairs[256*256][9]={};
extern "C" void wow_profile_pairs(double* out,int reset) {
 std::lock_guard<std::mutex> guard(prof_mutex);
 for(int i=0;i<256*256;++i)for(int j=0;j<9;++j){out[9*i+j]=prof_pairs[i][j];if(reset)prof_pairs[i][j]=0;}
}
'''
 at=s.index(anchor);s=s[:at]+decl+s[at:]
 needle='  auto begin = profile_surface ?';assert s.count(needle)==1;s=s.replace(needle,'  profile_surface=true; double prior_supp=surface_stats[2];\n'+needle)
 needle='  auto middle = profile_surface ?';s=s.replace(needle,'  int prof_original_count=count;\n'+needle)
 needle='  return count;\n}\n\nextern "C" int so101_install_contact_search';assert s.count(needle)==1
 s=s.replace(needle,'''  if(g1<256 && g2<256){
    std::lock_guard<std::mutex> guard(prof_mutex);double* q=prof_pairs[g1*256+g2];
    q[0]++;q[1]+=std::chrono::duration<double>(middle-begin).count();q[2]+=surface_stats[2]-prior_supp;
    q[3]+=prof_original_count;q[4]+=count;q[5]+=prof_original_count==0;q[6]+=count==0;q[7]=values[0];q[8]=values[1];
  }
'''+needle)
 c.write_text(s);subprocess.run(['bash',str(p/'build_contact_search.sh')],env=dict(os.environ,PYTHON=sys.executable),check=True)
original=b.load_backend;b.build_native=lambda directory:Path(directory);b.load_backend=lambda manifest:original(manifest,p)
from assembly_policy.wowrobo.env import BatchedAssemblyEnv,EnvConfig
bank=rel/'training-accepted-v13';meta=json.loads((bank/'manifest.json').read_text());e=BatchedAssemblyEnv(meta['task_names'],np.zeros(2,np.int64),EnvConfig(profile_path=meta['profile_path'],review_model=meta['review_model'],nthread=2))
lib=b.surface_library();stats=np.zeros((256*256,9));rows=[];arrays={}
if not plain:lib.wow_profile_pairs.argtypes=[ctypes.POINTER(ctypes.c_double),ctypes.c_int]
def read(reset=1):lib.wow_profile_pairs(stats.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),reset)
try:
 for task,stage in [('base_side_cover','00_loose'),('base_side_cover','05_cover_near'),('upper_arm_module','21_upper_far_side'),('forearm_module','31_final_near')]:
  ti=meta['task_names'].index(task)
  with np.load(bank/(task+'_train.npz')) as z:
   ids=np.flatnonzero(z['stage']==stage);fr=z['states'][ids,159:162];nom=np.flatnonzero(np.all(fr==1,axis=1));var=np.flatnonzero(np.any(fr!=1,axis=1));ix=[int(ids[nom[0] if len(nom) else 0]),int(ids[var[0] if len(var) else 0])];state=z['states'][ix]
  e.set_states(np.arange(2),state,np.full(2,ti,np.int64))
  if not plain:read()
  t=time.perf_counter();r,done,info=e.step(.005*np.sin(np.arange(24).reshape(2,12)));elapsed=time.perf_counter()-t
  arrays[stage+'_state']=e.state.copy();arrays[stage+'_reward']=r;arrays[stage+'_done']=done
  for key,value in info.items():arrays[stage+'_info_'+key]=value
  pairs=[]
  if not plain:
   read()
   for i in np.flatnonzero(stats[:,0]):
    g1,g2=divmod(int(i),256);pairs.append(dict(geoms=[g1,g2],names=[e.m.geom(g).name for g in [g1,g2]],bodies=[e.m.body(int(e.m.geom_bodyid[g])).name for g in [g1,g2]],stats=stats[i].tolist()))
   pairs.sort(key=lambda x:sum(x['stats'][1:3]),reverse=True)
  rows.append(dict(task=task,stage=stage,wall_s=elapsed,pairs=pairs));print(stage,elapsed,flush=True)
 np.savez_compressed(out/'outputs.npz',**arrays)
 (out/'pairs.json').write_text(json.dumps(dict(rows=rows,columns=['calls','sdf_s','supplement_s','sdf_contacts','returned_contacts','zero_sdf_calls','zero_final_calls','initpoints','iterations'],source_sha256=hashlib.sha256((p/'so101_contact_search.c').read_bytes()).hexdigest(),scope='Separate private instrumentation; sum across2native threads. Includes native rollout AND end-step refresh. Not primary throughput measurement.'),indent=2)+'\n')
finally:e.close()
