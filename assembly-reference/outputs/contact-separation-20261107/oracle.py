"""Private native search experiment. Never modifies a release or live process."""
from pathlib import Path
import sys,os,json,time,importlib.util,subprocess,hashlib,shutil
import numpy as np
import mujoco
root,out=map(Path,sys.argv[1:3]);tag=sys.argv[3];variant=tag.split('-')[0];stats=tag.endswith('-stats');out=out/tag;out.mkdir(parents=True,exist_ok=True)
rel=root/'native-v13';sys.path.insert(0,str(rel/'assembly-so101'));os.environ['WOWROBO_CANONICAL_ROOT']=str(root/'robotics-assembly-3.0')
s=importlib.util.spec_from_file_location('_wowrobo_training_native_backend',rel/'robotics-assembly-3.0/assembly_policy/so101/wowrobo_contact_backend.py');b=importlib.util.module_from_spec(s);sys.modules[s.name]=b;s.loader.exec_module(b)
reuse=len(sys.argv)>4 and sys.argv[4] in ['ppo','reuse']
if reuse:
 native=out/'native';assert native.is_dir()
elif variant=='baseline' and not stats:native=b.build_native(out/'native')
else:
 native=out/'native';shutil.copytree(out.parent/'baseline/native',native,dirs_exist_ok=True)
if (variant!='baseline' or stats) and not reuse:
 c=native/'so101_contact_search.c';text=c.read_text()
 if variant in ['separation','both']:
  old='if(!e.internal)return false;';assert text.count(old)==1
  text=text.replace(old,'if(!e.internal) { if(!assemblenist_grid_domain_radius(m,d,first,e.radius))return false; }')
  # Existing certificate covers zero-sublevel volumes. Positive margins keep full search.
  old='surface_mode!=2 && wow_sdf_pair_clear(m,d,g1,g2)';assert text.count(old)==1;text=text.replace(old,'surface_mode!=2 && margin<=0 && wow_sdf_pair_clear(m,d,g1,g2)')
 if variant in ['cache','both']:
  h=native/'so101_surface_tree.h';hs=h.read_text()
  hs=hs.replace('std::array<SweptSurfaceCandidates,1021> swept;', 'std::array<std::array<SweptSurfaceCandidates,4>,1021> swept;\n  unsigned next[1021]{};')
  old='  auto& region=surface_cache.swept[(unsigned(visual)*73856093u ^ unsigned(target)*19349663u)%1021];';assert hs.count(old)==1
  hs=hs.replace(old,'''  unsigned bucket=(unsigned(visual)*73856093u ^ unsigned(target)*19349663u)%1021;
  auto& ways=surface_cache.swept[bucket];
  int selected=-1;
  for(int way=0;way<4;++way) {
    const auto& r=ways[way];
    bool hit=r.valid && r.visual==visual && r.target==target && r.threshold==threshold && motion_guard<sweep/4;
    for(int row=0;row<3 && hit;++row) {
      double bound=fabs(translation[row]-r.translation[row]);
      for(int col=0;col<3;++col)bound+=fabs(rotation[3*row+col]-r.rotation[3*row+col])*radius[col];
      hit=bound+motion_guard<sweep;
    }
    if(hit){selected=way;break;}
  }
  if(selected<0)selected=surface_cache.next[bucket]++%4;
  auto& region=ways[selected];''')
  h.write_text(hs)
 if stats:
  h=native/'so101_surface_tree.h';hs=h.read_text()
  hs=hs.replace('static thread_local SurfaceCache surface_cache;', '''#include <atomic>
static std::atomic<unsigned long long> probe_counts[3]{};
extern "C" void probe_surface_counts(unsigned long long out[3]) {
 for(int i=0;i<3;++i)out[i]=probe_counts[i].load(std::memory_order_relaxed);
}
static thread_local SurfaceCache surface_cache;''')
  hs=hs.replace('  if(covered) {', '  if(covered) { probe_counts[0].fetch_add(1,std::memory_order_relaxed);')
  hs=hs.replace('  surface_cache.swept_stats[1]++;','  probe_counts[1].fetch_add(1,std::memory_order_relaxed);\n  surface_cache.swept_stats[1]++;')
  hs=hs.replace('  surface_cache.swept_stats[2]+=region.ids.size();','  probe_counts[2].fetch_add(region.ids.size(),std::memory_order_relaxed);\n  surface_cache.swept_stats[2]+=region.ids.size();')
  h.write_text(hs)
 c.write_text(text)
 subprocess.run(['bash',str(native/'build_contact_search.sh')],env=dict(os.environ,PYTHON=sys.executable),check=True)
original=b.load_backend;b.build_native=lambda directory:Path(directory);b.load_backend=lambda manifest:original(manifest,native)
from assembly_policy.wowrobo.env import BatchedAssemblyEnv,EnvConfig
bank=rel/'training-accepted-v13';meta=json.loads((bank/'manifest.json').read_text());e=BatchedAssemblyEnv(meta['task_names'],np.zeros(2,np.int64),EnvConfig(profile_path=meta['profile_path'],review_model=meta['review_model'],nthread=2))
lib=b.surface_library();lib.so101_surface_mode(2)
checked=0
try:
 for ti,task in enumerate(meta['task_names']):
  with np.load(bank/(task+'_train.npz')) as z:
   for stage in np.unique(z['stage']):
    ids=np.flatnonzero(z['stage']==stage);state=z['states'][[ids[0],ids[-1]]]
    lib.so101_surface_mode(2);e.set_states(np.arange(2),state,np.full(2,ti,np.int64))
    def capture():
     return [[d.qacc.copy(),d.sensordata.copy(),np.array([np.r_[c.dist,c.pos,c.frame.ravel(),c.geom] for c in d.contact])] for d in e.data]
    expected=capture();lib.so101_surface_mode(1);e._refresh();actual=capture()
    for aa,bb in zip(expected,actual):
     for x,y in zip(aa,bb):np.testing.assert_array_equal(x,y);checked+=1
    print(task,str(stage),'PASS',flush=True)
 (out/'oracle.json').write_text(json.dumps(dict(status='PASS',arrays=checked,scope='All25bankstages,2savedworlds each; main-thread exhaustive surface mode2 vs accelerated mode1, includes native acceleration/sensors/contact arrays'))+'\n')
finally:e.close()
