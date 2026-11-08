"""Build a private delegating timer with counters; numerical statements unchanged."""
from common import *
import subprocess

mod=backend('instrumented')
p=mod.build_native(OUT/'instrumented/native')
c=p/'so101_contact_search.c';s=c.read_text()
anchor='static int contact_search(const mjModel* m, mjData* d, mjPreContact* con,'
at=s.index(anchor)
decl='''
static thread_local double pair_stats[256*256][14] = {};
extern "C" void wow_profile_pairs(double* out,int reset) {
  for(int i=0;i<256*256;++i) for(int j=0;j<14;++j) {
    out[i*14+j]=pair_stats[i][j]; if(reset) pair_stats[i][j]=0;
  }
}
'''
s=s[:at]+decl+s[at:]
s=s.replace('  auto begin = profile_surface ?', '  double prior[7]; for(int j=0;j<7;++j) prior[j]=surface_stats[j];\n  auto begin = profile_surface ?',1)
s=s.replace('  auto middle = profile_surface ?','  int original_count=count;\n  auto middle = profile_surface ?',1)
needle='  return count;\n}\n\nextern "C" int so101_install_contact_search'
replacement='''  if(profile_surface && g1<256 && g2<256) {
    double* q=pair_stats[g1*256+g2];
    q[0]++;q[1]+=std::chrono::duration<double>(middle-begin).count();
    q[2]+=surface_stats[2]-prior[2];
    for(int j=3;j<7;++j) q[j]+=surface_stats[j]-prior[j];
    q[7]+=original_count;q[8]+=count;q[9]+=count-original_count;
    q[10]=values[0];q[11]=values[1];q[12]+=original_count==0;q[13]+=count==0;
  }
  return count;
}

extern "C" int so101_install_contact_search'''
assert s.count(needle)==1;s=s.replace(needle,replacement)
c.write_text(s)
subprocess.run(['bash',str(p/'build_contact_search.sh')],env=dict(os.environ,PYTHON=sys.executable),check=True)
write(OUT/'instrumentation.json',dict(instrumented_source_sha256=sha(c),baseline_source_hashes=json.loads((p/'build-receipt.json').read_text())['identity'],libraries={x.name:sha(x) for x in (p/'build').glob('*.so')},columns=['calls','sdf_s','supplement_s','tree_nodes','leaf_samples','aabb_rejected_nodes','grid_rejected_nodes','sdf_contacts','returned_contacts','supplement_net_added','initpoints','iterations','sdf_zero_calls','returned_zero_calls']))
