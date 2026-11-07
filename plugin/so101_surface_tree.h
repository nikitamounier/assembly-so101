// Exact existing sample set, immutable for a compiled model's lifetime.
// Per-thread ownership avoids shared mutable cache state and bounds memory to
// the most recently used compiled model. Mutating mesh arrays requires clearing.
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <numeric>
#include <unordered_map>
#include <vector>

extern "C" int assemblenist_grid_box_separated(const mjModel*, const mjData*, int,
                                               const mjtNum[3], const mjtNum[3], mjtNum);
struct SurfaceNode {
  std::array<mjtNum,3> lo, hi;
  int start, end, left=-1, right=-1;
};
struct SurfaceTree {
  std::vector<std::array<mjtNum,3>> points;
  std::vector<int> order;
  std::vector<SurfaceNode> nodes;
  int build(int start, int end) {
    SurfaceNode node; node.start=start; node.end=end;
    node.lo=points[order[start]]; node.hi=node.lo;
    for (int i=start+1;i<end;++i) for (int k=0;k<3;++k) {
      node.lo[k]=std::min(node.lo[k],points[order[i]][k]);
      node.hi[k]=std::max(node.hi[k],points[order[i]][k]);
    }
    int id=int(nodes.size()); nodes.push_back(node);
    if (end-start>32) {
      int axis=0;
      for(int k=1;k<3;++k) if(node.hi[k]-node.lo[k]>node.hi[axis]-node.lo[axis]) axis=k;
      int mid=(start+end)/2;
      std::nth_element(order.begin()+start,order.begin()+mid,order.begin()+end,
        [&](int a,int b){ return points[a][axis]<points[b][axis]; });
      int left=build(start,mid),right=build(mid,end);
      nodes[id].left=left; nodes[id].right=right;
    }
    return id;
  }
  SurfaceTree(const mjModel* m,int mesh) {
    int va=m->mesh_vertadr[mesh],fa=m->mesh_faceadr[mesh];
    for(int i=0;i<m->mesh_vertnum[mesh];++i) {
      std::array<mjtNum,3> p;
      for(int k=0;k<3;++k) p[k]=m->mesh_vert[3*(va+i)+k];
      points.push_back(p);
    }
    for(int i=0;i<m->mesh_facenum[mesh];++i) {
      std::array<mjtNum,3> p{};
      // Retain the original float division and accumulation order exactly.
      for(int j=0;j<3;++j) for(int k=0;k<3;++k)
        p[k]+=m->mesh_vert[3*(va+m->mesh_face[3*(fa+i)+j])+k]/3;
      points.push_back(p);
    }
    order.resize(points.size());std::iota(order.begin(),order.end(),0);
    if(!points.empty()) build(0,int(points.size()));
  }
};
struct SurfaceCache {
  uint64_t signature=0;
  const void* buffer=nullptr;
  std::unordered_map<int,SurfaceTree> trees;
};
static thread_local SurfaceCache surface_cache;
// Benchmark oracle selection is read once per thread, never on the hot path.
static int initial_surface_mode() {
  const char* mode=std::getenv("SO101_SURFACE_MODE");
  return mode && !strcmp(mode,"0") ? 0 : (mode && !strcmp(mode,"2") ? 2 : 1);
}
static thread_local int surface_mode=initial_surface_mode();
extern "C" int so101_surface_mode(int mode) {
  int previous=surface_mode; if(mode>=0 && mode<=2) surface_mode=mode; return previous;
}
extern "C" void so101_surface_clear_cache(void) { surface_cache=SurfaceCache{}; }

static void tree_surface(const mjModel* m,const mjData* d,int visual,int target,
                         int direction,mjtNum threshold,mjPreContact bins[12]) {
  if(surface_cache.signature!=m->signature || surface_cache.buffer!=m->buffer) {
    surface_cache=SurfaceCache{};
    surface_cache.signature=m->signature;surface_cache.buffer=m->buffer;
  }
  int mesh=m->geom_dataid[visual];
  auto entry=surface_cache.trees.find(mesh);
  if(entry==surface_cache.trees.end())
    entry=surface_cache.trees.emplace(mesh,SurfaceTree(m,mesh)).first;
  const SurfaceTree& tree=entry->second;
  if(surface_mode==2) {
    for(const auto& point:tree.points) sample_surface(m,d,visual,target,point.data(),direction,threshold,bins);
    return;
  }
  std::vector<int> candidates;
  std::vector<int> stack;
  if(!tree.nodes.empty()) stack.push_back(0);
  while(!stack.empty()) {
    const auto& node=tree.nodes[stack.back()];stack.pop_back();
    if(profile_surface) surface_stats[3]++;
    mjtNum lo[3]={INFINITY,INFINITY,INFINITY},hi[3]={-INFINITY,-INFINITY,-INFINITY};
    double scale=1;
    for(int k=0;k<3;++k) scale+=fabs(node.lo[k])+fabs(node.hi[k])+
      fabs(d->geom_xpos[3*visual+k])+fabs(d->geom_xpos[3*target+k]);
    // Bound affine extrema with the same two transforms used by sample_surface.
    for(int corner=0;corner<8;++corner) {
      mjtNum p[3],world[3],delta[3],local[3];
      for(int k=0;k<3;++k) p[k]=(corner&(1<<k))?node.hi[k]:node.lo[k];
      mju_mulMatVec3(world,d->geom_xmat+9*visual,p);
      mju_addTo3(world,d->geom_xpos+3*visual);
      mju_sub3(delta,world,d->geom_xpos+3*target);
      mju_mulMatTVec3(local,d->geom_xmat+9*target,delta);
      for(int k=0;k<3;++k){lo[k]=std::min(lo[k],local[k]);hi[k]=std::max(hi[k],local[k]);}
    }
    double pad=1e-12+128*std::numeric_limits<mjtNum>::epsilon()*scale;
    const mjtNum* box=m->geom_aabb+6*target;
    bool outside=false;
    for(int k=0;k<3;++k) {
      lo[k]-=pad;hi[k]+=pad;
      outside|=lo[k]>box[k]+box[k+3]+.0001 || hi[k]<box[k]-box[k+3]-.0001;
    }
    if(outside) { if(profile_surface) surface_stats[5]++; continue; }
    if(assemblenist_grid_box_separated(m,d,target,lo,hi,threshold)) {
      if(profile_surface) { surface_stats[6]++; }
      continue;
    }
    if(node.left<0) candidates.insert(candidates.end(),tree.order.begin()+node.start,tree.order.begin()+node.end);
    else {stack.push_back(node.right);stack.push_back(node.left);}
  }
  std::sort(candidates.begin(),candidates.end());
  for(int id:candidates) sample_surface(m,d,visual,target,tree.points[id].data(),direction,threshold,bins);
}
