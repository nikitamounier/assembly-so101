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
    if (end-start>8) {
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
extern "C" int assemblenist_grid_box_separated_fine(const mjModel*,const mjData*,int,
    const mjtNum[3],const mjtNum[3],mjtNum);
struct SweptSurfaceCandidates {
  int visual=-1,target=-1;
  bool valid=false;
  mjtNum threshold=0,rotation[9]{},translation[3]{};
  std::vector<int> ids;
};
struct SurfaceCache {
  uint64_t signature=0;
  const void* buffer=nullptr;
  std::unordered_map<int,SurfaceTree> trees;
  std::array<SweptSurfaceCandidates,127> swept;
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
  // Compose once per pair. Leaf evaluation retains sample_surface's original
  // arithmetic; this transform is used only for conservative rejection bounds.
  mjtNum rotation[9], translation[3], delta[3];
  const mjtNum* source_rotation=d->geom_xmat+9*visual;
  const mjtNum* target_rotation=d->geom_xmat+9*target;
  mju_sub3(delta,d->geom_xpos+3*visual,d->geom_xpos+3*target);
  mju_mulMatTVec3(translation,target_rotation,delta);
  for(int row=0;row<3;++row) for(int col=0;col<3;++col) {
    rotation[3*row+col]=0;
    for(int k=0;k<3;++k)
      rotation[3*row+col]+=target_rotation[3*k+row]*source_rotation[3*k+col];
  }
  // A certificate covers a 50um target-local box around each anchor sample.
  // Relative transforms allow common rigid world motion. Per-axis displacement
  // bounds cover every source point, not just the last contact locations.
  constexpr mjtNum sweep=0.00005;
  double radius[3]={},scale=1;
  if(!tree.nodes.empty()) for(int k=0;k<3;++k)
    radius[k]=std::max(fabs(tree.nodes[0].lo[k]),fabs(tree.nodes[0].hi[k]));
  for(int k=0;k<3;++k) scale+=radius[k]+fabs(d->geom_xpos[3*visual+k])+fabs(d->geom_xpos[3*target+k]);
  const double motion_guard=1e-12+1024*std::numeric_limits<mjtNum>::epsilon()*scale;
  auto& region=surface_cache.swept[(unsigned(visual)*73856093u ^ unsigned(target)*19349663u)%127];
  bool covered=region.valid && region.visual==visual && region.target==target &&
    region.threshold==threshold && motion_guard<sweep/4;
  for(int row=0;row<3 && covered;++row) {
    double bound=fabs(translation[row]-region.translation[row]);
    for(int col=0;col<3;++col) bound+=fabs(rotation[3*row+col]-region.rotation[3*row+col])*radius[col];
    covered=bound+motion_guard<sweep;
  }
  if(covered) {
    for(int id:region.ids) sample_surface(m,d,visual,target,tree.points[id].data(),direction,threshold,bins);
    return;
  }
  // Separate per-thread scratch from the model cache. Collision callbacks do
  // not recurse into tree_surface; clear sizes but retain allocated capacity.
  static thread_local std::vector<int> candidates, stack;
  candidates.clear();stack.clear();
  if(!tree.nodes.empty()) stack.push_back(0);
  while(!stack.empty()) {
    const auto& node=tree.nodes[stack.back()];stack.pop_back();
    if(profile_surface) surface_stats[3]++;
    mjtNum lo[3],hi[3],center[3],extent[3];
    double scale=1;
    for(int k=0;k<3;++k) {
      center[k]=node.lo[k]/2+node.hi[k]/2;
      extent[k]=node.hi[k]/2-node.lo[k]/2;
      scale+=fabs(node.lo[k])+fabs(node.hi[k])+
        fabs(d->geom_xpos[3*visual+k])+fabs(d->geom_xpos[3*target+k]);
    }
    // For rigid transforms, |R|*extent encloses every transformed corner.
    // The generous roundoff guard covers composition, center/extent rounding,
    // and the uncomposed two-transform arithmetic used by sample_surface.
    double pad=1e-12+256*std::numeric_limits<mjtNum>::epsilon()*scale;
    for(int row=0;row<3;++row) {
      mjtNum c=translation[row],e=0;
      for(int col=0;col<3;++col) {
        c+=rotation[3*row+col]*center[col];
        e+=fabs(rotation[3*row+col])*extent[col];
      }
      lo[row]=c-e;hi[row]=c+e;
    }
    const mjtNum* box=m->geom_aabb+6*target;
    bool outside=false;
    for(int k=0;k<3;++k) {
      lo[k]-=pad+sweep;hi[k]+=pad+sweep;
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
  region.ids.clear();
  // Refine loose BVH leaves once, with a fine swept voxel certificate. Keep
  // every uncertain sample in original order; its exact contact math runs at
  // each new pose. The coverage bound above forces rebuilding after motion.
  for(int id:candidates) {
    const auto& point=tree.points[id];
    mjtNum world[3],local[3],offset[3],lo[3],hi[3];
    mju_mulMatVec3(world,d->geom_xmat+9*visual,point.data());
    mju_addTo3(world,d->geom_xpos+3*visual);
    mju_sub3(offset,world,d->geom_xpos+3*target);
    mju_mulMatTVec3(local,d->geom_xmat+9*target,offset);
    const mjtNum* box=m->geom_aabb+6*target;
    bool outside=false;
    for(int k=0;k<3;++k) {
      lo[k]=local[k]-sweep-motion_guard;hi[k]=local[k]+sweep+motion_guard;
      outside|=lo[k]>box[k]+box[k+3]+.0001 || hi[k]<box[k]-box[k+3]-.0001;
    }
    if(!outside && !assemblenist_grid_box_separated_fine(m,d,target,lo,hi,threshold))
      region.ids.push_back(id);
  }
  region.visual=visual;region.target=target;region.threshold=threshold;
  std::copy(rotation,rotation+9,region.rotation);std::copy(translation,translation+3,region.translation);
  region.valid=motion_guard<sweep/4;
  for(int id:region.ids) sample_surface(m,d,visual,target,tree.points[id].data(),direction,threshold,bins);
}
