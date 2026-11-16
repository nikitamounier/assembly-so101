// Conservative rejection helper; it never changes grid_eval or its callbacks.
#include <limits>
static constexpr int kBoundBlock=4;
struct GridBounds {
  int bx, by, bz;
  double max_abs=0;
  std::vector<double> block_min;
};
static const GridBounds* grid_bounds(const Grid* g) {
  static std::mutex mutex;
  static std::unordered_map<const Grid*, std::unique_ptr<GridBounds>> all;
  thread_local std::unordered_map<const Grid*, const GridBounds*> cached;
  auto entry=cached.find(g);
  if(entry!=cached.end()) return entry->second;
  std::lock_guard<std::mutex> lock(mutex);
  auto found=all.find(g);
  if(found!=all.end()) return cached[g]=found->second.get();
  auto owner=std::make_unique<GridBounds>();
  GridBounds* b=owner.get();
  // Immutable minima over 4x4x4 grid samples, including boundary blocks.
  b->bx = (g->nx + kBoundBlock-1)/kBoundBlock; b->by = (g->ny + kBoundBlock-1)/kBoundBlock; b->bz = (g->nz + kBoundBlock-1)/kBoundBlock;
  b->block_min.assign(size_t(b->bx)*b->by*b->bz, INFINITY);
  for (int z = 0; z < g->nz; ++z) for (int y = 0; y < g->ny; ++y)
    for (int x = 0; x < g->nx; ++x) {
      double value = h2f(g->v[(size_t(z)*g->ny+y)*g->nx+x]);
      b->max_abs = std::max(b->max_abs, std::fabs(value));
      auto& low = b->block_min[(size_t(z/kBoundBlock)*b->by+y/kBoundBlock)*b->bx+x/kBoundBlock];
      low = std::min(low, std::isfinite(value) ? value : -INFINITY);
    }
  const GridBounds* result=b;
  all[g]=std::move(owner);
  return cached[g]=result;
}

extern "C" int assemblenist_grid_box_separated(const mjModel* m, const mjData* d,
    int target, const mjtNum lo[3], const mjtNum hi[3], mjtNum threshold) {
  int instance = m->geom_plugin[target];
  if (instance < 0 || !d->plugin_data[instance]) return 0;
  const mjpPlugin* plugin = mjp_getPluginAtSlot(m->plugin[instance]);
  if (!plugin || std::strcmp(plugin->name, "assemblenist.sdf.grid")) return 0;
  const Grid* g = get_grid(int(reinterpret_cast<Inst*>(d->plugin_data[instance])->a[0]));
  const GridBounds* bounds=grid_bounds(g);
  const int n[3] = {g->nx,g->ny,g->nz};
  const double o[3] = {g->ox,g->oy,g->oz};
  int first[3], last[3];
  for (int k=0; k<3; ++k) {
    if (!std::isfinite(lo[k]) || !std::isfinite(hi[k])) return 0;
    // Match grid_eval clamping and include both corners of every possible cell.
    first[k] = int(std::floor(std::clamp((lo[k]-o[k])/g->h,0.0,n[k]-1.000001)))/kBoundBlock;
    last[k] = (int(std::floor(std::clamp((hi[k]-o[k])/g->h,0.0,n[k]-1.000001)))+1)/kBoundBlock;
  }
  // Rounded trilinear interpolation of half values stays above the corner
  // minimum up to rounding. This deliberately generous absolute guard dwarfs
  // interpolation error in metre-scale grids; outside-grid extension is >=0.
  const double guard = 1e-12 + 128*std::numeric_limits<double>::epsilon()*bounds->max_abs;
  for (int z=first[2];z<=last[2];++z) for (int y=first[1];y<=last[1];++y)
    for (int x=first[0];x<=last[0];++x)
      if (!(bounds->block_min[(size_t(z)*bounds->by+y)*bounds->bx+x] > threshold+guard)) return 0;
  return 1;
}

// Fine voxel-corner certificate for a small swept sample box. Rounded
// trilinear interpolation is bounded below by these corners; outside-grid
// extension is nonnegative. Unknown/non-grid targets always use the fallback.
extern "C" int assemblenist_grid_box_separated_fine(const mjModel* m,const mjData* d,
    int target,const mjtNum lo[3],const mjtNum hi[3],mjtNum threshold) {
  int instance=m->geom_plugin[target];
  if(instance<0 || !d->plugin_data[instance]) return 0;
  const auto* plugin=mjp_getPluginAtSlot(m->plugin[instance]);
  if(!plugin || std::strcmp(plugin->name,"assemblenist.sdf.grid")) return 0;
  const Grid* g=get_grid(int(reinterpret_cast<Inst*>(d->plugin_data[instance])->a[0]));
  const int n[3]={g->nx,g->ny,g->nz};const double o[3]={g->ox,g->oy,g->oz};
  int first[3],last[3];
  for(int k=0;k<3;++k) {
    if(!std::isfinite(lo[k]) || !std::isfinite(hi[k]) || lo[k]>hi[k]) return 0;
    first[k]=int(std::floor(std::clamp((lo[k]-o[k])/g->h,0.0,n[k]-1.000001)));
    last[k]=int(std::floor(std::clamp((hi[k]-o[k])/g->h,0.0,n[k]-1.000001)))+1;
  }
  double guard=1e-12+128*std::numeric_limits<double>::epsilon()*grid_bounds(g)->max_abs;
  for(int z=first[2];z<=last[2];++z) for(int y=first[1];y<=last[1];++y)
    for(int x=first[0];x<=last[0];++x)
      if(!(double(h2f(g->v[(size_t(z)*g->ny+y)*g->nx+x]))>threshold+guard)) return 0;
  return 1;
}

// Certify disjoint zero-sublevel volumes, not merely separated mesh samples.
// Source boundary positivity proves that no source solid lies outside its grid.
// Unknown, touching, budget-exhausted and non-grid cases return false.
extern "C" int assemblenist_grid_pair_separated(const mjModel* m,const mjData* d,
    int source,int target,const mjtNum rotation[9],const mjtNum translation[3],
    mjtNum sweep) {
  int instance=m->geom_plugin[source];
  if(instance<0 || !d->plugin_data[instance]) return 0;
  const auto* plugin=mjp_getPluginAtSlot(m->plugin[instance]);
  if(!plugin || std::strcmp(plugin->name,"assemblenist.sdf.grid")) return 0;
  const Grid* g=get_grid(int(reinterpret_cast<Inst*>(d->plugin_data[instance])->a[0]));
  // Immutable boundary minima are amortized across all pairs and threads.
  static std::mutex mutex;
  static std::unordered_map<const Grid*,bool> boundary;
  bool closed;
  {
    std::lock_guard<std::mutex> lock(mutex);
    auto found=boundary.find(g);
    if(found==boundary.end()) {
      double minimum=INFINITY;
      auto at=[&](int x,int y,int z){return double(h2f(g->v[(size_t(z)*g->ny+y)*g->nx+x]));};
      for(int z=0;z<g->nz;++z) for(int y=0;y<g->ny;++y)
        minimum=std::min(minimum,std::min(at(0,y,z),at(g->nx-1,y,z)));
      for(int z=0;z<g->nz;++z) for(int x=0;x<g->nx;++x)
        minimum=std::min(minimum,std::min(at(x,0,z),at(x,g->ny-1,z)));
      for(int y=0;y<g->ny;++y) for(int x=0;x<g->nx;++x)
        minimum=std::min(minimum,std::min(at(x,y,0),at(x,y,g->nz-1)));
      // grid_eval clamps just inside the final face. Cover that tiny layer
      // by requiring all six two-sample boundary slabs positive as well.
      const int n[3]={g->nx,g->ny,g->nz};const double o[3]={g->ox,g->oy,g->oz};
      bool positive=minimum>1e-10;
      for(int axis=0;axis<3 && positive;++axis) for(int side=0;side<2;++side) {
        mjtNum lo[3],hi[3];
        for(int k=0;k<3;++k){lo[k]=o[k];hi[k]=o[k]+(n[k]-1)*g->h;}
        if(side)lo[axis]=o[axis]+(n[axis]-2)*g->h;
        else hi[axis]=o[axis]+g->h;
        positive=assemblenist_grid_box_separated_fine(m,d,source,lo,hi,0)!=0;
        if(!positive)break;
      }
      found=boundary.emplace(g,positive).first;
    }
    closed=found->second;
  }
  if(!closed)return 0;
  struct Box{int lo[3],hi[3];};
  std::vector<Box> stack(1);stack[0]={{0,0,0},{g->nx-1,g->ny-1,g->nz-1}};
  const double o[3]={g->ox,g->oy,g->oz};int visits=0;
  while(!stack.empty()) {
    if(++visits>2048)return 0;
    Box box=stack.back();stack.pop_back();mjtNum lo[3],hi[3],center[3],extent[3];
    for(int k=0;k<3;++k){lo[k]=o[k]+box.lo[k]*g->h;hi[k]=o[k]+box.hi[k]*g->h;center[k]=(lo[k]+hi[k])/2;extent[k]=(hi[k]-lo[k])/2;}
    if(assemblenist_grid_box_separated(m,d,source,lo,hi,0))continue;
    mjtNum other_lo[3],other_hi[3];
    for(int row=0;row<3;++row){double c=translation[row],e=0;
      for(int col=0;col<3;++col){c+=rotation[3*row+col]*center[col];e+=fabs(rotation[3*row+col])*extent[col];}
      other_lo[row]=c-e-sweep;other_hi[row]=c+e+sweep;
    }
    if(assemblenist_grid_box_separated(m,d,target,other_lo,other_hi,0))continue;
    int axis=0;for(int k=1;k<3;++k)if(box.hi[k]-box.lo[k]>box.hi[axis]-box.lo[axis])axis=k;
    if(box.hi[axis]-box.lo[axis]<=1) {
      if(assemblenist_grid_box_separated_fine(m,d,source,lo,hi,0) ||
         assemblenist_grid_box_separated_fine(m,d,target,other_lo,other_hi,0))continue;
      return 0;
    }
    int mid=(box.lo[axis]+box.hi[axis])/2;
    Box right=box;right.lo[axis]=mid;box.hi[axis]=mid;
    stack.push_back(right);stack.push_back(box);
  }
  return 1;
}
extern "C" int assemblenist_grid_domain_radius(const mjModel* m,const mjData* d,int geom,mjtNum radius[3]) {
  int instance=m->geom_plugin[geom];if(instance<0 || !d->plugin_data[instance])return 0;
  const auto* p=mjp_getPluginAtSlot(m->plugin[instance]);
  if(!p || std::strcmp(p->name,"assemblenist.sdf.grid"))return 0;
  const Grid* g=get_grid(int(reinterpret_cast<Inst*>(d->plugin_data[instance])->a[0]));
  const int n[3]={g->nx,g->ny,g->nz};const double o[3]={g->ox,g->oy,g->oz};
  for(int k=0;k<3;++k)radius[k]=std::max(fabs(o[k]),fabs(o[k]+(n[k]-1)*g->h));
  return 1;
}
