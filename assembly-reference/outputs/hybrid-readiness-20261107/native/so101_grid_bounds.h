// Conservative rejection helper; it never changes grid_eval or its callbacks.
#include <limits>
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
  // Immutable minima over 8x8x8 grid samples, including boundary blocks.
  b->bx = (g->nx + 7)/8; b->by = (g->ny + 7)/8; b->bz = (g->nz + 7)/8;
  b->block_min.assign(size_t(b->bx)*b->by*b->bz, INFINITY);
  for (int z = 0; z < g->nz; ++z) for (int y = 0; y < g->ny; ++y)
    for (int x = 0; x < g->nx; ++x) {
      double value = h2f(g->v[(size_t(z)*g->ny+y)*g->nx+x]);
      b->max_abs = std::max(b->max_abs, std::fabs(value));
      auto& low = b->block_min[(size_t(z/8)*b->by+y/8)*b->bx+x/8];
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
    first[k] = int(std::floor(std::clamp((lo[k]-o[k])/g->h,0.0,n[k]-1.000001)))/8;
    last[k] = (int(std::floor(std::clamp((hi[k]-o[k])/g->h,0.0,n[k]-1.000001)))+1)/8;
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
