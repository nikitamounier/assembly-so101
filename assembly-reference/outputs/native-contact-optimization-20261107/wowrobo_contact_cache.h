// Exact-pose supplement memo, per thread and bounded to 128 recently used pairs.
// Grid SDF callbacks depend only on local position and immutable model grids.
// No contacts are predicted across motion; even a one-bit pose change misses.
#include <array>
#include <cstring>
#include <mujoco/mujoco.h>

struct WowSupplementMemo {
  int first=-1,second=-1;
  mjtNum threshold=0;
  std::array<mjtNum,24> pose{};
  mjPreContact bins[12]{};
  bool valid=false,matches=false;
};
static WowSupplementMemo* wow_supplement_memo(const mjModel* m,const mjData* d,
                                             int first,int second,mjtNum threshold) {
  static thread_local std::array<WowSupplementMemo,128> entries;
  static thread_local const void* buffer=nullptr;
  static thread_local uint64_t signature=0;
  if(buffer!=m->buffer || signature!=m->signature) {
    entries={};buffer=m->buffer;signature=m->signature;
  }
  // Direct mapping bounds memory and never changes correctness on eviction.
  auto& entry=entries[(unsigned(first)*65537u+unsigned(second))%entries.size()];
  std::array<mjtNum,24> pose;
  memcpy(pose.data(),d->geom_xpos+3*first,3*sizeof(mjtNum));
  memcpy(pose.data()+3,d->geom_xmat+9*first,9*sizeof(mjtNum));
  memcpy(pose.data()+12,d->geom_xpos+3*second,3*sizeof(mjtNum));
  memcpy(pose.data()+15,d->geom_xmat+9*second,9*sizeof(mjtNum));
  entry.matches=entry.valid && entry.first==first && entry.second==second &&
    !memcmp(&entry.threshold,&threshold,sizeof(threshold)) &&
    !memcmp(entry.pose.data(),pose.data(),sizeof(pose));
  if(!entry.matches) {
    entry.valid=false;entry.first=first;entry.second=second;
    entry.threshold=threshold;entry.pose=pose;
  }
  return &entry;
}
