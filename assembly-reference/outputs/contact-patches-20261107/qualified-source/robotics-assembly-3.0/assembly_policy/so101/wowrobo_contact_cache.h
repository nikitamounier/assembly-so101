// Exact-pose supplement memo, per thread and bounded to 128 recently used pairs.
// Grid SDF callbacks depend only on local position and immutable model grids.
// No contacts are predicted across motion; even a one-bit pose change misses.
#include <array>
#include <limits>
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

#include <cmath>
#include <algorithm>
extern "C" int assemblenist_grid_pair_separated(const mjModel*,const mjData*,int,int,
    const mjtNum[9],const mjtNum[3],mjtNum);
extern "C" int assemblenist_grid_domain_radius(const mjModel*,const mjData*,int,mjtNum[3]);
struct WowHingeCertificate {
  int first=-1,second=-1,cooldown=0;
  bool clear=false,internal=false;
  mjtNum rotation[9]{},translation[3]{},radius[3]{};
};
static thread_local unsigned long long hinge_counts[4]={};
extern "C" void wow_hinge_certificate_stats(unsigned long long out[4]) {
  std::copy(hinge_counts,hinge_counts+4,out);
}
static bool wow_sdf_pair_clear(const mjModel* m,const mjData* d,int first,int second) {
  static thread_local std::array<WowHingeCertificate,127> entries;
  static thread_local const void* buffer=nullptr;
  static thread_local uint64_t signature=0;
  if(buffer!=m->buffer || signature!=m->signature){entries={};buffer=m->buffer;signature=m->signature;}
  auto& e=entries[(unsigned(first)*73856093u ^ unsigned(second)*19349663u)%127];
  if(e.first!=first || e.second!=second) {
    e={};e.first=first;e.second=second;
    for(int i=0;i<m->npair;++i)if((m->pair_geom1[i]==first && m->pair_geom2[i]==second) ||
                                  (m->pair_geom1[i]==second && m->pair_geom2[i]==first)) {
      const char* name=mj_id2name(m,mjOBJ_PAIR,i);
      e.internal=name && !strncmp(name,"passive_motor_stop_",19);
    }
    if(e.internal && !assemblenist_grid_domain_radius(m,d,first,e.radius))e.internal=false;
  }
  // The same conservative grid proof applies to external pairs too.
  if(!e.internal) { if(!assemblenist_grid_domain_radius(m,d,first,e.radius))return false; }
  mjtNum rotation[9],translation[3],delta[3];
  mju_sub3(delta,d->geom_xpos+3*first,d->geom_xpos+3*second);
  mju_mulMatTVec3(translation,d->geom_xmat+9*second,delta);
  for(int r=0;r<3;++r)for(int c=0;c<3;++c){rotation[3*r+c]=0;
    for(int k=0;k<3;++k)rotation[3*r+c]+=d->geom_xmat[9*second+3*k+r]*d->geom_xmat[9*first+3*k+c];}
  double scale=1;for(int k=0;k<3;++k)scale+=e.radius[k]+fabs(d->geom_xpos[3*first+k])+fabs(d->geom_xpos[3*second+k]);
  const double guard=1e-12+2048*std::numeric_limits<mjtNum>::epsilon()*scale;
  constexpr double sweep=.00005;
  bool covered=e.clear && guard<sweep/4;
  for(int r=0;r<3 && covered;++r){double move=fabs(translation[r]-e.translation[r]);
    for(int c=0;c<3;++c)move+=fabs(rotation[3*r+c]-e.rotation[3*r+c])*e.radius[c];
    covered=move+guard<sweep;}
  if(covered){hinge_counts[0]++;return true;}
  e.clear=false;
  if(e.cooldown>0){--e.cooldown;return false;}
  if(guard>=sweep/4)return false;
  hinge_counts[1]++;
  e.clear=assemblenist_grid_pair_separated(m,d,first,second,rotation,translation,sweep+guard)!=0;
  if(!e.clear){hinge_counts[3]++;e.cooldown=63;return false;}
  hinge_counts[2]++;std::copy(rotation,rotation+9,e.rotation);std::copy(translation,translation+3,e.translation);
  return true;
}
