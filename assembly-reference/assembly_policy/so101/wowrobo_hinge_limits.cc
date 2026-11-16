// Per-world native joint-limit variations. Never mutate shared mjModel arrays.
#include <mujoco/mujoco.h>
#include <algorithm>
#include <cmath>
#include <vector>
static const mjModel* target=nullptr;
static int joints[2],offset=0;
static double noise_cap=0;
static mjfGeneric previous=nullptr;
static void limit_control(const mjModel* m,mjData* d) {
  if(previous)previous(m,d);  // Preserve native episode-friction rebuilding.
  if(m!=target)return;
  bool changed=false,near=false;
  for(int k=0;k<2;++k){int j=joints[k];double q=d->qpos[m->jnt_qposadr[j]];
    double lo=m->jnt_range[2*j],hi=m->jnt_range[2*j+1];
    double dl=d->userdata[offset+2*k],du=d->userdata[offset+2*k+1];
    if(!std::isfinite(dl)||!std::isfinite(du)||fabs(dl)>noise_cap||fabs(du)>noise_cap)
      mju_error("Invalid per-world hinge limit noise");
    changed|=dl!=0 || du!=0;
    near|=q<std::max(lo,lo+dl)+m->jnt_margin[j] || q>std::min(hi,hi+du)-m->jnt_margin[j];
  }
  if(!changed || !near)return;
  static thread_local std::vector<mjtNum> ranges;
  ranges.assign(m->jnt_range,m->jnt_range+2*m->njnt);
  for(int k=0;k<2;++k){int j=joints[k];ranges[2*j]+=d->userdata[offset+2*k];ranges[2*j+1]+=d->userdata[offset+2*k+1];
    if(!(ranges[2*j]<ranges[2*j+1]))mju_error("Inverted hinge limit interval");}
  mjModel view=*m;view.jnt_range=ranges.data();
  mj_makeConstraint(&view,d);
  // Range values have been materialized into native constraint positions.
  // Use the original model for all remaining callbacks/dynamics.
  mj_island(m,d);mj_projectConstraint(m,d);mj_transmission(m,d);mj_fwdVelocity(m,d);
}
extern "C" int wowrobo_install_hinge_limits(const mjModel* m,const int ids[2],int userdata_offset,double cap) {
  if(target || userdata_offset<0 || m->nuserdata<userdata_offset+4 || !(cap>=0 && cap<.01))return 0;
  for(int k=0;k<2;++k)if(ids[k]<0 || ids[k]>=m->njnt || m->jnt_type[ids[k]]!=mjJNT_HINGE || !m->jnt_limited[ids[k]])return 0;
  target=m;joints[0]=ids[0];joints[1]=ids[1];offset=userdata_offset;noise_cap=cap;
  previous=mjcb_control;mjcb_control=limit_control;return 1;
}
extern "C" void wowrobo_remove_hinge_limits() {
  if(mjcb_control==limit_control)mjcb_control=previous;
  target=nullptr;previous=nullptr;
}
