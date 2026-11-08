// Per-world contact friction at the native control hook, before force solving.
#include <mujoco/mujoco.h>
#include <cmath>
#include <vector>
static const mjModel* target = nullptr;
static std::vector<int> roles;
static int offset = 0;
static mjfGeneric previous = nullptr;
static void friction_control(const mjModel* m, mjData* d) {
  if (previous) previous(m,d);
  if (m != target) return;
  const mjtNum* f = d->userdata + offset;
  // Uninitialized work data and nominal episodes retain the original fast path.
  if (!(f[0]>0 && f[1]>0 && f[2]>0) ||
      (f[0]==1 && f[1]==1 && f[2]==1)) return;
  bool changed=false;
  for (int i=0;i<d->ncon;i++) {
    mjContact* c=d->contact+i;
    if (c->geom[0]<0 || c->geom[1]<0) continue;
    int a=roles[c->geom[0]],b=roles[c->geom[1]],k=-1;
    if ((a==1&&b==2)||(a==2&&b==1)) k=0;
    else if ((a==2&&b==3)||(a==3&&b==2)) k=1;
    else if (a==2&&b==2) k=2;
    if (k>=0 && f[k]!=1) {
      c->friction[0]*=f[k];c->friction[1]*=f[k];changed=true;
    }
  }
  if (changed) {
    mj_makeConstraint(m,d);mj_island(m,d);mj_projectConstraint(m,d);
    mj_transmission(m,d);mj_fwdVelocity(m,d);
  }
}
extern "C" int wowrobo_install_friction(const mjModel* m,const int* r,int userdata_offset) {
  if (target || mjcb_control || userdata_offset<0 || m->nuserdata<userdata_offset+3) return 0;
  target=m;roles.assign(r,r+m->ngeom);offset=userdata_offset;
  previous=mjcb_control;mjcb_control=friction_control;return 1;
}
extern "C" void wowrobo_remove_friction() {
  if (mjcb_control==friction_control) mjcb_control=previous;
  target=nullptr;roles.clear();previous=nullptr;
}
