#include <string>
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
  if(!e.internal)return false;
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

// MuJoCo 4.0: reserve mjMAXCONPAIR through the model option, then use
// per-model search budgets inside the native collider. No preload/interposition.
#include <math.h>
#include <chrono>
static thread_local bool profile_surface=false;
static thread_local double surface_stats[7]={};
extern "C" void so101_surface_stats(double* out, int reset) {
  for(int i=0;i<7;++i) { out[i]=surface_stats[i]; if(reset) surface_stats[i]=0; }
  profile_surface=true;
}
#include <string.h>
#include <mujoco/mujoco.h>

static mjfCollision original[mjNGEOMTYPES][mjNGEOMTYPES];
static int installed;

extern "C" int assembly_contact_capacity_guard(void) { return 4; }

static int finger(const char* name) {
  return name && (strstr(name, "wrist_roll_follower_so101_v1_collision_sdf") ||
                  strstr(name, "moving_jaw_so101_v1_collision_sdf"));
}

static int free_object(const mjModel* m, int geom) {
  int root = m->body_rootid[m->geom_bodyid[geom]];
  return root > 0 && m->body_jntnum[root] > 0 &&
         m->jnt_type[m->body_jntadr[root]] == mjJNT_FREE;
}

// Reconstruct support continuously within the ordinary native contact margin.
// Pair impedance supplies compliance; a penetration-gap switch would make
// supporting contacts appear and disappear with a finite force jump.
static void sample_surface(const mjModel* m, const mjData* d, int visual, int target,
                           const mjtNum point[3], int direction, mjtNum threshold,
                           mjPreContact bins[12]) {
  if(profile_surface) surface_stats[4]++;
  mjtNum world[3], local[3], delta[3];
  mju_mulMatVec3(world, d->geom_xmat + 9*visual, point);
  mju_addTo3(world, d->geom_xpos + 3*visual);
  mju_sub3(delta, world, d->geom_xpos + 3*target);
  mju_mulMatTVec3(local, d->geom_xmat + 9*target, delta);
  const mjtNum* aabb = m->geom_aabb + 6*target;
  for (int k = 0; k < 3; k++) {
    if (fabs(local[k] - aabb[k]) > aabb[k+3] + 0.0001) return;
  }
  int instance = m->geom_plugin[target];
  if (instance < 0) return;
  const mjpPlugin* plugin = mjp_getPluginAtSlot(m->plugin[instance]);
  if (!plugin || !plugin->sdf_distance || !plugin->sdf_gradient) return;
  mjtNum distance = plugin->sdf_distance(local, d, instance);
  if (!isfinite(distance) || distance >= threshold) return;
  mjtNum gradient[3], normal[3];
  plugin->sdf_gradient(gradient, local, d, instance);
  mjtNum length = mju_normalize3(gradient);
  if (!isfinite(length) || length < 1e-10) return;
  mju_mulMatVec3(normal, d->geom_xmat + 9*target, gradient);

  // Keep the deepest sample for each signed normal axis, in both directions.
  // The existing collider retains its ordinary manifold; these are supplements.
  int axis = 0;
  for (int k = 1; k < 3; k++) {
    if (fabs(normal[k]) > fabs(normal[axis])) axis = k;
  }
  int bin = 6*direction + 2*axis + (normal[axis] > 0);
  if (distance >= bins[bin].dist) return;
  mjPreContact* contact = bins + bin;
  contact->dist = distance;
  mjtNum sign = direction == 0 ? 1.0 : -1.0;
  for (int k = 0; k < 3; k++) {
    contact->pos[k] = world[k] - .5*distance*normal[k];
    contact->normal[k] = sign*normal[k];
    contact->tangent[k] = 0;
  }
}

#include "so101_surface_tree.h"

static int supplement_mating_surfaces(const mjModel* m, mjData* d, mjPreContact* con,
                                  int g1, int g2, int count, mjtNum margin) {
  const char* a = mj_id2name(m, mjOBJ_GEOM, g1);
  const char* b = mj_id2name(m, mjOBJ_GEOM, g2);
  if (!a || !b) return count;
  // WowRobo routing only: sampling/binning/culling below is unchanged.
  int forward = 0, reverse = 0;
  int first_visual = -1, second_visual = -1;
  const char* suffix = "__contact";
  size_t na = strlen(a), nb = strlen(b), ns = strlen(suffix);
  int generic_flag = mj_name2id(m, mjOBJ_NUMERIC, "wowrobo_surface_contact_guard");
  if (generic_flag >= 0 && m->numeric_data[m->numeric_adr[generic_flag]] != 0 &&
      na > ns && nb > ns && !strcmp(a+na-ns,suffix) && !strcmp(b+nb-ns,suffix)) {
    std::string av(a,na-ns), bv(b,nb-ns);
    first_visual = mj_name2id(m,mjOBJ_GEOM,av.c_str());
    second_visual = mj_name2id(m,mjOBJ_GEOM,bv.c_str());
    forward = 1;
  } else {
  forward = !strcmp(a, "base_contact") && !strcmp(b, "motor_contact");
  reverse = !strcmp(a, "motor_contact") && !strcmp(b, "base_contact");
  const char* first_name = "base_visual";
  const char* second_name = "motor_visual";
  const char* flag_name = "so101_material_contact_guard";
  if (!forward && !reverse) {
    forward = !strcmp(a, "motor_contact") && !strcmp(b, "horn_contact");
    reverse = !strcmp(a, "horn_contact") && !strcmp(b, "motor_contact");
    first_name = "motor_visual";
    second_name = "horn_visual";
    flag_name = "so101_horn_contact_guard";
  }
  if (!forward && !reverse) return count;
  int flag = mj_name2id(m, mjOBJ_NUMERIC, flag_name);
  if (flag < 0 || m->numeric_data[m->numeric_adr[flag]] == 0) return count;
  first_visual = mj_name2id(m, mjOBJ_GEOM, first_name);
  second_visual = mj_name2id(m, mjOBJ_GEOM, second_name);
  if (first_visual < 0 || second_visual < 0) return count;
  }
  if (first_visual < 0 || second_visual < 0) return count;
  mjtNum threshold = margin;
  mjPreContact bins[12];
  memset(bins, 0, sizeof(bins));
  for (int i = 0; i < 12; i++) bins[i].dist = 1e30;
  WowSupplementMemo* memo = wow_supplement_memo(m,d,g1,g2,threshold);
  if (memo->matches && surface_mode != 2) {
    memcpy(bins,memo->bins,sizeof(bins));
  } else {
  for (int direction = 0; direction < 2; direction++) {
    int visual = (direction == 0) == forward ? second_visual : first_visual;
    int target = direction == 0 ? g1 : g2;
    if (surface_mode != 0) {
      tree_surface(m,d,visual,target,direction,threshold,bins);
      continue;
    }
    int mesh = m->geom_dataid[visual];
    int vertices = m->mesh_vertadr[mesh], faces = m->mesh_faceadr[mesh];
    for (int i = 0; i < m->mesh_vertnum[mesh]; i++) {
      mjtNum point[3];
      for (int k = 0; k < 3; k++) point[k] = m->mesh_vert[3*(vertices+i)+k];
      sample_surface(m, d, visual, target, point, direction, threshold, bins);
    }
    for (int i = 0; i < m->mesh_facenum[mesh]; i++) {
      mjtNum point[3] = {0};
      for (int j = 0; j < 3; j++) {
        for (int k = 0; k < 3; k++) {
          point[k] += m->mesh_vert[3*(vertices+m->mesh_face[3*(faces+i)+j])+k]/3;
        }
      }
      sample_surface(m, d, visual, target, point, direction, threshold, bins);
    }
  }
  memcpy(memo->bins,bins,sizeof(bins));
  memo->valid=true;
  }
  for (int i = 0; i < 12; i++) {
    if (bins[i].dist >= threshold) continue;
    if (count < mjMAXCONPAIR) {
      con[count++] = bins[i];
    } else {
      int shallow = 0;
      for (int j = 1; j < count; j++) {
        if (con[j].dist > con[shallow].dist) shallow = j;
      }
      if (bins[i].dist < con[shallow].dist) con[shallow] = bins[i];
    }
  }
  return count;
}

static int contact_search(const mjModel* m, mjData* d, mjPreContact* con,
                          int g1, int g2, mjtNum margin) {
  mjfCollision collider = original[m->geom_type[g1]][m->geom_type[g2]];
  int id = mj_name2id(m, mjOBJ_NUMERIC, "so101_sdf_search");
  if (id < 0) return collider(m, d, con, g1, g2, margin);
  const char* a = mj_id2name(m, mjOBJ_GEOM, g1);
  const char* b = mj_id2name(m, mjOBJ_GEOM, g2);
  int wow_helper_role = 0;
  int wow = mj_name2id(m, mjOBJ_NUMERIC, "wowrobo_surface_contact_guard");
  if (wow >= 0 && a && b && strstr(a,"__contact") && strstr(b,"__contact")) {
    int assembly = mj_name2id(m,mjOBJ_NUMERIC,"so101_assembly_search");
    if (assembly >= 0) id=assembly;
    // Preserve explicit mating/passive-stop budgets. Helpers have separate
    // grasp and hardware budgets; original target-target mating stays 40x40.
    int helper_a=strstr(a,"physical_helper_")!=nullptr;
    int helper_b=strstr(b,"physical_helper_")!=nullptr;
    int role=mj_name2id(m,mjOBJ_NUMERIC,
        helper_a && helper_b ? "wowrobo_hardware_search" : "wowrobo_grasp_search");
    bool explicit_pair=false;
    for(int pair=0;pair<m->npair;++pair)
      explicit_pair|=(m->pair_geom1[pair]==g1 && m->pair_geom2[pair]==g2) ||
                     (m->pair_geom1[pair]==g2 && m->pair_geom2[pair]==g1);
    if ((helper_a || helper_b) && !explicit_pair) {
      if(role>=0) id=role;
      else wow_helper_role=helper_a && helper_b ? 2 : 1;
    }
  } else
  if (a && b && ((!strcmp(a, "base_contact") && !strcmp(b, "motor_contact")) ||
                 (!strcmp(a, "motor_contact") && !strcmp(b, "base_contact"))))
    id = mj_name2id(m, mjOBJ_NUMERIC, "so101_motor_base_search");
  else if (a && b && ((!strcmp(a, "horn_contact") && !strcmp(b, "motor_contact")) ||
                      (!strcmp(a, "motor_contact") && !strcmp(b, "horn_contact")))) {
    int horn = mj_name2id(m, mjOBJ_NUMERIC, "so101_horn_search");
    if (horn >= 0) id = horn;
  }
  else if ((finger(a) && free_object(m, g2)) || (finger(b) && free_object(m, g1))) {
    int grasp = mj_name2id(m, mjOBJ_NUMERIC, "so101_grasp_search");
    if (grasp >= 0) id = grasp;
  }
  else if (free_object(m, g1) && free_object(m, g2)) {
    int assembly = mj_name2id(m, mjOBJ_NUMERIC, "so101_assembly_search");
    if (assembly >= 0) id = assembly;
  }
  const mjtNum* values = m->numeric_data + m->numeric_adr[id];
  mjModel view = *m;  // Only local options change; shared arrays are read-only.
  view.opt.sdf_initpoints = (int)values[0];
  view.opt.sdf_iterations = (int)values[1];
  if (wow_helper_role) {
    view.opt.sdf_initpoints = 40;
    view.opt.sdf_iterations = 40;
  }

  auto begin = profile_surface ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
  int count = surface_mode!=2 && wow_sdf_pair_clear(m,d,g1,g2) ? 0 : collider(&view, d, con, g1, g2, margin);
  auto middle = profile_surface ? std::chrono::steady_clock::now() : begin;
  count = supplement_mating_surfaces(m, d, con, g1, g2, count, margin);
  if(profile_surface) {
    surface_stats[0]++;
    surface_stats[1]+=std::chrono::duration<double>(middle-begin).count();
    surface_stats[2]+=std::chrono::duration<double>(std::chrono::steady_clock::now()-middle).count();
  }
  return count;
}

extern "C" int so101_install_contact_search(void) {
  if (installed) return 4;
  for (int a = 0; a < mjNGEOMTYPES; ++a)
    for (int b = 0; b < mjNGEOMTYPES; ++b)
      original[a][b] = mjCOLLISIONFUNC[a][b];
  for (int a = 0; a < mjNGEOMTYPES; ++a)
    for (int b = 0; b < mjNGEOMTYPES; ++b)
      if ((a == mjGEOM_SDF || b == mjGEOM_SDF) && original[a][b])
        mjCOLLISIONFUNC[a][b] = contact_search;
  installed = 1;
  return 4;
}

extern "C" double wowrobo_sdf_distance(const mjModel* m, const mjData* d, int geom, const mjtNum world[3]) {
  mjtNum local[3], delta[3];
  mju_sub3(delta,world,d->geom_xpos+3*geom);
  mju_mulMatTVec3(local,d->geom_xmat+9*geom,delta);
  int instance=m->geom_plugin[geom];
  if (instance<0) return NAN;
  const mjpPlugin* p=mjp_getPluginAtSlot(m->plugin[instance]);
  return p && p->sdf_distance ? p->sdf_distance(local,d,instance) : NAN;
}
