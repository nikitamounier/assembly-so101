// MuJoCo 3.17: reserve mjMAXCONPAIR through the model option, then use
// per-model search budgets inside the native collider. No preload/interposition.
#include <math.h>
#include <string.h>
#include <mujoco/mujoco.h>

static mjfCollision original[mjNGEOMTYPES][mjNGEOMTYPES];
static int installed;

int assembly_contact_capacity_guard(void) { return 4; }

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

static int supplement_motor_base(const mjModel* m, mjData* d, mjPreContact* con,
                                  int g1, int g2, int count, mjtNum margin) {
  int flag = mj_name2id(m, mjOBJ_NUMERIC, "so101_material_contact_guard");
  if (flag < 0 || m->numeric_data[m->numeric_adr[flag]] == 0) return count;
  const char* a = mj_id2name(m, mjOBJ_GEOM, g1);
  const char* b = mj_id2name(m, mjOBJ_GEOM, g2);
  if (!a || !b) return count;
  int forward = !strcmp(a, "base_contact") && !strcmp(b, "motor_contact");
  int reverse = !strcmp(a, "motor_contact") && !strcmp(b, "base_contact");
  if (!forward && !reverse) return count;
  int base_visual = mj_name2id(m, mjOBJ_GEOM, "base_visual");
  int motor_visual = mj_name2id(m, mjOBJ_GEOM, "motor_visual");
  if (base_visual < 0 || motor_visual < 0) return count;
  mjtNum threshold = margin;
  mjPreContact bins[12];
  memset(bins, 0, sizeof(bins));
  for (int i = 0; i < 12; i++) bins[i].dist = 1e30;
  for (int direction = 0; direction < 2; direction++) {
    int visual = (direction == 0) == forward ? motor_visual : base_visual;
    int target = direction == 0 ? g1 : g2;
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
  int count = collider(&view, d, con, g1, g2, margin);
  return supplement_motor_base(m, d, con, g1, g2, count, margin);
}

int so101_install_contact_search(void) {
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
