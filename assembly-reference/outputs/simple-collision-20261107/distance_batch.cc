#include <cstddef>
#include <mujoco/mujoco.h>
using Distance = double (*)(const mjModel*, const mjData*, int, const mjtNum[3]);
static_assert(sizeof(mjtNum) == sizeof(double));
extern "C" void proxy_distance_batch(Distance distance, const mjModel* m,
    const mjData* d, int geom, const mjtNum* points, size_t n, mjtNum* output) {
  for (size_t i=0;i<n;++i) output[i]=distance(m,d,geom,points+3*i);
}
