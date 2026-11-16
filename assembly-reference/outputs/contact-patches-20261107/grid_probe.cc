#include <mujoco/mujoco.h>
#include <cstring>
#include <cmath>
extern "C" const void* assemblenist_grid_contact_handle(const mjModel*,const mjData*,int);
extern "C" mjtNum assemblenist_grid_contact_eval(const void*,const mjtNum[3],mjtNum,mjtNum[3]);
extern "C" int compare_grid_samples(const mjModel* m,const mjData* d,int instance,const mjtNum* points,int count,mjtNum threshold) {
 const auto* plugin=mjp_getPluginAtSlot(m->plugin[instance]);
 const void* handle=assemblenist_grid_contact_handle(m,d,instance);if(!handle)return -1;
 for(int i=0;i<count;++i){const mjtNum* p=points+3*i;mjtNum reference=plugin->sdf_distance(p,d,instance),grad[3],actual_grad[3];
   mjtNum actual=assemblenist_grid_contact_eval(handle,p,threshold,actual_grad);
   if(std::memcmp(&reference,&actual,sizeof(mjtNum)))return 1+2*i;
   if(std::isfinite(reference)&&reference<threshold){plugin->sdf_gradient(grad,p,d,instance);if(std::memcmp(grad,actual_grad,3*sizeof(mjtNum)))return 2+2*i;}
 }
 return 0;
}
