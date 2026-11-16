#include "../../assembly_policy/so101/wowrobo_contact_cache.h"
#include <cassert>
#include <thread>
#include <vector>
static void check() {
  mjModel model{};mjData data{};
  mjtNum pos[9]={},rot[27]={};
  int identity;model.buffer=&identity;model.signature=123;
  data.geom_xpos=pos;data.geom_xmat=rot;
  auto* p=wow_supplement_memo(&model,&data,0,1,.001);
  assert(!p->matches);p->bins[0].dist=.123;p->valid=true;
  p=wow_supplement_memo(&model,&data,0,1,.001);assert(p->matches && p->bins[0].dist==.123);
  pos[3]=1e-16;p=wow_supplement_memo(&model,&data,0,1,.001);assert(!p->matches);p->valid=true;
  p=wow_supplement_memo(&model,&data,0,1,.002);assert(!p->matches);p->valid=true;
  rot[9]=-0.;p=wow_supplement_memo(&model,&data,0,1,.002);assert(!p->matches);p->valid=true;
  p=wow_supplement_memo(&model,&data,1,0,.002);assert(!p->matches);p->valid=true;
  model.signature++;p=wow_supplement_memo(&model,&data,1,0,.002);assert(!p->matches);
}
int main(){std::vector<std::thread> ts;for(int i=0;i<8;++i)ts.emplace_back(check);for(auto& t:ts)t.join();}
