// Experimental ordinary-manifold continuation. Supplements always run afterwards.
#include <array>
struct Continuation {
 const mjData* data=nullptr; const void* buffer=nullptr; uint64_t signature=0;
 int a=-1,b=-1,n=0,age=0; mjtNum time=-1,margin=0;
 mjtNum pose[24]{}, anchors[mjMAXCONPAIR][2][3]{};
};
static thread_local std::array<Continuation,257> continuation;
static void local_point(const mjData*d,int g,const mjtNum*w,mjtNum*p) {
 mjtNum t[3];mju_sub3(t,w,d->geom_xpos+3*g);mju_mulMatTVec3(p,d->geom_xmat+9*g,t);
}
static void world_point(const mjData*d,int g,const mjtNum*p,mjtNum*w) {
 mju_mulMatVec3(w,d->geom_xmat+9*g,p);mju_addTo3(w,d->geom_xpos+3*g);
}
static int continue_contacts(mjfCollision collider,const mjModel*m,mjData*d,mjPreContact*con,int a,int b,mjtNum margin) {
 auto&e=continuation[(uintptr_t(d)/64+unsigned(a)*65537u+unsigned(b))%continuation.size()];
 const void* ha=assemblenist_grid_contact_handle(m,d,m->geom_plugin[a]);
 const void* hb=assemblenist_grid_contact_handle(m,d,m->geom_plugin[b]);
 if(!ha || !hb)return collider(m,d,con,a,b,margin);
 mjtNum pose[24];memcpy(pose,d->geom_xpos+3*a,3*sizeof(mjtNum));memcpy(pose+3,d->geom_xmat+9*a,9*sizeof(mjtNum));
 memcpy(pose+12,d->geom_xpos+3*b,3*sizeof(mjtNum));memcpy(pose+15,d->geom_xmat+9*b,9*sizeof(mjtNum));
 bool use=e.data==d && e.buffer==m->buffer && e.signature==m->signature && e.a==a && e.b==b && e.margin==margin && e.n>0 && e.age<7 && d->time>e.time && d->time-e.time<=m->opt.timestep*1.01;
 // Conservative motion trigger (absolute motion intentionally over-invalidates).
 for(int i=0;i<24 && use;++i)if(fabs(pose[i]-e.pose[i])>((i<3 || (i>=12&&i<15))?.00005:.001))use=false;
 int n=0;
 if(use)for(int i=0;i<e.n;++i) {
  mjtNum wa[3],wb[3],w[3],pa[3],pb[3],ga[3],gb[3],na[3],nb[3];
  world_point(d,a,e.anchors[i][0],wa);world_point(d,b,e.anchors[i][1],wb);
  for(int k=0;k<3;++k)w[k]=(wa[k]+wb[k])*.5;
  mjtNum da=0,db=0;
  for(int it=0;it<6;++it) {
   local_point(d,a,w,pa);local_point(d,b,w,pb);
   da=assemblenist_grid_contact_eval(ha,pa,INFINITY,ga);db=assemblenist_grid_contact_eval(hb,pb,INFINITY,gb);
   mju_mulMatVec3(na,d->geom_xmat+9*a,ga);mju_mulMatVec3(nb,d->geom_xmat+9*b,gb);
   mjtNum v[3];mju_sub3(v,na,nb);mjtNum norm=mju_dot3(v,v);
   if(!isfinite(da+db+norm) || norm<1e-12){use=false;break;}
   if(fabs(da-db)<1e-9)break;
   mjtNum scale=(da-db)/norm;for(int k=0;k<3;++k)w[k]-=scale*v[k];
  }
  local_point(d,a,w,pa);local_point(d,b,w,pb);
  da=assemblenist_grid_contact_eval(ha,pa,INFINITY,ga);db=assemblenist_grid_contact_eval(hb,pb,INFINITY,gb);
  mju_mulMatVec3(na,d->geom_xmat+9*a,ga);mju_mulMatVec3(nb,d->geom_xmat+9*b,gb);
  mju_normalize3(na);mju_normalize3(nb);mjtNum normal[3];mju_sub3(normal,na,nb);
  if(!use || !isfinite(da+db) || fabs(da-db)>1e-7 || fmax(da,db)>margin || mju_normalize3(normal)<1e-10){use=false;break;}
  con[n]={};con[n].dist=fmax(da,db);mju_copy3(con[n].pos,w);mju_copy3(con[n].normal,normal);++n;
 }
 if(!use){n=collider(m,d,con,a,b,margin);e.age=0;}else{
  // Fresh ordinary seeds detect additional patches between periodic full refreshes.
  mjModel explore=*m;explore.opt.sdf_initpoints=4;explore.opt.sdf_iterations=6;
  mjPreContact fresh[mjMAXCONPAIR];int nf=collider(&explore,d,fresh,a,b,margin);
  for(int i=0;i<nf && n<mjMAXCONPAIR;++i){bool duplicate=false;for(int j=0;j<n;++j)if(mju_dist3(con[j].pos,fresh[i].pos)<1e-5)duplicate=true;if(!duplicate)con[n++]=fresh[i];}
  ++e.age;
 }
 e.data=d;e.buffer=m->buffer;e.signature=m->signature;e.a=a;e.b=b;e.margin=margin;e.time=d->time;e.n=n;memcpy(e.pose,pose,sizeof(pose));
 for(int i=0;i<n;++i){local_point(d,a,con[i].pos,e.anchors[i][0]);local_point(d,b,con[i].pos,e.anchors[i][1]);}
 return n;
}
