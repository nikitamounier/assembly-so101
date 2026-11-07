#include <mujoco/mujoco.h>
#include <stdint.h>
#include <stdio.h>
#include <time.h>
#include <string.h>
#include <stdlib.h>
// Synchronous compiler hooks: no signals, ptrace, or asynchronous unwinding.
// Run one simulation thread. Instrument one out of every 64 calls per pair.
#define NODES 32768
typedef struct {int parent; void *fn; uint64_t ns, calls;} Node;
static Node nodes[NODES]; static int nn=1,enabled=0;
static int table[131072];
static int depth, stack[256]; static uint64_t starts[256];
static uint64_t count[512][512],total[512][512],contacts[512][512],selected[512][512];
static mjfCollision original[mjNGEOMTYPES][mjNGEOMTYPES],stock[mjNGEOMTYPES][mjNGEOMTYPES];
static const mjModel *model;
static unsigned stride=64;
void np_stride(unsigned value){stride=value?value:64;}
static uint64_t now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return (uint64_t)t.tv_sec*1000000000+t.tv_nsec;}
static int node(int parent,void*fn){unsigned h=(((uintptr_t)fn>>3)*2654435761u+parent*97u)&131071;while(table[h]){int i=table[h];if(nodes[i].parent==parent&&nodes[i].fn==fn)return i;h=(h+1)&131071;}if(nn==NODES)abort();int i=nn++;table[h]=i;nodes[i]=(Node){.parent=parent,.fn=fn};return i;}
static void enter(void*fn){if(depth>=256)abort();int n=node(depth?stack[depth-1]:0,fn);stack[depth]=n;starts[depth++]=now();nodes[n].calls++;}
static void leave(void){uint64_t t=now();if(depth<1)abort();--depth;nodes[stack[depth]].ns+=t-starts[depth];}
void __cyg_profile_func_enter(void *fn,void*caller){if(enabled)enter(fn);}
void __cyg_profile_func_exit(void *fn,void*caller){if(enabled)leave();}
// The upstream internal accessor is hidden in the wheel. Same registered plugin,
// accessed through its public checked API in this diagnostic copy only.
const mjpPlugin* mjp_getPluginAtSlotUnsafe(int slot,int nslot){return mjp_getPluginAtSlot(slot);}
extern int mjc_SDF(const mjModel*,mjData*,mjPreContact*,int,int,mjtNum);
extern int mjc_MeshSDF(const mjModel*,mjData*,mjPreContact*,int,int,mjtNum);
extern int mjc_HFieldSDF(const mjModel*,mjData*,mjPreContact*,int,int,mjtNum);
void np_engine(void){
 memcpy(stock,mjCOLLISIONFUNC,sizeof(stock));
 for(int a=0;a<mjNGEOMTYPES;a++)for(int b=0;b<mjNGEOMTYPES;b++)if(stock[a][b]&&a!=mjGEOM_PLANE&&b!=mjGEOM_PLANE&&(a==mjGEOM_SDF||b==mjGEOM_SDF)){
  mjCOLLISIONFUNC[a][b]=(a==mjGEOM_MESH||b==mjGEOM_MESH)?mjc_MeshSDF:(a==mjGEOM_HFIELD||b==mjGEOM_HFIELD)?mjc_HFieldSDF:mjc_SDF;
 }
}
static int wrapped(const mjModel*m,mjData*d,mjPreContact*c,int a,int b,mjtNum margin){
 int lo=a<b?a:b,hi=a<b?b:a;uint64_t call=++count[lo][hi];
 int sample=call%stride==1;uint64_t t=now();
 if(sample){enabled=1;enter((void*)(uintptr_t)(1+lo*512+hi));selected[lo][hi]++;}
 int n=original[m->geom_type[a]][m->geom_type[b]](m,d,c,a,b,margin);
 if(sample){leave();enabled=0;}
 total[lo][hi]+=now()-t;contacts[lo][hi]+=n;return n;
}
void np_start(const mjModel*m){model=m;if(m->ngeom>=512)abort();memcpy(original,mjCOLLISIONFUNC,sizeof(original));for(int a=0;a<mjNGEOMTYPES;a++)for(int b=0;b<mjNGEOMTYPES;b++)if(original[a][b])mjCOLLISIONFUNC[a][b]=wrapped;}
void np_stop(void){memcpy(mjCOLLISIONFUNC,original,sizeof(original));}
void np_stock(void){memcpy(mjCOLLISIONFUNC,stock,sizeof(stock));}
void np_reset(void){memset(nodes,0,sizeof(nodes));memset(table,0,sizeof(table));nn=1;depth=enabled=0;memset(count,0,sizeof(count));memset(total,0,sizeof(total));memset(contacts,0,sizeof(contacts));memset(selected,0,sizeof(selected));}
void np_dump(const char*path){FILE*f=fopen(path,"w");if(!f)abort();fprintf(f,"node,parent,address,nanoseconds,calls\n");for(int i=1;i<nn;i++)fprintf(f,"%d,%d,%p,%llu,%llu\n",i,nodes[i].parent,nodes[i].fn,(unsigned long long)nodes[i].ns,(unsigned long long)nodes[i].calls);fclose(f);}
void np_pairs(const char*path){FILE*f=fopen(path,"w");if(!f)abort();fprintf(f,"a,b,calls,nanoseconds,contacts,selected\n");for(int a=0;a<model->ngeom;a++)for(int b=a;b<model->ngeom;b++)if(count[a][b])fprintf(f,"%d,%d,%llu,%llu,%llu,%llu\n",a,b,(unsigned long long)count[a][b],(unsigned long long)total[a][b],(unsigned long long)contacts[a][b],(unsigned long long)selected[a][b]);fclose(f);}
uintptr_t np_base(void){extern char __ehdr_start;return (uintptr_t)&__ehdr_start;}
