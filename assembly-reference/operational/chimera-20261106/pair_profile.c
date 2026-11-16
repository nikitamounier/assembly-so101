#include <mujoco/mujoco.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
static mjfCollision original[mjNGEOMTYPES][mjNGEOMTYPES];
static uint64_t *duration,*calls,*contacts;
static int ngeom,installed,enabled;
static _Thread_local int depth;
static uint64_t now_ns(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return (uint64_t)t.tv_sec*1000000000ULL+t.tv_nsec;}
static int timed(const mjModel*m,mjData*d,mjPreContact*c,int a,int b,mjtNum margin){
  mjfCollision f=original[m->geom_type[a]][m->geom_type[b]];
  int measure=enabled && depth==0;depth++;
  uint64_t begin=measure?now_ns():0;
  int n=f(m,d,c,a,b,margin);
  uint64_t ns=measure?now_ns()-begin:0;depth--;
  if(measure && a<ngeom && b<ngeom){int lo=a<b?a:b,hi=a<b?b:a;size_t i=(size_t)lo*ngeom+hi;
    __atomic_fetch_add(&duration[i],ns,__ATOMIC_RELAXED);__atomic_fetch_add(&calls[i],1,__ATOMIC_RELAXED);
    if(n>0)__atomic_fetch_add(&contacts[i],n,__ATOMIC_RELAXED);
  }return n;
}
int pair_start(int n){
 if(installed)return -1;ngeom=n;size_t size=(size_t)n*n;
 duration=calloc(size,sizeof(uint64_t));calls=calloc(size,sizeof(uint64_t));contacts=calloc(size,sizeof(uint64_t));
 if(!duration||!calls||!contacts)return -2;
 for(int a=0;a<mjNGEOMTYPES;a++)for(int b=0;b<mjNGEOMTYPES;b++){original[a][b]=mjCOLLISIONFUNC[a][b];if(original[a][b])mjCOLLISIONFUNC[a][b]=timed;}
 installed=1;enabled=1;return 0;
}
void pair_reset(void){size_t size=(size_t)ngeom*ngeom;for(size_t i=0;i<size;i++)duration[i]=calls[i]=contacts[i]=0;}
int pair_dump(const char*path){enabled=0;FILE*f=fopen(path,"w");if(!f)return -1;fprintf(f,"geom1,geom2,calls,nanoseconds,contacts\n");for(int a=0;a<ngeom;a++)for(int b=a;b<ngeom;b++){size_t i=(size_t)a*ngeom+b;if(calls[i])fprintf(f,"%d,%d,%llu,%llu,%llu\n",a,b,(unsigned long long)calls[i],(unsigned long long)duration[i],(unsigned long long)contacts[i]);}fclose(f);for(int a=0;a<mjNGEOMTYPES;a++)for(int b=0;b<mjNGEOMTYPES;b++)mjCOLLISIONFUNC[a][b]=original[a][b];return 0;}
