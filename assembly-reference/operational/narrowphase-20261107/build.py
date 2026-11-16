import os,subprocess,pathlib,sys,json,hashlib
assert os.environ.get('SLURM_JOB_ID')=='3604284'
ROOT=pathlib.Path('/large_storage/goodarzilab/nikitamounier/assembly/source-85c006a')
OUT=ROOT.parent/'narrowphase-20261107';os.chdir(OUT)
import mujoco
MJ=pathlib.Path(mujoco.__file__).parent;LIB=next(MJ.glob('libmujoco.so.*'))
common=['-O2','-g','-fPIC','-I'+str(MJ/'include'),'-I'+str(OUT/'mujoco-3.17.0/src')]
sources=[('gcc',OUT/'engine_collision_sdf.c','sdf'),('gcc',ROOT/'plugin/so101_contact_search.c','contact'),('g++',ROOT/'plugin/assemblenist_sdf.cc','plugin'),('g++',ROOT/'plugin/so101_driver.cc','driver')]
manifest={}
for cc,source,name in sources:
 flags=['-finstrument-functions','-finstrument-functions-exclude-file-list=/usr/include,/include/mujoco']
 if cc=='g++':flags+=['-std=c++17','-mf16c']
 subprocess.run([cc,*common,*flags,'-c',str(source),'-o',name+'.o'],check=True)
 manifest[str(source)]=hashlib.sha256(source.read_bytes()).hexdigest()
subprocess.run(['gcc',*common,'-c',str(OUT/'trace.c'),'-o','trace.o'],check=True)
subprocess.run(['g++','-shared','-Wl,-Bsymbolic','-Wl,-z,defs','sdf.o','contact.o','plugin.o','driver.o','trace.o',str(LIB),'-Wl,-rpath,'+str(MJ),'-o','diagnostic.so'],check=True)
subprocess.run(['gcc','-shared','-Wl,-Bsymbolic','-Wl,-z,defs','sdf.o','trace.o','-lm',str(LIB),'-Wl,-rpath,'+str(MJ),'-o','observer.so'],check=True)
(OUT/'symbols.txt').write_bytes(subprocess.check_output(['nm','-anC','diagnostic.so']))
(OUT/'build.json').write_text(json.dumps(dict(sources=manifest,mujoco=mujoco.__version__,compiler=subprocess.check_output(['gcc','--version'],text=True),flags=common),indent=2))
# A separate coarse-timer adapter keeps the ORIGINAL MuJoCo and plugin binaries.
# No per-function hooks; three timestamps per collider call and plain counters.
source=(ROOT/'plugin/so101_contact_search.c').read_text()
prefix='''
#include <stdint.h>
#include <stdio.h>
#include <time.h>
static uint64_t stages[512][512][4], events[5];
static uint64_t stamp(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return (uint64_t)t.tv_sec*1000000000+t.tv_nsec;}
void stage_reset(void){memset(stages,0,sizeof(stages));memset(events,0,sizeof(events));}
void stage_dump(const char*path){FILE*f=fopen(path,"w");fprintf(f,"a,b,selector_ns,collider_ns,supplement_ns,calls\\n");for(int a=0;a<512;a++)for(int b=a;b<512;b++)if(stages[a][b][3])fprintf(f,"%d,%d,%llu,%llu,%llu,%llu\\n",a,b,(unsigned long long)stages[a][b][0],(unsigned long long)stages[a][b][1],(unsigned long long)stages[a][b][2],(unsigned long long)stages[a][b][3]);fclose(f);}
void stage_events(const char*path){FILE*f=fopen(path,"w");for(int i=0;i<5;i++)fprintf(f,"%llu\\n",(unsigned long long)events[i]);fclose(f);}
'''
source=source.replace('#include <mujoco/mujoco.h>','#include <mujoco/mujoco.h>\n'+prefix)
source=source.replace('  mjtNum world[3], local[3], delta[3];','  events[0]++;\n  mjtNum world[3], local[3], delta[3];')
source=source.replace('if (fabs(local[k] - aabb[k]) > aabb[k+3] + 0.0001) return;','if (fabs(local[k] - aabb[k]) > aabb[k+3] + 0.0001) {events[1]++;return;}')
source=source.replace('  mjtNum distance = plugin->sdf_distance','  events[2]++;\n  mjtNum distance = plugin->sdf_distance')
source=source.replace('  plugin->sdf_gradient(gradient','  events[3]++;\n  plugin->sdf_gradient(gradient')
source=source.replace('  contact->dist = distance;','  events[4]++;\n  contact->dist = distance;')
source=source.replace('  mjfCollision collider = original','  uint64_t t0=stamp();\n  mjfCollision collider = original')
old='''  int count = collider(&view, d, con, g1, g2, margin);
  return supplement_motor_base(m, d, con, g1, g2, count, margin);'''
new='''  uint64_t t1=stamp();
  int count = collider(&view, d, con, g1, g2, margin);
  uint64_t t2=stamp();
  int result=supplement_motor_base(m, d, con, g1, g2, count, margin);
  uint64_t t3=stamp();int lo=g1<g2?g1:g2,hi=g1<g2?g2:g1;
  stages[lo][hi][0]+=t1-t0;stages[lo][hi][1]+=t2-t1;stages[lo][hi][2]+=t3-t2;stages[lo][hi][3]++;
  return result;'''
assert old in source
(OUT/'stage_contact.c').write_text(source.replace(old,new))
subprocess.run(['gcc',*common,'-shared',str(OUT/'stage_contact.c'),str(LIB),'-Wl,-rpath,'+str(MJ),'-o','stage.so'],check=True)
print('BUILD PASS',flush=True)
