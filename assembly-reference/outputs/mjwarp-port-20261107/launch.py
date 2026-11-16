import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[2]
files = {
    "wowrobo_warp.py": (root / "assembly_policy/so101/wowrobo_warp.py").read_text(),
    "runner.py": (root / "outputs/mjwarp-port-20261107/runner.py").read_text(),
}
code = (
    'from pathlib import Path\nimport subprocess,os,json\np=Path("/large_storage/goodarzilab/nikitamounier/assembly/mjwarp-port-20261107")\nfiles='
    + repr(files)
    + "\nfor n,s in files.items():(p/n).write_text(s)\n"
)
if len(sys.argv) > 2 and sys.argv[2] == "setup":
    code += '''
f=p/'assembly-so101/plugin/so101_contact_search.c'
s=f.read_text()
if 'warp_reference_grid' not in s:
 s+=r"""
extern "C" int warp_reference_grid(const mjModel* m,const mjData* d,int geom,const double* point,double* out) {
 int instance=m->geom_plugin[geom];if(instance<0)return 0;
 const mjpPlugin* plugin=mjp_getPluginAtSlot(m->plugin[instance]);
 out[0]=plugin->sdf_distance(point,d,instance);plugin->sdf_gradient(out+1,point,d,instance);return 1;
}
extern "C" int warp_reference_supplement(const mjModel* m,mjData* d,int a,int b,double margin,double* out) {
 mjPreContact contacts[mjMAXCONPAIR];int n=supplement_mating_surfaces(m,d,contacts,a,b,0,margin);
 for(int i=0;i<n;++i){out[7*i]=contacts[i].dist;for(int k=0;k<3;++k){out[7*i+1+k]=contacts[i].pos[k];out[7*i+4+k]=contacts[i].normal[k];}}return n;
}
"""
 f.write_text(s)
py='/large_storage/goodarzilab/nikitamounier/assembly/mjwarp-h100-20261107/.venv/bin/python'
subprocess.run([py,'-m','pip','install','libigl','trimesh','scipy'],check=True)
'''
if sys.argv[1] != "deploy":
    code += (
        """
py='/large_storage/goodarzilab/nikitamounier/assembly/mjwarp-h100-20261107/.venv/bin/python'
env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',SO101_SURFACE_MODE='1',WOWROBO_WARP_DEBUG=os.environ.get('WOWROBO_WARP_DEBUG',''),CUDA_LAUNCH_BLOCKING=os.environ.get('CUDA_LAUNCH_BLOCKING','0'))
cmd=[py,'-u',str(p/'runner.py')]+"""
        + repr(sys.argv[1].split())
        + """
name='-'.join("""
        + repr(sys.argv[1].split())
        + """).replace('/','_')
with (p/(name+'.log')).open('w') as f:
 r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,env=env,timeout=1600)
print((p/(name+'.log')).read_text()[-22000:]);print('exit',r.returncode)
raise SystemExit(r.returncode)
"""
    )
subprocess.run(
    [
        "python3",
        str(Path(__file__).with_name("remote.py")),
        sys.argv[1].replace(" ", "_"),
    ],
    input=code,
    text=True,
    check=True,
)
