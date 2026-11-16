from pathlib import Path
from patch import apply as fuse

def apply(native):
 native=Path(native);fuse(native,'fused')
 p=native/'assemblenist_sdf.cc';s=p.read_text()+'''
extern "C" const void* assemblenist_grid_contact_handle(const mjModel* m,const mjData* d,int instance) {
 if(instance<0 || !d->plugin_data[instance])return nullptr;
 const auto* plugin=mjp_getPluginAtSlot(m->plugin[instance]);
 if(!plugin || strcmp(plugin->name,"assemblenist.sdf.grid"))return nullptr;
 return get_grid(int(reinterpret_cast<Inst*>(d->plugin_data[instance])->a[0]));
}
extern "C" mjtNum assemblenist_grid_contact_eval(const void* handle,const mjtNum p[3],mjtNum threshold,mjtNum gradient[3]) {
 return grid_eval(p,static_cast<const Grid*>(handle),gradient,threshold);
}
''';p.write_text(s)
 p=native/'so101_contact_search.c';s=p.read_text();needle='static void sample_surface(';s=s.replace(needle,'extern "C" const void* assemblenist_grid_contact_handle(const mjModel*,const mjData*,int);\nextern "C" mjtNum assemblenist_grid_contact_eval(const void*,const mjtNum[3],mjtNum,mjtNum[3]);\n'+needle,1)
 s=s.replace('mjtNum threshold,\n                           mjPreContact bins[12]) {','mjtNum threshold,\n                           mjPreContact bins[12],const void* grid_context=nullptr) {',1)
 a=s.index('  int instance = m->geom_plugin[target];',s.index('static void sample_surface('));b=s.index('  mjtNum length =',a)
 s=s[:a]+'''  mjtNum distance,gradient[3],normal[3];
  if(grid_context)distance=assemblenist_grid_contact_eval(grid_context,local,threshold,gradient);
  else {
    int instance=m->geom_plugin[target];if(instance<0)return;
    const auto* plugin=mjp_getPluginAtSlot(m->plugin[instance]);
    if(!plugin || !plugin->sdf_distance || !plugin->sdf_gradient)return;
    distance=plugin->sdf_distance(local,d,instance);
    if(isfinite(distance) && distance<threshold)plugin->sdf_gradient(gradient,local,d,instance);
  }
  if(!isfinite(distance) || distance>=threshold)return;
'''+s[b:];p.write_text(s)
 p=native/'so101_surface_tree.h';s=p.read_text();anchor='  const SurfaceTree& tree=entry->second;';s=s.replace(anchor,anchor+'\n  const void* grid_context=assemblenist_grid_contact_handle(m,d,m->geom_plugin[target]);',1)
 s=s.replace('direction,threshold,bins);','direction,threshold,bins,grid_context);');p.write_text(s)
