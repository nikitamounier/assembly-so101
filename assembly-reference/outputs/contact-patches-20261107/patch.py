"""Apply isolated, ablatable contact candidates to a private compiled-source tree."""
from pathlib import Path

def apply(native,variant):
 native=Path(native);c=native/'so101_contact_search.c';s=c.read_text()
 if variant in ['fused','combined','winner']:
  p=native/'assemblenist_sdf.cc';g=p.read_text();g=g.replace('const Grid* G, mjtNum* grad) {','const Grid* G, mjtNum* grad, double gradient_threshold=INFINITY) {',1);g=g.replace('  if (grad) {','  if (grad && (gradient_threshold==INFINITY || d+out < gradient_threshold)) {',1)
  g+='''\nextern "C" int assemblenist_grid_contact_sample(const mjModel* m,const mjData* d,int instance,const mjtNum p[3],mjtNum threshold,mjtNum* distance,mjtNum gradient[3]) {
 if(instance<0 || !d->plugin_data[instance])return 0;
 const auto* plugin=mjp_getPluginAtSlot(m->plugin[instance]);
 if(!plugin || strcmp(plugin->name,"assemblenist.sdf.grid"))return 0;
 const Grid* grid=get_grid(int(reinterpret_cast<Inst*>(d->plugin_data[instance])->a[0]));
 *distance=grid_eval(p,grid,gradient,threshold);return 1;
}\n''';p.write_text(g)
  old='  mjtNum distance = plugin->sdf_distance(local, d, instance);\n  if (!isfinite(distance) || distance >= threshold) return;\n  mjtNum gradient[3], normal[3];\n  plugin->sdf_gradient(gradient, local, d, instance);';assert s.count(old)==1
  s=s.replace(old,'''  mjtNum distance,gradient[3],normal[3];
  if(!assemblenist_grid_contact_sample(m,d,instance,local,threshold,&distance,gradient)) {
    distance=plugin->sdf_distance(local,d,instance);
    if(isfinite(distance) && distance<threshold)plugin->sdf_gradient(gradient,local,d,instance);
  }
  if(!isfinite(distance) || distance>=threshold)return;''')
  anchor='static void sample_surface('
  s=s.replace(anchor,'extern "C" int assemblenist_grid_contact_sample(const mjModel*,const mjData*,int,const mjtNum[3],mjtNum,mjtNum*,mjtNum[3]);\n'+anchor,1)
 if variant in ['weld','combined']:
  anchor='  mjfCollision collider = original[m->geom_type[g1]][m->geom_type[g2]];'
  assert s.count(anchor)==1
  s=s.replace(anchor,'''  // mjData activation varies independently between rollout worlds.
  static thread_local const void* attachment_model=nullptr;
  static thread_local uint64_t attachment_signature=0;
  struct Link { int eq,a,b; };
  static thread_local Link links[3];
  if(attachment_model!=m->buffer || attachment_signature!=m->signature) {
    const char* names[3]={"installed_cover_to_base","installed_connector_to_output_horns","installed_servo2_case_to_connector"};
    for(int i=0;i<3;++i){int eq=mj_name2id(m,mjOBJ_EQUALITY,names[i]);links[i]={eq,eq>=0?m->eq_obj1id[eq]:-1,eq>=0?m->eq_obj2id[eq]:-1};}
    attachment_model=m->buffer;attachment_signature=m->signature;
  }
  for(const auto& link:links)if(link.eq>=0 && m->eq_type[link.eq]==mjEQ_WELD && d->eq_active[link.eq]) {
    int a=m->geom_bodyid[g1],b=m->geom_bodyid[g2];
    if((a==link.a && b==link.b)||(b==link.a && a==link.b))return 0;
  }
'''+anchor)
 if variant=='winner':
  # Exact lower bound: a sample cannot win any bin if its distance is at
  # least the largest existing bin depth. Missing bins keep original cutoff.
  anchor='  int instance = m->geom_plugin[target];'
  assert s.count(anchor)>=1
  s=s.replace(anchor,'''  mjtNum largest=-INFINITY;
  for(int bin=6*direction;bin<6*direction+6;++bin)largest=std::max(largest,bins[bin].dist);
  threshold=std::min(threshold,largest);
'''+anchor,1)
 c.write_text(s)
