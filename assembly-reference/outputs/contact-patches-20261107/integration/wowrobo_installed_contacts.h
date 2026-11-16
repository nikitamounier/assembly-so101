// Opt-in per-world installed-interface filter; insertion contacts remain active.
static bool wowrobo_skip_installed_contacts(const mjModel* m,const mjData* d,int g1,int g2) {
  // mjData activation varies independently between rollout worlds.
  static thread_local const void* attachment_model=nullptr;
  static thread_local uint64_t attachment_signature=0;
  struct Link { int eq,a,b; };
  static thread_local Link links[3];
  if(attachment_model!=m->buffer || attachment_signature!=m->signature) {
    int flag=mj_name2id(m,mjOBJ_NUMERIC,"wowrobo_installed_contact_filter");
    bool enabled=flag>=0 && m->numeric_data[m->numeric_adr[flag]]!=0;
    const char* names[3]={"installed_cover_to_base","installed_connector_to_output_horns","installed_servo2_case_to_connector"};
    for(int i=0;i<3;++i){int eq=enabled?mj_name2id(m,mjOBJ_EQUALITY,names[i]):-1;links[i]={eq,eq>=0?m->eq_obj1id[eq]:-1,eq>=0?m->eq_obj2id[eq]:-1};}
    attachment_model=m->buffer;attachment_signature=m->signature;
  }
  for(const auto& link:links)if(link.eq>=0 && m->eq_type[link.eq]==mjEQ_WELD && d->eq_active[link.eq]) {
    int a=m->geom_bodyid[g1],b=m->geom_bodyid[g2];
    if((a==link.a && b==link.b)||(b==link.a && a==link.b))return true;
  }
  return false;
}
