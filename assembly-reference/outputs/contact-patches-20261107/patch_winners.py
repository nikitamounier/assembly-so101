"""Conservative seeded, chunk-level branch-and-bound for unchanged contact bins."""
from pathlib import Path

def apply(native):
 native=Path(native);c=native/'so101_contact_search.c';s=c.read_text();a=s.index('static void sample_surface(');b=s.index('#include "so101_surface_tree.h"',a);f=s[a:b].replace('static void sample_surface(', 'static int sample_surface(').replace('return;', 'return -1;');at=f.rfind('}');f=f[:at]+'  return bin;\n'+f[at:];s=s[:a]+f+s[b:];c.write_text(s)
 p=native/'so101_surface_tree.h';h=p.read_text();h=h.replace('  std::vector<int> ids;','  std::vector<int> ids;\n  std::vector<SurfaceNode> chunks;\n  std::array<int,6> winners{{-1,-1,-1,-1,-1,-1}};',1)
 anchor='static void tree_surface('
 helper='''static void evaluate_winning_chunks(const mjModel* m,const mjData* d,int visual,int target,int direction,mjtNum threshold,mjPreContact bins[12],const SurfaceTree& tree,SweptSurfaceCandidates& region) {
  mjPreContact seeds[12];for(auto& seed:seeds)seed.dist=1e30;
  for(int id:region.winners)if(id>=0)sample_surface(m,d,visual,target,tree.points[id].data(),direction,threshold,seeds);
  mjtNum cutoff=-INFINITY;
  for(int bin=6*direction;bin<6*direction+6;++bin)cutoff=std::max(cutoff,seeds[bin].dist);
  cutoff=std::min(cutoff,threshold);
  std::array<int,6> winners{{-1,-1,-1,-1,-1,-1}};
  // Seed evaluations only establish bounds. Actual emission keeps original
  // sample order, including strict tie behavior and normal-axis classification.
  for(const auto& chunk:region.chunks) {
    bool clear=false;
    if(cutoff<threshold) {
      mjtNum lo[3]={INFINITY,INFINITY,INFINITY},hi[3]={-INFINITY,-INFINITY,-INFINITY};
      double scale=1;
      for(int corner=0;corner<8;++corner) {
        mjtNum point[3],world[3],delta[3],local[3];
        for(int k=0;k<3;++k)point[k]=(corner&(1<<k))?chunk.hi[k]:chunk.lo[k];
        mju_mulMatVec3(world,d->geom_xmat+9*visual,point);mju_addTo3(world,d->geom_xpos+3*visual);mju_sub3(delta,world,d->geom_xpos+3*target);mju_mulMatTVec3(local,d->geom_xmat+9*target,delta);
        for(int k=0;k<3;++k){lo[k]=std::min(lo[k],local[k]);hi[k]=std::max(hi[k],local[k]);scale+=fabs(world[k])+fabs(point[k])+fabs(d->geom_xpos[3*target+k]);}
      }
      double guard=1e-12+2048*std::numeric_limits<mjtNum>::epsilon()*scale;
      for(int k=0;k<3;++k){lo[k]-=guard;hi[k]+=guard;}
      clear=assemblenist_grid_box_separated(m,d,target,lo,hi,cutoff)!=0;
    }
    if(clear)continue;
    for(int offset=chunk.start;offset<chunk.end;++offset) {
      int id=region.ids[offset];int bin=sample_surface(m,d,visual,target,tree.points[id].data(),direction,threshold,bins);
      if(bin>=0)winners[bin%6]=id;
    }
  }
  region.winners=winners;
}

'''
 assert h.count(anchor)==1;h=h.replace(anchor,helper+anchor)
 old='for(int id:region.ids) sample_surface(m,d,visual,target,tree.points[id].data(),direction,threshold,bins);';assert h.count(old)==2;h=h.replace(old,'evaluate_winning_chunks(m,d,visual,target,direction,threshold,bins,tree,region);')
 h=h.replace('  region.ids.clear();','  region.ids.clear();region.chunks.clear();region.winners.fill(-1);',1)
 anchor='  surface_cache.swept_stats[2]+=region.ids.size();'
 h=h.replace(anchor,'''  for(int start=0;start<int(region.ids.size());start+=32) {
    SurfaceNode chunk;chunk.start=start;chunk.end=std::min(start+32,int(region.ids.size()));
    chunk.lo=tree.points[region.ids[start]];chunk.hi=chunk.lo;
    for(int offset=start+1;offset<chunk.end;++offset)for(int k=0;k<3;++k){mjtNum v=tree.points[region.ids[offset]][k];chunk.lo[k]=std::min(chunk.lo[k],v);chunk.hi[k]=std::max(chunk.hi[k],v);}
    region.chunks.push_back(chunk);
  }
'''+anchor)
 p.write_text(h)
