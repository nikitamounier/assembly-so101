from pathlib import Path
import json,time
import cadquery as cq,numpy as np,trimesh
ROOT=Path.cwd();DIR=ROOT/'outputs/insertion-fix-20261107/coherence/conforming-mesh';manifest=json.loads((DIR/'manifest.json').read_text());rows=[]
for part in manifest['parts']:
 solid=cq.importers.importStep(part['source_STEP']).val();shells=cq.Compound.makeCompound(solid.Faces());mesh=trimesh.load_mesh(part['output_OBJ'],process=False);rng=np.random.default_rng(5084);sample=mesh.triangles[rng.choice(len(mesh.faces),100,replace=False)]*1000;points=np.vstack((np.array([p['point_mm']for p in part['worst_support_projection_queries']]),sample.mean(1),(sample[:,0]+sample[:,1])/2));dist=[];start=time.monotonic()
 for i,q in enumerate(points):
  dist.append(float(cq.Vertex.makeVertex(*q).distance(shells)))
  if i%50==0:print(part['name'],i,'/',len(points),'maxmm',max(dist),flush=True)
 row={'name':part['name'],'queries':len(points),'method':'ExactOCCpoint-to-compound-of-trimmed-faces distance;50worstsupportprojectionpoints+100deterministicrandomtrianglecentroids+100edgemidpoints.','maximum_trimmed_surface_distance_mm':max(dist),'p99_mm':float(np.quantile(dist,.99)),'passes40um':max(dist)<=.04,'elapsed_s':time.monotonic()-start};rows.append(row)
(DIR/'trimmed-surface-spotcheck.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(rows,indent=2),flush=True)
