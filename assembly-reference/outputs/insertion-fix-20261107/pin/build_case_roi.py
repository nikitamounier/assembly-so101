"""Source-triangle exact-distance/winding fine field in the pin interaction ROI."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib,json,struct,time,os
import igl,numpy as np,trimesh
ROOT=Path('/Users/nikitamounier/Documents/ChatGPT/robot assembly 2/robotics-assembly-3.0');OUT=Path(__file__).resolve().parent
LOW=np.array([-.007,-.0155,.044]);HIGH=np.array([.007,.0005,.077]);H=.0001

def slab(job):
 path,target,start,end,shape=job;mesh=trimesh.load_mesh(path,process=False);v=np.asarray(mesh.vertices);f=np.asarray(mesh.faces,dtype=np.int64);tree=igl.AABB();tree.init(v,f);nx,ny,nz=shape;xy=np.stack(np.meshgrid(LOW[0]+np.arange(nx)*H,LOW[1]+np.arange(ny)*H,indexing='xy'),-1).reshape(-1,2);grid=np.memmap(target,dtype='<f2',mode='r+',offset=44,shape=(nz,ny,nx))
 for k in range(start,end):
  q=np.column_stack([xy,np.full(len(xy),LOW[2]+k*H)]);sq,_,_=tree.squared_distance(v,f,q);w=igl.fast_winding_number(v,f,q);grid[k]=((np.sqrt(sq)*np.where(w>.5,-1,1)).reshape(ny,nx)).astype('<f2')
 grid.flush();return[start,end]

def main():
 begin=time.monotonic();base=json.loads((ROOT/'outputs/linear-insertion-updated-20261107/grids/manifest.json').read_text());record=dict(base['meshes']['servo_1_case']);path=record['source_file'];shape=np.rint((HIGH-LOW)/H).astype(int)+1;target=OUT/'fields/case-roi.bin';partial=target.with_suffix('.partial')
 with partial.open('wb')as f:f.write(struct.pack('<3i4d',*shape,*LOW,H));f.truncate(44+int(np.prod(shape))*2)
 n=int(shape[2]);chunks=[(path,str(partial),start,min(start+12,n),tuple(shape))for start in range(0,n,12)]
 with ProcessPoolExecutor(max_workers=4)as pool:
  for future in as_completed([pool.submit(slab,j)for j in chunks]):print('completedz',future.result(),flush=True)
 partial.replace(target);mesh=trimesh.load_mesh(path,process=False);record.update(name='servo_1_case',spacing=H,origin=LOW.tolist(),shape=shape.tolist(),grid_file=str(target),grid_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),source_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),grid_bytes=target.stat().st_size,source_watertight=bool(mesh.is_watertight),scope='Exactwhole-source-trianglefieldrestrictedtopinROI;out-of-ROIcontactqueriesmustreject',valid_contact_roi_m=[LOW.tolist(),HIGH.tolist()],elapsed_s=time.monotonic()-begin)
 (OUT/'fields/case-roi.json').write_text(json.dumps(record,indent=2)+'\n')
 for kind,pinlabel in [('fit','fit_pin'),('negative','old_negative_pin')]:
  directory=OUT/f'roi-{kind}-grids';directory.mkdir(exist_ok=True);manifest=json.loads(json.dumps(base));manifest.update(grid_directory=str(directory),grid_dir=str(directory),status='complete',profile=f'pin_source_ROI_{kind}_0.1mm',local_case_field_roi_m=[LOW.tolist(),HIGH.tolist()])
  for name,old in base['meshes'].items():
   row=dict(record)if name=='servo_1_case'else json.loads((OUT/'fields'/f'{pinlabel}.json').read_text())if name=='pin'else dict(old);row['name']=name;dest=directory/f"grid_{old['grid_id']}.bin";source=Path(row['grid_file'])
   if dest.exists():dest.unlink()
   os.link(source,dest);row['grid_file']=str(dest);manifest['meshes'][name]=row
  manifest['same_case_field_sha256']=record['grid_sha256'];(directory/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 print(json.dumps({'elapsed':time.monotonic()-begin,'shape':shape.tolist(),'case_sha':record['grid_sha256']}))
if __name__=='__main__':main()
