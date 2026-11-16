from pathlib import Path
import json,copy,hashlib,shutil,xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor,as_completed
import mujoco
from scripts.build_wowrobo_contact_grids import build_one
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).parent

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 olddir=ROOT/'outputs/linear-insertion-updated-20261107/grids';old=json.loads((olddir/'manifest.json').read_text());pinpath=ROOT/'outputs/insertion-fix-20261107/pin/consistent-v3/grids/manifest.json';pin=json.loads(pinpath.read_text());upper=ROOT/'outputs/insertion-fix-20261107/upper/combined-clearance-v4';assets=OUT/'assets';assets.mkdir(exist_ok=True);grids=OUT/'grids';grids.mkdir(exist_ok=True);manifest=copy.deepcopy(old);source_root=ET.parse(ROOT/'outputs/linear-insertion-20261107/source-snapshot/scene.xml').getroot();frozen={}
 for name in ['shoulder_connector_printed_0','shoulder_connector_printed_1']:
  src=upper/f'{name}.obj';dst=assets/f'{name}.obj';shutil.copy2(src,dst);frozen[name]={'source':str(src),'sha256':digest(dst),'frozen_asset':str(dst)};source_root.find(f"asset/mesh[@name='{name}']").set('file',str(dst.resolve()))
 pinrec=pin['meshes']['pin'];pin_src=Path(pinrec['source_file']);pin_dst=assets/'pin.obj';shutil.copy2(pin_src,pin_dst);source_root.find("asset/mesh[@name='pin']").set('file',str(pin_dst.resolve()));frozen['pin']={'source':str(pin_src),'sha256':digest(pin_dst),'frozen_asset':str(pin_dst)}
 source=OUT/'source-scene.xml';ET.ElementTree(source_root).write(source,encoding='unicode');m=mujoco.MjModel.from_xml_path(str(source));jobs=[]
 for name,row in manifest['meshes'].items():
  path=grids/f"grid_{row['grid_id']}.bin"
  if name in ['pin','servo_1_case']:
   row=copy.deepcopy(pin['meshes'][name]);manifest['meshes'][name]=row
  if name in frozen:row['source_file']=frozen[name]['frozen_asset']
  row['grid_file']=str(path.resolve())
  mid=m.mesh(name).id;row['native_mesh_pos']=m.mesh_pos[mid].tolist();row['native_mesh_quat_wxyz']=m.mesh_quat[mid].tolist()
  if name.startswith('shoulder_connector_printed_'):
   row['spacing']=.0002
   for key in ['origin','shape','grid_sha256','source_sha256']:row.pop(key,None)
   jobs.append(row)
  else:
   src=Path(pin['meshes'][name]['grid_file']) if name in ['pin','servo_1_case'] else Path(old['meshes'][name]['grid_file'])
   if not path.exists():path.symlink_to(src.resolve())
   if digest(path)!=row['grid_sha256']:raise RuntimeError('Mismatched linked field '+name)
 manifest.update(grid_dir=str(grids.resolve()),grid_directory=str(grids.resolve()),source_scene=str(source.resolve()),source_scene_sha256=digest(source),status='building',profile='unified-source-preserving-review-candidate-v1',frozen_assets=frozen,qualification='Upper combined CAD validity/source guard pending; native integrated tests pending; not a production default.')
 target=grids/'manifest.json';target.write_text(json.dumps(manifest,indent=2)+'\n')
 with ProcessPoolExecutor(max_workers=2) as pool:
  for future in as_completed([pool.submit(build_one,s) for s in jobs]):
   row=future.result();manifest['meshes'][row['name']]=row;target.write_text(json.dumps(manifest,indent=2)+'\n');print(row['name'],row['elapsed_s'],row['grid_sha256'],flush=True)
 manifest['status']='complete';target.write_text(json.dumps(manifest,indent=2)+'\n')
 from assembly_policy.so101.wowrobo_contact_backend import prepare_scene
 print(json.dumps(prepare_scene(source,target,OUT/'scene.xml'),indent=2))
if __name__=='__main__':main()
