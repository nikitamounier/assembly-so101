from pathlib import Path
import json,hashlib,copy,xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor,as_completed
import mujoco
from scripts.build_wowrobo_contact_grids import build_one
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).parent;OLD=ROOT/'outputs/linear-insertion-updated-20261107/grids';SOURCE=ROOT/'outputs/linear-insertion-20261107/source-snapshot/scene.xml';NAME='shoulder_connector_printed_0'
def main():
 grids=OUT/'clearance-review-v4/grids';grids.mkdir(parents=True,exist_ok=True);original=json.loads((OLD/'manifest.json').read_text());manifest=copy.deepcopy(original);raw=ET.parse(SOURCE).getroot();raw.find("asset/mesh[@name='"+NAME+"']").set('file',str((OUT/'clearance-review-v4/shoulder_connector_printed_0.obj').resolve()));source=OUT/'clearance-review-v4/source-scene.xml';ET.ElementTree(raw).write(source,encoding='unicode');model=mujoco.MjModel.from_xml_path(str(source));manifest.update(grid_directory=str(grids.resolve()),grid_dir=str(grids.resolve()),source_scene=str(source.resolve()),source_scene_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),profile='connector-two-horn-clearance-review-v4',status='building');jobs=[]
 for name,row in manifest['meshes'].items():
  path=grids/f"grid_{row['grid_id']}.bin";row['grid_file']=str(path.resolve());mid=model.mesh(name).id;row['native_mesh_pos']=model.mesh_pos[mid].tolist();row['native_mesh_quat_wxyz']=model.mesh_quat[mid].tolist()
  if name == NAME:
   if name==NAME:row['source_file']=str((OUT/'clearance-review-v4/shoulder_connector_printed_0.obj').resolve())
   row['spacing']=.0004
   for key in ['grid_sha256','origin','shape','source_sha256']:row.pop(key,None)
   jobs.append(row)
  elif not path.exists():path.symlink_to(OLD/f"grid_{row['grid_id']}.bin")
 target=grids/'manifest.json';target.write_text(json.dumps(manifest,indent=2)+'\n')
 with ProcessPoolExecutor(max_workers=2) as pool:
  for future in as_completed([pool.submit(build_one,x) for x in jobs]):
   row=future.result();manifest['meshes'][row['name']]=row;target.write_text(json.dumps(manifest,indent=2)+'\n');print(row['name'],row['elapsed_s'],row['sampled_error_max_m'],flush=True)
 manifest['status']='complete';target.write_text(json.dumps(manifest,indent=2)+'\n')
 from assembly_policy.so101.wowrobo_contact_backend import prepare_scene
 result=prepare_scene(source,target,OUT/'clearance-review-v4/scene.xml');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
