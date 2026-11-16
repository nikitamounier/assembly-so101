"""Freeze the single final candidate geometry and build/reframe its native fields."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import copy,json,hashlib,shutil,xml.etree.ElementTree as ET
import mujoco
from scripts.build_wowrobo_contact_grids import build_one
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).parent
PRINTED=['shoulder_connector_printed_0','shoulder_connector_printed_1','upper_arm_module_printed_0']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 original_path=ROOT/'outputs/linear-insertion-updated-20261107/grids/manifest.json';original=json.loads(original_path.read_text());pin_path=ROOT/'outputs/insertion-fix-20261107/pin/consistent-v3/grids/manifest.json';pin=json.loads(pin_path.read_text());conforming=ROOT/'outputs/insertion-fix-20261107/coherence/conforming-mesh';freeze=json.loads((conforming/'freeze.json').read_text());assets=OUT/'assets';assets.mkdir(exist_ok=True);grids=OUT/'base-grids';grids.mkdir(exist_ok=True);raw=ET.parse(ROOT/'outputs/linear-insertion-20261107/source-snapshot/scene.xml').getroot();frozen={}
 for name in PRINTED+['pin']:
  src=conforming/f'{name}.obj' if name in PRINTED else Path(pin['meshes']['pin']['source_file']);dst=assets/f'{name}.obj';shutil.copy2(src,dst);frozen[name]={'source':str(src.resolve()),'frozen_asset':str(dst.resolve()),'sha256':sha(dst)};raw.find(f"asset/mesh[@name='{name}']").set('file',str(dst.resolve()))
 source=OUT/'source-scene.xml';ET.ElementTree(raw).write(source,encoding='unicode');m=mujoco.MjModel.from_xml_path(str(source));manifest=copy.deepcopy(original);jobs=[]
 for name,row in manifest['meshes'].items():
  if name=='pin':row=copy.deepcopy(pin['meshes'][name]);manifest['meshes'][name]=row
  if name in frozen:row['source_file']=frozen[name]['frozen_asset']
  target=grids/f"grid_{row['grid_id']}.bin";row['grid_file']=str(target.resolve());mi=m.mesh(name).id;row['native_mesh_pos']=m.mesh_pos[mi].tolist();row['native_mesh_quat_wxyz']=m.mesh_quat[mi].tolist()
  if name in PRINTED:
   row['spacing']=.0004 if name=='upper_arm_module_printed_0' else .0002
   for key in ['origin','shape','grid_sha256','source_sha256']:row.pop(key,None)
   jobs.append(row)
  else:
   src=Path(pin['meshes'][name]['grid_file']) if name=='pin' else Path(original['meshes'][name]['grid_file'])
   if not target.exists():target.symlink_to(src.resolve())
   if sha(target)!=row['grid_sha256']:raise RuntimeError('Linked grid identity changed '+name)
 manifest.update(grid_dir=str(grids.resolve()),grid_directory=str(grids.resolve()),source_scene=str(source.resolve()),source_scene_sha256=sha(source),status='building',profile='final-common-conforming-clearance-candidate-v1',frozen_assets=frozen,conforming_freeze=freeze,scope='Derived review CAD and fit pin, originals retained. One common full51 candidate for all insertion tests. Not hardware manufacture approval or robot-policy training admission.')
 dest=grids/'manifest.json';dest.write_text(json.dumps(manifest,indent=2)+'\n')
 with ProcessPoolExecutor(max_workers=3) as pool:
  for future in as_completed([pool.submit(build_one,row) for row in jobs]):
   row=future.result();manifest['meshes'][row['name']]=row;dest.write_text(json.dumps(manifest,indent=2)+'\n');print(row['name'],row['elapsed_s'],row['grid_sha256'],flush=True)
 manifest['status']='complete';dest.write_text(json.dumps(manifest,indent=2)+'\n')
 from assembly_policy.so101.wowrobo_case_grid_reuse import build_profile,reframe_scene
 from assembly_policy.so101.wowrobo_contact_backend import prepare_scene
 final_manifest=build_profile(dest,pin_path,OUT/'grids');prepare_scene(source,final_manifest,OUT/'scene-unframed.xml');reframe_scene(OUT/'scene-unframed.xml',final_manifest,OUT/'scene.xml');receipt={'source_scene_sha256':sha(source),'scene_sha256':sha(OUT/'scene.xml'),'manifest_sha256':sha(final_manifest),'base_manifest_sha256':sha(dest),'frozen_assets':frozen,'fine_case_reframe_applied_exactly_once':True,'status':'Fields built; final native frame and integrated mechanical checks pending.'};(OUT/'build-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))
if __name__=='__main__':main()
