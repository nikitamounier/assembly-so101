from pathlib import Path
import json,hashlib,os,xml.etree.ElementTree as ET
import mujoco
from scripts.build_wowrobo_contact_grids import build_one
ROOT=Path('/Users/nikitamounier/Documents/ChatGPT/robot assembly 2/robotics-assembly-3.0');OUT=Path(__file__).resolve().parent

def main():
 old=json.loads((OUT.parent/'fit-grids/manifest.json').read_text());record=dict(old['meshes']['pin']);record.update(name='pin',source_file=str(OUT/'assets/pin.obj'),grid_file=str(OUT/'pin-v3.bin'),spacing=.0001);row=build_one(record);(OUT/'pin-v3-field.json').write_text(json.dumps(row,indent=2)+'\n')
 tree=ET.parse(OUT.parent/'fit/source-scene.xml');tree.find("asset/mesh[@name='pin']").set('file',str(OUT/'assets/pin.obj'));source=OUT/'source-scene.xml';tree.write(source,encoding='unicode');m=mujoco.MjModel.from_xml_path(str(source));mid=m.mesh('pin').id;row['native_mesh_pos']=m.mesh_pos[mid].tolist();row['native_mesh_quat_wxyz']=m.mesh_quat[mid].tolist();(OUT/'pin-v3-field.json').write_text(json.dumps(row,indent=2)+'\n')
 for label,base in [('grids',OUT.parent/'fit-grids/manifest.json'),('grids-original-receivers',OUT.parent/'fit-pin-only-grids/manifest.json')]:
  manifest=json.loads(base.read_text());folder=OUT/label;folder.mkdir(exist_ok=True)
  for name,previous in list(manifest['meshes'].items()):
   item=dict(row)if name=='pin'else dict(previous);dest=folder/f"grid_{previous['grid_id']}.bin";origin=Path(item['grid_file'])
   if dest.exists():dest.unlink()
   os.link(origin,dest);item['grid_file']=str(dest);manifest['meshes'][name]=item
  manifest.update(grid_dir=str(folder),grid_directory=str(folder),profile=f'consistent-connector-fit-pin-v3-{label}',source_scene=str(source),source_scene_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),pin_mesh=str(OUT/'assets/pin.obj'),pin_registration=str(OUT/'registration.json'));(folder/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 print(json.dumps({'field_sha':row['grid_sha256'],'elapsed_s':row['elapsed_s'],'field_pitch_m':row['spacing'],'grids':str(OUT/'grids/manifest.json'),'originalreceivers':str(OUT/'grids-original-receivers/manifest.json')},indent=2))
if __name__=='__main__':main()
