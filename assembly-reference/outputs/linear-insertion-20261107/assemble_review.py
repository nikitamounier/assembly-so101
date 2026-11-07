from pathlib import Path
import hashlib,json,subprocess,math
BASE=Path(__file__).resolve().parent
FF='/Users/nikitamounier/.local/bin/ffmpeg'
steps=['cover','connector','pin','upper','final']
records=[]
for step in steps:
 clips=[]
 for mode in ['off','convex']:
  folder=BASE/step/mode
  r=json.loads((folder/'receipt.json').read_text())
  assert r['step']==step and r['contact_mode']==mode
  assert len(r['trajectory'])==96
  assert all(math.isfinite(t['axial_mm']) for t in r['trajectory'])
  assert abs(sum(x*x for x in r['axis_world'])-1)<1e-8
  assert any(sum(f*f for f in t['force_N'])>0 for t in r['trajectory'])
  if mode=='off': assert r['peak_contacts']==0
  assert r['physical_success'] is False
  clip=folder/'insertion.mp4';assert clip.stat().st_size>10000
  clips.append(clip)
  records.append({k:r[k] for k in ['step','contact_mode','axis_world','force_max_N','final_axial_mm','authored_reference_axial_mm','peak_contacts','stopped','source_scene_sha256','script_sha256']})
 concat=BASE/step/'concat.txt'
 concat.write_text(''.join(f"file '{p}'\n" for p in clips))
 subprocess.run([FF,'-y','-loglevel','error','-f','concat','-safe','0','-i',str(concat),'-c','copy','-movflags','+faststart',str(BASE/step/'comparison.mp4')],check=True)
concat=BASE/'concat-all.txt'
concat.write_text(''.join(f"file '{BASE/step/'comparison.mp4'}'\n" for step in steps))
subprocess.run([FF,'-y','-loglevel','error','-f','concat','-safe','0','-i',str(concat),'-c','copy','-movflags','+faststart',str(BASE/'all-insertions.mp4')],check=True)
summary={'order':steps,'mode_order':['off','convex'],'force_N':.5,'all_modes':records,'qualification':'Force-driven fixture probes only; contacts-off pass-through and convex cavity artifacts prevent seating qualification.'}
(BASE/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(records,indent=2))
print('Combined:',BASE/'all-insertions.mp4')
