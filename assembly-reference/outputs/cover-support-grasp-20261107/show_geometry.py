import json
from pathlib import Path
import numpy as np
import mujoco
from assembly_policy.so101.wowrobo_randomization import WowRoboRandomizer
r=WowRoboRandomizer(config='configs/wowrobo_randomization_repaired.json');m=r.model;s=r.sequence;d=r.restore('18_upper_floor_grasp')
for piece in ['base_servo','base_side_cover','shoulder_connector']:
 b=m.body(piece).id
 print(piece,'pose',d.xpos[b],d.xmat[b].reshape(3,3),'grip',s.layout.grips.get(piece))
 print('bounds',s.layout.vertices[piece].min(0),s.layout.vertices[piece].max(0))
print('side cover spec',s.side_grasps['base_side_cover'])
print('base support',s.installed_base_grip)
print('leftsite',d.site_xpos[m.site('left_gripperframe').id],d.site_xmat[m.site('left_gripperframe').id].reshape(3,3))
