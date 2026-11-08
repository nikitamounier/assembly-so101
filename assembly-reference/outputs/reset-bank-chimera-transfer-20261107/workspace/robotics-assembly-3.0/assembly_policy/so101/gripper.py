"""Bare-CAD fingertip self-contact, excluding the intentional hinge overlap."""
from functools import lru_cache
from pathlib import Path
import struct
import xml.etree.ElementTree as ET
import numpy as np


@lru_cache(maxsize=2)
def distal_triangles(path, y_max):
    """Use original distal STL faces; no pad, scaling or visual modification."""
    path=Path(path)
    with path.open('rb') as stream:
        stream.seek(80);count=struct.unpack('<I',stream.read(4))[0]
        dtype=np.dtype([('normal','<f4',(3,)),('vertices','<f4',(3,3)),('attribute','<u2')])
        if path.stat().st_size!=84+50*count:raise ValueError('Expected pinned binary SO101 STL')
        faces=np.fromfile(stream,dtype=dtype,count=count)['vertices'].astype(float)
    faces=faces[np.all(faces[:,:,1]<y_max,axis=1)]
    if len(faces)<10:raise ValueError('No distal finger contact surface found')
    vertices,inverse=np.unique(faces.reshape(-1,3),axis=0,return_inverse=True)
    return vertices,inverse.reshape(-1,3)


def configure(root, cfg, assets):
    def numbers(values):return ' '.join(format(float(x),'.17g') for x in np.asarray(values).ravel())
    vertices,faces=distal_triangles(str(assets/'so101/assets/moving_jaw_so101_v1.stl'),cfg['distal_source_y_max_m'])
    ET.SubElement(root.find('asset'),'mesh',name='so101_distal_finger_mesh',vertex=numbers(vertices),face=numbers(faces))
    for side in ('left','right'):
        for suffix in ('wrist_roll_follower_so101_v1','moving_jaw_so101_v1'):
            contact=root.find(f".//geom[@name='{side}_{suffix}_collision_sdf']")
            # Priority selects finger parameters against ordinary objects,
            # including nominal 0.7 sliding friction. Explicit pairs override it.
            contact.set('priority','1')
            contact.set('solref',numbers(cfg['object_solref']))
            contact.set('solimp',numbers(cfg['object_solimp']))
        joint=root.find(f".//joint[@name='{side}_gripper']")
        joint.set('range',numbers(cfg['joint_range_rad']))
        root.find(f"actuator/*[@name='{side}_gripper']").set('ctrlrange',numbers([cfg['closed_command_rad'],cfg['open_command_rad']]))
        body=root.find(f".//body[@name='{side}_moving_jaw_so101_v1']")
        visual=body.find(f"geom[@mesh='{side}_moving_jaw_so101_v1']")
        name=side+'_distal_finger_contact'
        attrs={key:visual.get(key) for key in ('pos','quat') if visual.get(key)}
        ET.SubElement(body,'geom',name=name,type='mesh',mesh='so101_distal_finger_mesh',
                      contype='0',conaffinity='0',density='0',group='3',**attrs)
        # Explicit pairing bypasses parent-child filtering only at the real tips.
        # The connected hinge still has its normal exclusion. One relative DOF
        # needs a normal stop; object/jaw friction remains native contact.
        ET.SubElement(root.find('contact'),'pair',name=side+'_finger_stop',
                      geom1=side+'_wrist_roll_follower_so101_v1_collision_sdf',geom2=name,
                      condim='1',solref=numbers(cfg['stop_solref']),solimp=numbers(cfg['stop_solimp']))
