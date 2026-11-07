"""Published HOTO envelope plus actual native button-contact and drop checks."""

from assembly_policy.so101.limits import PENETRATION_LIMIT_M
import xml.etree.ElementTree as ET
import mujoco
import numpy as np
import pytest
from assembly_policy.so101 import scene
from assembly_policy.so101.robot import home
from assembly_policy.so101.tool import add_driver,GRASP_Z,PUBLISHED_DIMENSIONS_M,PUBLISHED_MASS_KG


def tool_bench(button=None):
    scene.load_plugin();root=ET.Element('mujoco')
    ET.SubElement(root,'compiler',angle='radian',autolimits='true')
    ET.SubElement(root,'option',timestep='.0005',integrator='implicitfast',gravity='0 0 0',cone='elliptic',iterations='100')
    default=ET.SubElement(root,'default');ET.SubElement(default,'geom',friction='.7 .005 .0001',solref='.001 1',solimp='.99 .999 .0001')
    asset=ET.SubElement(root,'asset');world=ET.SubElement(root,'worldbody')
    extension=ET.SubElement(root,'extension');contact=ET.SubElement(root,'contact')
    tool=add_driver(world,extension,contact,position=(0,0,.25),quaternion=(1,0,0,0),asset=asset)
    if button:
        tool.remove(tool.find('freejoint'))  # A bolted test fixture, never a grasp.
        height={'button_forward':GRASP_Z,'button_reverse':.017,'trigger_saddle':-.021}[button]
        finger=ET.SubElement(world,'body',name='test_press',pos=f'0 .016 {.25+height}')
        ET.SubElement(finger,'joint',name='test_press',type='slide',axis='0 -1 0',range='0 .009',damping='1')
        ET.SubElement(finger,'geom',type='box',size='.002 .001 .002',mass='.015')
        actuator=ET.SubElement(root,'actuator');ET.SubElement(actuator,'motor',joint='test_press',ctrlrange='-3 3')
    return mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'))


def test_reconstruction_matches_published_envelope_and_mass_without_the_bit():
    m=tool_bench();d=mujoco.MjData(m);mujoco.mj_forward(m,d);points=[];triangles=0
    for g in range(m.ngeom):
        if m.geom_group[g]!=2 or m.geom_bodyid[g]==m.body('driver_bit').id:continue
        mesh=m.geom_dataid[g]
        if mesh<0:continue
        a,n=m.mesh_vertadr[mesh],m.mesh_vertnum[mesh]
        start,count=m.mesh_faceadr[mesh],m.mesh_facenum[mesh]
        triangles_local=m.mesh_vert[a:a+n][m.mesh_face[start:start+count]]
        volume=np.einsum('ij,ij->i',triangles_local[:,0],np.cross(triangles_local[:,1],triangles_local[:,2])).sum()/6
        assert volume>0,(m.geom(g).name,'inward mesh normals')
        points.append(m.mesh_vert[a:a+n]@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g])
        triangles+=int(m.mesh_facenum[mesh])
    extent=np.ptp(np.concatenate(points),axis=0)
    np.testing.assert_allclose(extent[[2,0,1]],PUBLISHED_DIMENSIONS_M,rtol=0,atol=.0003)
    assert m.body_subtreemass[m.body('driver').id]-m.body_mass[m.body('driver_bit').id]==pytest.approx(PUBLISHED_MASS_KG)
    assert triangles<16000 and m.neq==0
    assert not any((m.geom(g).name or '').startswith(('sleeve_','trigger_guide_')) for g in range(m.ngeom))


@pytest.mark.parametrize('button,direction',[('button_forward',1),('button_reverse',-1),('trigger_saddle',0)])
def test_physical_contact_depresses_the_real_control_and_release_stops_it(button,direction):
    m=tool_bench(button);d=mujoco.MjData(m);spindle=int(m.joint('spindle').dofadr[0])
    d.ctrl[0]=2.;mujoco.mj_step(m,d,nstep=1000)
    assert d.joint(button).qpos[0]>.0006
    if direction:assert direction*d.qvel[spindle]>20.
    else:assert abs(d.qvel[spindle])<.001  # Selector has no electrical ON channel.
    other='button_reverse' if button=='button_forward' else 'button_forward'
    assert abs(d.joint(other).qpos[0])<.0001
    d.ctrl[0]=-2.;mujoco.mj_step(m,d,nstep=2000)
    assert abs(d.joint(button).qpos[0])<.0001
    assert abs(d.qvel[spindle])<.1
    assert np.isfinite(d.qpos).all() and not d.warning.number.any()


@pytest.mark.parametrize('height',[.02,.06])
def test_bare_tool_drops_and_settles_freely(height):
    m=scene.model(contact_monitor=True);d=mujoco.MjData(m);home(m,d)
    q=int(m.joint('driver_free').qposadr[0]);v=int(m.joint('driver_free').dofadr[0]);d.qpos[q+2]+=height
    initial=d.qpos[q+2];peak=0.
    for _ in range(300):
        mujoco.mj_step(m,d,nstep=20)
        distances=[c.dist for c in d.contact if any(m.body_rootid[m.geom_bodyid[int(g)]]==m.body('driver').id for g in c.geom)]
        peak=max(peak,max(0.,-float(min(distances,default=0))))
    assert initial-d.qpos[q+2]>.015
    assert np.linalg.norm(d.qvel[v:v+3])<.01
    assert peak<PENETRATION_LIMIT_M,peak
    assert m.neq==0 and not d.warning.number.any() and np.isfinite(d.qpos).all()
