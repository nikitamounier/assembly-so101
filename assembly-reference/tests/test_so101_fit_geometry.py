"""Bearing contact patches must coincide with the original CAD surface."""
import numpy as np


def test_bearing_patches_are_inside_original_horn_and_have_exact_floor():
    import igl,mujoco
    from assembly_policy.so101 import scene
    from assembly_policy.so101.fit_geometry import horn_bearing_patches
    meta=scene.metadata()
    with np.load(scene.ASSETS/'parts/horn.npz') as z:v,f=z['vertices'],z['faces']
    signs=np.array(np.meshgrid([-1,1],[-1,1],[-1,1])).reshape(3,-1).T
    for patch in horn_bearing_patches(meta):
        R=np.empty(9);mujoco.mju_quat2Mat(R,np.array(patch['quat']));R=R.reshape(3,3)
        points=(signs*(np.array(patch['size'])-1e-6))@R.T+patch['pos']
        assert np.all(igl.winding_number(v,f,points)>.99)
        bottom=(signs[signs[:,2]<0]*patch['size'])@R.T+patch['pos']
        assert np.sqrt(igl.point_mesh_squared_distance(bottom,v,f)[0]).max()<1e-7
        assert patch['density']==patch['contype']==patch['conaffinity']==0


def test_base_rail_patches_are_inside_original_cad_with_exact_contact_faces():
    import igl
    from assembly_policy.so101 import scene,workcell
    patches=[g for g in workcell.configuration()['contact_surfaces'] if g['name'].startswith('base_rail_')]
    assert len(patches)==4
    with np.load(scene.ASSETS/'parts/base.npz') as z:v,f=z['vertices'],z['faces']
    grid=np.array(np.meshgrid(*[np.linspace(-1,1,5)]*3)).reshape(3,-1).T
    def vector(value):return np.asarray(value,float) if isinstance(value,list) else np.fromstring(value,sep=' ')
    for patch in patches:
        position,size=vector(patch['pos']),vector(patch['size'])
        assert np.all(igl.winding_number(v,f,position+grid*(size-1e-6))>.99)
        normal=1 if position[2]<0 else -1
        face=position+grid[grid[:,2]==normal]*size
        assert np.sqrt(igl.point_mesh_squared_distance(face,v,f)[0]).max()<1e-7
        assert patch['density']==patch['contype']==patch['conaffinity']==0
