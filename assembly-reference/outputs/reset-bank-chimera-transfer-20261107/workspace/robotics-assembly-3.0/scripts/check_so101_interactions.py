"""Diverse native interaction probes; failures are retained, not fitted away.

Fixed-palm grasps qualify aligned contact mechanics, not tabletop pickup.
Production geometry/actuators/solver are retained; object initialization, gravity
and friction/load perturbations are explicit diagnostic conditions.
"""

from assembly_policy.so101.limits import PENETRATION_LIMIT_M
import argparse
import copy
import gc
import hashlib
import itertools
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from assembly_policy.so101 import scene
from assembly_policy.so101.robot import home,ik,down_rotation
from assembly_policy.so101.workcell import configuration,environment_identity


def fixed_palm(shape='box',width=.012,part=None):
    """Extract the actual left hand; retain exact geometry and native actuation."""
    scene.load_plugin();root=ET.fromstring(scene.build_xml(contact_monitor=True))
    world=root.find('worldbody');palm=copy.deepcopy(world.find(".//body[@name='left_gripper']"))
    palm.remove(palm.find('joint'));palm.set('pos','0 0 .25');palm.set('quat','1 0 0 0')
    selected=copy.deepcopy(world.find(f"body[@name='{part}']")) if part else None
    table=copy.deepcopy(world.find("geom[@name='table']"));world.clear();world.extend([table,palm])
    if selected is None:
        selected=ET.SubElement(world,'body',name='object',pos='0 0 .15')
        ET.SubElement(selected,'freejoint',name='object_free')
        attrs=dict(name='object_geom',type=shape,density='1200',rgba='.2 .5 .8 1')
        if shape=='box':attrs['size']=' '.join([str(width/2)]*3)
        elif shape=='sphere':attrs['size']=str(width/2)
        else:attrs.update(size=f'{width/2} .006',quat='.7071067811865476 .7071067811865476 0 0')
        ET.SubElement(selected,'geom',**attrs)
    else:world.append(selected)
    kept_bodies={b.get('name') for b in world.iter('body')}
    kept_geoms={g.get('name') for g in world.iter('geom')}
    contact=root.find('contact')
    for element in list(contact):
        names=[element.get(k) for k in ('geom1','geom2')] if element.tag=='pair' else [element.get(k) for k in ('body1','body2')]
        if not all(name in (kept_geoms if element.tag=='pair' else kept_bodies) for name in names):contact.remove(element)
    ET.SubElement(contact,'exclude',body1='left_gripper',body2='left_moving_jaw_so101_v1')
    acts=root.find('actuator')
    for element in list(acts):
        if element.get('name')!='left_gripper':acts.remove(element)
    meshes={g.get('mesh') for g in world.iter('geom') if g.get('mesh')}
    asset=root.find('asset')
    for element in list(asset):
        if element.tag=='mesh' and element.get('name') not in meshes:asset.remove(element)
    instances={p.get('instance') for p in root.findall('.//geom/plugin')+root.findall('asset/mesh/plugin')}
    ext=root.find('extension')
    for plugin in list(ext):
        if plugin.get('plugin')=='so101.contact_driver':
            if part!='driver':ext.remove(plugin)
            continue
        for instance in list(plugin):
            if instance.get('name') not in instances:plugin.remove(instance)
        if not len(plugin):ext.remove(plugin)
    m=mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'))
    if part in configuration()['free_body_damping']:
        v=int(m.joint(part+'_free').dofadr[0]);damping=configuration()['free_body_damping'][part]
        m.dof_damping[v:v+3]=damping['linear'];m.dof_damping[v+3:v+6]=damping['angular']
    return m


def finger_forces(m,d,body,side='left'):
    result={'fixed':0.,'moving':0.,'other':0.}
    for i,c in enumerate(d.contact):
        a,b=map(int,c.geom)
        inside_a=m.body_rootid[m.geom_bodyid[a]]==body
        inside_b=m.body_rootid[m.geom_bodyid[b]]==body
        if inside_a and inside_b:continue
        if inside_a:other=b
        elif inside_b:other=a
        else:continue
        f=np.zeros(6);mujoco.mj_contactForce(m,d,i,f)
        name=m.geom(other).name
        key='fixed' if name==side+'_wrist_roll_follower_so101_v1_collision_sdf' else 'moving' if name==side+'_moving_jaw_so101_v1_collision_sdf' else 'other'
        result[key]+=abs(float(f[0]))
    return result


def primitive_mesh_overlap(m,qpos,shape):
    """Independent original-finger triangles versus analytic object surfaces."""
    import igl
    d=mujoco.MjData(m);d.qpos[:]=qpos;mujoco.mj_kinematics(m,d)
    g=m.geom('object_geom').id;size=m.geom_size[g];R=d.geom_xmat[g].reshape(3,3);centre=d.geom_xpos[g]
    if shape=='sphere':
        i=np.arange(1024);z=1-2*(i+.5)/len(i);theta=i*np.pi*(3-np.sqrt(5));r=np.sqrt(1-z*z)
        points=np.column_stack((r*np.cos(theta),r*np.sin(theta),z))*size[0]
    elif shape=='box':
        points=[]
        for axis in range(3):
            others=[x for x in range(3) if x!=axis]
            for sign in (-1,1):
                for u,v in itertools.product(np.linspace(-1,1,15),repeat=2):
                    p=np.zeros(3);p[axis]=sign*size[axis];p[others]=[u*size[others[0]],v*size[others[1]]];points.append(p)
        points=np.asarray(points)
    else:
        angle=np.linspace(0,2*np.pi,64,endpoint=False);points=[]
        for z in np.linspace(-size[1],size[1],15):points.extend(np.column_stack((size[0]*np.cos(angle),size[0]*np.sin(angle),np.full(len(angle),z))))
        for z in (-size[1],size[1]):
            for radius in np.linspace(0,size[0],8):points.extend(np.column_stack((radius*np.cos(angle),radius*np.sin(angle),np.full(len(angle),z))))
        points=np.asarray(points)
    world=points@R.T+centre;worst=0.
    for name in ('left_wrist_roll_follower_so101_v1','left_moving_jaw_so101_v1'):
        geom=next(g for g in range(m.ngeom) if m.geom_type[g]==mujoco.mjtGeom.mjGEOM_MESH and m.mesh(m.geom_dataid[g]).name==name)
        mesh=m.geom_dataid[geom];v=m.mesh_vert[m.mesh_vertadr[mesh]:m.mesh_vertadr[mesh]+m.mesh_vertnum[mesh]];f=m.mesh_face[m.mesh_faceadr[mesh]:m.mesh_faceadr[mesh]+m.mesh_facenum[mesh]]
        local=(world-d.geom_xpos[geom])@d.geom_xmat[geom].reshape(3,3)
        inside=igl.fast_winding_number(v,f,local)>.5
        if inside.any():worst=max(worst,float(np.sqrt(igl.point_mesh_squared_distance(local[inside],v,f)[0]).max()))
        finger=(np.vstack((v,v[f].mean(1)))@d.geom_xmat[geom].reshape(3,3).T+d.geom_xpos[geom]-centre)@R
        if shape=='sphere':distance=np.linalg.norm(finger,axis=1)-size[0]
        elif shape=='box':
            delta=np.abs(finger)-size;distance=np.linalg.norm(np.maximum(delta,0),axis=1)+np.minimum(np.max(delta,axis=1),0)
        else:
            delta=np.column_stack((np.linalg.norm(finger[:,:2],axis=1)-size[0],np.abs(finger[:,2])-size[1]));distance=np.linalg.norm(np.maximum(delta,0),axis=1)+np.minimum(np.max(delta,axis=1),0)
        worst=max(worst,max(0.,-float(distance.min())))
    return worst


def mesh_surfaces(m,d,geoms):
    for g in geoms:
        mesh=m.geom_dataid[g]
        if mesh<0:continue
        a,n=m.mesh_vertadr[mesh],m.mesh_vertnum[mesh];b,nf=m.mesh_faceadr[mesh],m.mesh_facenum[mesh]
        v=m.mesh_vert[a:a+n]@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g]
        f=m.mesh_face[b:b+nf]
        yield v,f,np.vstack((v,v[f].mean(1)))


def part_mesh_overlap(m,qpos,part,side='left'):
    """Original visual mesh samples; screw uses its nominal analytic host mesh."""
    import igl
    d=mujoco.MjData(m);d.qpos[:]=qpos;mujoco.mj_kinematics(m,d)
    body=m.body(part).id
    obj=list(mesh_surfaces(m,d,[g for g in range(m.ngeom) if m.body_rootid[m.geom_bodyid[g]]==body and m.geom_group[g]==2]))
    if not obj:return None  # Driver consists of primitives, not vendor CAD.
    fingers=list(mesh_surfaces(m,d,[g for g in range(m.ngeom) if m.geom_group[g]==2 and m.geom_type[g]==mujoco.mjtGeom.mjGEOM_MESH and m.mesh(m.geom_dataid[g]).name in (side+'_wrist_roll_follower_so101_v1',side+'_moving_jaw_so101_v1')]))
    worst=0.
    for sources,targets in ((obj,fingers),(fingers,obj)):
        for _,_,points in sources:
            for v,f,_ in targets:
                p=points[np.all((points>=v.min(0))&(points<=v.max(0)),axis=1)]
                if not len(p):continue
                inside=igl.fast_winding_number(v,f,p)>.5
                if inside.any():worst=max(worst,float(np.sqrt(igl.point_mesh_squared_distance(p[inside],v,f)[0]).max()))
    return worst


def grasp_case(shape,width,depth,mu,*,part=None,closed=None,dt=.0005,load=1.,search=None,stiff=False,impedance=.99,solver=None):
    m=fixed_palm(shape,width,part);m.opt.timestep=dt;d=mujoco.MjData(m)
    for key,value in (solver or {}).items():setattr(m.opt,key,value)
    if search is not None:
        m.numeric('so101_sdf_search').data[:]=search
        m.numeric('so101_grasp_search').data[:]=search
    if stiff:
        m.geom_solref[:]=[.001,1.];m.geom_solimp[:]=[impedance,.9999,.0001,.5,2.]
    name=part or 'object';body=m.body(name).id;q=int(m.joint(name+'_free').qposadr[0])
    grip=int(m.joint('left_gripper').qposadr[0]);d.qpos[grip]=.65
    rotation=np.eye(3);offset=np.zeros(3)
    if part:
        bounds=np.array(scene.metadata()[part]['bounds']) if part!='driver' else np.array([[-.014,-.015,-.078],[.014,.018,.105]])
        if part=='screw':
            # The nominal M3 supplement differs from the incomplete source CAD.
            d.qpos[q:q+7]=[0,0,0,1,0,0,0];mujoco.mj_kinematics(m,d)
            points=np.vstack([v for v,_,_ in mesh_surfaces(m,d,[m.geom(n).id for n in ('male_contact','screw_head_contact')])])
            bounds=np.array([points.min(0),points.max(0)])
        if part=='base':offset=[bounds[0,0],0,.03]  # outside face of the thin side wall
        elif part=='driver':
            from assembly_policy.so101.tool import GRASP_CENTER,PUBLISHED_DIMENSIONS_M
            rotation=Rotation.from_quat([-.5,-.5,-.5,.5]).as_matrix();offset=GRASP_CENTER;width=PUBLISHED_DIMENSIONS_M[2]
        else:
            axis=int(np.argmin(np.ptp(bounds,axis=0)));rotation=Rotation.from_euler('z',90,degrees=True).as_matrix() if axis==1 else Rotation.from_euler('y',90,degrees=True).as_matrix() if axis==2 else np.eye(3)
            offset=bounds.mean(0)
            width=float(np.ptp(bounds@rotation.T,axis=0)[0])
    centre=np.array([-.0079+width/2+.0003,0.,.25+depth])
    if part=='base':centre[0]=-.0079+.0003
    d.qpos[q:q+3]=centre-rotation@np.asarray(offset)
    quat=Rotation.from_matrix(rotation).as_quat();d.qpos[q+3:q+7]=quat[[3,0,1,2]]
    if mu is not None:m.geom_friction[:,0]=mu
    m.opt.gravity[:]=0;d.ctrl[0]=configuration()['gripper']['closed_command_rad'] if closed is None else closed
    mujoco.mj_forward(m,d);initial_depth=max(0.,-float(min(d.contact.dist,default=0)))
    if initial_depth>1e-6:
        return dict(family='aligned_grasp',shape=part or shape,width_m=width,depth_in_palm_m=depth,friction=mu,
                    initial_native_overlap_m=initial_depth,rejected_initialization=True,passed=None,
                    scope='Invalid overlapping fixture; not simulated or counted as a physics failure.')
    peak_depth=initial_depth;peak_speed=0.;peak_pose=d.qpos.copy()
    for _ in range(round(1.2/dt)):
        mujoco.mj_step(m,d)
        depth_now=max(0.,-float(min(d.contact.dist,default=0)))
        if depth_now>peak_depth:peak_depth=depth_now;peak_pose=d.qpos.copy()
    closure_pose=d.qpos.copy();before=d.body(name).xpos.copy();r0=d.body(name).xmat.reshape(3,3).copy();minimum_force=float('inf')
    m.opt.gravity[:]=[0,0,-9.81*load]
    for k in range(round(1./dt)):
        mujoco.mj_step(m,d)
        if k%20==0:
            depth_now=max(0.,-float(min(d.contact.dist,default=0)))
            if depth_now>peak_depth:peak_depth=depth_now;peak_pose=d.qpos.copy()
            velocity=np.zeros(6);mujoco.mj_objectVelocity(m,d,mujoco.mjtObj.mjOBJ_BODY,body,velocity,0);peak_speed=max(peak_speed,float(np.linalg.norm(velocity[3:])))
            if k>round(.8/dt):
                forces=finger_forces(m,d,body);minimum_force=min(minimum_force,forces['fixed'],forces['moving'])
    displacement=float(np.linalg.norm(d.body(name).xpos-before))
    angle=float(Rotation.from_matrix(d.body(name).xmat.reshape(3,3)@r0.T).magnitude())
    hold_pose=d.qpos.copy()
    retention_angle=0. if not part and shape=='sphere' else angle
    if not part and shape=='cylinder':
        retention_angle=float(np.arccos(np.clip((d.body(name).xmat.reshape(3,3)@np.array([0,1,0]))@(r0@np.array([0,1,0])),-1,1)))
    forces=finger_forces(m,d,body)
    finite=bool(np.isfinite(d.qpos).all() and not d.warning.number.any() and abs(d.time-2.2)<1e-8)
    held=bool(displacement<.001 and retention_angle<.03 and minimum_force>.001 and forces['other']<.01)
    d.ctrl[0]=configuration()['gripper']['open_command_rad'];released_from=d.body(name).xpos.copy()
    mujoco.mj_step(m,d,nstep=round(.5/dt));release_distance=float(np.linalg.norm(d.body(name).xpos-released_from))
    finite=bool(finite and np.isfinite(d.qpos).all() and not d.warning.number.any() and abs(d.time-2.7)<1e-8)
    material_samples=[part_mesh_overlap(m,pose,part) if part else primitive_mesh_overlap(m,pose,shape)
                      for pose in (peak_pose,closure_pose,hold_pose)]
    material=max(material_samples) if all(x is not None for x in material_samples) else None
    material_limit=min(PENETRATION_LIMIT_M,.05*width)
    return dict(family='aligned_grasp',shape=part or shape,width_m=width,depth_in_palm_m=depth,friction=mu,timestep_s=dt,load_g=load,
                old_command_control=closed is not None,search_override=search,stiff_contact_override=stiff,solver_override=solver,
                mass_kg=float(m.body_subtreemass[body]),clock_error_s=float(d.time-2.7),warnings=d.warning.number.tolist(),
                sampled_material_overlap_m=material,material_pose_samples=3,material_limit_m=material_limit,
                initial_native_overlap_m=initial_depth,peak_native_depth_m=peak_depth,
                hold_translation_m=displacement,hold_rotation_rad=angle,retention_rotation_rad=retention_angle,minimum_bilateral_force_N=minimum_force,
                final_contact_forces_N=forces,peak_speed_m_s=peak_speed,release_distance_m=release_distance,
                held=held,released=release_distance>.005,numerically_valid=finite,
                passed=bool(held and release_distance>.005 and finite and peak_depth<PENETRATION_LIMIT_M and material is not None and material<material_limit),
                scope='Aligned fixed-palm mechanics; sampled geometry check, not continuous collision or tabletop pickup. Driver geometry qualification unavailable.' if material is None else 'Aligned fixed-palm mechanics; sampled geometry check, not continuous collision or tabletop pickup.')


def table_cases():
    m=scene.model(contact_monitor=True);rows=[]
    for name in (*scene.PARTS,'driver'):
        for height in (.02,.06):
            d=mujoco.MjData(m);home(m,d);q=int(m.joint(name+'_free').qposadr[0]);d.qpos[q+2]+=height
            d.ctrl[:]=d.qpos[:12];mujoco.mj_forward(m,d);peak=0.;floor_peak=0.;body=m.body(name).id
            geoms=[g for g in range(m.ngeom) if m.body_rootid[m.geom_bodyid[g]]==body and m.geom_group[g]==2 and m.geom_dataid[g]>=0]
            for _ in range(300):
                mujoco.mj_step(m,d,nstep=20)
                distances=[c.dist for c in d.contact if any(m.body_rootid[m.geom_bodyid[int(g)]]==body for g in c.geom)]
                peak=max(peak,max(0.,-float(min(distances,default=0))))
                for v,_,_ in mesh_surfaces(m,d,geoms):floor_peak=max(floor_peak,max(0.,-float(v[:,2].min())))
            velocity=np.zeros(6);mujoco.mj_objectVelocity(m,d,mujoco.mjtObj.mjOBJ_BODY,m.body(name).id,velocity,0)
            valid=bool(np.isfinite(d.qpos).all() and not d.warning.number.any())
            rows.append(dict(family='table_drop',part=name,height_m=height,peak_native_depth_m=peak,sampled_mesh_floor_overlap_m=floor_peak if geoms else None,final_speed_m_s=float(np.linalg.norm(velocity[3:])),
                             numerically_valid=valid,passed=bool(valid and np.linalg.norm(velocity[3:])<.01 and peak<PENETRATION_LIMIT_M and floor_peak<PENETRATION_LIMIT_M)))
        for ratio in (.5,2.):
            d=mujoco.MjData(m);home(m,d);d.ctrl[:]=d.qpos[:12];mujoco.mj_step(m,d,nstep=4000)
            start=d.body(name).xpos.copy();body=m.body(name).id;force=ratio*.7*m.body_subtreemass[body]*9.81
            d.xfrc_applied[body,0]=force;mujoco.mj_step(m,d,nstep=400);d.xfrc_applied[:]=0;mujoco.mj_step(m,d,nstep=400)
            rows.append(dict(family='table_push',part=name,force_over_nominal_friction=ratio,force_N=float(force),
                             displacement_m=float(np.linalg.norm(d.body(name).xpos-start)),
                             numerically_valid=bool(np.isfinite(d.qpos).all() and not d.warning.number.any()),passed=None,
                             scope='Response measurement; curved/irregular parts can roll, so no universal static-friction displacement gate.'))
    return rows


def reach_cases():
    m=scene.model();d=mujoco.MjData(m);home(m,d);mujoco.mj_step(m,d,nstep=1000);rows=[]
    for side,part,height in itertools.product(('left','right'),('base','motor'),(.02,.05,.08)):
        target=d.body(part).xpos+[0,0,height]
        try:ik(m,d.qpos,side,target);passed=True;detail='position-only approach reachable'
        except ValueError as error:passed=False;detail=str(error)
        rows.append(dict(family='approach_reach',side=side,part=part,height_above_centre_m=height,passed=passed,detail=detail))
    return rows


def pickup_cases():
    """Scripted whole-arm attempts; actuator commands only after initialization."""
    rows=[]
    for yaw in (0.,np.pi/2,np.pi):
        m=scene.model(contact_monitor=True);d=mujoco.MjData(m);home(m,d);mujoco.mj_step(m,d,nstep=2000)
        acts=[m.actuator('right_'+j).id for j in scene.JOINTS];R=down_rotation(yaw)
        bounds=np.array(scene.metadata()['motor']['bounds']);corners=np.array(np.meshgrid(*zip(*bounds))).T.reshape(-1,3)
        points=corners@d.body('motor').xmat.reshape(3,3).T+d.body('motor').xpos;centre=points.mean(0)
        width=float(np.ptp(points@R[:,2]));goal=centre-R[:,2]*(width/2+.0003)
        row=dict(family='tabletop_pickup',part='motor',side='right',yaw_rad=yaw,phases=[],passed=False)
        peak=0.;poses=[]
        def drive(target,seconds,label):
            nonlocal peak
            start=d.ctrl[acts].copy();steps=round(seconds/m.opt.timestep)
            for k in range(steps):
                t=(k+1)/steps;blend=t*t*(3-2*t);d.ctrl[acts]=start+(target-start)*blend;mujoco.mj_step(m,d)
                if k%20==0:peak=max(peak,max(0.,-float(min(d.contact.dist,default=0))))
            poses.append(d.qpos.copy())
            row['phases'].append(dict(phase=label,time_s=float(d.time),motor_position=d.body('motor').xpos.tolist(),site_position=d.site('right_gripperframe').xpos.tolist()))
        try:
            target=d.ctrl[acts].copy();target[:5]=ik(m,d.qpos,'right',goal+[0,0,.015],R)
            target[5]=configuration()['gripper']['open_command_rad'];drive(target,2.,'approach')
            target[:5]=ik(m,d.qpos,'right',goal,R);drive(target,1.,'lower')
            target[5]=configuration()['gripper']['closed_command_rad'];drive(target,1.2,'close')
            held_from=d.body('motor').xpos.copy();lift_goal=d.site('right_gripperframe').xpos+[0,0,.025]
            target[:5]=ik(m,d.qpos,'right',lift_goal);drive(target,1.,'lift');drive(target,.5,'hold')
            forces=finger_forces(m,d,m.body('motor').id,'right')
            lifted=float(d.body('motor').xpos[2]-held_from[2]);site_error=float(np.linalg.norm(d.site('right_gripperframe').xpos-lift_goal))
            material=max(part_mesh_overlap(m,pose,'motor','right') for pose in poses)
            released_from=d.body('motor').xpos.copy();target[5]=configuration()['gripper']['open_command_rad'];drive(target,.5,'release')
            release=float(np.linalg.norm(d.body('motor').xpos-released_from))
            row.update(lift_m=lifted,site_error_m=site_error,hold_contact_forces_N=forces,release_translation_m=release,sampled_material_overlap_m=material,
                       passed=bool(lifted>.02 and site_error<.003 and min(forces['fixed'],forces['moving'])>.001 and forces['other']<.01 and release>.005 and peak<PENETRATION_LIMIT_M and material<PENETRATION_LIMIT_M))
        except ValueError as error:row['error']=str(error)
        valid=bool(np.isfinite(d.qpos).all() and not d.warning.number.any())
        row.update(peak_native_depth_m=peak,warnings=d.warning.number.tolist(),numerically_valid=valid,
                   scope='Scripted native-actuator attempt; sampled phase-end geometry, no pose actuation, welds or learned policy.')
        row['passed']=row['passed'] and valid;rows.append(row)
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,default=scene.ROOT/'artifacts_local/interaction-audit')
    p.add_argument('--sections',default='grasps+parts+table+reach+pickup');a=p.parse_args();a.out.mkdir(exist_ok=True,parents=True)
    report=dict(environment=environment_identity(),audit_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                scope='Native diagnostic interactions, not a learned policy or hardware calibration',
                criteria=dict(native_depth_m=PENETRATION_LIMIT_M,material_overlap_m='min(0.0005, 0.05*object_width)',hold_translation_m=.001,
                              asymmetric_rotation_rad=.03,release_translation_m=.005,initial_overlap_reject_m=1e-6),cases=[])
    def save(row):
        report['cases'].append(row);(a.out/'interactions.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(row),flush=True)
    sections=a.sections.split('+')
    if 'grasps' in sections:
        for shape,width,depth,mu in itertools.product(('box','sphere','cylinder'),(.004,.012,.024),(-.100,-.082),(.35,.7)):
            save(grasp_case(shape,width,depth,mu));gc.collect()
        for shape,width in itertools.product(('box','sphere'),(.004,.024)):
            save(grasp_case(shape,width,-.100,.7,dt=.00025))
            save(grasp_case(shape,width,-.100,.7,closed=-.1))
    if 'parts' in sections:
        for part,depth in itertools.product((*scene.PARTS,'driver'),(-.100,-.085)):
            save(grasp_case('box',.006,depth,None,part=part));gc.collect()
    if 'table' in sections:
        for row in table_cases():save(row)
    if 'reach' in sections:
        for row in reach_cases():save(row)
    if 'pickup' in sections:
        for row in pickup_cases():save(row)
    summary={}
    for family in sorted({r['family'] for r in report['cases']}):
        rows=[r for r in report['cases'] if r['family']==family]
        summary[family]=dict(cases=len(rows),passed=sum(r['passed'] is True for r in rows),failed=sum(r['passed'] is False for r in rows),measurements=sum(r['passed'] is None for r in rows))
    report['summary']=summary;(a.out/'interactions.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
