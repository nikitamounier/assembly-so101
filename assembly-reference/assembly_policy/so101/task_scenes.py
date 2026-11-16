"""Native-qualified task completion states for explicit scene initialization."""

from assembly_policy.so101.limits import PENETRATION_LIMIT_M
import hashlib
import json
from pathlib import Path
import tempfile

import mujoco
import numpy as np
from .scene import metadata, ASSETS, ASSEMBLY_TASKS
from .robot import home
from .fastener import origins, bearing_pose, target_rotations
from .goals import placement_errors, RECEIVERS
from .contact_audit import sampled_assembly_overlaps
from .workcell import configuration, environment_identity

TASKS = tuple(configuration()['prepared_scenes']['validated_tasks'])


def initial_scene(model):
    data=mujoco.MjData(model);home(model,data)
    return dict(qpos=data.qpos.copy(),qvel=data.qvel.copy(),ctrl=data.ctrl.copy(),
                preparation=dict(kind='initial',completed_tasks=[],scope='Initial loose physical scene'))


def geometric_scene(model, task):
    """Seed the current part and predecessors from the shared reward geometry."""
    if task not in TASKS:raise ValueError('Unknown or unqualified assembly task')
    data=mujoco.MjData(model);home(model,data);meta=metadata();targets=origins(meta);rotations=target_rotations(meta)
    settings=configuration()['prepared_scenes']
    quaternion=np.asarray(settings['completion_base_quaternions'][task],float)
    matrix=np.empty(9);mujoco.mju_quat2Mat(matrix,quaternion);rotation=matrix.reshape(3,3)
    bounds=np.asarray(meta['base']['bounds'])
    corners=np.array(np.meshgrid(*bounds.T,indexing='ij')).reshape(3,-1).T
    bq=int(model.joint('base_free').qposadr[0]);position=data.qpos[bq:bq+3].copy()
    position[2]=-float((corners@rotation.T)[:,2].min())+settings['table_clearance_m']
    data.qpos[bq:bq+3]=position;data.qpos[bq+3:bq+7]=quaternion
    for name in ASSEMBLY_TASKS[:ASSEMBLY_TASKS.index(task)+1]:
        q=int(model.joint(name+'_free').qposadr[0])
        data.qpos[q:q+3]=position+rotation@(targets[name]-targets['base'])
        local=np.empty(4);mujoco.mju_mat2Quat(local,rotations[name].ravel())
        mujoco.mju_mulQuat(data.qpos[q+3:q+7],quaternion,local)
    mujoco.mj_forward(model,data)
    return data


def seed_scene(model,task):
    """Explicit target seed retaining the already-qualified predecessor assembly."""
    data=geometric_scene(model,task)
    index=ASSEMBLY_TASKS.index(task)
    if index:
        # Retain the actual completed prefix, including the screw's seated
        # phase. Reinitializing predecessors at ideal CAD poses discarded it.
        previous=task_scene(model,ASSEMBLY_TASKS[index-1])
        prior=mujoco.MjData(model);prior.qpos[:]=previous['qpos'];mujoco.mj_kinematics(model,prior)
        old_base=prior.body('base');new_base=data.body('base')
        anchor=new_base.xpos.copy();rotation=new_base.xmat.reshape(3,3)@old_base.xmat.reshape(3,3).T
        inverse=old_base.xquat.copy();inverse[1:]*=-1;delta=np.empty(4)
        mujoco.mju_mulQuat(delta,new_base.xquat,inverse)
        settings=configuration()['prepared_scenes']['completion_base_quaternions']
        if np.allclose(settings[task],settings[ASSEMBLY_TASKS[index-1]],rtol=0,atol=1e-12):
            anchor=old_base.xpos.copy();rotation=np.eye(3);delta=np.array([1.,0,0,0])
        for name in ('base',*ASSEMBLY_TASKS[:index]):
            q=int(model.joint(name+'_free').qposadr[0]);body=prior.body(name)
            data.qpos[q:q+3]=anchor+rotation@(body.xpos-old_base.xpos)
            mujoco.mju_mulQuat(data.qpos[q+3:q+7],delta,body.xquat)
        mujoco.mj_kinematics(model,data)
        if task in ('horn','yoke'):
            name=RECEIVERS[task];receiver=data.body(name);q=int(model.joint(task+'_free').qposadr[0]);meta=metadata()
            targets=origins(meta);rotations=target_rotations(meta);frame=receiver.xmat.reshape(3,3)@rotations[name].T
            data.qpos[q:q+3]=receiver.xpos+frame@(targets[task]-targets[name])
            mujoco.mju_mat2Quat(data.qpos[q+3:q+7],(frame@rotations[task]).ravel())
        mujoco.mj_forward(model,data)
    return data


def _measure(model, data, task):
    errors=placement_errors(data,task,metadata(),origins(metadata()))
    depth=max([0.,*[-float(c.dist) for c in data.contact]])
    completed=ASSEMBLY_TASKS[:ASSEMBLY_TASKS.index(task)+1]
    material=sampled_assembly_overlaps(data,['base',*completed],ASSETS)
    overlap=max(material.values(),default=0.)
    if (not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all()
            or data.warning.number.any() or max(errors.values())>=.001 or depth>PENETRATION_LIMIT_M or overlap>PENETRATION_LIMIT_M):
        raise ValueError(f'Invalid {task} target: geometry={max(errors.values()):g}, contact={depth:g}, overlap={overlap:g}')
    horn_axis=data.body('horn').xmat.reshape(3,3)[:,2]
    motor_axis=data.body('motor').xmat.reshape(3,3)[:,2]
    return dict(kind='reward_target',task=task,completed_tasks=list(completed),
                predecessors=list(completed[:-1]),placement_error_m=errors,
                base_quaternion=data.body('base').xquat.tolist(),
                horn_flatness_error_rad=float(np.arccos(np.clip(motor_axis@horn_axis,-1,1))) if 'horn' in completed else None,
                screw_insertion_axis_world=(-motor_axis).tolist(),
                max_contact_depth_m=depth,sampled_motor_base_overlap_m=material['base/motor'],
                sampled_material_overlap_m=material,
                scope='Explicit initialization qualified by native policy reward; not executed or learned assembly')


def audit_reward(model, prepared, task, *, ticks=15, capture=False):
    """Exercise unchanged policy release, thread, safety and 0.4 s pose gates."""
    from .env import BatchedAssemblyEnv,EnvConfig,STATE_SPEC
    data=mujoco.MjData(model);data.qpos[:]=prepared['qpos'];data.qvel[:]=prepared['qvel'];data.ctrl[:]=prepared['ctrl']
    mujoco.mj_forward(model,data)
    state=np.empty(mujoco.mj_stateSize(model,STATE_SPEC));mujoco.mj_getState(model,data,state,STATE_SPEC)
    env=BatchedAssemblyEnv([task],[0],EnvConfig(nthread=1,stability_metric='pose',max_contact_depth=PENETRATION_LIMIT_M),model_factory=lambda **_:model)
    try:
        env.set_states([0],state[None]);history=[];qualified=None
        for _ in range(ticks):
            reward,done,info=env.step(np.zeros((1,12)))
            history.append(dict(success=bool(info['success'][0]),safety=bool(info['safety'][0]),
                                placement_error_m=float(info['d_asm'][0]),reward=float(reward[0])))
            if history[-1]['safety']:break
            if history[-1]['success']:
                if capture:
                    d=env.data[0];qualified=dict(qpos=d.qpos.copy(),qvel=d.qvel.copy(),ctrl=env.last_ctrl[0].copy())
                    break
        report=dict(tested=True,physics_seconds=len(history)*.04,
                    passed=any(x['success'] for x in history) and not any(x['safety'] for x in history),
                    stability_metric='pose',max_contact_depth_m=PENETRATION_LIMIT_M,final=history[-1],history=history)
        return (report,qualified) if capture else report
    finally:env.close()


def task_scene(model, task):
    """Return only a state that actually earned native reward under free physics."""
    if task not in TASKS:raise ValueError('Unknown or unqualified assembly task')
    # Local, ignored cache binds both the full physical identity and qualification
    # code. Fresh hosts qualify independently; generated states never enter git.
    code=b''.join((Path(__file__).with_name(n)).read_bytes() for n in ('limits.py','task_scenes.py','env.py','goals.py','contact_audit.py','fastener.py'))
    identity=environment_identity();key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()+code).hexdigest()
    cache=ASSETS/'task_targets'/key/(task+'.npz')
    if cache.exists():
        with np.load(cache,allow_pickle=False) as archive:
            result={n:archive[n].copy() for n in ('qpos','qvel','ctrl')}
            result['preparation']=json.loads(str(archive['preparation']))
        data=mujoco.MjData(model);data.qpos[:]=result['qpos'];data.qvel[:]=result['qvel'];mujoco.mj_forward(model,data)
        _measure(model,data,task)
        if not result['preparation']['native_reward']['passed']:raise ValueError('Unqualified cached target')
        return result
    data=seed_scene(model,task)
    # Free settling is preparation, outside the later reward-validation episode.
    mujoco.mj_step(model,data,nstep=1200)
    if task=='screw':
        q=int(model.joint('screw_free').qposadr[0])
        position,quaternion=bearing_pose(model,data,metadata())
        data.qpos[q:q+3]=position;data.qpos[q+3:q+7]=quaternion
        data.qvel[:]=0
        mujoco.mj_forward(model,data)
    seed=dict(qpos=data.qpos.copy(),qvel=data.qvel.copy(),ctrl=data.ctrl.copy())
    _measure(model,data,task)
    report,result=audit_reward(model,seed,task,ticks=60,capture=True)
    if not report['passed']:raise ValueError(f'{task} failed native reward validation: {report["final"]}')
    data.qpos[:]=result['qpos'];data.qvel[:]=result['qvel'];mujoco.mj_forward(model,data)
    result['preparation']=dict(_measure(model,data,task),native_reward=report,environment_identity=identity)
    cache.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=cache.parent,prefix=task+'.',suffix='.npz',delete=False) as stream:
        temporary=Path(stream.name)
        np.savez_compressed(stream,**{n:result[n] for n in ('qpos','qvel','ctrl')},preparation=json.dumps(result['preparation']))
    try:temporary.replace(cache)
    finally:temporary.unlink(missing_ok=True)
    return result
