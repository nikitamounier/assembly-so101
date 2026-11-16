"""Meta Quest owner on the native Hiwonder assembly plant (simulation only)."""
from pathlib import Path

import mujoco
import numpy as np

from .control import SimulationController
from .runtime import NativeFrames, PreviewRuntime
from assembly_policy.so101 import scene
from assembly_policy.so101.robot import home as initialize_home
from assembly_policy.so101.workcell import configuration, environment_identity
from assembly_policy.so101.task_scenes import TASKS, initial_scene, task_scene
from .runtime import MAX_AGE_NS
from .visual_scene import scene_payload

HERE = Path(__file__).resolve().parent
ROOT = scene.ROOT


def load_scene():
    """Use the exact native training factory; add only teleoperation metadata."""
    model = scene.model(contact_monitor=True)
    data = mujoco.MjData(model)
    initialize_home(model, data)
    settings = configuration()
    camera = settings['main_camera']
    main = model.camera('workspace_camera').id
    rotation = np.empty(9)
    mujoco.mju_quat2Mat(rotation, model.cam_quat[main])
    descriptor = dict(arms={}, home_qpos=data.qpos.tolist(), home_ctrl=data.ctrl.tolist(),
        environment_identity=environment_identity(), physical_settings=settings,
        camera='workspace_camera', cameras={'main':'workspace_camera', 'left':'left_wrist_view', 'right':'right_wrist_view'},
        main_camera=dict(camera, rotation=rotation.reshape(3,3).tolist(), height_above_table_m=camera['position'][2]),
        frame='+x into workspace, +y left, +z up; metres/radians',
        tool_definition='Source gripperframe; nominal gripping centre; position-priority five-axis IK')
    for side in ('left', 'right'):
        joints = [side+'_'+n for n in scene.JOINTS[:5]]
        jids = [model.joint(n).id for n in joints]
        descriptor['arms'][side] = dict(joint_names=joints, actuator_names=joints,
            joint_ids=jids, actuator_ids=[model.actuator(n).id for n in joints],
            qpos_indices=model.jnt_qposadr[jids].tolist(), dof_indices=model.jnt_dofadr[jids].tolist(),
            gripper_joint=side+'_gripper', gripper_actuator=side+'_gripper',
            gripper_qpos_index=int(model.joint(side+'_gripper').qposadr[0]),
            gripper_actuator_id=model.actuator(side+'_gripper').id,
            gripper_closed=settings['gripper']['closed_command_rad'], gripper_open=settings['gripper']['open_command_rad'], tool_site=side+'_gripperframe',
            base_body=side+'_base', wrist_camera=True)
    return model, descriptor


class AssemblyController(SimulationController):
    stop_on_joint_velocity = False

    def __init__(self, model, description):
        super().__init__(model, description)
        self.native_substeps = round(.002/model.opt.timestep)
        if self.native_substeps < 1 or abs(self.native_substeps*model.opt.timestep-.002)>1e-12:
            raise ValueError('Physics timestep must divide the 2 ms controller period')
        if 'initial_qvel' in description:
            self.data.qvel[:] = description['initial_qvel']
            mujoco.mj_forward(model, self.data)
        self.robot_side = {}
        for body in range(1, model.nbody):
            name = model.body(body).name
            if name in ('left_base', 'right_base'):
                self.robot_side[body] = name.split('_')[0]
            elif int(model.body_parentid[body]) in self.robot_side:
                self.robot_side[body] = self.robot_side[int(model.body_parentid[body])]
        self.fingers = {model.body(s+n).id for s in ('left','right')
                        for n in ('_gripper','_moving_jaw_so101_v1')}
        self.mounts = {model.body(s+'_base').id for s in ('left','right')}
        self.speed = 1.2
        self.acceleration = 16.
        self.cartesian_speed = .5
        self.mapper.acquire_radius = None

    def collisions(self):
        """User-selected free simulation: no software contact checks or Holds."""
        return []


class AssemblyRuntime(PreviewRuntime):
    def __init__(self, model, description, **kwargs):
        super().__init__(model, description, controller_factory=AssemblyController, receipt_timing=True, require_both_controllers=False, **kwargs)
        self.visual_scene = scene_payload(model, self.owner_id)
        self.input_poll_steps = 2
        self.stage_index = 0
        self.task_index = 0
        self.scene_generation = 0
        self.prepared_scenes = {}
        self.preparation = None

    def prepare_tasks(self):
        self.prepared_scenes = {'initial': initial_scene(self.model),
                               **{task: task_scene(self.model, task) for task in TASKS}}

    def _fresh(self, now):
        # Stereo shows measured geometry. Hidden wrist JPEGs must not revoke
        # its controls; the unchanged 200 ms owner/input leases still apply.
        with self.lock:
            return (self.owner_acquired_ns is not None and 0 <= now-self.owner_acquired_ns < MAX_AGE_NS
                    and self.render_alive and not self.render_error)

    def apply_scene_action(self, action):
        self._owned()
        stage = 0 if action == 'environment_reset' else (self.stage_index+1) % (len(TASKS)+1)
        task = TASKS[stage-1] if stage else 'initial'
        try:
            prepared = self.prepared_scenes.get(task)
            if prepared is None:
                prepared = self.prepared_scenes[task] = task_scene(self.model, task) if stage else initial_scene(self.model)
        except ValueError as error:
            self.controller.enabled = False
            self.controller.hold('prepared_scene_rejected:'+str(error))
            return
        self.controller.stop()
        if self.controller._torque_disable_owned:
            self.model.opt.disableflags = int(self.model.opt.disableflags) & ~self.controller._actuation_disable_bit
        description = dict(self.description, home_qpos=prepared['qpos'], initial_qvel=prepared['qvel'])
        self.controller = self.controller_factory(self.model, description)
        self.controller.reason = 'environment_reset' if not stage else 'task_target_loaded'
        self.stage_index = stage
        self.task_index = max(0, stage-1)
        self.preparation = prepared['preparation']
        self.scene_generation += 1
        self._record('events', dict(event=self.controller.reason, task=task, scene_generation=self.scene_generation,
                                   preparation=self.preparation, server_ns=self.clock()))

    def get_visual_scene(self):
        return self.visual_scene

    def scene_state(self):
        data = self.controller.data
        return dict(simulation_time_s=float(data.time),execution='simulation',
                    task=TASKS[self.task_index], task_index=self.task_index, stage_index=self.stage_index,
                    scene_generation=self.scene_generation, preparation=self.preparation,
                    body_poses=np.concatenate((data.xpos,data.xquat),axis=1).tolist())

    def get_config(self):
        return dict(super().get_config(), backend='assembly', execution='simulation', application='robotics-assembly-3.0/teleop',
                    parts=list(scene.PARTS)+['driver'], tracking_profile='direct-clutch-sim',
                    collision_monitor=False, measured_speed_stop=False, input_timing="server_receipt", acquire_from_anywhere=True, simple_y=True, require_both_controllers=False,
                    motion_limits=dict(joint_reference_rad_s=1.2,joint_acceleration_rad_s2=16.,cartesian_m_s=.5),
                    environment_identity=self.description['environment_identity'],
                    contact_search=dict(self.description['physical_settings']['contact_search'],
                        motor_base=self.description['physical_settings']['motor_base_search'],
                        horn=self.description['physical_settings']['horn_search'],
                        assembly=self.description['physical_settings']['assembly_search'],
                        contact_capacity=int(self.model.opt.sdf_initpoints)),
                    immersive_view=True, contact_capacity_guard=True,
                    task_scenes=True, tasks=list(TASKS), environment_reset_hold_ms=2000,
                    physics_solver=dict(iterations=int(self.model.opt.iterations),tolerance=float(self.model.opt.tolerance),
                        friction_iterations=int(self.model.opt.noslip_iterations),sleeping=bool(self.model.opt.enableflags&int(mujoco.mjtEnableBit.mjENBL_SLEEP))),
                    camera_scope='Head-tracked stereo geometry; optional mono wrist close-ups')


class AssemblyFrames(NativeFrames):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scene_option.geomgroup[3] = 0  # contact debug meshes, not the source visuals
