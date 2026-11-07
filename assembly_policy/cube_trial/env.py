"""Sparse cube task with stock MuJoCo rollout and complete per-instance user state.

Actions integrate the previous position command, so neutral input retains the
reset's physical grasp preload. Rollout exposes full-physics endpoints but not
per-instance final solver warmstarts; the initial boundary warmstart is retained.
"""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from mujoco import rollout

from .core import JOINTS, RIGHT_GRASP_LOOKBACK_S, cube_inside_target, model, placement_contacts, success

HISTORY = 5
PROPRIO_DIM = 50
ACT_DIM = 12
ARM_ACTIONS = np.array([0, 1, 2, 3, 4, 6, 7, 8, 9, 10])
PREVIOUS_ACTION_START = 38
TASK_NAMES = ("cube_handoff",)
STATE_SPEC = mujoco.mjtState.mjSTATE_INTEGRATION
PHYSICS_SPEC = mujoco.mjtState.mjSTATE_FULLPHYSICS
CONTROL_SPEC = (
    mujoco.mjtState.mjSTATE_CTRL
    | mujoco.mjtState.mjSTATE_MOCAP_POS
    | mujoco.mjtState.mjSTATE_MOCAP_QUAT
    | mujoco.mjtState.mjSTATE_QFRC_APPLIED
    | mujoco.mjtState.mjSTATE_XFRC_APPLIED
)


@dataclass
class EnvConfig:
    suite: str = "cube_trial"
    control_dt: float = 0.04
    horizon_s: float = 10.0
    action_scale_joints: tuple = (0.035,) * 5
    gripper_scale: float = 0.05
    nthread: int = 2
    proprio_dim: int = PROPRIO_DIM
    success_frames: int = 3
    cost_coef: float = 2e-5
    action_reference: str = "previous_command"
    reset_context_path: str | None = None


def parse_scales(value):
    values = tuple(float(v) for v in value.replace(",", "+").split("+"))
    if len(values) != 5 or not np.isfinite(values).all() or min(values) <= 0:
        raise ValueError("Five finite positive joint scales required")
    return values


def resolve_tasks(value):
    if value in ("all", "cube_trial", "cube_handoff"):
        return list(TASK_NAMES)
    raise ValueError("Cube task is cube_handoff")


class BatchedCubeEnv:
    def __init__(self, task_names, task_of_env, ecfg=None, *, model_factory=None):
        self.ecfg = ecfg or EnvConfig()
        if list(task_names) != list(TASK_NAMES):
            raise ValueError("Expected only cube_handoff")
        if self.ecfg.action_reference != "previous_command":
            raise ValueError(
                "Cube deltas must integrate previous commanded positions to retain grasp preload"
            )
        if not np.isfinite(self.ecfg.cost_coef) or self.ecfg.cost_coef < 0:
            raise ValueError("Control cost must be finite and nonnegative")
        if self.ecfg.nthread < 1 or self.ecfg.success_frames < 1:
            raise ValueError("Positive simulation/thread budgets required")
        self.m = (model_factory or model)()
        if self.m.nplugin or self.m.neq or np.any(self.m.geom_type == mujoco.mjtGeom.mjGEOM_SDF):
            raise ValueError("Cube backend requires plugin-free native contacts and no equality attachments")
        self.reset_context = {}
        if self.ecfg.reset_context_path:
            path = Path(self.ecfg.reset_context_path)
            manifest = json.loads(path.with_name("manifest.json").read_text())
            if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["reset_context_sha256"]:
                raise ValueError("Reset context checksum differs")
            with np.load(path) as context:
                for j, key in enumerate(context["state_sha256"]):
                    self.reset_context[str(key)] = (
                        context["histories"][j].copy(), context["previous_actions"][j].copy(),
                        float(context["right_grasp_age_s"][j]), bool(context["carry_started"][j]),
                    )
        self.task_names = list(task_names)
        self.task = np.asarray(task_of_env, np.int64).copy()
        self.N = len(self.task)
        if np.any(self.task != 0):
            raise ValueError("Only task index0 exists")
        self.data = [mujoco.MjData(self.m) for _ in range(self.N)]
        self.work = [mujoco.MjData(self.m) for _ in range(self.ecfg.nthread)]
        self.engine = rollout.Rollout(nthread=self.ecfg.nthread if self.ecfg.nthread > 1 else 0)
        self.nstate = mujoco.mj_stateSize(self.m, STATE_SPEC)
        self.state = np.zeros((self.N, self.nstate))
        self.prev_action = np.zeros((self.N, ACT_DIM))
        self.last_ctrl = np.zeros((self.N, ACT_DIM))
        self.hist = np.zeros((self.N, HISTORY, PROPRIO_DIM), np.float32)
        self.t = np.zeros(self.N, np.int64)
        self.stable = np.zeros(self.N, np.int64)
        self.grounded = np.zeros(self.N, bool)
        self.right_touchdown = np.zeros(self.N, bool)
        self.last_right_grasp_time = np.full(self.N, -np.inf)
        self.carry_started = np.zeros(self.N, bool)
        self.continuity_broken = np.zeros(self.N, bool)
        self.off_target_table_contact = np.zeros(self.N, bool)
        self.delivery_touchdown = np.zeros(self.N, bool)
        self.touchdown_data = mujoco.MjData(self.m)
        self.cube_geom = self.m.geom("cube_geom").id
        self.table_geom = self.m.geom("table").id
        self.right_pads = {self.m.geom("right_" + p + "_pad").id for p in ("fixed", "moving")}
        self.horizon = round(self.ecfg.horizon_s / self.ecfg.control_dt)
        self.substeps = round(self.ecfg.control_dt / self.m.opt.timestep)
        if (
            self.horizon < 1
            or self.substeps < 1
            or not np.isclose(self.substeps * self.m.opt.timestep, self.ecfg.control_dt)
        ):
            raise ValueError("Control dt must be a positive integer multiple of native timestep")
        self.qi = np.array(
            [self.m.joint(s + "_" + j).qposadr[0] for s in ("left", "right") for j in JOINTS], int
        )
        self.sites = [self.m.site(s + "_gripperframe").id for s in ("left", "right")]
        self.fingers = [
            self.m.body(s + "_" + b).id for s in ("left", "right") for b in ("gripper", "moving_jaw_so101_v1")
        ]
        self.initial = np.empty((self.N, mujoco.mj_stateSize(self.m, PHYSICS_SPEC)))
        self.control = np.empty((self.N, mujoco.mj_stateSize(self.m, CONTROL_SPEC)))
        self.warmstart = np.zeros((self.N, self.m.nv))
        self.frames = np.zeros((self.N, PROPRIO_DIM), np.float32)

    def close(self):
        self.engine.close()

    def set_states(self, ids, states, task_ids=None):
        ids = np.asarray(ids, int)
        states = np.asarray(states, float)
        if states.shape != (len(ids), self.nstate) or not np.isfinite(states).all():
            raise ValueError("Expected finite complete mjSTATE_INTEGRATION rows")
        if task_ids is not None:
            if np.any(np.asarray(task_ids) != 0):
                raise ValueError("Only cube task0 exists")
            self.task[ids] = task_ids
        for row, i in enumerate(ids):
            d = self.data[i]
            mujoco.mj_setState(self.m, d, states[row], STATE_SPEC)
            d.time = 0.0
            self.last_ctrl[i] = d.ctrl
            mujoco.mj_forward(self.m, d)
            contacts = placement_contacts(self.m, d)
            ground, held = contacts["table_touching"], contacts["right_held"]
            # Mid-trajectory banks have an assumed valid preceding carry.
            self.carry_started[i] = np.any(contacts["pad_touching"]) and (not ground or held)
            self.continuity_broken[i] = False
            self.off_target_table_contact[i] = (
                self.carry_started[i] and ground and not cube_inside_target(self.m, d)
            )
            self.delivery_touchdown[i] = ground and held and cube_inside_target(self.m, d)
            self.grounded[i] = ground
            self.last_right_grasp_time[i] = 0.0 if held else -np.inf
            self.right_touchdown[i] = ground and held
        self.prev_action[ids] = 0
        self.t[ids] = 0
        self.stable[ids] = 0
        self._refresh()
        self.hist[ids] = self.frames[ids, None]
        for row, i in enumerate(ids):
            context = self.reset_context.get(hashlib.sha256(states[row].tobytes()).hexdigest())
            if context is not None:
                history, previous_action, grasp_age, carry_started = context
                self.prev_action[i] = previous_action
                self.hist[i] = history
                self.last_right_grasp_time[i] = -grasp_age
                self.carry_started[i] = carry_started
        if self.reset_context:
            self._refresh()

    def _support_contacts(self, d):
        contacts = placement_contacts(self.m, d)
        return contacts["table_touching"], contacts["right_held"]

    def _touchdown_sample(self, i, now, ground, held):
        if held:
            self.last_right_grasp_time[i] = now
        if ground and not self.grounded[i]:
            elapsed = now - self.last_right_grasp_time[i]
            self.right_touchdown[i] = 0 <= elapsed <= RIGHT_GRASP_LOOKBACK_S + 1e-12
        self.grounded[i] = ground

    def _continuity_sample(self, i, ground, left_touch, any_touch, inside_target):
        if not self.carry_started[i] and not ground and left_touch:
            self.carry_started[i] = True
        if self.carry_started[i] and ground and not inside_target:
            self.off_target_table_contact[i] = True
        if not self.carry_started[i] or self.delivery_touchdown[i]:
            return
        # Only outside-target table contact invalidates the preceding carry.
        if ground and inside_target and self.right_touchdown[i]:
            self.delivery_touchdown[i] = True

    def _track_touchdown(self, i, trajectory):
        d = self.touchdown_data
        mujoco.mj_setState(self.m, d, self.state[i], STATE_SPEC)
        d.ctrl[:] = self.data[i].ctrl
        for state in trajectory:
            # Observe every native substep, including grasps above the table.
            mujoco.mj_setState(self.m, d, state, PHYSICS_SPEC)
            mujoco.mj_forward(self.m, d)
            contacts = placement_contacts(self.m, d)
            ground, held = contacts["table_touching"], contacts["right_held"]
            self._touchdown_sample(i, float(d.time), ground, held)
            touching = contacts["pad_touching"]
            inside = bool(ground and self.carry_started[i] and cube_inside_target(self.m, d))
            self._continuity_sample(i, ground, bool(np.any(touching[:2])), bool(np.any(touching)), inside)

    def _refresh(self):
        prop = []
        tcp_p = []
        tcp_R = []
        part_p = []
        part_q = []
        for i, d in enumerate(self.data):
            mujoco.mj_forward(self.m, d)
            row = []
            for site in self.sites:
                q = np.empty(4)
                mujoco.mju_mat2Quat(q, d.site_xmat[site])
                q *= 1 if q[0] >= 0 else -1
                vel = np.empty(6)
                mujoco.mj_objectVelocity(self.m, d, mujoco.mjtObj.mjOBJ_SITE, site, vel, 0)
                row.extend(np.r_[d.site_xpos[site], q, vel[3:], vel[:3]])
            row.extend(d.qpos[self.qi])
            row.extend(self.prev_action[i])
            prop.append(row)
            tcp_p.append([d.xpos[b].copy() for b in self.fingers])
            tcp_R.append([d.xmat[b].reshape(3, 3).copy() for b in self.fingers])
            part_p.append([d.body("cube").xpos.copy(), d.body("target").xpos.copy()])
            part_q.append([d.body("cube").xquat.copy(), d.body("target").xquat.copy()])
            mujoco.mj_getState(self.m, d, self.state[i], STATE_SPEC)
        self.frames = np.asarray(prop, np.float32)
        self.compact = dict(
            tcp_p=np.asarray(tcp_p, np.float32),
            tcp_R=np.asarray(tcp_R, np.float32),
            part_p=np.asarray(part_p, np.float32),
            part_q=np.asarray(part_q, np.float32),
            finger=np.array([d.qpos[self.qi[[5, 11]]] for d in self.data], np.float32),
        )

    def observe(self):
        return dict(
            proprio=self.hist.reshape(self.N, -1).copy(),
            task=self.task.copy(),
            **{k: v.copy() for k, v in self.compact.items()},
        )

    def step(self, action):
        action = np.asarray(action, float)
        if action.shape != (self.N, ACT_DIM) or not np.isfinite(action).all():
            raise ValueError("Expected finite (N,12) actions")
        current = self.last_ctrl
        scales = np.tile(np.r_[self.ecfg.action_scale_joints, self.ecfg.gripper_scale], 2)
        ctrl = np.clip(
            current + scales * action, self.m.actuator_ctrlrange[:, 0], self.m.actuator_ctrlrange[:, 1]
        )
        expected = np.array([d.time for d in self.data]) + self.ecfg.control_dt
        for i, d in enumerate(self.data):
            d.ctrl[:] = ctrl[i]
            mujoco.mj_getState(self.m, d, self.initial[i], PHYSICS_SPEC)
            mujoco.mj_getState(self.m, d, self.control[i], CONTROL_SPEC)
            self.warmstart[i] = d.qacc_warmstart
        trajectory, _ = self.engine.rollout(
            self.m,
            self.work,
            self.initial,
            self.control[:, None, :],
            control_spec=CONTROL_SPEC,
            nstep=self.substeps,
            initial_warmstart=self.warmstart,
        )
        if not np.isfinite(trajectory).all() or not np.allclose(
            trajectory[:, -1, 0], expected, atol=1e-8, rtol=0
        ):
            raise FloatingPointError("Native MuJoCo reset or produced nonfinite trajectory")
        for i, d in enumerate(self.data):
            self._track_touchdown(i, trajectory[i])
            mujoco.mj_setState(self.m, d, trajectory[i, -1], PHYSICS_SPEC)
        self.last_ctrl[:] = ctrl
        self.prev_action[:] = action
        self.t += 1
        self._refresh()
        candidate = (
            self.right_touchdown
            & self.delivery_touchdown
            & ~self.off_target_table_contact
            & np.array([success(self.m, d) for d in self.data], bool)
        )
        self.stable = np.where(candidate, self.stable + 1, 0)
        won = self.stable >= self.ecfg.success_frames
        failed = self.off_target_table_contact.copy()
        timeout = (self.t >= self.horizon) & ~won & ~failed
        self.hist = np.concatenate((self.hist[:, 1:], self.frames[:, None]), axis=1)
        false = np.zeros(self.N, bool)
        velocity = np.array([d.qvel[self.m.jnt_dofadr[self.m.actuator_trnid[:, 0]]] for d in self.data])
        torque = np.array([d.actuator_force for d in self.data])
        limit = self.m.actuator_forcerange[:, 1]
        cost = self.ecfg.cost_coef * (
            (action**2).sum(1) + (velocity**2 / 25).sum(1) + ((torque / limit) ** 2).sum(1)
        )
        distance = np.array(
            [np.linalg.norm(d.body("cube").xpos[:2] - d.site("target_center").xpos[:2]) for d in self.data],
            np.float32,
        )
        info = dict(success=won, safety=false, timeout=timeout, d_asm=distance, cost=cost, unstable=false)
        return (won.astype(np.float32) - cost).astype(np.float32), won | failed | timeout, info


BatchedAssemblyEnv = BatchedCubeEnv
