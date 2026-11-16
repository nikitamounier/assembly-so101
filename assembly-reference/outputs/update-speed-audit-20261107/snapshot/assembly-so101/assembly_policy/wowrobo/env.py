"""Current native assembly: task rewards, real contacts, 12 policy actions.

+1 terminates once after 0.4 s stable seating. All native substeps are screened
for deep contacts; transient pass-through is never paid as success.
"""

from dataclasses import dataclass

import mujoco
import numpy as np
from mujoco import rollout

from ..so101.env import (
    ACT_DIM as ACT_DIM,
)
from ..so101.env import (
    ARM_ACTIONS as ARM_ACTIONS,
)
from ..so101.env import (
    HISTORY as HISTORY,
)
from ..so101.env import PREVIOUS_ACTION_START as PREVIOUS_ACTION_START
from ..so101.env import (
    PROPRIO_DIM as PROPRIO_DIM,
)
from ..so101.env import parse_scales as parse_scales
from ..so101.wowrobo_obs import GRIPS, WOW_TASKS
from ..so101.wowrobo_reward import RewardRegions
from .runtime import DEFAULT_PROFILE, DEFAULT_REVIEW, load_runtime

TASK_NAMES = WOW_TASKS
STATE_SPEC = mujoco.mjtState.mjSTATE_FULLPHYSICS


def resolve_tasks(value):
    names = list(TASK_NAMES) if value in ("all", "wowrobo") else value.split("+")
    if not names or any(n not in TASK_NAMES for n in names):
        raise ValueError(f"Expected current assembly tasks: {TASK_NAMES}")
    return names


@dataclass
class EnvConfig:
    suite: str = "wowrobo"
    profile_path: str = str(DEFAULT_PROFILE)
    review_model: str = str(DEFAULT_REVIEW)
    control_dt: float = 0.04
    horizon_s: float = 15.0
    action_scale_joints: tuple = (0.035,) * 5
    gripper_scale: float = 0.05
    cost_coef: float = 2e-5
    nthread: int = 1
    max_contact_depth: float = 0.0005
    stability_metric: str = "pose"
    safety_penalty: float = 0.0
    stable_translation: float = 0.00025
    stable_rotation: float = 0.01
    settle_s: float = 0.4

    def __post_init__(self):
        self.action_scale_joints = tuple(self.action_scale_joints)
        if self.control_dt <= 0 or self.horizon_s <= 0 or self.settle_s < 0:
            raise ValueError("Episode durations must be positive; settling must be nonnegative")


class BatchedAssemblyEnv:
    def __init__(self, task_names, task_of_env, ecfg=None):
        self.ecfg = ecfg or EnvConfig()
        self.task_names = list(task_names)
        self.task = np.asarray(task_of_env, dtype=np.int64)
        self.N = len(self.task)
        self.m, self.profile, self.model_signature = load_runtime(self.ecfg.profile_path)
        if self.ecfg.max_contact_depth is None or self.ecfg.max_contact_depth <= 0:
            raise ValueError("Current assembly requires a positive contact guard")
        self.data = [mujoco.MjData(self.m) for _ in range(self.N)]
        self.work = [mujoco.MjData(self.m) for _ in range(max(1, self.ecfg.nthread))]
        self.engine = rollout.Rollout(nthread=self.ecfg.nthread if self.ecfg.nthread > 1 else 0)
        self.full_size = mujoco.mj_stateSize(self.m, STATE_SPEC)
        self.extra_reset = self.profile.get("training_reset_payload") == "fullphysics-eq-ctrl-warmstart-v1"
        self.eq_end = self.full_size + self.m.neq
        self.episode_friction = self.profile.get("training_episode_friction") == "native-userdata-v1"
        self.nstate = self.eq_end + (self.m.nu + self.m.nv if self.extra_reset else 0) + (3 if self.episode_friction else 0)
        self.reset_bias = np.zeros((self.N, self.m.nu))
        self.initial_warmstart = np.zeros((self.N, self.m.nv))
        self.needs_warmstart = np.zeros(self.N, bool)
        self.state = np.zeros((self.N, self.nstate))
        self.t = np.zeros(self.N, np.int64)
        self.horizon = round(self.ecfg.horizon_s / self.ecfg.control_dt)
        self.substeps = round(self.ecfg.control_dt / self.m.opt.timestep)
        if self.substeps < 1 or not np.isclose(self.substeps * self.m.opt.timestep, self.ecfg.control_dt):
            raise ValueError("Control interval must be a timestep multiple")
        self.ctrl_adr = self.m.jnt_qposadr[self.m.actuator_trnid[:, 0]]
        self.prev_action = np.zeros((self.N, ACT_DIM))
        self.last_ctrl = np.zeros_like(self.prev_action)
        self.hist = np.zeros((self.N, HISTORY, PROPRIO_DIM), np.float32)
        review = mujoco.MjModel.from_xml_path(self.ecfg.review_model)
        self.obs_names = tuple(review.body(i).name for i in range(1, review.nbody))
        self.reward = RewardRegions(self.m, self.profile["source_model_sha256"])
        if not self.reward.compatible:
            raise ValueError("Reward contract differs from physical profile")
        self.stable = np.zeros(self.N, np.int64)
        self.terminated = np.zeros(self.N, bool)
        self.stable_pose = [None] * self.N
        sid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SENSOR, "training_min_contact_distance")
        self.sensor_adr = int(self.m.sensor_adr[sid])
        self.frames = None
        self.native_warning = np.zeros(self.N, bool)

    def close(self):
        self.engine.close()

    def set_states(self, ids, states, task_ids=None):
        ids = np.asarray(ids)
        states = np.asarray(states)
        if states.shape != (len(ids), self.nstate):
            raise ValueError("Reset schema must include full physics and explicit equality activation")
        if not np.isfinite(states).all():
            raise ValueError("Nonfinite reset")
        self.state[ids] = states
        self.state[ids, 0] = 0
        if task_ids is not None:
            self.task[ids] = task_ids
        self.prev_action[ids] = 0
        self.t[ids] = 0
        self.stable[ids] = 0
        self.terminated[ids] = False
        for i in ids:
            self.stable_pose[i] = None
            mujoco.mj_setState(self.m, self.data[i], self.state[i, : self.full_size], STATE_SPEC)
            self.last_ctrl[i] = (self.state[i, self.eq_end:self.eq_end+self.m.nu] if self.extra_reset else self.data[i].qpos[self.ctrl_adr])
            self.reset_bias[i] = self.last_ctrl[i] - self.data[i].qpos[self.ctrl_adr]
            if self.extra_reset:
                self.initial_warmstart[i] = self.state[i, self.eq_end+self.m.nu:self.eq_end+self.m.nu+self.m.nv]
                self.needs_warmstart[i] = True
        self._refresh()
        self.hist[ids] = self.frames[ids, None]

    def _refresh(self):
        frames = []
        pp = []
        pq = []
        gp = []
        gr = []
        for i, d in enumerate(self.data):
            mujoco.mj_setState(self.m, d, self.state[i, : self.full_size], STATE_SPEC)
            d.eq_active[:] = self.state[i, self.full_size : self.eq_end]
            d.ctrl[:] = self.last_ctrl[i]
            if self.episode_friction: d.userdata[:3] = self.state[i,-3:]
            for k in range(int(mujoco.mjtWarning.mjNWARNING)):
                d.warning[k].number = 0
            mujoco.mj_forward(self.m, d)
            self.native_warning[i] = any(
                d.warning[k].number for k in range(int(mujoco.mjtWarning.mjNWARNING))
            )
            row = []
            for side in ("left", "right"):
                sid = self.m.site(side + "_gripperframe").id
                q = np.empty(4)
                mujoco.mju_mat2Quat(q, d.site_xmat[sid])
                q *= 1 if q[0] >= 0 else -1
                velocity = np.zeros(6)
                mujoco.mj_objectVelocity(self.m, d, mujoco.mjtObj.mjOBJ_SITE, sid, velocity, 0)
                row.extend(np.r_[d.site_xpos[sid], q, velocity[3:], velocity[:3]])
            row.extend(d.qpos[self.ctrl_adr])
            row.extend(self.prev_action[i])
            frames.append(row)
            pp.append([d.body(n).xpos.copy() for n in self.obs_names])
            pq.append([d.body(n).xquat.copy() for n in self.obs_names])
            gp.append([d.body(n).xpos.copy() for n in GRIPS])
            gr.append([d.body(n).xmat.reshape(3, 3).copy() for n in GRIPS])
        self.frames = np.asarray(frames, np.float32)
        self.compact = dict(
            tcp_p=np.asarray(gp, np.float32),
            tcp_R=np.asarray(gr, np.float32),
            part_p=np.asarray(pp, np.float32),
            part_q=np.asarray(pq, np.float32),
            finger=np.asarray([d.qpos[self.ctrl_adr[[5, 11]]] for d in self.data], np.float32),
        )

    def observe(self):
        return dict(
            proprio=self.hist.reshape(self.N, -1).copy(),
            task=self.task.copy(),
            **{k: v.copy() for k, v in self.compact.items()},
        )

    def step(self, action):
        warning_before = self.native_warning.copy()
        for work in self.work:
            for k in range(int(mujoco.mjtWarning.mjNWARNING)):
                work.warning[k].number = 0
        a = np.asarray(action, float)
        if a.shape != (self.N, ACT_DIM) or not np.isfinite(a).all():
            raise ValueError("Expected finite (N,12) actions")
        ctrl = self.last_ctrl.copy() if self.extra_reset else np.asarray([d.qpos[self.ctrl_adr] for d in self.data])
        ctrl[:, ARM_ACTIONS] += np.tile(self.ecfg.action_scale_joints, 2) * a[:, ARM_ACTIONS]
        ctrl[:, [5, 11]] += self.ecfg.gripper_scale * a[:, [5, 11]]
        ctrl = np.clip(ctrl, self.m.actuator_ctrlrange[:, 0], self.m.actuator_ctrlrange[:, 1])
        controls = np.concatenate([ctrl, self.state[:, self.full_size : self.eq_end], *([self.state[:,-3:]] if self.episode_friction else [])], axis=1)
        trajectories, sensors = self.engine.rollout(
            self.m,
            self.work,
            self.state[:, : self.full_size].copy(),
            np.repeat(controls[:, None, :], self.substeps, axis=1),
            control_spec=mujoco.mjtState.mjSTATE_CTRL | mujoco.mjtState.mjSTATE_EQ_ACTIVE | (mujoco.mjtState.mjSTATE_USERDATA if self.episode_friction else 0),
            nstep=self.substeps,
            initial_warmstart=np.where(self.needs_warmstart[:,None], self.initial_warmstart, 0.0),
        )
        self.needs_warmstart[:] = False
        warning_during = any(
            work.warning[k].number
            for work in self.work
            for k in range(int(mujoco.mjtWarning.mjNWARNING))
        )
        if not np.isfinite(trajectories).all() or not np.isfinite(sensors).all():
            raise FloatingPointError("Nonfinite current-assembly trajectory")
        expected = self.state[:, 0] + self.ecfg.control_dt
        unstable = abs(trajectories[:, -1, 0] - expected) > 1e-6
        self.state[:, : self.full_size] = trajectories[:, -1]
        self.prev_action = a.copy()
        self.last_ctrl = ctrl
        self.t += 1
        self._refresh()
        # A rollout-thread warning cannot be attributed to one world. Reject
        # the whole slice rather than reward dynamics with truncated constraints.
        unstable |= warning_before | self.native_warning | warning_during
        penetration = np.maximum(0, -sensors[:, :, self.sensor_adr].min(1))
        penetration = np.maximum(
            penetration, np.asarray([max(0, -min(d.contact.dist, default=0)) for d in self.data])
        )
        safety = unstable | (penetration > self.ecfg.max_contact_depth)
        velocities = trajectories[:, :, 1 + self.m.nq : 1 + self.m.nq + self.m.nv]
        safety |= (abs(velocities[:, :, :12]) > 5).any((1, 2))
        for task in TASK_NAMES:
            address = int(self.m.joint(task + "_free").dofadr[0])
            safety |= (np.linalg.norm(velocities[:, :, address : address + 3], axis=2) > 1).any(1)
            safety |= (np.linalg.norm(velocities[:, :, address + 3 : address + 6], axis=2) > 30).any(1)
        success = np.zeros(self.N, bool)
        distance = np.zeros(self.N)
        for i, d in enumerate(self.data):
            safety[i] |= bool(np.any(abs(d.qvel[:12]) > 5))
            for task in TASK_NAMES:
                adr = int(self.m.joint(task + "_free").dofadr[0])
                safety[i] |= (
                    np.linalg.norm(d.qvel[adr : adr + 3]) > 1
                    or np.linalg.norm(d.qvel[adr + 3 : adr + 6]) > 30
                )
            result = self.reward.snapshot(d, dict(task=self.task_names[self.task[i]]))
            distance[i] = max(max(abs(np.asarray(c["position_error_m"]))) for c in result["checks"])
            goal = self.reward.config["tasks"][self.task_names[self.task[i]]]
            gid = self.m.geom(goal["geom"]).id
            pose = (d.geom_xpos[gid].copy(), d.geom_xmat[gid].copy())
            previous = self.stable_pose[i]
            close = (
                previous is not None
                and np.linalg.norm(pose[0] - previous[0]) <= self.ecfg.stable_translation
                and np.arccos(
                    np.clip((np.trace(pose[1].reshape(3, 3).T @ previous[1].reshape(3, 3)) - 1) / 2, -1, 1)
                )
                <= self.ecfg.stable_rotation
            )
            if result["reward"] and not safety[i]:
                if not close:
                    self.stable_pose[i] = pose
                    self.stable[i] = 1
                else:
                    self.stable[i] += 1
            else:
                self.stable[i] = 0
                self.stable_pose[i] = None
            success[i] = self.stable[i] >= round(self.ecfg.settle_s / self.ecfg.control_dt) + 1
        success &= ~self.terminated
        safety &= ~self.terminated
        timeout = (self.t >= self.horizon) & ~success & ~safety & ~self.terminated
        self.terminated |= success | safety | timeout
        cost = self.ecfg.cost_coef * (a * a).sum(1)
        self.hist = np.concatenate([self.hist[:, 1:], self.frames[:, None]], axis=1)
        return (
            (success.astype(np.float32) - cost - self.ecfg.safety_penalty * safety).astype(np.float32),
            success | safety | timeout,
            dict(
                success=success,
                safety=safety,
                timeout=timeout,
                d_asm=distance,
                cost=cost,
                unstable=unstable,
                penetration=penetration,
            ),
        )
