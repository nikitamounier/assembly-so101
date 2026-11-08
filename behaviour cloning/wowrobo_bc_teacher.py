"""Retarget the side-cover video reference onto a native WowRobo reset."""

import numpy as np


def control_scales(config):
    return np.tile([*config.action_scale_joints, config.gripper_scale], 2)


def control_base(env):
    """Return the action origin for the current native reset payload."""
    if getattr(env, "extra_reset", False):
        return env.last_ctrl[0]
    return env.data[0].qpos[env.ctrl_adr]


def action_for_target(env, target, limit=1.):
    base = control_base(env)
    lo, hi = env.m.actuator_ctrlrange.T
    scales = control_scales(env.ecfg)
    target = np.clip(target, lo, hi)
    action = np.clip((target-base) / scales, -limit, limit)
    expected = np.clip(base + scales*action, lo, hi)
    return action, expected


class CoverReference:
    def __init__(self, scene, targets, trajectory, audit):
        import json

        import mujoco
        from scipy.spatial.transform import Rotation

        self.mj = mujoco
        self.rotation = Rotation
        self.model = mujoco.MjModel.from_xml_path(str(scene))
        self.data = mujoco.MjData(self.model)
        with np.load(targets, allow_pickle=False) as archive:
            self.targets = archive["base_side_cover"].astype(float)
        with np.load(trajectory, allow_pickle=False) as archive:
            self.q0 = archive["qpos"][0].copy()
        self.meta = json.loads(audit.read_text())
        self.dt = self.meta["nominal_control_dt_s"]
        self.pickup_s = next(x["retimed_pickup_reference_s"] for x in self.meta["tasks"]
                             if x["task"] == "base_side_cover")
        self.arm_names = [
            self.model.joint(int(j)).name for j in self.model.actuator_trnid[:, 0]
        ]
        self.adr = self.model.jnt_qposadr[self.model.actuator_trnid[:, 0]]
        if self.targets.shape[1] != 12 or self.q0.shape != (self.model.nq,):
            raise ValueError("Reference trajectory or actuator layout differs from review model")
        self.fk(self.q0)
        self.cover0 = self.pose(self.data.body("base_side_cover"))
        self.base0 = self.pose(self.data.body("base_servo"))

    def fk(self, q):
        self.data.qpos[:] = q
        self.mj.mj_kinematics(self.model, self.data)

    def pose(self, item):
        return item.xpos.copy(), self.rotation.from_matrix(item.xmat.reshape(3, 3))

    def retarget(self, model, data, indices):
        """Return joint targets. This modifies separate FK data only."""
        from scipy.optimize import least_squares
        from scipy.spatial.transform import Slerp

        native_names = [model.joint(int(j)).name for j in model.actuator_trnid[:, 0]]
        if self.arm_names != native_names:
            raise ValueError("Native and reference actuator orders differ")
        adr = model.jnt_qposadr[model.actuator_trnid[:, 0]]
        fk_data = self.mj.MjData(model)
        fk_data.qpos[:] = data.qpos
        initial = data.qpos[adr].copy()
        cover = self.pose(data.body("base_side_cover"))
        base = self.pose(data.body("base_servo"))

        def mapping(original, actual, p, r):
            p0, r0 = original
            p1, r1 = actual
            dr = r1*r0.inv()
            return p1 + dr.apply(p-p0), dr*r

        lo, hi = model.actuator_ctrlrange[:5].T
        previous = np.clip(initial[:5], lo+1e-8, hi-1e-8)
        frames = []
        for i in indices:
            ref = self.targets[i]
            q = self.q0.copy()
            q[self.adr] = ref
            self.fk(q)
            p, r = self.pose(self.data.site("left_gripperframe"))
            loose_p, loose_r = mapping(self.cover0, cover, p, r)
            goal_p, goal_r = mapping(self.base0, base, p, r)
            weight = np.clip((i*self.dt-self.pickup_s-.65)/1.15, 0., 1.)
            dest = loose_p + weight*(goal_p-loose_p)
            orient = Slerp([0, 1], self.rotation.concatenate([loose_r, goal_r]))(weight)

            def residual(joints):
                fk_data.qpos[adr[:5]] = joints
                self.mj.mj_kinematics(model, fk_data)
                point, rot = self.pose(fk_data.site("left_gripperframe"))
                return np.r_[point-dest, .026*(orient.inv()*rot).as_rotvec(),
                             .001*(joints-previous)]

            previous = least_squares(residual, previous, bounds=(lo, hi), max_nfev=36).x
            ctrl = initial.copy()
            ctrl[:5] = previous
            ctrl[5] = ref[5]
            frames.append(ctrl)
        return np.asarray(frames)
