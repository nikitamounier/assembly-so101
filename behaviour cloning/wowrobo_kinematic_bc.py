"""Bake a CAD model once, then generate and fit kinematic cover demos without a simulator."""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DT = .04
SCALES = np.tile([.035]*5 + [.05], 2)


def rots(q):
    from scipy.spatial.transform import Rotation

    return Rotation.from_quat(np.roll(q, -1, axis=-1).reshape(-1, 4)).as_matrix().reshape(q.shape[:-1]+(3, 3))


def apply(r, v):
    return np.einsum("...ij,...j->...i", r, v)


class Geometry:
    def __init__(self, arrays):
        self.a = arrays
        self.body_r = rots(arrays["body_quat"])
        self.site_r = rots(arrays["site_quat"])
        self.ctrl = arrays["ctrl"]

    def fk(self, q):
        from scipy.spatial.transform import Rotation

        q = np.asarray(q)
        shape = q.shape[:-1]
        a = self.a
        n = len(a["parent"])
        p = np.zeros(shape+(n, 3))
        r = np.broadcast_to(np.eye(3), shape+(n, 3, 3)).copy()
        for i in range(1, n):
            parent = a["parent"][i]
            p[..., i, :] = p[..., parent, :] + apply(r[..., parent, :, :], a["body_pos"][i])
            r[..., i, :, :] = r[..., parent, :, :] @ self.body_r[i]
            j = a["body_joint"][i]
            if j < 0:
                continue
            start = a["qadr"][j]
            if a["jtype"][j] == 0:
                p[..., i, :] = q[..., start:start+3]
                r[..., i, :, :] = rots(q[..., start+3:start+7])
            elif a["jtype"][j] == 3:
                axis = a["jaxis"][j]
                pivot = a["jpos"][j]
                delta = (q[..., start]-a["q0"][start])[..., None]*axis
                rr = Rotation.from_rotvec(delta.reshape(-1, 3)).as_matrix().reshape(shape+(3, 3))
                old = r[..., i, :, :]
                p[..., i, :] += apply(old, pivot-apply(rr, pivot))
                r[..., i, :, :] = old @ rr
            else:
                raise ValueError("Unsupported joint type in baked kinematic model")
        return p, r

    def tcp(self, p, r):
        b = self.a["site_body"]
        return p[..., b, :] + apply(r[..., b, :, :], self.a["site_pos"]), r[..., b, :, :] @ self.site_r

    def cloud(self, p, r):
        # The published tables use 384 part/receiver samples and four 16-point pads.
        a = self.a
        p, r = p.astype(np.float32), r.astype(np.float32)
        part_ids, fingers = a["point_body"], a["finger_body"]
        part = apply(r[..., part_ids, :, :], a["local_points"]) + p[..., part_ids, :]
        pads = (np.einsum("...gij,gnj->...gni", r[..., fingers, :, :], a["pad_points"])
                + p[..., fingers, None, :]).reshape(p.shape[:-2]+(64, 3))
        world = np.concatenate([part, pads], axis=-2)
        left = fingers[0]
        local = np.einsum("...ji,...nj->...ni", r[..., left, :, :], world-p[..., left, None, :])
        return (10*local).reshape(p.shape[:-2]+(1344,))

    def observations(self, q, action):
        from scipy.spatial.transform import Rotation

        p, r = self.fk(q)
        tp, tr = self.tcp(p, r)
        quat = np.roll(Rotation.from_matrix(tr.reshape(-1, 3, 3)).as_quat(), 1, axis=-1)
        quat = quat.reshape(len(q), 2, 4)
        quat *= np.where(quat[..., :1] < 0., -1., 1.)
        linear = np.zeros_like(tp)
        angular = np.zeros_like(tp)
        linear[1:] = np.diff(tp, axis=0)/DT
        delta = tr[1:] @ np.swapaxes(tr[:-1], -2, -1)
        angular[1:] = Rotation.from_matrix(delta.reshape(-1, 3, 3)).as_rotvec().reshape(-1, 2, 3)/DT
        previous = np.vstack([np.zeros((1, 12)), action])
        prop = np.concatenate([np.concatenate([tp, quat, linear, angular], axis=-1).reshape(len(q), 26),
                               q[:, self.ctrl], previous], axis=1)
        history = np.maximum(0, np.arange(len(q)-1)[:, None] + np.arange(-4, 1))
        return self.cloud(p[:-1], r[:-1]), prop[history].reshape(-1, 250).astype(np.float32)


def bake(args):
    import mujoco
    import torch

    sys.path.insert(0, str(args.encoder_source.resolve()))
    from assembly_policy.so101.wowrobo_obs import GRIPS, PointTables

    m = mujoco.MjModel.from_xml_path(str(args.scene))
    d = mujoco.MjData(m)
    tables = PointTables(m, json.loads(args.point_features.read_text()))
    sites = [m.site(side+"_gripperframe").id for side in ("left", "right")]
    ctrl = m.jnt_qposadr[m.actuator_trnid[:, 0]]
    if np.any(m.body_jntnum > 1) or m.nq != 51 or len(ctrl) != 12:
        raise ValueError("Expected the published viewer joint layout")
    with np.load(args.trajectory, allow_pickle=False) as z:
        source_q, fps = z["qpos"], z["fps"].item()
    report = json.loads(args.chapters.read_text())
    pickup = next(c["frame"] for c in report["checkpoints"] if c["part"] == "base_side_cover" and c["kind"] == "pickup")
    seat = next(c["frame"] for c in report["checkpoints"] if c["part"] == "base_side_cover" and c["kind"] == "seat")
    q = source_q[:seat+1].copy()
    # Stretch any interval with a large nominal action before sampling at 25 Hz.
    widths = np.maximum(1/fps, (np.abs(np.diff(q[:, ctrl], axis=0))/SCALES).max(axis=1)*DT)
    t = np.r_[0., np.cumsum(widths)]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pack = {
        "parent": m.body_parentid.copy(), "body_quat": m.body_quat.copy(), "body_pos": m.body_pos.copy(),
        "body_joint": m.body_jntadr.copy(), "jtype": m.jnt_type.copy(), "qadr": m.jnt_qposadr.copy(),
        "jaxis": m.jnt_axis.copy(), "jpos": m.jnt_pos.copy(), "q0": m.qpos0.copy(), "ctrl": ctrl,
        "cover_adr": m.joint("base_side_cover_free").qposadr.item(),
        "site_body": m.site_bodyid[sites], "site_pos": m.site_pos[sites], "site_quat": m.site_quat[sites],
        "local_points": tables.points[0].numpy(), "point_body": tables.which[0].numpy()+1,
        "pad_points": tables.grip.numpy(), "finger_body": np.array([m.body(n).id for n in GRIPS]),
        "q": q, "time": t, "pickup_time": t[pickup], "joint_bounds": m.actuator_ctrlrange.copy(),
        "encoder_signature": tables.signature, "scene_sha256": hashlib.sha256(args.scene.read_bytes()).hexdigest(),
    }
    geometry = Geometry(pack)
    errors = []
    for j in np.linspace(0, seat, 15, dtype=int):
        d.qpos[:] = source_q[j]
        mujoco.mj_kinematics(m, d)
        p, r = geometry.fk(source_q[j])
        np.testing.assert_allclose(p, d.xpos, atol=2e-8)
        np.testing.assert_allclose(r, d.xmat.reshape(-1, 3, 3), atol=2e-8)
        fingers = pack["finger_body"]
        v = lambda x: torch.as_tensor(x[None], dtype=torch.float32)
        expected = tables(torch.tensor([0]), v(d.xpos[fingers]), v(d.xmat[fingers].reshape(4, 3, 3)),
                          v(d.xpos[1:]), v(d.xquat[1:]), v(d.qpos[ctrl[[5, 11]]])).numpy()[0]
        error = float(np.max(np.abs(geometry.cloud(p, r)-expected)))
        errors.append(error)
    if max(errors) > 5e-6:
        raise ValueError(f"Point encoder parity failed: {max(errors):g}")
    np.savez_compressed(args.out, **pack)
    print(json.dumps({"frames_checked": len(errors), "max_cloud_error": max(errors)}), flush=True)


class MujocoFK:
    """Reuse the compiled model for pose queries. No collision or dynamics calls."""

    def __init__(self, scene, geometry):
        import mujoco

        self.mj = mujoco
        if hashlib.sha256(scene.read_bytes()).hexdigest() != geometry.a["scene_sha256"].item():
            raise ValueError("Scene does not match the baked model")
        self.model = mujoco.MjModel.from_xml_path(str(scene))
        self.model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
        self.data = mujoco.MjData(self.model)
        self.data.qpos[:] = geometry.a["q"][0]
        self.site = self.model.site("left_gripperframe").id
        self.indices = geometry.ctrl[:5]

    def left_tcp(self, x):
        from scipy.spatial.transform import Rotation

        self.data.qpos[self.indices] = x
        self.mj.mj_kinematics(self.model, self.data)
        return self.data.site_xpos[self.site].copy(), Rotation.from_matrix(
            self.data.site_xmat[self.site].reshape(3, 3))


class BatchedIK:
    """Linearize the reference arm once, then solve nearby poses in batches."""

    def __init__(self, geometry):
        self.geometry = geometry
        a = geometry.a
        q = a["q"]
        self.indices = geometry.ctrl[:5]
        self.lo, self.hi = a["joint_bounds"][:5].T
        t = a["time"]
        self.rows = np.flatnonzero(t < a["pickup_time"] + 1.75)
        q = q[self.rows]
        p, r = self.tcp(q)
        perturb = np.repeat(q[None], 5, axis=0)
        eps = 1e-5
        for j, index in enumerate(self.indices):
            perturb[j, :, index] += eps
        pp, rr = self.tcp(perturb)
        jacobian = np.concatenate([(pp-p)/eps, .025*self.rot_delta(rr, r)/eps], axis=-1)
        jacobian = jacobian.transpose(1, 2, 0)
        transpose = jacobian.transpose(0, 2, 1)
        self.inverse = np.linalg.solve(transpose @ jacobian + 4e-6*np.eye(5), transpose)

    def tcp(self, q):
        p, r = self.geometry.tcp(*self.geometry.fk(q))
        return p[..., 0, :], r[..., 0, :, :]

    @staticmethod
    def rot_delta(target, actual):
        from scipy.spatial.transform import Rotation

        delta = target @ actual.swapaxes(-1, -2)
        return Rotation.from_matrix(delta.reshape(-1, 3, 3)).as_rotvec().reshape(delta.shape[:-2]+(3,))

    def retarget(self, q, p_target, r_target):
        rows = self.rows
        local = q[rows].copy()
        for _ in range(3):
            p, r = self.tcp(local)
            error = np.concatenate([p_target[rows]-p, .025*self.rot_delta(r_target[rows], r)], axis=-1)
            local[:, self.indices] = np.clip(local[:, self.indices] + apply(self.inverse, error),
                                             self.lo, self.hi)
        q[np.ix_(rows, self.indices)] = local[:, self.indices]


def variant(geometry, rng, native_fk=None, batched=None):
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation

    a = geometry.a
    q = a["q"].copy()
    indices = geometry.ctrl[:5]
    t = a["time"]
    # Translate and yaw only the loose cover and taper the correction after the lift.
    # At the seat the reference is the original receiver-relative pose.
    origin = q[0, a["cover_adr"]:a["cover_adr"]+3].copy()
    shift = np.r_[rng.uniform(-.006, .006, 2), 0.]
    yaw = rng.uniform(-3., 3.)
    angle = np.deg2rad(yaw)
    fade = np.clip((t-a["pickup_time"].item()-.65)/1.1, 0., 1.)
    weight = 1.-fade*fade*(3.-2.*fade)
    turn = Rotation.from_rotvec(weight[:, None]*[0., 0., angle])
    p, r = geometry.fk(q)
    tp, tr = geometry.tcp(p, r)
    goal = origin + turn.apply(tp[:, 0]-origin) + weight[:, None]*shift
    orientation = turn*Rotation.from_matrix(tr[:, 0])
    k = a["cover_adr"].item()
    q[:, k:k+3] = origin+turn.apply(q[:, k:k+3]-origin)+weight[:, None]*shift
    old = Rotation.from_quat(np.roll(q[:, k+3:k+7], -1, axis=1))
    q[:, k+3:k+7] = np.roll((turn*old).as_quat(), 1, axis=1)

    if batched is not None:
        batched.retarget(q, goal, orientation.as_matrix())
    else:
        # Numerical IK only where the reference is shifted. The unaffected receiver
        # suffix stays exact. Five arm coordinates constrain six pose dimensions.
        # Use just the left branch's joints and ancestral bodies in the inner loop.
        site = a["site_body"][0]
        chain = []
        while site:
            chain.append(site)
            site = a["parent"][site]
        lo, hi = a["joint_bounds"][:5].T
        slots = {int(v): j for j, v in enumerate(indices)}
        chain = chain[::-1]

        def left_tcp(x):
            p0, r0 = np.zeros(3), np.eye(3)
            for body in chain:
                p0 += r0 @ a["body_pos"][body]
                r0 = r0 @ geometry.body_r[body]
                j = a["body_joint"][body]
                if j < 0:
                    continue
                adr = a["qadr"][j]
                pivot = a["jpos"][j]
                rr = Rotation.from_rotvec(a["jaxis"][j]*(x[slots[adr]]-a["q0"][adr])).as_matrix()
                p0 += r0 @ (pivot-rr@pivot)
                r0 = r0 @ rr
            return p0 + r0 @ a["site_pos"][0], Rotation.from_matrix(r0 @ geometry.site_r[0])

        query_tcp = left_tcp if native_fk is None else native_fk.left_tcp
        for i in np.flatnonzero(weight > 1e-8):
            ref = q[i, indices].copy()
            dest, rot = goal[i], orientation[i]

            def error(x):
                pp, rr = query_tcp(x)
                return np.r_[pp-dest, .025*(rot.inv()*rr).as_rotvec(), .002*(x-ref)]

            q[i, indices] = least_squares(error, np.clip(ref, lo+1e-9, hi-1e-9),
                                          bounds=(lo, hi), max_nfev=18).x
    p1, r1 = geometry.fk(q)
    measured, _ = geometry.tcp(p1, r1)
    # Measure grip alignment at the reference close, not after interpolation.
    close = int(np.argmin(np.abs(t-a["pickup_time"])))
    err = float(np.linalg.norm(measured[close, 0]-goal[close]))
    max_step = (np.abs(np.diff(q[:, geometry.ctrl], axis=0))/SCALES).max(axis=1)*DT
    clock = np.r_[0., np.cumsum(np.maximum(np.diff(t), max_step))]
    ticks = np.arange(np.ceil(clock[-1]/DT)+1)*DT
    out = np.stack([np.interp(ticks, clock, col) for col in q.T], axis=-1)
    adr = a["cover_adr"].item()
    quat = out[:, adr+3:adr+7]
    quat /= np.linalg.norm(quat, axis=1, keepdims=True)
    action = (np.diff(out[:, geometry.ctrl], axis=0)/SCALES).astype(np.float32)
    if not np.isfinite(out).all() or np.max(np.abs(action)) > 1.00001 or err > .005:
        raise ValueError(f"Kinematic retarget rejected: pickup residual {err*1000:.2f} mm")
    x, p = geometry.observations(out, action)
    return dict(points=x, proprio=p, actions=action, qpos=out, shift=shift, yaw_deg=yaw,
                pickup_error_m=err)


def generate(args):
    start = time.perf_counter()
    with np.load(args.model, allow_pickle=False) as archive:
        geometry = Geometry(dict(archive))
    native_fk = MujocoFK(args.scene, geometry) if args.scene else None
    batched = BatchedIK(geometry) if not native_fk and not args.iterative else None
    backend = "mujoco_fk" if native_fk else ("batched_numpy" if batched else "numpy_iterative")
    args.out.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(args.seed)
    accepted = []
    for episode in range(args.episodes):
        try:
            result = variant(geometry, rng, native_fk, batched)
        except ValueError as exc:
            print(f"Episode {episode}: {exc}", flush=True)
            continue
        np.savez_compressed(args.out / f"ep-{episode:03d}.npz", **result)
        accepted.append({"episode": episode, "steps": len(result["actions"]),
                         "pickup_error_m": result["pickup_error_m"],
                         "xy_mm": (result["shift"][:2]*1000).tolist(), "yaw_deg": result["yaw_deg"]})
    receipt = {"kind": "kinematic_only", "backend": backend,
               "physics_steps": 0, "seed": args.seed, "attempts": args.episodes,
               "accepted": accepted, "seconds": time.perf_counter()-start,
               "baked_model_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest()}
    (args.out/"manifest.json").write_text(json.dumps(receipt, indent=2)+"\n")
    print(json.dumps({k: v for k, v in receipt.items() if k != "accepted"}
                     | {"episodes": len(accepted), "samples": sum(x["steps"] for x in accepted)}))


def fit(args):
    import torch

    sys.path.insert(0, str(ROOT))
    from wowrobo_bc import actor_net, train_actor

    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    manifest = json.loads((args.data/"manifest.json").read_text())
    if manifest["kind"] != "kinematic_only":
        raise ValueError("Expected the kinematic dataset")
    rows = np.random.default_rng(args.seed).permutation([e["episode"] for e in manifest["accepted"]])
    if len(rows) < 5:
        raise ValueError("Need at least five retargeted episodes")

    def load(ids):
        buffer = [[], [], []]
        for episode in ids:
            with np.load(args.data / f"ep-{episode:03d}.npz", allow_pickle=False) as z:
                for dest, key in zip(buffer, ("points", "proprio", "actions")):
                    dest.append(z[key])
        return [torch.from_numpy(np.concatenate(x)) for x in buffer]

    nval = max(1, len(rows)//5)
    x, p, y = load(rows[nval:])
    vx, vp, vy = load(rows[:nval])
    model = actor_net()

    @torch.no_grad()
    def mse():
        return float((model.actor(vx, model.norm(vp))-vy).square().mean())

    before, zero = mse(), float(vy.square().mean())
    start = time.perf_counter()
    train_actor(model, x, p, y, steps=args.steps, batch=256, seed=args.seed)
    result = {"kind": "kinematic_only", "training_episodes": rows[nval:].tolist(),
              "validation_episodes": rows[:nval].tolist(), "validation_mse_before": before,
              "validation_mse_zero": zero, "validation_mse_after": mse(),
              "training_examples": len(x), "validation_examples": len(vx),
              "seconds": time.perf_counter()-start}
    args.out.mkdir(parents=True, exist_ok=False)
    torch.save({"kind": "wowrobo-kinematic-only-v1", "actor": model.actor.state_dict(),
                "norm": model.norm.state_dict(), "metrics": result}, args.out/"actor.pt")
    (args.out/"fit.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    for mode, fn, required in (
        ("bake", bake, ("scene", "point-features", "trajectory", "chapters", "encoder-source")),
        ("generate", generate, ("model",)),
        ("fit", fit, ("data",)),
    ):
        p = sub.add_parser(mode)
        p.set_defaults(fn=fn)
        p.add_argument("--out", required=True, type=Path)
        for key in required:
            p.add_argument("--"+key, required=True, type=Path)
        p.add_argument("--seed", type=int, default=43)
        if mode == "generate":
            p.add_argument("--episodes", type=int, default=20)
            backend = p.add_mutually_exclusive_group()
            backend.add_argument("--scene", type=Path, help="Use MuJoCo FK in the iterative IK solver")
            backend.add_argument("--iterative", action="store_true", help="Use the iterative NumPy IK reference")
        if mode == "fit":
            p.add_argument("--steps", type=int, default=700)
    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
