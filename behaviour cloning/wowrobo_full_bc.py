"""Generate and fit a shared actor on continuous, randomized WowRobo builds."""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

from wowrobo_kinematic_bc import DT, ROOT, SCALES, BatchedIK, Geometry

TASKS = ("base_side_cover", "shoulder_connector", "assembly_pin", "upper_arm_module", "forearm_module")
SIDES = (0, 1, 0, 1, 0)
OFFSET_SCALES = (1., .6, .4, .7, .35)


class RetargetRejected(ValueError):
    """A sampled offset falls outside the pickup retarget tolerance."""


def load_pack(path):
    with np.load(path, allow_pickle=False) as archive:
        return dict(archive)


def bake(args):
    import mujoco
    import torch

    sys.path.insert(0, str(args.encoder_source.resolve()))
    from assembly_policy.so101.wowrobo_obs import GRIPS, WOW_TASKS, PointTables

    if tuple(WOW_TASKS) != TASKS:
        raise ValueError("Unexpected WowRobo task order")
    pack = load_pack(args.cover_pack)
    scene_hash = hashlib.sha256(args.scene.read_bytes()).hexdigest()
    if scene_hash != pack["scene_sha256"].item():
        raise ValueError("Scene differs from the previously verified geometry pack")
    m = mujoco.MjModel.from_xml_path(str(args.scene))
    d = mujoco.MjData(m)
    tables = PointTables(m, json.loads(args.point_features.read_text()))
    if tables.signature != pack["encoder_signature"].item():
        raise ValueError("Point encoder differs from the side-cover export")
    with np.load(args.trajectory, allow_pickle=False) as z:
        q, fps = z["qpos"], z["fps"].item()
    report = json.loads(args.chapters.read_text())
    if tuple(c["part"] for c in report["chapters"]) != TASKS or q.shape != (report["frames"], m.nq):
        raise ValueError("Expected the saved continuous build")
    starts = np.array([c["first_frame"] for c in report["chapters"]])
    pickup = np.array([next(c["frame"] for c in report["checkpoints"]
                            if c["part"] == task and c["kind"] == "pickup") for task in TASKS])
    seats = np.array([next(c["frame"] for c in report["checkpoints"]
                           if c["part"] == task and c["kind"] == "seat") for task in TASKS])
    # The saved renderer uses these four consecutive pickup segments.
    approach_frames = sum(round(t*fps) for t in (1.1, .66, .60, .58))
    approach = pickup - approach_frames + 1
    if not np.all(starts <= approach):
        raise ValueError("Pickup segment metadata differs from the reference build")
    # Drop the identical opening frames. The actor has no episode clock.
    initial_motion = np.flatnonzero(np.max(abs(np.diff(q[:, pack["ctrl"]], axis=0)), axis=1) > 1e-8)[0]
    crop = min(int(initial_motion), int(approach[0]))
    q = q[crop:].copy()
    starts = np.maximum(0, starts-crop)
    pickup, seats, approach = pickup-crop, seats-crop, approach-crop
    widths = np.maximum(1/fps, (np.abs(np.diff(q[:, pack["ctrl"]], axis=0))/SCALES).max(axis=1)*DT)
    clock = np.r_[0., np.cumsum(widths)]
    pack.update(q=q, time=clock, starts=starts, pickups=pickup, seats=seats, approaches=approach,
                task_names=np.array(TASKS), active_sides=np.array(SIDES),
                part_qadr=np.array([m.joint(t+"_free").qposadr.item() for t in TASKS]),
                all_local_points=tables.points.numpy(), all_point_body=tables.which.numpy()+1,
                source_commit=report["source_commit"])
    g = BuildGeometry(pack)
    max_error = 0.
    for task, frame in enumerate(pickup):
        d.qpos[:] = q[frame]
        mujoco.mj_kinematics(m, d)
        p, r = g.fk(q[frame])
        np.testing.assert_allclose(p, d.xpos, atol=2e-8)
        np.testing.assert_allclose(r, d.xmat.reshape(-1, 3, 3), atol=2e-8)
        fingers = [m.body(n).id for n in GRIPS]

        def v(x):
            return torch.as_tensor(x[None], dtype=torch.float32)

        target = tables(torch.tensor([task]), v(d.xpos[fingers]), v(d.xmat[fingers].reshape(4, 3, 3)),
                        v(d.xpos[1:]), v(d.xquat[1:]), v(d.qpos[pack["ctrl"][[5, 11]]])).numpy()[0]
        max_error = max(max_error, float(abs(g.task_geometry[task].cloud(p, r)-target).max()))
    if max_error > 5e-6:
        raise ValueError(f"Full build point parity failed: {max_error:g}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **pack)
    print(json.dumps({"task_point_parity_max_error": max_error, "reference_s": clock[-1],
                      "cropped_idle_frames": crop}), flush=True)


def smooth(x):
    x = np.clip(x, 0., 1.)
    return x*x*(3.-2.*x)


class BuildGeometry(Geometry):
    def __init__(self, arrays):
        super().__init__(arrays)
        if tuple(arrays["task_names"]) != TASKS:
            raise ValueError("Unexpected baked task order")
        self.task_geometry = [
            Geometry(dict(arrays, local_points=arrays["all_local_points"][i],
                          point_body=arrays["all_point_body"][i]))
            for i in range(len(TASKS))
        ]

    def features(self, q, actions, tasks):
        p, r = self.fk(q)
        clouds = np.empty((len(actions), 1344), dtype=np.float32)
        for task, geometry in enumerate(self.task_geometry):
            rows = np.flatnonzero(tasks == task)
            clouds[rows] = geometry.cloud(p[rows], r[rows])
        return clouds, self.proprio(q, actions, p, r)


class BuildTeacher:
    def __init__(self, geometry):
        self.g = geometry
        a = geometry.a
        self.t = a["time"]
        self.p, self.r = geometry.tcp(*geometry.fk(a["q"]))
        self.blends = []
        for i, side in enumerate(SIDES):
            pickup = self.t[a["pickups"][i]]
            approach = self.t[a["approaches"][i]]
            last = min(pickup+1.75, self.t[a["seats"][i]])
            first = min(approach+.75, pickup)
            rise = smooth((self.t-approach)/max(first-approach, DT))
            fade = 1.-smooth((self.t-pickup-.65)/max(last-pickup-.65, DT))
            arm = rise*fade
            rows = np.flatnonzero(arm > 1e-8)
            self.blends.append((fade, arm, BatchedIK(geometry, side=side, rows=rows)))

    def episode(self, rng):
        from scipy.spatial.transform import Rotation

        a = self.g.a
        q = a["q"].copy()
        offsets = []
        targets = []
        for i, (side, adr, (fade, arm, solver)) in enumerate(zip(SIDES, a["part_qadr"], self.blends)):
            # The narrow pin and module grasps receive smaller offsets.
            scale = OFFSET_SCALES[i]
            shift = np.r_[rng.uniform(-.006, .006, 2)*scale, 0.]
            yaw = rng.uniform(-3., 3.)*scale
            origin = q[0, adr:adr+3].copy()
            axis = np.array([0., 0., np.deg2rad(yaw)])
            part_rot = Rotation.from_rotvec(fade[:, None]*axis)
            q[:, adr:adr+3] = origin + part_rot.apply(q[:, adr:adr+3]-origin) + fade[:, None]*shift
            original = Rotation.from_quat(np.roll(q[:, adr+3:adr+7], -1, axis=1))
            q[:, adr+3:adr+7] = np.roll((part_rot*original).as_quat(), 1, axis=1)
            hand_rot = Rotation.from_rotvec(arm[:, None]*axis)
            target_p = origin + hand_rot.apply(self.p[:, side]-origin) + arm[:, None]*shift
            target_r = (hand_rot*Rotation.from_matrix(self.r[:, side])).as_matrix()
            solver.retarget(q, target_p, target_r)
            targets.append(target_p[a["pickups"][i]])
            offsets.append([float(shift[0]*1000), float(shift[1]*1000), float(yaw)])

        actual, _ = self.g.tcp(*self.g.fk(q[a["pickups"]]))
        errors = np.linalg.norm(actual[np.arange(len(TASKS)), SIDES]-targets, axis=1)
        if np.max(errors) > .005:
            raise RetargetRejected(f"Pickup IK residuals exceed 5 mm: {errors*1000}")
        minimum = (np.abs(np.diff(q[:, self.g.ctrl], axis=0))/SCALES).max(axis=1)*DT
        clock = np.r_[0., np.cumsum(np.maximum(np.diff(self.t), minimum))]
        ticks = np.arange(int(np.ceil(clock[-1]/DT))+1)*DT
        out = np.stack([np.interp(ticks, clock, col) for col in q.T], axis=-1)
        for adr in a["part_qadr"]:
            quat = out[:, adr+3:adr+7]
            quat /= np.linalg.norm(quat, axis=1, keepdims=True)
        actions = (np.diff(out[:, self.g.ctrl], axis=0)/SCALES).astype(np.float32)
        tasks = np.searchsorted(clock[a["starts"]][1:], ticks[:-1], side="right").astype(np.int8)
        points, proprio = self.g.features(out, actions, tasks)
        if not np.isfinite(out).all() or np.max(abs(actions)) > 1.00001:
            raise ValueError("Invalid generated trajectory")
        np.testing.assert_allclose(out[-1], a["q"][-1], atol=1e-8)
        return (dict(points=points, proprio=proprio, actions=actions, tasks=tasks, qpos=out),
                offsets, errors.tolist())


def generate(args):
    start = time.perf_counter()
    pack = load_pack(args.model)
    teacher = BuildTeacher(BuildGeometry(pack))
    args.out.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(args.seed)
    episodes = []
    rejected = 0
    for i in range(args.episodes):
        for _ in range(20):
            try:
                arrays, offsets, errors = teacher.episode(rng)
            except RetargetRejected:
                rejected += 1
            else:
                break
        else:
            raise ValueError("Could not retarget a build within 20 attempts")
        np.savez_compressed(args.out/f"build-{i:03d}.npz", **arrays)
        episodes.append({"episode": i, "steps": len(arrays["actions"]),
                         "steps_by_task": np.bincount(arrays["tasks"], minlength=len(TASKS)).tolist(),
                         "offsets_mm_deg": offsets, "pickup_ik_error_m": errors})
    report = {"kind": "kinematic_full_build", "seed": args.seed, "task_names": list(TASKS),
              "encoder_signature": pack["encoder_signature"].item(),
              "model_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
              "physics_steps": 0, "episodes": episodes, "rejected_offsets": rejected,
              "seconds": time.perf_counter()-start}
    (args.out/"manifest.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({"builds": len(episodes), "samples": sum(e["steps"] for e in episodes),
                      "seconds": report["seconds"]}), flush=True)


def fit(args):
    import torch

    sys.path.insert(0, str(ROOT))
    from wowrobo_bc import actor_net, train_actor

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    report = json.loads((args.data/"manifest.json").read_text())
    if report["kind"] != "kinematic_full_build" or report["task_names"] != list(TASKS):
        raise ValueError("Expected a full-build dataset")
    ids = np.random.default_rng(args.seed).permutation([e["episode"] for e in report["episodes"]])
    if len(ids) < 5:
        raise ValueError("Generate at least five builds")
    nval = max(1, len(ids)//5)

    def load(rows):
        data = {key: [] for key in ("points", "proprio", "actions", "tasks")}
        for i in rows:
            with np.load(args.data/f"build-{i:03d}.npz", allow_pickle=False) as z:
                for key in data:
                    data[key].append(z[key])
        return [torch.from_numpy(np.concatenate(v)) for v in data.values()]

    x, p, y, train_task = load(ids[nval:])
    vx, vp, vy, val_task = load(ids[:nval])
    model = actor_net()

    @torch.no_grad()
    def metrics():
        loss = torch.cat([(model.actor(a, model.norm(b))-c).square().mean(1)
                          for a, b, c in zip(vx.split(1024), vp.split(1024), vy.split(1024))])
        return {"all": float(loss.mean()), **{name: float(loss[val_task == i].mean())
                                             for i, name in enumerate(TASKS)}}

    before = metrics()
    start = time.perf_counter()
    train_actor(model, x, p, y, steps=args.steps, batch=256, seed=args.seed)
    result = {"training_builds": ids[nval:].tolist(), "validation_builds": ids[:nval].tolist(),
              "training_examples": len(x), "validation_examples": len(vx),
              "training_examples_by_task": torch.bincount(train_task.to(torch.long)).tolist(),
              "validation_mse_before": before, "validation_mse_after": metrics(),
              "steps": args.steps, "seconds": time.perf_counter()-start}
    signature = hashlib.sha256((report["encoder_signature"]+repr(list(TASKS))).encode()).hexdigest()
    args.out.mkdir(parents=True, exist_ok=False)
    torch.save({"kind": "wowrobo-kinematic-full-build-v1", "actor": model.actor.state_dict(),
                "norm": model.norm.state_dict(), "task_names": list(TASKS),
                "control_dt": DT, "action_scales": SCALES.tolist(),
                "observation_signature": signature, "metrics": result}, args.out/"actor.pt")
    (args.out/"fit.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    for name, fn, files in (
        ("bake", bake, ("scene", "point-features", "trajectory", "chapters", "cover-pack", "encoder-source")),
        ("generate", generate, ("model",)),
        ("fit", fit, ("data",)),
    ):
        p = sub.add_parser(name)
        p.set_defaults(fn=fn)
        p.add_argument("--out", type=Path, required=True)
        for key in files:
            p.add_argument("--"+key, type=Path, required=True)
        if name != "bake":
            p.add_argument("--seed", type=int, default=43)
        if name == "generate":
            p.add_argument("--episodes", type=int, default=60)
        if name == "fit":
            p.add_argument("--steps", type=int, default=3500)
            p.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    for key in ("episodes", "steps", "threads"):
        if getattr(args, key, 1) < 1:
            parser.error(f"--{key} must be positive")
    args.fn(args)


if __name__ == "__main__":
    main()
