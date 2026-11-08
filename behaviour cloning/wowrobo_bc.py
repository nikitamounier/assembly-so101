"""Collect, fit, and evaluate a side-cover behavior-cloning pilot in native WowRobo."""

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from wowrobo_bc_teacher import CoverReference, action_for_target

ROOT = Path(__file__).resolve().parents[1]
TASK = "base_side_cover"
POINT_KEYS = ("task", "tcp_p", "tcp_R", "part_p", "part_q", "finger")


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n")


def setup_native(args):
    import torch

    runtime = args.runtime.resolve()
    if not (runtime / "assembly_policy/wowrobo/env.py").is_file():
        raise FileNotFoundError(f"Native WowRobo package not found in {runtime}")
    sys.path.insert(0, str(runtime))
    from assembly_policy.wowrobo import env as native
    from assembly_policy.wowrobo.bank import directory_signature
    from assembly_policy.wowrobo.obs import PointTables

    cfg = read(args.config)
    bank = read(args.bank / "manifest.json")
    tasks = native.resolve_tasks(cfg["tasks"])
    if cfg.get("suite") != "wowrobo" or TASK not in tasks:
        raise ValueError("Expected the current WowRobo run configuration including side cover")
    sig = directory_signature(args.bank)
    if sig != cfg["bank_directory_signature"]:
        raise ValueError("Reset bank identity differs from training config")
    options = {
        **cfg["env"], "profile_path": bank["profile_path"],
        "review_model": bank["review_model"], "nthread": 1,
    }
    for key in ("profile_path", "review_model"):
        if not Path(options[key]).is_file():
            raise FileNotFoundError(f"Native training asset missing: {options[key]}")
    torch.set_num_threads(args.threads)
    env = native.BatchedAssemblyEnv(tasks, [tasks.index(TASK)], native.EnvConfig(**options))
    try:
        if env.model_signature != cfg["model_signature"] or env.model_signature != bank["model_signature"]:
            raise ValueError("Native physics identity differs")
        tables = PointTables(tasks, review_model=bank["review_model"])
        if tables.signature != cfg["observation_signature"]:
            raise ValueError("Native point encoder identity differs")
    except BaseException:
        env.close()
        raise
    contract = {
        "tasks": tasks,
        "model_signature": env.model_signature,
        "observation_signature": tables.signature,
        "bank_directory_signature": sig,
        "env": cfg["env"],
    }
    return env, tables, contract


def points(tables, observation):
    import torch

    args = (torch.as_tensor(observation[key], dtype=torch.long if key == "task" else torch.float32)
            for key in POINT_KEYS)
    return tables(*args).detach().numpy()[0]


def reset_bank(path, split, size):
    with np.load(path / f"{TASK}_{split}.npz", allow_pickle=False) as archive:
        states = archive["states"]
        stages = archive["stage"].astype(str) if "stage" in archive.files else None
    if states.ndim != 2 or states.shape[1] != size:
        raise ValueError("Unexpected reset payload shape")
    return states, stages


def select_rows(args, states, stages):
    if args.rows:
        rows = list(dict.fromkeys(int(x) for x in args.rows.split(",")))
        if any(x < 0 or x >= len(states) for x in rows):
            raise ValueError("Reset row is out of range")
    else:
        if stages is None:
            raise ValueError("This bank has no stage field. Specify checked --rows explicitly.")
        rows = np.flatnonzero([any(stage.startswith(prefix) for prefix in args.stage)
                               for stage in stages])
        if not len(rows):
            raise ValueError("No resets matched. Inspect bank stages or specify --rows.")
        rows = np.random.default_rng(args.seed).permutation(rows)[:args.limit].tolist()
    return rows[args.shard_index::args.shard_count]


def collect(args):
    env, tables, contract = setup_native(args)
    try:
        states, stages = reset_bank(args.bank, "train", env.nstate)
        rows = select_rows(args, states, stages)
        if not rows:
            raise ValueError("No rows assigned to this shard")
        args.out.mkdir(parents=True, exist_ok=False)
        write(args.out / "contract.json", contract)
        reference = CoverReference(args.scene, args.targets, args.trajectory, args.audit)
        if not np.isclose(reference.dt, env.ecfg.control_dt):
            raise ValueError("Retimed reference uses a different action period")
        start_index = round(args.from_reference_s / reference.dt)
        if not 0 <= start_index < len(reference.targets):
            raise ValueError("Invalid reference suffix")
        for index in rows:
            start = time.perf_counter()
            env.set_states([0], states[index:index+1])
            target = reference.retarget(env.m, env.data[0], range(start_index, len(reference.targets)))
            initial = env.data[0].qpos[env.ctrl_adr].copy()
            buffers = {key: [] for key in ("points", "proprio", "actions")}
            qpos = [env.data[0].qpos.copy()]
            n_blend = round(args.blend_s/env.ecfg.control_dt)
            outcome = {}
            peak = 0.
            initial_z = float(env.data[0].body(TASK).xpos[2])
            max_lift = 0.
            for tick in range(env.horizon):
                if tick < n_blend:
                    fraction = (tick+1)/n_blend
                    pos = initial + fraction*(target[0]-initial)
                else:
                    pos = target[min(tick-n_blend, len(target)-1)]
                obs = env.observe()
                action, expected_ctrl = action_for_target(env, pos, args.action_limit)
                _, done, info = env.step(action[None])
                if not np.allclose(env.last_ctrl[0], expected_ctrl, rtol=0., atol=1e-8):
                    raise RuntimeError("Native actuator update differs from the teacher action contract")
                qpos.append(env.data[0].qpos.copy())
                max_lift = max(max_lift, float(env.data[0].body(TASK).xpos[2])-initial_z)
                buffers["points"].append(points(tables, obs))
                buffers["proprio"].append(obs["proprio"][0])
                buffers["actions"].append(action)
                peak = max(peak, float(info["penetration"][0]))
                if done[0]:
                    outcome = {key: bool(info[key][0]) for key in ("success", "safety", "timeout")}
                    break
            receipt = {
                "row": index, "split": "train", "stage": None if stages is None else str(stages[index]),
                "reference_start_s": args.from_reference_s, "ticks": len(qpos)-1,
                "seconds": time.perf_counter()-start, "peak_penetration_m": peak,
                "max_cover_root_rise_m": max_lift, **outcome,
            }
            episode = args.out / f"row-{index:05d}"
            if outcome.get("success") and not outcome.get("safety"):
                arrays = {k: np.asarray(v, np.float32) for k, v in buffers.items()}
                np.savez_compressed(episode.with_suffix(".npz"), **arrays)
            if args.save_replay:
                joints = [env.m.joint(i).name for i in range(env.m.njnt)]
                np.savez_compressed(episode.with_suffix(".replay.npz"), qpos=np.asarray(qpos),
                                    dt=env.ecfg.control_dt, joint_names=np.asarray(joints),
                                    addresses=env.m.jnt_qposadr.copy(), source="native_rollout")
            write(episode.with_suffix(".json"), receipt)
            print(json.dumps(receipt), flush=True)
    finally:
        env.close()


def actor_net(n_points=448, proprio=250, actions=12):
    from assembly_policy.rl.nets import ActorCritic

    return ActorCritic(n_points, proprio_dim=proprio, act_dim=actions)


def train_actor(model, x, p, y, *, steps, batch, seed):
    import torch

    generator = torch.Generator().manual_seed(seed)
    model.norm.update(p)
    optimizer = torch.optim.AdamW(model.actor.parameters(), lr=3e-4, weight_decay=1e-4)
    for _ in range(steps):
        indices = torch.randint(len(x), (batch,), generator=generator).to(x.device)
        prediction = model.actor(x[indices], model.norm(p[indices]))
        loss = (prediction-y[indices]).square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.actor.parameters(), 1.)
        optimizer.step()
    return float(loss.detach().cpu())


def fit(args):
    import torch

    sys.path.insert(0, str(args.runtime.resolve()))
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    receipts = {}
    contract = None
    for folder in args.data:
        candidate = read(folder / "contract.json")
        if contract is None:
            contract = candidate
        if contract != candidate:
            raise ValueError("Mixed native observation, reset or control contracts")
        for path in sorted(folder.glob("row-*.json")):
            result = read(path)
            key = result["row"]
            if (result.get("split") == "train" and result.get("success")
                    and not result.get("safety") and path.with_suffix(".npz").is_file()):
                receipts.setdefault(key, path.with_suffix(".npz"))
    if len(receipts) < 4:
        raise ValueError("Need at least four distinct successful training resets to fit and hold out episodes")
    rows = np.random.default_rng(args.seed).permutation(sorted(receipts))
    nval = max(1, len(rows)//5)

    def tensors(keys):
        arrays = {k: [] for k in ("points", "proprio", "actions")}
        for key in keys:
            with np.load(receipts[key], allow_pickle=False) as file:
                for name, buffer in arrays.items():
                    value = file[name]
                    if not np.isfinite(value).all():
                        raise ValueError(f"Invalid {name} for row {key}")
                    buffer.append(value)
        return tuple(torch.as_tensor(np.concatenate(buffer), device=args.device, dtype=torch.float32)
                     for buffer in arrays.values())

    val = tensors(rows[:nval])
    x, p, y = tensors(rows[nval:])
    if (x.shape[1:], p.shape[1:], y.shape[1:]) != ((1344,), (250,), (12,)):
        raise ValueError("Dataset differs from the 448-point, 250-proprio, 12-action policy")
    if not len(x) or not len(val[0]):
        raise ValueError("Training and validation splits must both contain transitions")
    model = actor_net().to(args.device)

    @torch.no_grad()
    def mse():
        vx, vp, vy = val
        squared = [(model.actor(a, model.norm(b))-c).square().sum()
                   for a, b, c in zip(vx.split(1024), vp.split(1024), vy.split(1024))]
        return float(torch.stack(squared).sum().cpu())/vy.numel()

    before = mse()
    start = time.perf_counter()
    last = train_actor(model, x, p, y, steps=args.steps, batch=args.batch, seed=args.seed)
    metrics = {
        "training_rows": rows[nval:].tolist(), "validation_rows": rows[:nval].tolist(),
        "training_transitions": len(x), "validation_transitions": len(val[0]),
        "validation_mse_before": before, "validation_mse_after": mse(),
        "last_batch_mse": last, "optimizer_steps": args.steps, "seconds": time.perf_counter()-start,
    }
    args.out.mkdir(parents=True, exist_ok=False)
    torch.save({"kind": "wowrobo-bc-actor-v1", "contract": contract,
                "actor": {k: v.cpu() for k, v in model.actor.state_dict().items()},
                "norm": {k: v.cpu() for k, v in model.norm.state_dict().items()},
                "metrics": metrics}, args.out / "actor.pt")
    write(args.out / "fit.json", metrics)
    print(json.dumps(metrics), flush=True)


def policy_from(path, contract, n_points, proprio, act_dim, *, trusted_ppo=False):
    import torch

    model = actor_net(n_points, proprio, act_dim)
    if path is not None:
        checkpoint = torch.load(path, map_location="cpu", weights_only=not trusted_ppo)
        if checkpoint.get("kind") == "wowrobo-bc-actor-v1":
            if checkpoint["contract"] != contract:
                raise ValueError("Actor checkpoint contract differs")
            model.actor.load_state_dict(checkpoint["actor"])
            model.norm.load_state_dict(checkpoint["norm"])
        else:
            config = checkpoint["args"]
            for key in ("model_signature", "observation_signature", "bank_directory_signature"):
                if config.get(key) != contract[key]:
                    raise ValueError(f"Baseline checkpoint differs on {key}")
            model.load_state_dict(checkpoint["model"])
    return model.eval()


def evaluate(args):
    import torch

    torch.manual_seed(args.seed)
    env, tables, contract = setup_native(args)
    try:
        states, stages = reset_bank(args.bank, "eval", env.nstate)
        rows = select_rows(args, states, stages)
        policies = {
            "bc": policy_from(args.actor, contract, tables.n_pts,
                              env.hist.shape[1]*env.hist.shape[2], env.m.nu),
            "random_init": policy_from(None, contract, tables.n_pts,
                                       env.hist.shape[1]*env.hist.shape[2], env.m.nu),
        }
        if args.baseline:
            policies["ppo_baseline"] = policy_from(args.baseline, contract, tables.n_pts,
                                                   env.hist.shape[1]*env.hist.shape[2], env.m.nu,
                                                   trusted_ppo=True)
        args.out.mkdir(parents=True, exist_ok=False)
        for name, policy in policies.items():
            outcomes = []
            for index in rows:
                env.set_states([0], states[index:index+1])
                initial_z = float(env.data[0].body(TASK).xpos[2])
                max_lift = 0.
                for tick in range(env.horizon):
                    obs = env.observe()
                    with torch.no_grad():
                        action = policy.actor(torch.from_numpy(points(tables, obs))[None],
                                              policy.norm(torch.from_numpy(obs["proprio"]))).numpy()
                    _, done, info = env.step(action)
                    max_lift = max(max_lift, float(env.data[0].body(TASK).xpos[2])-initial_z)
                    if done[0]:
                        break
                outcomes.append({
                    "row": index, "ticks": tick+1, "max_cover_root_rise_m": max_lift,
                    **{k: bool(info[k][0]) for k in ("success", "safety", "timeout")},
                })
            write(args.out / f"{name}.json", outcomes)
            print(name, sum(x["success"] for x in outcomes), "/", len(outcomes), flush=True)
        write(args.out / "evaluation.json", {"actor_sha256": sha(args.actor), "contract": contract,
                                              "split": "eval", "rows": rows})
    finally:
        env.close()


def benchmark(args):
    import torch

    sys.path.insert(0, str(args.runtime.resolve()))
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    model = actor_net().to(device)
    # Synthetic data for optimizer throughput only. No model is saved.
    points_ = torch.randn((args.samples, 448*3), device=device)
    prop = torch.randn((args.samples, 250), device=device)
    action = torch.randn((args.samples, 12), device=device)
    start = time.perf_counter()
    last = train_actor(model, points_, prop, action, steps=args.steps, batch=args.batch, seed=args.seed)
    if device.type == "mps":
        torch.mps.synchronize()
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter()-start
    print(json.dumps({"mode": "synthetic_optimizer_benchmark", "device": args.device,
                      "threads": args.threads, "steps": args.steps, "batch": args.batch,
                      "seconds": elapsed, "examples_per_s": args.steps*args.batch/elapsed,
                      "last_batch_mse": last}, indent=2))


def video(args):
    """Render recorded post-step states without simulating or modifying the recording."""
    import os

    os.environ.setdefault("MUJOCO_GL", "glfw")
    import mujoco
    from PIL import Image, ImageDraw

    model = mujoco.MjModel.from_xml_path(str(args.scene))
    data = mujoco.MjData(model)
    receipt = read(args.receipt)
    with np.load(args.replay, allow_pickle=False) as archive:
        if archive["source"].item() != "native_rollout":
            raise ValueError("Expected a recorded native replay")
        q = archive["qpos"]
        fps = 1/float(archive["dt"])
        sources = dict(zip(archive["joint_names"].tolist(),
                           np.split(q, archive["addresses"][1:], axis=1)))
    mapped = np.broadcast_to(model.qpos0, (len(q), model.nq)).copy()
    for i in range(model.njnt):
        start = model.jnt_qposadr[i]
        end = model.jnt_qposadr[i+1] if i+1 < model.njnt else model.nq
        track = sources[model.joint(i).name]
        if track.shape[1] != end-start:
            raise ValueError(f"Joint shape differs for {model.joint(i).name}")
        mapped[:, start:end] = track
    if not np.isfinite(mapped).all():
        raise ValueError("Nonfinite replay")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    w, h = 960, 540
    model.vis.global_.offwidth = w
    model.vis.global_.offheight = h
    option = mujoco.MjvOption()
    option.geomgroup[:] = [1, 0, 1, 0, 0, 0]
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [-.03, .06, .09]
    camera.distance, camera.azimuth, camera.elevation = .72, 96, -40
    status = ("SEATED" if receipt.get("success") else
              "SAFETY STOP" if receipt.get("safety") else "TIMEOUT")
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
        "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(args.out),
    ]
    with mujoco.Renderer(model, height=h, width=w) as renderer, subprocess.Popen(
            command, stdin=subprocess.PIPE) as encoder:
        for i, pose in enumerate(mapped):
            data.qpos[:] = pose
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=camera, scene_option=option)
            frame = Image.fromarray(renderer.render())
            draw = ImageDraw.Draw(frame)
            draw.rectangle((0, 0, w, 31), fill="#111b29")
            draw.text((12, 9), f"NATIVE TEACHER  /  RESET {receipt['row']}  /  "
                                f"{i/fps:05.2f} s  /  FINAL OUTCOME {status}", fill="white")
            encoder.stdin.write(frame.tobytes())
        encoder.stdin.close()
        if encoder.wait():
            raise RuntimeError("Failed to encode replay")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    for name, fn in (("collect", collect), ("fit", fit), ("evaluate", evaluate), ("benchmark", benchmark)):
        p = sub.add_parser(name)
        p.set_defaults(fn=fn)
        p.add_argument("--runtime", type=Path, default=ROOT)
        p.add_argument("--seed", type=int, default=17)
        p.add_argument("--threads", type=int, default=1 if name in ("collect", "evaluate") else 4)
        if name in ("collect", "evaluate"):
            p.add_argument("--bank", type=Path, required=True)
            p.add_argument("--config", type=Path, required=True)
            p.add_argument("--rows", help="Comma-separated reset row indices for this split")
            p.add_argument("--stage", nargs="+", default=["00"], help="Accepted stage prefixes")
            p.add_argument("--limit", type=int, default=12)
            p.add_argument("--shard-count", type=int, default=1)
            p.add_argument("--shard-index", type=int, default=0)
        if name in ("benchmark", "fit"):
            p.add_argument("--device", default="cpu")
            p.add_argument("--steps", type=int, default=500)
            p.add_argument("--batch", type=int, default=256)
        if name != "benchmark":
            p.add_argument("--out", type=Path, required=True)
        if name == "collect":
            for key in ("scene", "targets", "trajectory", "audit"):
                p.add_argument("--"+key, type=Path, required=True)
            p.add_argument("--from-reference-s", type=float, default=0.)
            p.add_argument("--blend-s", type=float, default=.5)
            p.add_argument("--action-limit", type=float, default=1.)
            p.add_argument("--save-replay", action="store_true")
        if name == "fit":
            p.add_argument("--data", type=Path, nargs="+", required=True)
        if name == "evaluate":
            p.add_argument("--actor", type=Path, required=True)
            p.add_argument("--baseline", type=Path, help="Trusted existing PPO checkpoint")
        if name == "benchmark":
            p.add_argument("--samples", type=int, default=8192)
    p = sub.add_parser("video")
    p.set_defaults(fn=video)
    for key in ("scene", "replay", "receipt", "out"):
        p.add_argument("--"+key, type=Path, required=True)
    args = parser.parse_args()
    for option in ("threads", "steps", "batch", "limit", "samples", "action_limit"):
        if hasattr(args, option) and getattr(args, option) <= 0:
            parser.error(f"--{option.replace('_', '-')} must be positive")
    if hasattr(args, "shard_count") and not 0 <= args.shard_index < args.shard_count:
        parser.error("--shard-index must be between 0 and --shard-count minus one")
    if hasattr(args, "blend_s") and args.blend_s < 0:
        parser.error("--blend-s must be nonnegative")
    args.fn(args)


if __name__ == "__main__":
    main()
