"""Check asset provenance, task resets, observation/control plumbing, and source preservation."""

import argparse
import ast
import hashlib
import json
import subprocess
from pathlib import Path

import mujoco
import numpy as np
import torch

from assembly_policy.so101.env import PROPRIO_DIM, TASK_NAMES, BatchedAssemblyEnv, EnvConfig
from assembly_policy.so101.obs import PointTables
from assembly_policy.so101.resets import check_bank, directory_signature, signature
from assembly_policy.so101.scene import ASSETS, ROOT
from assembly_policy.so101.workcell import assert_matches_reference, default_bank_directory
from scripts.so101_source_checks import sdf_equations_unchanged

UPSTREAM = "eb5d15e004591d380c0d521da67e025ebee37703"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bank-dir", type=Path, default=default_bank_directory())
    ap.add_argument("--out", type=Path, default=ROOT / "artifacts_local/preflight.json")
    ap.add_argument("--max-contact-depth", type=float, default=None)
    ap.add_argument("--stability", choices=["velocity", "pose"], default="pose")
    ap.add_argument("--safety-penalty", type=float, default=0.0)
    ap.add_argument("--physics", choices=["native"], default="native")
    ap.add_argument("--physics-model", type=Path)
    ap.add_argument(
        "--allow-task-mechanics-failure",
        action="store_true",
        help="Report screw pitch failure without blocking learner debugging; safety checks remain required",
    )
    a = ap.parse_args()
    bank = a.bank_dir
    check_bank(bank)
    result = {
        "model_signature": signature(),
        "bank_directory_signature": directory_signature(bank),
        "bank_dir": str(bank.resolve()),
        "max_contact_depth": a.max_contact_depth,
        "stability_metric": a.stability,
        "safety_penalty": a.safety_penalty,
        "allow_task_mechanics_failure": a.allow_task_mechanics_failure,
        "checks": {},
    }
    result["physics_backend"] = a.physics
    result["physics_signature"] = result["model_signature"]

    def check(name, condition):
        result["checks"][name] = bool(condition)
        if not condition:
            a.out.parent.mkdir(parents=True, exist_ok=True)
            a.out.write_text(json.dumps(result, indent=2) + "\n")
            raise AssertionError(name)

    cad = ASSETS / "hiwonder/follower.3dm"
    cad_hash = "f675f6049f94206c7b74ab73c40485913916babff1a6f4fc145dc9040672a2f8"
    if cad.exists():
        check("official_cad_hash", hashlib.sha256(cad.read_bytes()).hexdigest() == cad_hash)
    else:
        pinned = json.loads((ROOT / "configs/runtime_bundle.json").read_text())["files"]

        def matches(name, record):
            with (ROOT / name).open("rb") as stream:
                return hashlib.file_digest(stream, "sha256").hexdigest() == record["sha256"]

        check(
            "pinned_runtime_derivatives",
            all(matches(n, r) for n, r in pinned.items() if n.startswith("assets_local/")),
        )
        check(
            "imported_cad_provenance",
            json.loads((ASSETS / "parts/manifest.json").read_text())["cad_sha256"] == cad_hash,
        )
    for f in ("assembly_policy/rl/nets.py", "assembly_policy/rl/sampler.py"):
        check(
            f + "_unchanged",
            subprocess.check_output(["git", "show", UPSTREAM + ":" + f], cwd=ROOT) == (ROOT / f).read_bytes(),
        )
    original = subprocess.check_output(
        ["git", "show", UPSTREAM + ":plugin/assemblenist_sdf.cc"], cwd=ROOT, text=True
    )
    current = (ROOT / "plugin/assemblenist_sdf.cc").read_text()
    check("sdf_equations_unchanged", sdf_equations_unchanged(original, current))

    def update(text):
        return next(
            f
            for n in ast.parse(text).body
            if isinstance(n, ast.ClassDef) and n.name == "Trainer"
            for f in n.body
            if isinstance(f, ast.FunctionDef) and f.name == "update"
        )

    def objective(text):
        names = {"ratio", "A", "pi_loss", "v_clip", "vf_loss"}
        terms = [
            ast.dump(n)
            for n in ast.walk(update(text))
            if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names for t in n.targets)
        ]
        terms += [
            ast.dump(n)
            for n in ast.walk(update(text))
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "backward"
        ]
        if len(terms) != 6:
            raise ValueError("Unexpected PPO objective structure")
        return terms

    check(
        "ppo_objective_unchanged",
        objective(
            subprocess.check_output(
                ["git", "show", UPSTREAM + ":assembly_policy/train.py"], cwd=ROOT, text=True
            )
        )
        == objective((ROOT / "assembly_policy/train.py").read_text()),
    )
    config = EnvConfig(
        nthread=1,
        max_contact_depth=a.max_contact_depth,
        stability_metric=a.stability,
        safety_penalty=a.safety_penalty,
    )
    env = BatchedAssemblyEnv(TASK_NAMES, np.arange(len(TASK_NAMES)), config)
    if a.physics == "native":
        result["workcell_reference"] = assert_matches_reference(env.m)
        check("native_scene_matches_reference", result["workcell_reference"]["passed"])
    states = []
    for name in TASK_NAMES:
        tr = np.load(bank / f"{name}_train.npz")
        ev = np.load(bank / f"{name}_eval.npz")
        check(name + "_disjoint_seeds", not set(tr["seed"]) & set(ev["seed"]))
        states.append(ev["states"][0])
    env.set_states(np.arange(len(TASK_NAMES)), np.stack(states))
    o = env.observe()
    check("observation_shape", o["proprio"].shape == (len(TASK_NAMES), PROPRIO_DIM * 5))
    check("twelve_robot_actuators", env.m.nu == 12)
    check("no_attachment_constraints", env.m.neq == 0)
    for name in (*TASK_NAMES, "base"):
        check(name + "_free", int(env.m.joint(name + "_free").type[0]) == int(mujoco.mjtJoint.mjJNT_FREE))
    points = PointTables(TASK_NAMES)
    p = points(*[torch.tensor(o[k]) for k in ("task", "tcp_p", "tcp_R", "part_p", "part_q", "finger")])
    check(
        "points_finite_448",
        points.n_pts == 448 and p.shape == (len(TASK_NAMES), 1344) and bool(torch.isfinite(p).all()),
    )
    reward, done, info = env.step(np.zeros((len(TASK_NAMES), 12)))
    check("loose_resets_not_successful", not info["success"].any())
    check("zero_control_stable", not info["safety"].any() and bool(np.isfinite(reward).all()))
    # Timeouts and autoreset consumers must receive the real terminal state.
    env.t[:] = env.horizon - 1
    _, done, info = env.step(np.zeros((len(TASK_NAMES), 12)))
    check("timeouts_terminate", done.all() and info["timeout"].all())
    env.close()
    from scripts.so101_thread_check import bench

    result["thread_check"] = {"rotation": bench(), "axial": bench(True)}
    r = result["thread_check"]["rotation"]
    axial = result["thread_check"]["axial"]
    pitch_ok = abs(r["advance_m"] - r["expected_advance_m"]) < 0.00005 and abs(r["turns"]) > 1
    result["task_mechanics"] = {
        "thread_tracks_pitch": bool(pitch_ok),
        "rotation_turns": float(r["turns"]),
        "required_rotation_turns": 1.0,
        "pitch_error_m": abs(r["advance_m"] - r["expected_advance_m"]),
        "pitch_tolerance_m": 0.00005,
    }
    if not a.allow_task_mechanics_failure:
        check("thread_tracks_pitch", pitch_ok)
    check("thread_blocks_axial_push", abs(axial["advance_m"]) < 0.0001)
    result["thread_check"] = {
        k: {kk: vv for kk, vv in v.items() if kk != "trace"} for k, v in result["thread_check"].items()
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"PASS: {len(result['checks'])} checks", flush=True)
    if not pitch_ok:
        print(
            "KNOWN TASK FAILURE: screw rotation/pitch qualification; see task_mechanics in the report",
            flush=True,
        )


if __name__ == "__main__":
    main()
