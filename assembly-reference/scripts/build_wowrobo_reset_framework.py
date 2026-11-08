"""Prepare, inspect and run an object-relative, resumable 1,800-reset bank.

No training/cluster jobs are launched. Native CPU workers own independent stage
shards; only physical admission can fill a quota. Failed slots remain missing.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from assembly_policy.so101.wowrobo_bank import digest, write_json, write_npz
from assembly_policy.so101.wowrobo_reset_framework import (
    check_library,
    grasp_id,
    relative_grasp,
    resize_sampling,
)

ROOT = Path(__file__).resolve().parents[1]


def prepare(output, config_path, results, train=60, evaluation=12, nominal_source=None):
    import mujoco
    import numpy as np
    from publish_wowrobo_equilibrium_resets import publish

    from assembly_policy.so101.wowrobo_contact_backend import (
        load_backend,
        set_surface_mode,
    )
    from assembly_policy.so101.wowrobo_photo_resets import PhotoResetSequence

    output = output.resolve()
    if output.exists():
        raise ValueError("Preparation is immutable; choose a new output directory")
    config = json.loads(config_path.read_text())
    source = (
        nominal_source.resolve() if nominal_source else ROOT / config["nominal_bank"]
    )
    stages = json.loads((source / "stages/manifest.json").read_text())["task_names"]
    reports = {
        stage: json.loads((results / (stage + ".json")).read_text()) for stage in stages
    }
    signatures = {r["physics_profile"]["model_sha256"] for r in reports.values()}
    if len(signatures) != 1:
        raise ValueError("All seeds must use the same physical model")
    profile = next(iter(reports.values()))["physics_profile"]
    if digest(Path(profile["model_path"]).read_bytes()) not in signatures:
        raise ValueError("Seed model no longer matches its witness")
    # Preserve reference validity labels while also retaining failed seed poses
    # as geometric proposals. Failed seeds can NEVER bypass variation admission.
    publish(
        source,
        [results],
        output / "references",
        allow_incomplete=True,
        minimum_only=True,
    )
    nominal = output / "references"
    load_backend(profile["grid_manifest"])
    set_surface_mode(1)
    physical = mujoco.MjModel.from_xml_string(Path(profile["model_path"]).read_text())
    visual = mujoco.MjModel.from_xml_string((nominal / "scene.xml").read_text())
    sequence = PhotoResetSequence(ROOT / config["component_assets"])
    definitions = {s["id"]: s for s in sequence.plan["states"]}
    receipts = json.loads((nominal / "stage-receipts.json").read_text())
    by_stage = {s["id"]: s for s in receipts}
    library = {
        "schema_version": 1,
        "model_signature": digest((nominal / "scene.xml").read_bytes()),
        "physical_model_signature": next(iter(signatures)),
        "grasps": {},
        "bindings": {},
    }
    support_recalibration = {}
    for stage in stages:
        report = reports[stage]
        with np.load(results / (stage + ".npz")) as saved:
            state, controls = saved["states"][0], saved["controls"][0]
        if (
            digest(state.tobytes()) != report["state_sha256"]
            or digest(controls.tobytes()) != report["control_sha256"]
        ):
            raise ValueError(f"Witness snapshot hash mismatch: {stage}")
        d = mujoco.MjData(physical)
        mujoco.mj_setState(physical, d, state, mujoco.mjtState(report["state_spec"]))
        d.ctrl[:] = controls
        mujoco.mj_forward(physical, d)
        view = mujoco.MjData(visual)
        view.qpos[:], view.qvel[:], view.ctrl[:], view.time = (
            d.qpos,
            d.qvel,
            controls,
            d.time,
        )
        full = np.empty(
            mujoco.mj_stateSize(visual, mujoco.mjtState.mjSTATE_FULLPHYSICS)
        )
        mujoco.mj_getState(visual, view, full, mujoco.mjtState.mjSTATE_FULLPHYSICS)
        path = nominal / "stages" / (stage + "_train.npz")
        with np.load(path) as saved:
            arrays = {k: saved[k].copy() for k in saved.files}
        arrays["states"][0], arrays["controls"][0] = full, controls
        write_npz(path, **arrays)
        receipt = by_stage[stage]
        if stage in {
            "21_upper_far_side",
            "22_upper_half",
            "23_upper_near",
            "29_final_far_side",
            "30_final_half",
            "31_final_near",
        }:
            from assembly_policy.so101.wowrobo_repaired_resets import (
                RepairedResetSequence,
            )

            # The physical publisher migrates poses, not historical contact
            # annotations. Re-select the actual supported vertex in the native
            # printed far-end patch; do not pull noise toward an old pose.
            mujoco.mj_forward(visual, view)
            fitter = object.__new__(RepairedResetSequence)
            fitter.model = visual
            fitter.old_receipts = {stage: receipt}
            fitter.definitions = definitions
            support = fitter._support(view, stage)
            if support is None or support["upward_receiver_normal"] is None:
                raise ValueError(f"No actual far-end receiver patch: {stage}")
            normal = np.asarray(support["upward_receiver_normal"])
            receiver = visual.body(support["receiver_body"]).id
            receiver_r = view.xmat[receiver].reshape(3, 3)
            signed_offset = float(
                np.dot(
                    np.asarray(support["far_point_world_m"])
                    - np.asarray(support["nearest_receiver_surface_world_m"]),
                    normal,
                )
            )
            support["mating_evidence"].update(
                reference_signed_support_offset_m=signed_offset,
                support_normal_receiver_local=(receiver_r.T @ normal).tolist(),
                support_reference_state_sha256=report["state_sha256"],
                support_constraint="preserve measured dynamic reference offset; not a zero-gap/static support claim",
            )
            support_recalibration[stage] = {
                "old_mating": receipt.get("mating"),
                "refreshed_support": support,
                "source_state_sha256": report["state_sha256"],
                "geometry_changed": False,
            }
            receipt["support_contact"] = support
            receipt["mating"] = support["mating_evidence"]
        receipt["state_sha256"] = digest(full.tobytes())
        receipt["framework_seed_witness_pass"] = bool(report["minimum_validity_pass"])
        library["bindings"][stage] = {}
        definition = definitions[stage]
        for side, anchor in report["grip_anchor_by_side"].items():
            role, piece = definition[side + "_role"], definition["task"]
            override = receipt.get("support_grasp_by_side", {}).get(side)
            if override:
                calibration = dict(override["grasp_calibration"])
            elif role in ("grasp_active", "hold_connector"):
                calibration = dict(
                    sequence.side_grasps[
                        piece if role == "grasp_active" else "shoulder_connector"
                    ]
                )
            else:
                calibration = dict(
                    sequence.installed_upper_grip[1]
                    if role == "hold_upper_arm"
                    else sequence.installed_base_grip[1]
                )
            calibration["opening_radians"] = float(d.joint(side + "_gripper").qpos[0])
            body, site = (
                physical.body(anchor).id,
                physical.site(side + "_gripperframe").id,
            )
            record = {
                "anchor": anchor,
                "transform": relative_grasp(
                    d.xpos[body], d.xmat[body], d.site_xpos[site], d.site_xmat[site]
                ),
                "calibration": calibration,
                "source_witness_pass": bool(report["minimum_validity_pass"]),
                "source_stage": stage,
                "source_side": side,
                "role": role,
                "witness_state_sha256": report["state_sha256"],
                "qualification": "Measured seed grasp; every candidate requires native overlap/retention admission",
            }
            key = grasp_id(record)
            library["grasps"][key] = record
            library["bindings"][stage][side] = key
    write_json(nominal / "stage-receipts.json", receipts)
    write_json(output / "support-recalibration.json", support_recalibration)
    check_library(library)
    write_json(output / "grasps.json", library)
    config["sampling"] = resize_sampling(config["sampling"], train, evaluation)
    config["sampling"].update(
        train_total=train * len(stages), eval_total=evaluation * len(stages)
    )
    config.update(
        nominal_bank=str(nominal),
        grasp_library=str(output / "grasps.json"),
        completed_anchor_bank=str(nominal / "completed"),
        profile="wowrobo-object-relative-1800-v4",
        preserve_reference_velocities=True,
    )
    write_json(output / "config.json", config)
    unqualified = [s for s in stages if not reports[s]["minimum_validity_pass"]]
    plan = {
        "schema_version": 1,
        "config": str(output / "config.json"),
        "config_sha256": digest((output / "config.json").read_bytes()),
        "stage_ids": stages,
        "train_per_stage": train,
        "eval_per_stage": evaluation,
        "target_total": len(stages) * (train + evaluation),
        "source_witness_pass_count": len(stages) - len(unqualified),
        "unqualified_seed_stages": unqualified,
        "stationarity_required": False,
        "duplicate_padding": False,
        "physical_model_signature": next(iter(signatures)),
        "grasp_library_sha256": digest((output / "grasps.json").read_bytes()),
    }
    write_json(output / "plan.json", plan)
    return plan


def status(output, pilot=False):
    plan = json.loads((output / "plan.json").read_text())
    base = output / ("pilot" if pilot else "shards")
    rows = {}
    for stage in plan["stage_ids"]:
        path = base / stage / "run.json"
        if path.exists():
            run = json.loads(path.read_text())
            p = run.get("progress", {})
            rows[stage] = {
                "status": run["status"],
                "train": p.get("accepted_train", 0),
                "eval": p.get("accepted_eval", 0),
                "attempts": p.get("generated_count", 0),
            }
        else:
            rows[stage] = {
                "status": "not_started",
                "train": 0,
                "eval": 0,
                "attempts": 0,
            }
    total = sum(row["train"] + row["eval"] for row in rows.values())
    complete = not pilot and all(
        row["train"] == plan["train_per_stage"]
        and row["eval"] == plan["eval_per_stage"]
        for row in rows.values()
    )
    result = {
        "target": plan["target_total"],
        "accepted": total,
        "complete": complete,
        "pilot": pilot,
        "by_stage": rows,
        "stationarity_required": False,
    }
    write_json(output / ("pilot-progress.json" if pilot else "progress.json"), result)
    return result


def run(output, stages, workers=2, attempts=20, pilot=False):
    with (output / ".run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("Another coordinator owns this plan") from error
        return _run(output, stages, workers, attempts, pilot)


def _run(output, stages, workers=2, attempts=20, pilot=False):
    plan = json.loads((output / "plan.json").read_text())
    if digest(Path(plan["config"]).read_bytes()) != plan["config_sha256"]:
        raise ValueError("Prepared configuration changed; prepare a new plan")
    if digest((output / "grasps.json").read_bytes()) != plan["grasp_library_sha256"]:
        raise ValueError("Prepared grasp library changed")
    selected = stages or plan["stage_ids"]
    if set(selected) - set(plan["stage_ids"]) or len(set(selected)) != len(selected):
        raise ValueError("Invalid or repeated stage")
    if workers < 1 or attempts < 1:
        raise ValueError("Positive workers/attempt limit required")
    # Build once before any workers start. Absolute grid paths then give all
    # shards the exact same XML signature and prevent concurrent cache writes.
    from assembly_policy.so101.wowrobo_physical_resets import WowRoboPhysicalValidator

    config = json.loads(Path(plan["config"]).read_text())

    def resolve(key):
        value = Path(config[key])
        return value if value.is_absolute() else ROOT / value

    profile_dir = output / "physical"
    # Publication preserves the source XML but rebakes source OBJ filenames.
    # Reuse grids only for the exact same audit XML and verified grid bytes;
    # the physical builder independently checks each cached source mesh hash
    # and spacing against its newly baked source before accepting the cache.
    witness = json.loads(
        (
            resolve("nominal_bank")
            / "physical-replays"
            / (plan["stage_ids"][0] + ".json")
        ).read_text()
    )
    seed_profile = witness["physics_profile"]
    if digest(resolve("nominal_bank").joinpath("scene.xml").read_bytes()) == digest(
        Path(seed_profile["audit_model_path"]).read_bytes()
    ):
        cache = profile_dir / "grids"
        cache.mkdir(parents=True, exist_ok=True)
        for metadata in Path(seed_profile["grid_manifest"]).parent.glob("grid_*.json"):
            record = json.loads(metadata.read_text())
            grid = Path(record["grid_file"])
            if record.get("grid_sha256") != digest(grid.read_bytes()):
                raise ValueError("Witness grid cache identity changed")
            destination = cache / grid.name
            cached_metadata = cache / metadata.name
            if not cached_metadata.exists():
                if not destination.exists():
                    destination.symlink_to(grid)
                elif digest(destination.read_bytes()) != record["grid_sha256"]:
                    raise ValueError("Prepared grid cache differs from witness")
                record["grid_file"] = str(destination)
                write_json(cached_metadata, record)
    WowRoboPhysicalValidator(
        nominal_bank=resolve("nominal_bank"),
        output_dir=profile_dir,
        grid_manifest=resolve("grid_manifest"),
        pin_mesh=resolve("pin_mesh"),
        pin_registration_path=resolve("pin_registration_path"),
        completed_anchor_bank=resolve("completed_anchor_bank"),
        mating_profile=config.get("physical_mating_profile"),
        pin_pose_registered=config.get("pin_pose_registered", False),
        passive_motor_stops=True,
        exclude_internal_servo_contacts=True,
        source_mesh_table_contacts=True,
        center_motor_housings=True,
        helper_grid_profile=resolve("helper_grid_profile"),
    )
    base = output / ("pilot" if pilot else "shards")
    base.mkdir(exist_ok=True)

    def work(stage):
        cmd = [
            sys.executable,
            str(ROOT / "scripts/build_wowrobo_randomized_bank.py"),
            "--config",
            plan["config"],
            "--output",
            str(base / stage),
            "--stages",
            stage,
            "--validation",
            "physical",
            "--max-attempts",
            str(attempts),
            "--physical-profile",
            str(profile_dir),
            "--settle-seconds",
            ".08",
            "--hold-seconds",
            ".2",
        ]
        cmd += (
            ["--train-per-stage", "1", "--eval-per-stage", "1"] if pilot else ["--full"]
        )
        with (base / (stage + ".log")).open("a") as log:
            code = subprocess.run(
                cmd, check=False, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT
            ).returncode
        return stage, code

    # Independent subprocesses isolate MuJoCo native/plugin globals. A stage owns
    # its output/profile, so no two workers race on an attempt or model cache.
    failures = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for job in as_completed([pool.submit(work, stage) for stage in selected]):
            stage, code = job.result()
            if code:
                failures[stage] = code
            print(json.dumps({"stage": stage, "returncode": code}), flush=True)
            status(output, pilot)
    result = status(output, pilot)
    result["worker_failures"] = failures
    if result["complete"]:
        merge(output, plan)
    return result


def merge(output, plan):
    """Seal a loadable flat bank only after every stage and split fills its quota."""
    import numpy as np

    target = output / "bank"
    target.mkdir(exist_ok=True)
    manifests, seen = [], set()
    for stage in plan["stage_ids"]:
        shard = output / "shards" / stage / "qualified"
        manifest = json.loads((shard / "manifest.json").read_text())
        if not manifest["complete"]:
            raise ValueError("Incomplete shard cannot seal a bank")
        manifests.append(manifest)
        for split, count in (
            ("train", plan["train_per_stage"]),
            ("eval", plan["eval_per_stage"]),
        ):
            path = shard / (stage + "_" + split + ".npz")
            with np.load(path) as saved:
                if len(saved["states"]) != count:
                    raise ValueError("Shard quota mismatch")
                for state in saved["states"]:
                    fingerprint = digest(np.round(state, 10).tobytes())
                    if fingerprint in seen:
                        raise ValueError("Duplicate physical state across bank/splits")
                    seen.add(fingerprint)
            shutil.copy2(path, target / path.name)
    if (
        len({m["model_signature"] for m in manifests}) != 1
        or len({m["state_spec"] for m in manifests}) != 1
    ):
        raise ValueError("Shards use different physical models/state layouts")
    first = output / "shards" / plan["stage_ids"][0] / "qualified"
    shutil.copy2(first / "scene.xml", target / "scene.xml")
    write_json(
        target / "manifest.json",
        dict(
            manifests[0],
            task_names=plan["stage_ids"],
            complete=True,
            prototype_count=plan["target_total"],
            progress=status(output),
            framework_plan_sha256=digest((output / "plan.json").read_bytes()),
        ),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["prepare", "run", "status"])
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/wowrobo_randomization_minimum_valid.json",
    )
    p.add_argument(
        "--results",
        type=Path,
        default=ROOT / "outputs/equilibrium-reset-20261107/minimum-v6",
    )
    p.add_argument(
        "--nominal", type=Path, help="Explicit source reference bank for preparation"
    )
    p.add_argument("--train", type=int, default=60)
    p.add_argument("--eval", type=int, default=12)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--max-attempts", type=int, default=20)
    p.add_argument("--stages", nargs="+")
    p.add_argument("--pilot", action="store_true")
    a = p.parse_args()
    output = a.output.resolve()
    if a.action == "prepare":
        result = prepare(output, a.config, a.results, a.train, a.eval, a.nominal)
    elif a.action == "run":
        result = run(output, a.stages, a.workers, a.max_attempts, a.pilot)
    else:
        result = status(output, a.pilot)
    print(json.dumps(result, indent=2))
    return 2 if result.get("worker_failures") else 0


if __name__ == "__main__":
    raise SystemExit(main())
