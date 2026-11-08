"""Versioned, resumable physical equilibrium repair of every approved start."""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import mujoco
import numpy as np

from assembly_policy.so101.wowrobo_bank import write_json, write_npz
from assembly_policy.so101.wowrobo_equilibrium import settle_and_hold
from assembly_policy.so101.wowrobo_physical_resets import WowRoboPhysicalValidator
from assembly_policy.so101.wowrobo_randomization import WowRoboRandomizer

ROOT = Path(__file__).resolve().parents[1]


def initialize(
    config_path,
    physical,
    output,
    settle,
    hold,
    close,
    gravity,
    initializers=None,
    minimum_only=False,
):
    global SAMPLER, VALIDATOR, SETTINGS, INITIALIZERS
    config = json.loads(Path(config_path).read_text())
    SAMPLER = WowRoboRandomizer(config=config)

    def local(name):
        return ROOT / config[name]

    VALIDATOR = WowRoboPhysicalValidator(
        nominal_bank=SAMPLER.nominal_bank,
        output_dir=physical,
        grid_manifest=local("grid_manifest"),
        pin_mesh=local("pin_mesh"),
        pin_registration_path=local("pin_registration_path"),
        completed_anchor_bank=local("completed_anchor_bank"),
        mating_profile=config["physical_mating_profile"],
        pin_pose_registered=True,
        passive_motor_stops=True,
        exclude_internal_servo_contacts=True,
        source_mesh_table_contacts=bool(
            json.loads((Path(physical) / "profile.json").read_text()).get(
                "source_mesh_table_contacts"
            )
        ),
        center_motor_housings=bool(
            json.loads((Path(physical) / "profile.json").read_text()).get(
                "housing_registration"
            )
        ),
    )
    SETTINGS = (Path(output), settle, hold, close, gravity, minimum_only)
    INITIALIZERS = json.loads(Path(initializers).read_text()) if initializers else {}


def repair(stage):
    output, settle, hold, close, gravity, minimum_only = SETTINGS
    path = output / (stage + ".json")
    if path.exists():
        return json.loads(path.read_text())
    try:
        source = SAMPLER.restore(stage)
        applied = INITIALIZERS.get(stage, {})
        if "full_qpos" in applied:
            q = np.asarray(applied["full_qpos"], dtype=float)
            controls = np.asarray(applied["controls"], dtype=float)
            if (
                q.shape != (SAMPLER.model.nq,)
                or controls.shape != (SAMPLER.model.nu,)
                or not np.isfinite(q).all()
                or not np.isfinite(controls).all()
            ):
                raise ValueError("Initializer must match native qpos/control layout")
            source.qpos[:] = q
            source.ctrl[:] = controls
        if "full_qvel" in applied:
            velocity = np.asarray(applied["full_qvel"], dtype=float)
            if velocity.shape != source.qvel.shape or not np.isfinite(velocity).all():
                raise ValueError("Initializer velocity does not match native state")
            source.qvel[:] = velocity
        for side in ("left", "right"):
            if side not in applied:
                continue
            row = applied[side]
            q = np.asarray(row["qpos"], dtype=float)
            if q.shape != (6,) or not np.isfinite(q).all():
                raise ValueError("Expected six finite native helper joint positions")
            for name, value in zip(
                (
                    "shoulder_pan",
                    "shoulder_lift",
                    "elbow_flex",
                    "wrist_flex",
                    "wrist_roll",
                    "gripper",
                ),
                q,
                strict=True,
            ):
                joint = SAMPLER.model.joint(side + "_" + name)
                source.qpos[joint.qposadr[0]] = value
                actuator = int(
                    np.flatnonzero(SAMPLER.model.actuator_trnid[:, 0] == joint.id)[0]
                )
                source.ctrl[actuator] = value - (
                    row.get("jaw_preload_rad", 0.0) if name == "gripper" else 0.0
                )
        for address, value in applied.get("qpos_overrides", {}).items():
            values = np.atleast_1d(value)
            index = (
                int(address)
                if address.isdigit()
                else int(SAMPLER.model.joint(address).qposadr[0])
            )
            source.qpos[index : index + len(values)] = values
        for address, value in applied.get("ctrl_overrides", {}).items():
            index = (
                int(address)
                if address.isdigit()
                else SAMPLER.model.actuator(address).id
            )
            source.ctrl[index] = value
        if VALIDATOR.profile.get("housing_registration"):
            q = source.qpos.copy()
            if not applied.get("registered_model", False):
                from assembly_policy.so101.wowrobo_housing_registration import (
                    migrate_qpos,
                )

                q, migration = migrate_qpos(
                    SAMPLER.model,
                    VALIDATOR.model,
                    q,
                    propagate_final_with_receiver=int(stage.split("_")[0]) >= 29,
                )
                applied = dict(applied, housing_migration=migration)
            migrated = mujoco.MjData(VALIDATOR.model)
            migrated.qpos[:] = q
            migrated.qvel[:] = source.qvel
            migrated.ctrl[:] = source.ctrl
            source = migrated
        mujoco.mj_forward(
            VALIDATOR.model
            if VALIDATOR.profile.get("housing_registration")
            else SAMPLER.model,
            source,
        )
        saved, controls, report, end = settle_and_hold(
            VALIDATOR,
            stage,
            source,
            settle_seconds=settle,
            hold_seconds=hold,
            close_jaws=close,
            gravity_preload=gravity and not applied.get("gravity_preloaded", False),
            minimum_only=minimum_only,
        )
        report["physics_profile"] = VALIDATOR.profile
        report["source_bank"] = str(SAMPLER.nominal_bank)
        report["capture_initializer"] = applied
        write_npz(
            output / (stage + ".npz"),
            states=saved[None],
            controls=controls[None],
            replay_end=end,
            state_spec=np.array(int(mujoco.mjtState.mjSTATE_INTEGRATION)),
        )
    except (RuntimeError, ValueError, OSError, FloatingPointError) as exc:
        report = {
            "stage": stage,
            "equilibrium_pass": False,
            "rejection_reasons": ["generation_error"],
            "error": repr(exc),
            "training_eligible": False,
        }
    write_json(path, report)
    print(
        stage,
        report.get("minimum_validity_pass")
        if minimum_only
        else report["equilibrium_pass"],
        report["rejection_reasons"],
        flush=True,
    )
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/wowrobo_randomization_repaired_cover_support.json",
    )
    p.add_argument("--physical", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--stages", nargs="+")
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--settle", type=float, default=2.0)
    p.add_argument("--hold", type=float, default=0.5)
    p.add_argument("--no-jaw-closing", action="store_true")
    p.add_argument("--gravity-preload", action="store_true")
    p.add_argument("--initializers", type=Path)
    p.add_argument(
        "--minimum-only",
        action="store_true",
        help="Allow motion; require material clearance and retained grasps only",
    )
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    config = json.loads(a.config.read_text())
    nominal = ROOT / config["nominal_bank"]
    stages = (
        a.stages
        or json.loads((nominal / "stages/manifest.json").read_text())["task_names"]
    )
    with ProcessPoolExecutor(
        max_workers=a.workers,
        initializer=initialize,
        initargs=(
            a.config,
            a.physical,
            a.output,
            a.settle,
            a.hold,
            not a.no_jaw_closing,
            a.gravity_preload,
            a.initializers,
            a.minimum_only,
        ),
    ) as pool:
        results = [
            future.result()
            for future in as_completed([pool.submit(repair, stage) for stage in stages])
        ]
    write_json(
        a.output / "summary.json",
        {
            "requested_stages": stages,
            "equilibrium_pass_count": sum(
                r.get("equilibrium_pass") is True for r in results
            ),
            "minimum_validity_pass_count": sum(
                r.get("minimum_validity_pass") is True for r in results
            ),
            "count": len(results),
            "results": {
                r["stage"]: {
                    k: v
                    for k, v in r.items()
                    if k not in ("physics_profile", "boundary", "settling_trace")
                }
                for r in results
            },
            "training_eligible": False,
        },
    )


if __name__ == "__main__":
    main()
