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


def initialize(config_path, physical, output, settle, hold, close, gravity):
    global SAMPLER, VALIDATOR, SETTINGS
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
    )
    SETTINGS = (Path(output), settle, hold, close, gravity)


def repair(stage):
    output, settle, hold, close, gravity = SETTINGS
    path = output / (stage + ".json")
    if path.exists():
        return json.loads(path.read_text())
    try:
        saved, controls, report, end = settle_and_hold(
            VALIDATOR,
            stage,
            SAMPLER.restore(stage),
            settle_seconds=settle,
            hold_seconds=hold,
            close_jaws=close,
            gravity_preload=gravity,
        )
        report["physics_profile"] = VALIDATOR.profile
        report["source_bank"] = str(SAMPLER.nominal_bank)
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
    print(stage, report["equilibrium_pass"], report["rejection_reasons"], flush=True)
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
            "equilibrium_pass_count": sum(r["equilibrium_pass"] for r in results),
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
