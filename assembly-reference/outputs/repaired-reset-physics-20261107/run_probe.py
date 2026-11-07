"""Bounded full-scene dynamics on explicitly selected repaired geometry.

No force guide, helper weld, altered velocity reset, or admission waiver.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from assembly_policy.so101.wowrobo_physical_resets import WowRoboPhysicalValidator
from assembly_policy.so101.wowrobo_randomization import WowRoboRandomizer

ROOT = Path(__file__).resolve().parents[2]


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--nominal",
        type=Path,
        default=ROOT / "assets_local/wowrobo/photo-resets-repaired-v1",
    )
    p.add_argument(
        "--assets",
        type=Path,
        default=ROOT / "assets_local/wowrobo/components-repaired-v1",
    )
    p.add_argument(
        "--profile",
        type=Path,
        default=ROOT / "assets_local/wowrobo/physical-resets-repaired-v1",
    )
    p.add_argument(
        "--config", type=Path, default=ROOT / "configs/wowrobo_randomization.json"
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "bounded-probes",
    )
    p.add_argument(
        "--stages",
        nargs="+",
        default=["00_loose", "14_pin_half", "22_upper_half", "30_final_half"],
    )
    p.add_argument("--difficulty", type=float, default=0.0)
    p.add_argument("--settle", type=float, default=0.01)
    p.add_argument("--hold", type=float, default=0.02)
    args = p.parse_args()
    frozen = ROOT / "outputs/insertion-fix-20261107/final-common-v1"
    registration = (
        ROOT / "outputs/insertion-fix-20261107/pin/consistent-v3/registration.json"
    )
    validator = WowRoboPhysicalValidator(
        nominal_bank=args.nominal,
        output_dir=args.profile,
        grid_manifest=frozen / "grids/manifest.json",
        pin_mesh=frozen / "assets/pin.obj",
        pin_registration_path=registration,
        completed_anchor_bank=args.nominal / "completed",
        mating_profile="legacy_mapped_all",
        pin_pose_registered=True,
        settle_seconds=args.settle,
        hold_seconds=args.hold,
    )
    sampler = WowRoboRandomizer(
        assets=args.assets, nominal_bank=args.nominal, config=args.config
    )
    args.output.mkdir(exist_ok=True, parents=True)
    for stage in args.stages:
        data, metadata = sampler.sample(stage, 0, difficulty=args.difficulty)
        result = validator.validate(stage, data, metadata)
        state, controls = result.pop("settled_state"), result.pop("settled_controls")
        np.savez_compressed(
            args.output / (stage + ".npz"),
            input_qpos=data.qpos,
            input_controls=data.ctrl,
            settled_state=state,
            settled_controls=controls,
            state_spec=result["state_spec"],
        )
        result["sampler_metadata"] = metadata
        result["settled_state_sha256"] = hashlib.sha256(state.tobytes()).hexdigest()
        result["settled_controls_sha256"] = hashlib.sha256(
            controls.tobytes()
        ).hexdigest()
        (args.output / (stage + ".json")).write_text(
            json.dumps(result, indent=2, default=lambda x: x.tolist())
        )
        print(
            stage,
            result["physical_admission_pass"],
            result["rejection_reasons"],
            flush=True,
        )


if __name__ == "__main__":
    main()
