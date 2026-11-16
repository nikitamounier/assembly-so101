"""Model equivalence plus single-process runtime/cache smoke before parallel work."""

import argparse
import hashlib
import json
import os
import subprocess
import time
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    root = p.parse_args().root.resolve()
    project = root / "robotics-assembly-3.0"
    command = [
        sys.executable,
        str(root / "launch/parity-fixed.py"),
        "verify",
        "--profile",
        str(project / "outputs/reset-bank-1800-20261107-v4/physical/profile.json"),
        "--reference",
        str(
            project / "operational/reset-bank-chimera-20261107/source-model-arrays.npz"
        ),
        "--output",
        str(root / "native-array-parity.json"),
    ]
    subprocess.run(command, check=True)
    sys.path.insert(0, str(project))
    os.chdir(project)
    from assembly_policy.so101.wowrobo_bank import write_json
    from assembly_policy.so101.wowrobo_equilibrium_admission import (
        MinimumResetValidator,
    )
    from assembly_policy.so101.wowrobo_physical_resets import WowRoboPhysicalValidator
    from assembly_policy.so101.wowrobo_randomization import WowRoboRandomizer

    config_path = project / "outputs/reset-bank-1800-20261107-v4/config.json"
    config = json.loads(config_path.read_text())

    def path(key):
        value = Path(config[key])
        return value if value.is_absolute() else project / value

    # Warm the shared native-source authoring caches once. Parallel readers must
    # never race partial npz creation on their first initialization.
    started = time.monotonic()
    print("START sampler initialization", flush=True)
    sampler = WowRoboRandomizer(config=config_path)
    print("PASS sampler initialization", flush=True)
    import ctypes
    import gc

    gc.collect()
    ctypes.CDLL(None).malloc_trim(0)
    validator = WowRoboPhysicalValidator(
        nominal_bank=path("nominal_bank"),
        output_dir=project / "outputs/reset-bank-1800-20261107-v4/physical",
        grid_manifest=path("grid_manifest"),
        pin_mesh=path("pin_mesh"),
        pin_registration_path=path("pin_registration_path"),
        completed_anchor_bank=path("completed_anchor_bank"),
        mating_profile=config["physical_mating_profile"],
        pin_pose_registered=True,
        passive_motor_stops=True,
        exclude_internal_servo_contacts=True,
        source_mesh_table_contacts=True,
        center_motor_housings=True,
        helper_grid_profile=path("helper_grid_profile"),
    )
    ray_backends = sorted({type(mesh.ray).__module__ for mesh in validator.audit.meshes.values()})
    assert ray_backends == ["trimesh.ray.ray_pyembree"], ray_backends
    print("PASS physical validator initialization", ray_backends, flush=True)
    gc.collect()
    ctypes.CDLL(None).malloc_trim(0)
    admission = MinimumResetValidator(validator, settle_seconds=0.08, hold_seconds=0.2)
    stage = "21_upper_far_side"
    data, meta = sampler.sample(stage, 0, difficulty=0)
    print("START native retention admission", flush=True)
    import faulthandler
    faulthandler.dump_traceback_later(30, repeat=True)
    result = admission.validate(stage, data, meta)
    faulthandler.cancel_dump_traceback_later()
    print(
        "PASS native retention admission result",
        result["physical_admission_pass"],
        flush=True,
    )
    report = {
        "elapsed_seconds": time.monotonic() - started,
        "ray_backends": ray_backends,
        "stage": stage,
        "physical_admission_pass": result["physical_admission_pass"],
        "reasons": result["rejection_reasons"],
        "model": result["validation_model_signature"],
        "native_checks": result["minimum_validity"],
        "cache_patch_sha256": hashlib.sha256(
            (project / "assembly_policy/so101/wowrobo_supported_poses.py").read_bytes()
        ).hexdigest(),
    }
    write_json(root / "runtime-smoke.json", report)
    if not result["physical_admission_pass"]:
        raise RuntimeError(
            "Actual native smoke failed; parallel bank generation stays blocked"
        )
    print(
        json.dumps({"runtime_smoke_pass": True, "stage": stage, "cache_warmed": True}),
        flush=True,
    )


if __name__ == "__main__":
    main()
