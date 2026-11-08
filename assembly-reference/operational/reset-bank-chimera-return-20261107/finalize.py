"""Completion-only export of a sealed Linux-qualified bank; run in its Slurm job."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import tarfile
import time
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    tmp.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True, type=Path)
    a = p.parse_args()
    root = a.root.resolve()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Final publication must run in its allocated job")
    project = root / "robotics-assembly-3.0"
    plan = project / "outputs/reset-bank-1800-20261107-v4"
    spec = json.loads((plan / "plan.json").read_text())
    bank = plan / "bank"
    manifest = json.loads((bank / "manifest.json").read_text())
    import numpy as np

    if (
        not manifest["complete"]
        or len(spec["stage_ids"]) != 25
        or spec["train_per_stage"] != 60
        or spec["eval_per_stage"] != 12
    ):
        raise ValueError("Only a complete25x60/12 bank can be exported")
    total = 0
    for stage in spec["stage_ids"]:
        for split, count in [("train", 60), ("eval", 12)]:
            with np.load(bank / f"{stage}_{split}.npz") as saved:
                if (
                    len(saved["states"]) != count
                    or not np.isfinite(saved["states"]).all()
                ):
                    raise ValueError(f"Invalid sealed quota: {stage}/{split}")
                total += count
    assert total == 1800
    exports = root / "exports"
    exports.mkdir(exist_ok=True)
    archive = exports / "reset-bank-result.tar.gz"
    receipt = exports / "return-receipt.json"
    if receipt.exists():
        prior = json.loads(receipt.read_text())
        if archive.exists() and sha(archive) == prior["archive_sha256"]:
            print(json.dumps(prior))
            return
        raise ValueError(
            "Prior return export changed; preserve it and choose a new version"
        )
    staging = exports / "result"
    if staging.exists():
        staging.rename(exports / f"incomplete-publication-{time.time_ns()}")
    staging.mkdir()
    config = json.loads(Path(spec["config"]).read_text())
    publisher_spec = importlib.util.spec_from_file_location(
        "variation_publisher", project / "scripts/publish_wowrobo_variation_preview.py"
    )
    publisher = importlib.util.module_from_spec(publisher_spec)
    publisher_spec.loader.exec_module(publisher)
    publisher.publish(
        Path(config["nominal_bank"]),
        plan / "shards",
        staging / "review",
        spec["stage_ids"],
    )
    index = json.loads((staging / "review/stages/manifest.json").read_text())
    index["display_name"] = "WowRobo · 1,800 validated variations"
    index["progress"].update(
        status="complete",
        complete=True,
        preview_only=False,
        requested_component="All five assembly tasks",
        target_train=1500,
        target_eval=300,
        requested_train=1500,
        requested_eval=300,
    )
    for counts in index["progress"]["by_stage"].values():
        counts.update(target_train=60, target_eval=12)
    write(staging / "review/stages/manifest.json", index)
    write(staging / "review/progress.json", index["progress"])
    shutil.copytree(bank, staging / "native-bank")
    provenance = staging / "provenance"
    provenance.mkdir()
    for source, name in [
        (root / "release.json", "release.json"),
        (root / "transport-manifest.json", "transport-manifest.json"),
        (root / "environment.json", "environment.json"),
        (root / "native-array-parity.json", "native-array-parity.json"),
        (root / "relocation-complete.json", "relocation-complete.json"),
        (plan / "plan.json", "plan.json"),
        (plan / "config.json", "config.json"),
        (plan / "grasps.json", "grasps.json"),
        (plan / "support-recalibration.json", "support-recalibration.json"),
        (plan / "physical/profile.json", "physics-profile.json"),
    ]:
        shutil.copy2(source, provenance / name)
    for source in [
        root / "launch/preflight-amendment.json",
        root / "launch/wowrobo_supported_poses.json",
        root / "launch/parity-fixed.py",
        root / "launch/full_preflight.py",
        root / "component-pointer-amendment.json",
        root / "private-dependency-install.json",
        root / "mesh-portability-diagnostic.json",
        root / "runtime-smoke.json",
        root / "private-embreex-install.json",
        root / "launch/partition-proof.json",
        root / "launch/partition-assignment.json",
        root / "launch/partition_worker.py",
        root / "launch/farm_control.py",
        root / "launch/packed_worker.py",
        root / "launch/runtime-source-amendments.json",
    ]:
        if source.exists():
            shutil.copy2(source, provenance / source.name)
    transport = root / "transport-provenance"
    if transport.exists():
        shutil.copytree(transport, provenance / "original-witnesses")
    files = {
        str(path.relative_to(staging)): sha(path)
        for path in staging.rglob("*")
        if path.is_file()
    }
    result = {
        "schema_version": 1,
        "qualified_platform": "Linux",
        "qualification": "Native overlap and held-part retention; motion permitted",
        "accepted": 1800,
        "nominal_review_count": 25,
        "remote_root": str(root),
        "physical_model_signature": manifest["model_signature"],
        "files": files,
    }
    write(staging / "return-manifest.json", result)
    temp = exports / "reset-bank-result.partial.tar.gz"
    with tarfile.open(temp, "w:gz", compresslevel=1) as tar:
        for path in sorted(staging.rglob("*")):
            if path.is_file():
                tar.add(path, arcname=str(path.relative_to(staging)), recursive=False)
    temp.replace(archive)
    result = {
        "archive": str(archive),
        "archive_sha256": sha(archive),
        "bytes": archive.stat().st_size,
        "accepted": 1800,
        "status": "complete",
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
    }
    write(receipt, result)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
