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


def validate_bank(bank, spec, train, evaluation):
    manifest = json.loads((bank / "manifest.json").read_text())
    import numpy as np

    if sha(bank / "scene.xml") != manifest["model_signature"]:
        raise ValueError("Sealed native model bytes differ from its manifest")
    if (
        not manifest["complete"]
        or len(spec["stage_ids"]) != 25
        or len(set(spec["stage_ids"])) != 25
        or set(manifest["task_names"]) != set(spec["stage_ids"])
        or spec["train_per_stage"] != train
        or spec["eval_per_stage"] != evaluation
    ):
        raise ValueError(
            f"Only a complete25x({train}+{evaluation}) bank can be exported"
        )
    total = 0
    seen = set()
    for stage in spec["stage_ids"]:
        for split, count in [("train", train), ("eval", evaluation)]:
            with np.load(bank / f"{stage}_{split}.npz") as saved:
                if (
                    len(saved["states"]) != count
                    or not np.isfinite(saved["states"]).all()
                ):
                    raise ValueError(f"Invalid sealed quota: {stage}/{split}")
                if (
                    len(saved["controls"]) != count
                    or not np.isfinite(saved["controls"]).all()
                ):
                    raise ValueError(f"Invalid native controls: {stage}/{split}")
                if saved["friction_categories"].tolist() != [
                    "finger_part",
                    "part_table",
                    "part_part",
                ]:
                    raise ValueError(f"Unexpected friction categories: {stage}/{split}")
                if (
                    saved["friction_multipliers"].shape != (count, 3)
                    or not np.isfinite(saved["friction_multipliers"]).all()
                    or np.any(saved["friction_multipliers"] <= 0)
                ):
                    raise ValueError(
                        f"Invalid native friction payload: {stage}/{split}"
                    )
                for state in saved["states"]:
                    identity = hashlib.sha256(np.round(state, 10).tobytes()).hexdigest()
                    if identity in seen:
                        raise ValueError("Duplicate native state across bank/splits")
                    seen.add(identity)
                total += count
    assert total == 25 * (train + evaluation)
    return manifest, total


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True, type=Path)
    p.add_argument("--project", required=True, type=Path)
    p.add_argument("--plan", required=True, type=Path)
    p.add_argument("--source-root", required=True, type=Path)
    p.add_argument("--train", type=int, default=640)
    p.add_argument("--eval", type=int, default=128)
    a = p.parse_args()
    root = a.root.resolve()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Final publication must run in its allocated job")
    project, plan, source_root = (
        a.project.resolve(),
        a.plan.resolve(),
        a.source_root.resolve(),
    )
    if min(a.train, a.eval) <= 0:
        raise ValueError("Both split quotas must be positive")
    target = 25 * (a.train + a.eval)
    spec = json.loads((plan / "plan.json").read_text())
    bank = plan / "bank"
    manifest, target = validate_bank(bank, spec, a.train, a.eval)
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
    physical_profile = plan / "physical/profile.json"
    if not physical_profile.exists():
        physical_profile = (
            source_root
            / "robotics-assembly-3.0/outputs/reset-bank-1800-20261107-v4/physical/profile.json"
        )
    profile = json.loads(physical_profile.read_text())
    if (
        profile["model_sha256"] != manifest["model_signature"]
        or sha(Path(profile["model_path"])) != manifest["model_signature"]
    ):
        raise ValueError("Newbankdoesnotmatchtheunchangednativephysicalprofile")
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
    index["display_name"] = f"WowRobo · {target:,} validated variations"
    index["progress"].update(
        status="complete",
        complete=True,
        preview_only=False,
        requested_component="All five assembly tasks",
        target_train=25 * a.train,
        target_eval=25 * a.eval,
        requested_train=25 * a.train,
        requested_eval=25 * a.eval,
    )
    for counts in index["progress"]["by_stage"].values():
        counts.update(target_train=a.train, target_eval=a.eval)
    write(staging / "review/stages/manifest.json", index)
    write(staging / "review/progress.json", index["progress"])
    shutil.copytree(bank, staging / "native-bank")
    provenance = staging / "provenance"
    provenance.mkdir()
    for source, name in [
        (source_root / "release.json", "release.json"),
        (source_root / "transport-manifest.json", "transport-manifest.json"),
        (source_root / "environment.json", "environment.json"),
        (source_root / "native-array-parity.json", "native-array-parity.json"),
        (source_root / "relocation-complete.json", "relocation-complete.json"),
        (plan / "plan.json", "plan.json"),
        (plan / "config.json", "config.json"),
        (Path(config["grasp_library"]), "grasps.json"),
        (
            plan / "support-recalibration.json"
            if (plan / "support-recalibration.json").exists()
            else source_root
            / "robotics-assembly-3.0/outputs/reset-bank-1800-20261107-v4/support-recalibration.json",
            "support-recalibration.json",
        ),
        (physical_profile, "physics-profile.json"),
    ]:
        shutil.copy2(source, provenance / name)
    for source in [
        source_root / "launch/preflight-amendment.json",
        source_root / "launch/wowrobo_supported_poses.json",
        source_root / "launch/parity-fixed.py",
        source_root / "launch/full_preflight.py",
        source_root / "component-pointer-amendment.json",
        source_root / "private-dependency-install.json",
        source_root / "mesh-portability-diagnostic.json",
        source_root / "runtime-smoke.json",
        source_root / "private-embreex-install.json",
        source_root / "launch/partition-proof.json",
        source_root / "launch/partition-assignment.json",
        source_root / "launch/partition_worker.py",
        source_root / "launch/farm_control.py",
        source_root / "launch/packed_worker.py",
        source_root / "launch/runtime-source-amendments.json",
    ]:
        if source.exists():
            shutil.copy2(source, provenance / source.name)
    new_launch = provenance / "new-run-launch"
    new_launch.mkdir()
    for folder in [root, root / "launch"]:
        for source in folder.iterdir() if folder.exists() else []:
            if source.is_file() and source.suffix in {".json", ".py", ".sbatch", ".sh"}:
                shutil.copy2(source, new_launch / (folder.name + "-" + source.name))
    transport = source_root / "transport-provenance"
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
        "accepted": target,
        "nominal_review_count": 25,
        "remote_root": str(root),
        "runtime_source_root": str(source_root),
        "runtime_project": str(project),
        "train_per_stage": a.train,
        "eval_per_stage": a.eval,
        "stage_count": 25,
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
        "accepted": target,
        "status": "complete",
        "train_per_stage": a.train,
        "eval_per_stage": a.eval,
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
    }
    write(receipt, result)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
