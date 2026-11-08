"""Export an immutable accepted-only preview without pausing active generation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import shutil
import sys
import tarfile
import time
from pathlib import Path

SOURCE_ROOT = (
    "/large_storage/goodarzilab/nikitamounier/assembly/reset-bank-1800-20261107-v4"
)
MODEL_SHA = "1b071d9b92dc5e3e3e9ba33234a25f24bfc3b3049081a52ac673b1df423b952d"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def within(folder, relative):
    rel = Path(relative)
    if rel.is_absolute() or ".." in rel.parts:
        raise ValueError(f"Unsafe artifact path:{relative}")
    path = (folder / rel).resolve()
    if not path.is_relative_to(folder.resolve()) or not path.is_file():
        raise ValueError(f"Missing/outside artifact:{path}")
    return path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument(
        "--output", type=Path, required=True, help="Unused versioned snapshot directory"
    )
    p.add_argument("--source-root", type=Path, default=Path(SOURCE_ROOT))
    p.add_argument("--project", type=Path)
    a = p.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Snapshot computation requires its allocated step")
    root, output, source_root = (
        a.root.resolve(),
        a.output.resolve(),
        a.source_root.resolve(),
    )
    if output.exists():
        raise ValueError("Snapshots are immutable; choose a new output directory")
    settings = json.loads((root / "settings.json").read_text())
    project = (a.project or Path(settings["project"])).resolve()
    plan = Path(settings["plan"])
    spec = json.loads((plan / "plan.json").read_text())
    config = json.loads((plan / "config.json").read_text())
    stages, namespace = spec["stage_ids"], settings["rng_namespace"]
    if (
        len(stages) != 25
        or len(set(stages)) != 25
        or settings["model_sha256"] != MODEL_SHA
    ):
        raise ValueError("Unexpected stage/model identity")
    if (settings["train"], settings["eval"], settings["parts"], settings["target"]) != (
        640,
        128,
        64,
        19200,
    ):
        raise ValueError("Unexpected generation quotas/partition contract")
    if namespace != "wowrobo19200-v1":
        raise ValueError("Unexpected generation RNG namespace")
    sys.path.insert(0, str(project))
    import mujoco
    import numpy as np

    from assembly_policy.so101.wowrobo_bank import export_review, write_npz
    from assembly_policy.so101.wowrobo_contact_backend import (
        load_backend,
        set_surface_mode,
    )

    profile_path = Path(settings["physical"]) / "profile.json"
    profile = json.loads(profile_path.read_text())
    if (
        profile["model_sha256"] != MODEL_SHA
        or sha(Path(profile["model_path"])) != MODEL_SHA
    ):
        raise ValueError("Native physics changed")
    source = Path(config["nominal_bank"])
    source_manifest = json.loads((source / "stages/manifest.json").read_text())
    if set(source_manifest["task_names"]) != set(stages):
        raise ValueError("Reference and generated stage identities differ")
    started = time.time()
    # Take an explicit path list, then consume only atomically published accepted
    # receipts. Future completions remain for a later snapshot, never race writes.
    paths = sorted((plan / "partitions").glob("*/*/attempts/*.json"))
    selected, identities, duplicate_slots = {}, {}, []
    immutable_profiles = {}
    for path in paths:
        row = json.loads(path.read_text())
        if not row.get("training_eligible"):
            continue
        stage, part = path.parents[2].name, int(path.parents[1].name)
        split, slot = row["split"], int(row["slot"])
        if stage not in stages or row["stage"] != stage or not 0 <= part < 64:
            raise ValueError(f"Foreign stage/partition:{path}")
        limit = {"train": 640, "eval": 128}.get(split)
        if limit is None or not 0 <= slot < limit or slot % 64 != part:
            raise ValueError(f"Invalid split/slot ownership:{path}")
        if row["metadata"].get("rng_namespace") != namespace:
            raise ValueError(f"Wrong RNG namespace:{path}")
        physical = row["physical"]
        # Frozen physics profiles are repeated verbatim in every receipt. Share
        # identical decoded maps, never mutate them or their provenance values.
        for field in ("profile", "physics_profile"):
            value = physical.get(field)
            if isinstance(value, dict):
                key = json.dumps(value, sort_keys=True)
                if key not in immutable_profiles:
                    immutable_profiles[key] = value
                physical[field] = immutable_profiles[key]
        if (
            not physical.get("physical_admission_pass")
            or physical.get("validation_model_signature") != MODEL_SHA
            or row.get("candidate_rejected")
        ):
            raise ValueError(f"Missing native admission/model proof:{path}")
        if stage in identities and identities[stage] != row["identity"]:
            raise ValueError(f"Mixed generation identity within stage:{stage}")
        identities[stage] = row["identity"]
        key = (stage, split, slot)
        if key in selected:
            duplicate_slots.append(
                {"stage": stage, "split": split, "slot": slot, "skipped": str(path)}
            )
            continue
        selected[key] = (path, row)
    if not selected:
        raise ValueError("No native-admitted candidates currently available")
    load_backend(profile["grid_manifest"])
    set_surface_mode(1)
    native = mujoco.MjModel.from_xml_string(Path(profile["model_path"]).read_text())
    visual = mujoco.MjModel.from_xml_string((source / "scene.xml").read_text())
    if (native.nq, native.nv, native.nu, native.na) != (
        visual.nq,
        visual.nv,
        visual.nu,
        visual.na,
    ):
        raise ValueError("Native/display state layouts disagree")
    output.mkdir(parents=True)
    review = output / "review"
    shutil.copytree(source, review)
    native_bank = output / "native-bank"
    native_bank.mkdir()
    (native_bank / "scene.xml").write_bytes(Path(profile["model_path"]).read_bytes())
    rows, native_values = [], {}
    d = mujoco.MjData(native)
    view = mujoco.MjData(visual)
    print(
        json.dumps(
            {
                "selected": len(selected),
                "rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            }
        ),
        flush=True,
    )
    by_stage = {
        stage: {
            "accepted_train": 0,
            "accepted_eval": 0,
            "target_train": 640,
            "target_eval": 128,
            "generated_count": 0,
            "rejected_count": 0,
            "pending_count": 0,
        }
        for stage in stages
    }
    with tarfile.open(output / "accepted-source-artifacts.tar", "w") as artifacts:
        for (stage, split, slot), (path, original) in sorted(selected.items()):
            folder = path.parents[1]
            qualified_path = within(folder, original["qualified_archive"])
            display_path = within(folder, original["archive"])
            with np.load(qualified_path) as saved:
                state, controls = saved["states"][0].copy(), saved["controls"][0].copy()
            with np.load(display_path) as saved:
                old_display, old_controls = saved["states"][0], saved["controls"][0]
            for array in (state, controls, old_display, old_controls):
                if not np.isfinite(array).all():
                    raise ValueError(f"Nonfinite admitted artifact:{path}")
            if (
                hashlib.sha256(state.tobytes()).hexdigest()
                != original["qualified_state_sha256"]
                or hashlib.sha256(controls.tobytes()).hexdigest()
                != original["qualified_control_sha256"]
                or hashlib.sha256(old_display.tobytes()).hexdigest()
                != original["state_sha256"]
                or hashlib.sha256(old_controls.tobytes()).hexdigest()
                != original["control_sha256"]
            ):
                raise ValueError(f"Admitted state/control hash mismatch:{path}")
            state_spec = int(original["physical"]["state_spec"])
            mujoco.mj_setState(native, d, state, state_spec)
            if not np.array_equal(d.ctrl, controls):
                raise ValueError(
                    "Native INTEGRATION snapshot and control archive disagree"
                )
            view.time = d.time
            view.qpos[:], view.qvel[:], view.ctrl[:] = d.qpos, d.qvel, controls
            if visual.na:
                view.act[:] = d.act
            display = np.empty(
                mujoco.mj_stateSize(visual, mujoco.mjtState.mjSTATE_FULLPHYSICS)
            )
            mujoco.mj_getState(
                visual, view, display, mujoco.mjtState.mjSTATE_FULLPHYSICS
            )
            # All positions/velocities in the historical review archive must already
            # correspond to this admitted native snapshot; timestamp can differ.
            if not np.array_equal(old_display[1:], display[1:]) or not np.array_equal(
                old_controls, controls
            ):
                raise ValueError(
                    f"Display archive is not the native-admitted pose:{path}"
                )
            name = f"{stage}_{split}_{slot:06d}"
            row = dict(original)
            row["source_receipt"] = str(path)
            # The archived display pose has just been checked against the qualified
            # native snapshot (all qpos/qvel/controls exact; only clock may differ).
            # Export directly from that immutable source archive, avoiding thousands
            # of tiny NFS writes while the generators continue.
            row["archive"] = str(display_path)
            row["receipt"] = (
                f"accepted-source-artifacts.tar#source-receipts/{name}.json"
            )
            row["display_pose"] = (
                "native_admitted_pose_verified_against_qualified_integration"
            )
            artifacts.add(path, arcname=f"source-receipts/{name}.json", recursive=False)
            artifacts.add(
                qualified_path, arcname=f"native-qualified/{name}.npz", recursive=False
            )
            friction = original["metadata"]["friction_multipliers"]
            multipliers = [
                float(friction[k]) for k in ("finger_part", "part_table", "part_part")
            ]
            if not np.isfinite(multipliers).all() or min(multipliers) <= 0:
                raise ValueError("Invalid admitted friction multipliers")
            native_values.setdefault((stage, split), []).append(
                (state, controls, original["seed"], slot, multipliers)
            )
            rows.append(row)
            if len(rows) % 1000 == 0:
                update = {
                    "verified": len(rows),
                    "selected": len(selected),
                    "rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                    / 1024,
                }
                write(output / "snapshot-progress.json", update)
                print(json.dumps(update), flush=True)
            by_stage[stage]["accepted_" + split] += 1
            by_stage[stage]["generated_count"] += 1
    accepted = len(rows)
    progress = {
        "status": "accepted_partial_snapshot",
        "complete": False,
        "preview_only": True,
        "accepted_train": sum(d["accepted_train"] for d in by_stage.values()),
        "accepted_eval": sum(d["accepted_eval"] for d in by_stage.values()),
        "generated_count": accepted,
        "accepted": accepted,
        "target_train": 16000,
        "target_eval": 3200,
        "requested_train": 16000,
        "requested_eval": 3200,
        "target": 19200,
        "rejected_count": 0,
        "pending_count": 0,
        "by_stage": by_stage,
        "snapshot_started_unix": started,
        "snapshot_completed_unix": time.time(),
        "counted_source": "Atomic native-admitted partition receipts only",
    }
    export_review(review, source, stages, rows, progress)
    index = json.loads((review / "stages/manifest.json").read_text())
    index["display_name"] = (
        f"WowRobo · {accepted:,} accepted variations · live-build snapshot"
    )
    write(review / "stages/manifest.json", index)
    for (stage, split), values in native_values.items():
        write_npz(
            native_bank / f"{stage}_{split}.npz",
            states=np.stack([v[0] for v in values]),
            controls=np.stack([v[1] for v in values]),
            seed=np.array([v[2] for v in values]),
            slot=np.array([v[3] for v in values]),
            family=np.zeros(len(values), dtype=int),
            friction_multipliers=np.array([v[4] for v in values]),
            friction_categories=np.array(["finger_part", "part_table", "part_part"]),
        )
    write(
        native_bank / "manifest.json",
        {
            "model_signature": MODEL_SHA,
            "state_spec": state_spec,
            "task_names": stages,
            "complete": False,
            "preview_only": True,
            "progress": progress,
        },
    )
    provenance = output / "provenance"
    provenance.mkdir()
    for source_path, name in [
        (source_root / "transport-manifest.json", "transport-manifest.json"),
        (root / "native-array-parity.json", "native-array-parity.json"),
        (profile_path, "physics-profile.json"),
        (root / "settings.json", "settings.json"),
        (plan / "plan.json", "plan.json"),
        (plan / "config.json", "config.json"),
    ]:
        shutil.copy2(source_path, provenance / name)
    files = {
        str(path.relative_to(output)): sha(path)
        for path in output.rglob("*")
        if path.is_file()
    }
    manifest = {
        "schema_version": 1,
        "preview_only": True,
        "complete": False,
        "qualified_platform": "Linux",
        "accepted": accepted,
        "nominal_review_count": 25,
        "target": 19200,
        "train_per_stage": 640,
        "eval_per_stage": 128,
        "stage_count": 25,
        "rng_namespace": namespace,
        "physical_model_signature": MODEL_SHA,
        "remote_root": str(root),
        "runtime_source_root": str(source_root),
        "files": files,
        "by_stage": by_stage,
        "duplicate_slots_skipped": duplicate_slots,
        "snapshot_started_unix": started,
        "snapshot_completed_unix": time.time(),
    }
    write(output / "return-manifest.json", manifest)
    archive = output.parent / (output.name + ".tar.gz")
    temporary = archive.with_suffix(".partial.tar.gz")
    with tarfile.open(temporary, "w:gz", compresslevel=1) as tar:
        for path in sorted(output.rglob("*")):
            if path.is_file():
                tar.add(path, arcname=str(path.relative_to(output)), recursive=False)
    temporary.replace(archive)
    receipt = {
        "status": "accepted_partial_snapshot",
        "complete": False,
        "preview_only": True,
        "accepted": accepted,
        "target": 19200,
        "archive": str(archive),
        "archive_sha256": sha(archive),
        "bytes": archive.stat().st_size,
        "snapshot_directory": str(output),
        "job_id": os.environ["SLURM_JOB_ID"],
    }
    write(output.parent / (output.name + ".receipt.json"), receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
