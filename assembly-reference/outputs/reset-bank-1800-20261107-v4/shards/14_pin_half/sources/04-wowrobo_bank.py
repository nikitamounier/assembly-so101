"""Resumable WowRobo variation quotas and review export; no inferred admission."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def digest(data):
    return hashlib.sha256(data).hexdigest()


def changed_sources(expected):
    """Detect changes including creation/deletion of optional registration inputs."""
    return [
        str(path)
        for path, fingerprint in expected.items()
        if (digest(Path(path).read_bytes()) if Path(path).exists() else None)
        != fingerprint
    ]


def finite_json(value):
    if isinstance(value, dict):
        return {str(k): finite_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return finite_json(value.tolist())
    if isinstance(value, np.generic):
        return finite_json(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(finite_json(value), indent=2, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def write_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def seed_for_slot(slot, attempt, split, quota):
    """Retries stay in the same difficulty stratum; split hash is in sampler."""
    if split not in ("train", "eval") or min(slot, attempt) < 0:
        raise ValueError("Invalid split, slot or retry")
    return slot + attempt * quota


def sample_identity(state, friction):
    # Different friction makes a different physical reset even for equal qpos.
    return digest(
        np.asarray(state).tobytes() + json.dumps(friction, sort_keys=True).encode()
    )


def candidate_summary(row):
    meta = row["metadata"]
    return {
        "kind": "variation",
        "cad_revision": meta.get("cad_revision"),
        "support_grasp_by_side": meta.get("support_grasp_by_side", {}),
        "seed": row["seed"],
        "slot": row["slot"],
        "difficulty": meta.get("difficulty"),
        "friction_multipliers": meta.get("friction_multipliers"),
        "connector_swivel_degrees": meta.get("connector_swivel_degrees", 0),
        "training_eligible": row["training_eligible"],
        "candidate_rejected": row["candidate_rejected"],
        "rejection_reasons": row["rejection_reasons"],
        "qualification": row["qualification"],
        "physical_status": row["physical_status"],
        "validation_model_signature": row.get("physical", {}).get(
            "validation_model_signature"
        ),
        "physics_profile": row.get("physical", {}).get("physics_profile"),
        "physical_metrics": row.get("physical", {}).get("metrics", {}),
        "minimum_validity": row.get("physical", {}).get("minimum_validity"),
        "pin_pose_overlay": row.get("physical", {}).get("pin_pose_overlay"),
        "display_pose": row.get("display_pose", "authored_candidate"),
        "state_sha256": row["state_sha256"],
        "control_sha256": row["control_sha256"],
        "receipt": row["receipt"],
        "noise": {
            k: meta[k]
            for k in (
                "transforms",
                "tangent_yaw_noise_degrees",
                "insertion_noise",
                "transport_noise",
                "sag_degrees",
                "resting_contact",
                "pickup_requires_hold",
                "pickup_approach_gap_m",
                "projected_noise_constraints",
            )
            if k in meta
        },
    }


def export_review(output, nominal, stages, rows, progress, display_scene=None):
    """Publish authored candidates; qualified physics states stay separate."""
    output, nominal = Path(output), Path(nominal)
    source_manifest = json.loads((nominal / "stages/manifest.json").read_text())
    variation, count = {}, 0
    for stage in stages:
        variation[stage] = {}
        for split in ("train", "eval"):
            states, controls, seeds, metadata = [], [], [], []
            if split == "train":
                with np.load(nominal / "stages" / f"{stage}_train.npz") as saved:
                    states.append(saved["states"][0].copy())
                    controls.append(saved["controls"][0].copy())
                seeds.append(-1)
                reference = source_manifest.get("nominal_validation_by_task", {}).get(
                    stage, {}
                )
                metadata.append(
                    {
                        "kind": "nominal",
                        "cad_revision": source_manifest.get("cad_revision"),
                        "support_grasp_by_side": source_manifest.get(
                            "stage_by_task", {}
                        )
                        .get(stage, {})
                        .get("support_grasp_by_side", {}),
                        "training_eligible": False,
                        "candidate_rejected": bool(
                            reference.get("candidate_rejected", False)
                        ),
                        "rejection_reasons": reference.get("rejection_reasons", []),
                        "reference_validation": reference,
                        "qualification": "Nominal reference pose in physical overlay geometry; not physical admission"
                        if display_scene
                        else "Approved nominal geometry; not physical admission",
                        "physical_status": "reference",
                    }
                )
            for row in rows:
                if row["stage"] != stage or row["split"] != split:
                    continue
                with np.load(output / row["archive"]) as saved:
                    states.append(saved["states"][0].copy())
                    controls.append(saved["controls"][0].copy())
                seeds.append(row["seed"])
                metadata.append(candidate_summary(row))
            if states:
                write_npz(
                    output / "stages" / f"{stage}_{split}.npz",
                    states=np.stack(states),
                    controls=np.stack(controls),
                    seed=np.array(seeds),
                    family=np.zeros(len(states), dtype=int),
                )
                count += len(states)
            variation[stage][split] = metadata
    xml = Path(display_scene or nominal / "scene.xml").read_bytes()
    (output / "scene.xml").write_bytes(xml)
    manifest = dict(
        source_manifest,
        reset_distribution="photo_variations",
        display_name=source_manifest.get(
            "randomized_display_name",
            "WowRobo · randomized variations · admission status",
        ),
        variation_by_task=variation,
        progress=progress,
        default_viewer_collection=True,
    )
    manifest["task_names"] = stages
    manifest["model_signature"] = digest(xml)
    write_json(output / "stages/manifest.json", manifest)
    write_json(
        output / "manifest.json",
        {
            "profile": "wowrobo-randomized-review-v1",
            "model_signature": digest(xml),
            "bank": "stages",
            "count": count,
            "qualification": "Review collection: per-row physical status; not a training bank",
        },
    )
    write_json(output / "progress.json", progress)


def export_qualified(output, validator, stages, rows, progress):
    """Publish only accepted physical snapshots with their physical model identity."""
    output = Path(output)
    bank = output / "qualified"
    bank.mkdir(exist_ok=True)
    xml = Path(validator.scene_path).read_bytes()
    (bank / "scene.xml").write_bytes(xml)
    physical_spec = None
    for stage in stages:
        for split in ("train", "eval"):
            chosen = sorted(
                (
                    r
                    for r in rows
                    if r["training_eligible"]
                    and r["stage"] == stage
                    and r["split"] == split
                ),
                key=lambda r: r["slot"],
            )
            if not chosen:
                continue
            states, controls = [], []
            for row in chosen:
                physical_spec = row["physical"]["state_spec"]
                with np.load(output / row["qualified_archive"]) as saved:
                    states.append(saved["states"][0])
                    controls.append(saved["controls"][0])
            write_npz(
                bank / f"{stage}_{split}.npz",
                states=np.stack(states),
                controls=np.stack(controls),
                seed=np.array([r["seed"] for r in chosen]),
                family=np.zeros(len(chosen), dtype=int),
                friction_multipliers=np.array(
                    [
                        [
                            row["metadata"]["friction_multipliers"].get(category, 1.0)
                            for category in ("finger_part", "part_table", "part_part")
                        ]
                        for row in chosen
                    ]
                ),
                friction_categories=np.array(
                    ["finger_part", "part_table", "part_part"]
                ),
            )
    write_json(
        bank / "manifest.json",
        {
            "model_signature": digest(xml),
            "state_spec": physical_spec,
            "task_names": stages,
            "family_names": ["Physically admitted variation"],
            "reset_distribution": "photo_variations_qualified",
            "physics_profile": getattr(validator, "profile", {}),
            "progress": progress,
            "complete": progress["accepted_train"] == progress["target_train"]
            and progress["accepted_eval"] == progress["target_eval"],
        },
    )
