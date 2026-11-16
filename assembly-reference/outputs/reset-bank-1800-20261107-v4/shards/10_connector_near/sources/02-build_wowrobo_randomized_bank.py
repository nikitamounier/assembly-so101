"""Resumable rejection sampling, physical qualification and viewer publication."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from assembly_policy.so101.wowrobo_bank import (
    changed_sources,
    digest,
    export_qualified,
    export_review,
    sample_identity,
    seed_for_slot,
    write_json,
    write_npz,
)
from assembly_policy.so101.wowrobo_boundaries import NativeBoundaryAudit

ROOT = Path(__file__).resolve().parents[1]


def build(args):
    config = json.loads(args.config.read_text())
    args.assets = Path(
        args.assets
        or config.get("component_assets", "assets_local/wowrobo/components-v2")
    )
    args.nominal = Path(
        args.nominal
        or config.get("nominal_bank", "assets_local/wowrobo/photo-resets-v3")
    )
    if not args.assets.is_absolute():
        args.assets = ROOT / args.assets
    if not args.nominal.is_absolute():
        args.nominal = ROOT / args.nominal
    for name in (
        "grid_manifest",
        "pin_mesh",
        "pin_registration_path",
        "completed_anchor_bank",
    ):
        value = getattr(args, name, None) or config.get(name)
        if value:
            value = Path(value)
            if not value.is_absolute():
                value = ROOT / value
        setattr(args, name, value)
    args.mating_profile = getattr(args, "mating_profile", None) or config.get(
        "physical_mating_profile"
    )
    args.pin_pose_registered = bool(
        getattr(args, "pin_pose_registered", False)
        or config.get("pin_pose_registered", False)
    )
    asset_inputs = []
    asset_manifest_path = args.assets / "manifest.json"
    if asset_manifest_path.exists():
        asset_inputs.append(asset_manifest_path)
        asset_manifest = json.loads(asset_manifest_path.read_text())
        asset_inputs.extend(
            args.assets / record["mesh"]
            for record in asset_manifest.get("constituents", {}).values()
        )
    guarded_paths = [
        args.config,
        args.nominal / "scene.xml",
        Path(__file__).resolve(),
        ROOT / "assembly_policy/so101/wowrobo_randomization.py",
        ROOT / "assembly_policy/so101/wowrobo_bank.py",
    ]
    generator_dependencies = [
        ROOT / "assembly_policy/so101" / name
        for name in (
            "wowrobo_reset_framework.py",
            "wowrobo_photo_resets.py",
            "wowrobo_grasps.py",
            "wowrobo_boundaries.py",
            "wowrobo_supported_poses.py",
        )
    ]
    guarded_paths.extend(generator_dependencies)
    if config.get("grasp_library"):
        guarded_paths.append(ROOT / config["grasp_library"])
    guarded_paths += asset_inputs
    reference_inputs = [
        args.nominal / name
        for name in ("stage-receipts.json", "stages/manifest.json")
        if (args.nominal / name).exists()
    ]
    guarded_paths += reference_inputs
    if config.get("cad_revision"):
        guarded_paths.extend(
            ROOT / "assembly_policy/so101" / name
            for name in ("wowrobo_repaired_resets.py", "wowrobo_repaired_assets.py")
        )
    if config.get("frozen_cad_ready"):
        guarded_paths.append(ROOT / config["frozen_cad_ready"])
    if args.validation == "physical":
        guarded_paths += [
            ROOT / f"assembly_policy/so101/{name}.py"
            for name in [
                "wowrobo_physical_resets",
                "wowrobo_friction",
                "wowrobo_contact_backend",
                "wowrobo_pin_registration",
                "wowrobo_equilibrium",
                "wowrobo_equilibrium_admission",
                "wowrobo_housing_registration",
            ]
        ]
        guarded_paths.append(
            (args.physical_profile or args.output / "physical")
            / "goal-registration.json"
        )
        if args.pin_mesh:
            guarded_paths += [args.pin_mesh, args.pin_mesh.parent / "registration.json"]
        if args.pin_registration_path:
            guarded_paths.append(args.pin_registration_path)
        if args.grid_manifest:
            guarded_paths.append(args.grid_manifest)
        if args.completed_anchor_bank:
            guarded_paths.extend(sorted(args.completed_anchor_bank.glob("*.npz")))
        if args.mating_profile:
            guarded_paths.append(ROOT / "configs/wowrobo_mating_profile.json")
    source_guard = {
        path: digest(path.read_bytes()) if path.exists() else None
        for path in guarded_paths
    }
    from assembly_policy.so101.wowrobo_randomization import WowRoboRandomizer

    sampler = WowRoboRandomizer(args.assets, args.nominal, config=args.config)
    m = sampler.model
    config = json.loads(args.config.read_text())
    source_manifest = json.loads((args.nominal / "stages/manifest.json").read_text())
    stages = args.stages or source_manifest["task_names"]
    if set(stages) - set(source_manifest["task_names"]):
        raise ValueError("Unknown stage")
    quota = {
        "train": config["sampling"]["train_per_stage"],
        "eval": config["sampling"]["eval_per_stage"],
    }
    target = (
        quota
        if args.full
        else {"train": args.train_per_stage, "eval": args.eval_per_stage}
    )
    if any(x <= 0 for x in target.values()) or args.max_attempts <= 0:
        raise ValueError("Counts must be positive")
    if args.full and args.validation != "physical":
        raise ValueError(
            "--full requires physical validation; geometry-only output is a review pilot"
        )
    args.output.mkdir(parents=True, exist_ok=True)
    validator = None
    if args.validation == "physical":
        from assembly_policy.so101.wowrobo_physical_resets import (
            WowRoboPhysicalValidator,
        )

        validator = WowRoboPhysicalValidator(
            nominal_bank=args.nominal,
            output_dir=args.physical_profile or args.output / "physical",
            grid_manifest=args.grid_manifest,
            pin_mesh=args.pin_mesh,
            pin_registration_path=args.pin_registration_path,
            completed_anchor_bank=args.completed_anchor_bank,
            mating_profile=args.mating_profile,
            pin_pose_registered=args.pin_pose_registered,
            settle_seconds=args.settle_seconds,
            hold_seconds=args.hold_seconds,
            passive_motor_stops=bool(config.get("passive_motor_stops", False)),
            exclude_internal_servo_contacts=bool(
                config.get("exclude_internal_servo_contacts", False)
            ),
            source_mesh_table_contacts=bool(
                config.get("source_mesh_table_contacts", False)
            ),
            center_motor_housings=bool(config.get("center_motor_housings", False)),
            helper_grid_profile=(ROOT / config["helper_grid_profile"])
            if config.get("helper_grid_profile")
            else None,
        )
        if config.get("minimum_reset_required", False):
            from assembly_policy.so101.wowrobo_equilibrium_admission import (
                MinimumResetValidator,
            )

            validator = MinimumResetValidator(
                validator,
                settle_seconds=args.settle_seconds,
                hold_seconds=args.hold_seconds,
            )
        elif config.get("equilibrium_required", False):
            from assembly_policy.so101.wowrobo_equilibrium_admission import (
                EquilibriumRequiredValidator,
            )

            validator = EquilibriumRequiredValidator(
                validator,
                settle_seconds=args.settle_seconds,
                hold_seconds=args.hold_seconds,
                gravity_preload=bool(config.get("gravity_preload", False)),
            )
    display_scene = (
        Path(validator.profile.get("audit_model_path", args.nominal / "scene.xml"))
        if validator
        else None
    )
    if changed_sources(source_guard):
        raise ValueError("Inputs changed while sampler/physical model initialized")
    identity = hashlib.sha256()
    sources = [
        args.config,
        args.nominal / "scene.xml",
        Path(__file__).resolve(),
        ROOT / "assembly_policy/so101/wowrobo_randomization.py",
        ROOT / "assembly_policy/so101/wowrobo_bank.py",
    ]
    sources.extend(generator_dependencies)
    if config.get("grasp_library"):
        sources.append(ROOT / config["grasp_library"])
    sources.extend(asset_inputs)
    sources.extend(reference_inputs)
    if config.get("cad_revision"):
        sources.extend(
            ROOT / "assembly_policy/so101" / name
            for name in ("wowrobo_repaired_resets.py", "wowrobo_repaired_assets.py")
        )
    for path in (args.pin_registration_path, args.grid_manifest):
        if path and path.exists():
            sources.append(path)
    if args.completed_anchor_bank and args.completed_anchor_bank.exists():
        sources.extend(sorted(args.completed_anchor_bank.glob("*.npz")))
    if validator is not None:
        sources.extend(
            [
                ROOT / f"assembly_policy/so101/{name}.py"
                for name in [
                    "wowrobo_physical_resets",
                    "wowrobo_friction",
                    "wowrobo_contact_backend",
                    "wowrobo_equilibrium",
                    "wowrobo_equilibrium_admission",
                    "wowrobo_housing_registration",
                ]
            ]
        )
        sources.append(Path(validator.scene_path))
        sources.append(display_scene)
        identity.update(json.dumps(validator.profile, sort_keys=True).encode())
        identity.update(
            json.dumps(
                {
                    "settle_seconds": args.settle_seconds,
                    "hold_seconds": args.hold_seconds,
                },
                sort_keys=True,
            ).encode()
        )
    for path in sources:
        content = path.read_bytes()
        identity.update(content)
        source_guard.setdefault(path, digest(content))
    for stage in stages:
        identity.update((args.nominal / "stages" / f"{stage}_train.npz").read_bytes())
        path = args.nominal / "stages" / f"{stage}_train.npz"
        source_guard[path] = digest(path.read_bytes())
    identity.update(
        json.dumps(
            {
                "validation": args.validation,
                "stages": stages,
                "target": target,
                "full": args.full,
                "difficulty": args.difficulty,
            },
            sort_keys=True,
        ).encode()
    )
    identity = identity.hexdigest()
    run = args.output / "run.json"
    if run.exists() and json.loads(run.read_text())["identity"] != identity:
        raise ValueError(
            "Source/config/target identity changed; use a new output folder"
        )
    for index, path in enumerate(sources):
        snapshot = args.output / "sources" / f"{index:02d}-{path.name}"
        snapshot.parent.mkdir(exist_ok=True)
        snapshot.write_bytes(path.read_bytes())
    write_json(
        run,
        {
            "identity": identity,
            "validation": args.validation,
            "full": args.full,
            "target_per_stage": target,
            "stage_ids": stages,
            "status": "running",
            "nominal": str(args.nominal.resolve()),
        },
    )
    rows = []
    for path in sorted((args.output / "attempts").glob("*.json")):
        row = json.loads(path.read_text())
        if row.get("identity") != identity:
            raise ValueError("Mismatched attempt identity")
        with np.load(args.output / row["archive"]) as saved:
            if (
                digest(saved["states"][0].tobytes()) != row["state_sha256"]
                or digest(saved["controls"][0].tobytes()) != row["control_sha256"]
            ):
                raise ValueError("Attempt state/control hash mismatch")
        rows.append(row)
        if row["training_eligible"]:
            with np.load(args.output / row["qualified_archive"]) as saved:
                if (
                    digest(saved["states"][0].tobytes())
                    != row["qualified_state_sha256"]
                    or digest(saved["controls"][0].tobytes())
                    != row["qualified_control_sha256"]
                ):
                    raise ValueError("Qualified state/control hash mismatch")
        if row.get("physical_input_archive"):
            with np.load(args.output / row["physical_input_archive"]) as saved:
                if (
                    digest(saved["states"][0].tobytes())
                    != row["physical_input_state_sha256"]
                    or digest(saved["controls"][0].tobytes())
                    != row["physical_input_control_sha256"]
                ):
                    raise ValueError("Physical input state/control hash mismatch")
    by_key = {(r["stage"], r["split"], r["slot"], r["attempt"]): r for r in rows}
    seen = {r["sample_identity"] for r in rows}
    seen_inputs = {r.get("input_sample_identity", r["sample_identity"]) for r in rows}
    auditor = NativeBoundaryAudit(m) if validator is None else None
    spec = mujoco.mjtState.mjSTATE_FULLPHYSICS

    def publish(status="running"):
        changed = changed_sources(source_guard)
        if changed:
            write_json(
                run,
                dict(
                    json.loads(run.read_text()),
                    status="source_changed",
                    changed_source_paths=changed,
                ),
            )
            raise ValueError(
                "Source/config/input changed during run; output is not sealed, use a new folder"
            )
        by_stage = {}
        for stage in stages:
            selected = [r for r in rows if r["stage"] == stage]
            by_stage[stage] = {
                "target_train": quota["train"],
                "target_eval": quota["eval"],
                "accepted_train": sum(
                    r["training_eligible"] and r["split"] == "train" for r in selected
                ),
                "accepted_eval": sum(
                    r["training_eligible"] and r["split"] == "eval" for r in selected
                ),
                "rejected_count": sum(r["candidate_rejected"] for r in selected),
                "pending_count": sum(
                    not r["candidate_rejected"] and not r["training_eligible"]
                    for r in selected
                ),
                "generated_count": len(selected),
            }
        progress = {
            "status": status,
            "validation": args.validation,
            "identity": identity,
            "source_identity_verified": True,
            "by_stage": by_stage,
            "target_train": quota["train"] * len(stages),
            "target_eval": quota["eval"] * len(stages),
            "requested_train": target["train"] * len(stages),
            "requested_eval": target["eval"] * len(stages),
        }
        for key in [
            "accepted_train",
            "accepted_eval",
            "rejected_count",
            "pending_count",
            "generated_count",
        ]:
            progress[key] = sum(s[key] for s in by_stage.values())
        export_review(
            args.output,
            args.nominal,
            stages,
            rows,
            progress,
            display_scene=display_scene,
        )
        if validator is not None:
            export_qualified(args.output, validator, stages, rows, progress)
        write_json(
            run, dict(json.loads(run.read_text()), status=status, progress=progress)
        )
        return progress

    publish()
    for stage in stages:
        for split in ["train", "eval"]:
            for slot in range(target[split]):
                for attempt in range(args.max_attempts if args.full else 1):
                    key = (stage, split, slot, attempt)
                    if key in by_key:
                        if by_key[key]["training_eligible"] or not args.full:
                            break
                        continue
                    if args.max_total_attempts and len(rows) >= args.max_total_attempts:
                        publish("attempt_limit")
                        return 2
                    changed = changed_sources(source_guard)
                    if changed:
                        write_json(
                            run,
                            dict(
                                json.loads(run.read_text()),
                                status="source_changed",
                                changed_source_paths=changed,
                            ),
                        )
                        raise ValueError(
                            "Source/config/input changed during run; aborting before next sample"
                        )
                    seed = seed_for_slot(slot, attempt, split, quota[split])
                    data, meta = sampler.sample(
                        stage, seed, split=split, difficulty=args.difficulty
                    )
                    audit = (
                        auditor.audit(data)
                        if auditor
                        else {
                            "candidate_rejected": False,
                            "candidate_rejection_reasons": [],
                            "not_performed": True,
                            "reason": "Physical validator owns native start, interval and end boundary checks",
                        }
                    )
                    if config.get("cad_revision") and auditor is not None:
                        target_bounds = auditor._bounds(data, auditor.targets)
                        floor = min(
                            (
                                {
                                    "geom_name": auditor.info[g]["geom_name"],
                                    "minimum_z_m": float(target_bounds[g][0, 2]),
                                }
                                for g in auditor.targets
                            ),
                            key=lambda row: row["minimum_z_m"],
                        )
                        audit["target_table_minimum"] = floor
                        if floor["minimum_z_m"] < -0.0005:
                            audit["candidate_rejected"] = True
                            audit["candidate_rejection_reasons"].append("target_table")
                    reasons = []
                    if audit["candidate_rejected"] and validator is None:
                        reasons += [
                            "geometry:" + s
                            for s in audit["candidate_rejection_reasons"]
                        ]
                    if meta.get("candidate_rejected"):
                        reasons += list(
                            meta.get("rejection_reasons", ["sampler_constraint"])
                        )
                    if not meta.get("ik_position_pass", False):
                        reasons.append("ik_position")
                    state = np.empty(mujoco.mj_stateSize(m, spec))
                    mujoco.mj_getState(m, data, state, spec)
                    sample_hash = sample_identity(state, meta["friction_multipliers"])
                    input_sample_hash = sample_hash
                    if input_sample_hash in seen_inputs:
                        reasons.append("duplicate_reset")
                    physical = {
                        "training_eligible": False,
                        "rejection_reasons": [],
                        "metrics": {},
                    }
                    physical_status = (
                        "not_evaluated_geometry_rejected" if reasons else "pending"
                    )
                    if validator is not None and not reasons:
                        physical = validator.validate(stage, data, meta)
                        physical_status = (
                            "accepted" if physical["training_eligible"] else "rejected"
                        )
                        reasons += list(physical.get("rejection_reasons", []))
                    eligible = bool(physical.get("training_eligible")) and not reasons
                    name = f"{stage}_{split}_{slot:06d}_{attempt:04d}"
                    display_pose = "authored_candidate"
                    display_physical_state = physical.get("settled_state")
                    display_physical_controls = physical.get("settled_controls")
                    if (
                        display_physical_state is not None
                        and not np.isfinite(np.asarray(display_physical_state)).all()
                    ):
                        reasons.append("nonfinite_physical_state")
                        eligible = False
                        physical_status = "rejected"
                        write_npz(
                            args.output / "physical-failures" / (name + ".npz"),
                            states=[display_physical_state],
                        )
                        display_physical_state = physical.get("physical_input_state")
                        display_physical_controls = physical.get(
                            "physical_input_controls"
                        )
                        display_pose = "pre_settle_reference_nonfinite_physics"
                    if (
                        validator is not None
                        and display_physical_state is not None
                        and np.isfinite(np.asarray(display_physical_state)).all()
                    ):
                        physical_data = mujoco.MjData(validator.model)
                        mujoco.mj_setState(
                            validator.model,
                            physical_data,
                            np.asarray(display_physical_state),
                            physical["state_spec"],
                        )
                        data.qpos[:] = physical_data.qpos
                        data.qvel[:] = physical_data.qvel
                        if m.na:
                            data.act[:] = physical_data.act
                        data.ctrl[:] = display_physical_controls
                        mujoco.mj_getState(m, data, state, spec)
                        if display_pose == "authored_candidate":
                            display_pose = (
                                "post_settle_pose_in_physical_overlay_geometry"
                            )
                        sample_hash = sample_identity(
                            np.asarray(display_physical_state),
                            meta["friction_multipliers"],
                        )
                        if sample_hash in seen:
                            reasons.append("duplicate_settled_reset")
                            eligible = False
                            physical_status = "rejected"
                    archive = "attempts/" + name + ".npz"
                    receipt = "attempts/" + name + ".json"
                    write_npz(
                        args.output / archive,
                        states=[state],
                        controls=[data.ctrl.copy()],
                    )
                    serial_physical = {
                        k: v
                        for k, v in physical.items()
                        if k
                        not in (
                            "settled_state",
                            "settled_controls",
                            "physical_input_state",
                            "physical_input_controls",
                        )
                    }
                    row = {
                        "identity": identity,
                        "stage": stage,
                        "split": split,
                        "slot": slot,
                        "attempt": attempt,
                        "seed": seed,
                        "metadata": meta,
                        "audit": audit,
                        "physical": serial_physical,
                        "physical_status": physical_status,
                        "archive": archive,
                        "receipt": receipt,
                        "sample_identity": sample_hash,
                        "input_sample_identity": input_sample_hash,
                        "state_sha256": digest(state.tobytes()),
                        "control_sha256": digest(data.ctrl.tobytes()),
                        "training_eligible": eligible,
                        "candidate_rejected": bool(reasons),
                        "rejection_reasons": sorted(set(reasons)),
                        "qualification": "Physically qualified reset"
                        if eligible
                        else "Not admitted for training",
                        "display_pose": display_pose,
                    }
                    if physical.get("physical_input_state") is not None:
                        row["physical_input_archive"] = (
                            "physical-inputs/" + name + ".npz"
                        )
                        row["physical_input_state_sha256"] = digest(
                            np.asarray(physical["physical_input_state"]).tobytes()
                        )
                        row["physical_input_control_sha256"] = digest(
                            np.asarray(physical["physical_input_controls"]).tobytes()
                        )
                        write_npz(
                            args.output / row["physical_input_archive"],
                            states=[physical["physical_input_state"]],
                            controls=[physical["physical_input_controls"]],
                        )
                    if eligible:
                        settled = physical.get("settled_state")
                        if settled is None:
                            raise ValueError(
                                "Physical acceptance requires settled_state"
                            )
                        if physical.get("validation_model_signature") != digest(
                            Path(validator.scene_path).read_bytes()
                        ):
                            raise ValueError(
                                "Physical receipt model signature mismatch"
                            )
                        row["qualified_archive"] = "qualified/shards/" + name + ".npz"
                        row["qualified_state_sha256"] = digest(
                            np.asarray(settled).tobytes()
                        )
                        row["qualified_control_sha256"] = digest(
                            np.asarray(physical["settled_controls"]).tobytes()
                        )
                        write_npz(
                            args.output / row["qualified_archive"],
                            states=[settled],
                            controls=[physical["settled_controls"]],
                        )
                    write_json(args.output / receipt, row)
                    rows.append(row)
                    by_key[key] = row
                    seen.add(sample_hash)
                    seen_inputs.add(input_sample_hash)
                    print(
                        name,
                        "ACCEPT" if eligible else ("REJECT" if reasons else "PENDING"),
                        flush=True,
                    )
                    if len(rows) % 10 == 0:
                        publish()
                    if eligible or not args.full:
                        break
                    if args.max_total_attempts and len(rows) >= args.max_total_attempts:
                        publish("attempt_limit")
                        return 2
        publish()
    qualified = all(
        sum(
            r["training_eligible"] and r["stage"] == stage and r["split"] == split
            for r in rows
        )
        == target[split]
        for stage in stages
        for split in ["train", "eval"]
    )
    progress = publish(
        "complete"
        if qualified and args.full
        else ("pilot_complete" if not args.full else "incomplete_rejections")
    )
    write_json(
        run,
        dict(json.loads(run.read_text()), status=progress["status"], progress=progress),
    )
    return 0 if not args.full or qualified else 2


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--nominal",
        type=Path,
        help="Reference bank; defaults to the selected configuration",
    )
    p.add_argument(
        "--assets", type=Path, help="CAD assets; defaults to the selected configuration"
    )
    p.add_argument(
        "--config", type=Path, default=ROOT / "configs/wowrobo_randomization.json"
    )
    p.add_argument(
        "--output",
        type=Path,
        default=ROOT / "assets_local/wowrobo/randomization-bank-pilot-v1",
    )
    p.add_argument("--full", action="store_true")
    p.add_argument("--validation", choices=["geometry", "physical"], default="physical")
    p.add_argument("--stages", nargs="*")
    p.add_argument("--train-per-stage", type=int, default=2)
    p.add_argument("--eval-per-stage", type=int, default=1)
    p.add_argument(
        "--per-stage", type=int, help="Pilot shorthand for equal TRAIN/EVAL draw counts"
    )
    p.add_argument("--max-attempts", type=int, default=20)
    p.add_argument("--max-total-attempts", type=int, default=0)
    p.add_argument("--difficulty", type=float, default=None)
    p.add_argument("--grid-manifest", type=Path)
    p.add_argument(
        "--physical-profile",
        type=Path,
        help="Reuse an immutable prepared contact profile",
    )
    p.add_argument("--pin-mesh", type=Path)
    p.add_argument("--pin-registration-path", type=Path)
    p.add_argument("--completed-anchor-bank", type=Path)
    p.add_argument("--mating-profile", choices=["legacy_mapped_all"])
    p.add_argument("--pin-pose-registered", action="store_true")
    p.add_argument("--settle-seconds", type=float, default=0.15)
    p.add_argument("--hold-seconds", type=float, default=0.15)
    args = p.parse_args()
    if args.per_stage is not None:
        args.train_per_stage = args.eval_per_stage = args.per_stage
    return build(args)


if __name__ == "__main__":
    raise SystemExit(main())
