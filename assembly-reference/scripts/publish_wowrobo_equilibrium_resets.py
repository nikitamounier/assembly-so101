"""Publish witnessed minimum-valid or equilibrium resets with native replay state."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import mujoco
import numpy as np

from assembly_policy.so101.wowrobo_bank import write_json, write_npz
from assembly_policy.so101.wowrobo_contact_backend import load_backend, set_surface_mode


def publish(source, results, output, *, allow_incomplete=False, minimum_only=False):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists() and not (
        allow_incomplete
        and json.loads((output / "manifest.json").read_text()).get("repair_draft")
    ):
        raise ValueError(
            "Reference collections are versioned; choose a new output path"
        )
    manifest = json.loads((source / "stages/manifest.json").read_text())
    selected = {}
    observations = {}
    for folder in map(Path, results):
        for stage in manifest["task_names"]:
            path = folder / (stage + ".json")
            if not path.exists():
                continue
            report = json.loads(path.read_text())
            observations[stage] = report
            if report.get(
                "minimum_validity_pass" if minimum_only else "equilibrium_pass"
            ):
                selected[stage] = (folder, report)
    missing = sorted(set(manifest["task_names"]) - set(selected))
    if missing and not allow_incomplete:
        raise ValueError(f"Cannot publish unfinished reset references: {missing}")
    identities = {
        report["physics_profile"]["model_sha256"] for _, report in selected.values()
    }
    if len(identities) != 1:
        raise ValueError("All references must pass the same physical model")
    profile = next(iter(selected.values()))[1]["physics_profile"]
    if (
        hashlib.sha256(Path(profile["model_path"]).read_bytes()).hexdigest()
        not in identities
    ):
        raise ValueError("Witnessed physical model has changed")
    load_backend(profile["grid_manifest"])
    set_surface_mode(1)
    physical = mujoco.MjModel.from_xml_path(profile["model_path"])
    display_xml = Path(
        profile.get("audit_model_path", source / "scene.xml")
    ).read_bytes()
    nominal = mujoco.MjModel.from_xml_string(display_xml.decode())
    if (nominal.nq, nominal.nv, nominal.nu) != (physical.nq, physical.nv, physical.nu):
        raise ValueError("Physical and display state layouts differ")
    if not output.exists():
        shutil.copytree(source, output)
    (output / "scene.xml").write_bytes(display_xml)
    manifest["model_signature"] = hashlib.sha256(display_xml).hexdigest()
    if profile.get("housing_registration"):
        from assembly_policy.so101.wowrobo_housing_registration import migrate_qpos

        before_model = mujoco.MjModel.from_xml_string(
            (source / "scene.xml").read_text()
        )
        for path in (source / "completed").glob("*.npz"):
            with np.load(path) as old:
                values = {name: old[name].copy() for name in old.files}
            before = mujoco.MjData(before_model)
            mujoco.mj_setState(
                before_model,
                before,
                values["states"][0],
                mujoco.mjtState(manifest["state_spec"]),
            )
            after = mujoco.MjData(nominal)
            after.qpos[:], _ = migrate_qpos(
                before_model,
                nominal,
                before.qpos,
                propagate_final_with_receiver=path.name.startswith("32_"),
            )
            after.qvel[:] = before.qvel
            migrated = np.empty_like(values["states"][0])
            mujoco.mj_getState(
                nominal, after, migrated, mujoco.mjtState(manifest["state_spec"])
            )
            values["states"][0] = migrated
            write_npz(output / "completed" / path.name, **values)
    raw = output / "physical-replays"
    raw.mkdir(exist_ok=True)
    physical_bank = output.with_name(output.name + "-physical")
    if minimum_only:
        (physical_bank / "stages").mkdir(parents=True, exist_ok=True)
        (physical_bank / "scene.xml").write_bytes(
            Path(profile["model_path"]).read_bytes()
        )
    receipts = json.loads((output / "stage-receipts.json").read_text())
    by_stage = {r["id"]: r for r in receipts}
    validation = {}
    for stage in missing:
        report = observations.get(stage)
        summary = (
            {
                k: v
                for k, v in report.items()
                if k not in ("physics_profile", "settling_trace", "boundary")
            }
            if report
            else {"equilibrium_pass": None, "rejection_reasons": ["Hold check pending"]}
        )
        validation[stage] = {
            "candidate_rejected": bool(report),
            "rejection_reasons": report.get("rejection_reasons", []) if report else [],
            "equilibrium": summary,
            "training_eligible": False,
        }
    for stage, (folder, report) in selected.items():
        with np.load(folder / (stage + ".npz")) as saved:
            full = saved["states"][0].copy()
            controls = saved["controls"][0].copy()
            end = saved["replay_end"].copy()
        assert hashlib.sha256(full.tobytes()).hexdigest() == report["state_sha256"]
        assert (
            hashlib.sha256(controls.tobytes()).hexdigest() == report["control_sha256"]
        )
        d = mujoco.MjData(physical)
        mujoco.mj_setState(physical, d, full, mujoco.mjtState(report["state_spec"]))
        assert np.array_equal(d.ctrl, controls)
        view = mujoco.MjData(nominal)
        view.time = d.time
        view.qpos[:] = d.qpos
        view.qvel[:] = d.qvel
        view.ctrl[:] = d.ctrl
        if nominal.na:
            view.act[:] = d.act
        state = np.empty(
            mujoco.mj_stateSize(nominal, mujoco.mjtState(manifest["state_spec"]))
        )
        mujoco.mj_getState(
            nominal, view, state, mujoco.mjtState(manifest["state_spec"])
        )
        with np.load(output / "stages" / (stage + "_train.npz")) as old:
            values = {name: old[name].copy() for name in old.files}
        values["states"][0] = state
        values["controls"][0] = controls
        write_npz(output / "stages" / (stage + "_train.npz"), **values)
        write_npz(
            raw / (stage + ".npz"),
            states=full[None],
            controls=controls[None],
            replay_end=end,
            state_spec=np.array(report["state_spec"]),
        )
        if minimum_only:
            write_npz(
                physical_bank / "stages" / (stage + "_train.npz"),
                states=full[None],
                controls=controls[None],
                seed=np.array([0]),
                family=np.array([0]),
            )
        write_json(raw / (stage + ".json"), report)
        summary = {
            k: v
            for k, v in report.items()
            if k not in ("physics_profile", "settling_trace", "boundary")
        }
        validation[stage] = {
            "candidate_rejected": False,
            "rejection_reasons": [],
            "equilibrium": summary,
            "training_eligible": False,
            "qualification": "Passive unchanged-control hold passed; task admission remains separate",
        }
        if minimum_only:
            validation[stage].update(
                minimum_validity=summary,
                valid_for_initial_reset_bank=True,
                training_eligible=True,
                qualification="No unintended overlaps; held parts retained. Movement allowed.",
            )
            validation[stage].pop("equilibrium", None)
        receipt = by_stage[stage]
        receipt["equilibrium"] = summary
        if minimum_only:
            receipt.pop("equilibrium", None)
            receipt["minimum_validity"] = summary
        receipt["state_sha256"] = hashlib.sha256(state.tobytes()).hexdigest()
        receipt["control_sha256"] = report["control_sha256"]
        if minimum_only:
            receipt.update(
                registration_pass=True,
                registration_rejection_reasons=[],
                qualification="Valid initial reset: no unintended overlaps and retained grasps",
            )
        receipt["equilibrium_physical_archive"] = str(raw / (stage + ".npz"))
        for side in ("left", "right"):
            names = [
                "shoulder_pan",
                "shoulder_lift",
                "elbow_flex",
                "wrist_flex",
                "wrist_roll",
                "gripper",
            ]
            q = [float(view.joint(side + "_" + name).qpos[0]) for name in names]
            receipt.setdefault("arms", {}).setdefault(side, {}).update(
                joint_positions=q[:5], opening_radians=q[-1]
            )
        manifest["stage_by_task"][stage]["boundary_review"] = (
            "No unintended overlaps; held parts retained. Brackets stay passive; movement allowed."
            if minimum_only
            else "Passive brackets: unchanged-control equilibrium hold passed; task admission remains separate."
        )
    manifest.update(
        display_name=f"Initial resets · {len(selected)}/25 valid"
        if minimum_only
        else f"Passive equilibrium · {len(selected)}/25 holds passed",
        randomized_display_name="Valid reset variations"
        if minimum_only
        else "Passive equilibrium · reset variations",
        nominal_validation_by_task=validation,
        physics_profile=profile,
        equilibrium_reference_count=0 if minimum_only else len(selected),
        training_eligible=bool(minimum_only and not missing),
        repair_draft=allow_incomplete,
        render_only=minimum_only,
        training_bank=str(physical_bank) if minimum_only else None,
        family_names=["Passive-bracket reset review · per-state hold checks"],
        variation_by_task={
            stage: {
                "train": [
                    {
                        "kind": "nominal",
                        "training_eligible": bool(minimum_only and stage in selected),
                        "cad_revision": manifest.get("cad_revision"),
                        "support_grasp_by_side": by_stage[stage].get(
                            "support_grasp_by_side", {}
                        ),
                        "candidate_rejected": validation[stage]["candidate_rejected"],
                        "rejection_reasons": validation[stage]["rejection_reasons"],
                        "reference_validation": validation[stage],
                        (
                            "minimum_validity"
                            if minimum_only and stage in selected
                            else "equilibrium"
                        ): validation[stage].get(
                            "minimum_validity", validation[stage].get("equilibrium")
                        ),
                    }
                ],
                "eval": [],
            }
            for stage in manifest["task_names"]
        },
    )
    if minimum_only:
        manifest.update(
            family_names=["Valid initial reset · overlap and grasp checks"],
            minimum_validity_reference_count=len(selected),
            stationarity_required=False,
        )
    write_json(output / "stages/manifest.json", manifest)
    write_json(output / "stage-receipts.json", receipts)
    top = json.loads((output / "manifest.json").read_text())
    top.update(
        profile=output.name,
        model_signature=hashlib.sha256(display_xml).hexdigest(),
        source_bank=str(source),
        physics_profile=profile,
        equilibrium_reference_count=0 if minimum_only else len(selected),
        training_eligible=bool(minimum_only and not missing),
        repair_draft=allow_incomplete,
        render_only=minimum_only,
        training_bank=str(physical_bank) if minimum_only else None,
        qualification="Initial reset minimum: no unintended overlaps and held parts retained; movement allowed"
        if minimum_only
        else "Passive equilibrium review",
    )
    write_json(output / "manifest.json", top)
    if minimum_only:
        write_json(
            physical_bank / "stages/manifest.json",
            {
                "model_signature": profile["model_sha256"],
                "state_spec": int(mujoco.mjtState.mjSTATE_INTEGRATION),
                "task_names": manifest["task_names"],
                "family_names": ["Minimum-valid initial reset"],
                "reset_distribution": "minimum_valid_initial",
                "prototype_count": len(selected),
                "minimum_reset_required": True,
                "stationarity_required": False,
                "training_eligible": not missing,
                "complete": not missing,
                "physics_profile": profile,
                "stage_by_task": manifest["stage_by_task"],
            },
        )
        write_json(
            physical_bank / "manifest.json",
            {
                "profile": physical_bank.name,
                "model_signature": profile["model_sha256"],
                "bank": "stages",
                "count": len(selected),
                "qualification": "Initial reset minimum: overlap and gripper retention checks; movement allowed",
            },
        )
    with (
        output / ("VALIDITY_STATUS.md" if minimum_only else "EQUILIBRIUM_STATUS.md")
    ).open("w") as stream:
        stream.write(
            f"# Initial reset minimum validity\n\n{len(selected)}/25 references passed no-unintended-overlap and actual gripper-retention checks. Movement is allowed; brackets are unpowered. Physical integration states, controls, warmstarts and replay ends are in physical-replays/. Display states preserve measured velocities. No bracket-angle targets, helper welds or external holding forces were used.\n"
            if minimum_only
            else f"# Passive equilibrium references\n\n{len(selected)}/25 references passed an unchanged-control hold. Physical integration states and replay ends are in physical-replays/.\n"
        )
    return {
        "count": len(selected),
        "output": str(output),
        "physics_model_sha256": next(iter(identities)),
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--results", type=Path, nargs="+", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Publish an explicitly unfinished local review collection",
    )
    p.add_argument("--minimum-only", action="store_true")
    a = p.parse_args()
    print(
        json.dumps(
            publish(
                a.source,
                a.results,
                a.output,
                allow_incomplete=a.allow_incomplete,
                minimum_only=a.minimum_only,
            ),
            indent=2,
        )
    )
