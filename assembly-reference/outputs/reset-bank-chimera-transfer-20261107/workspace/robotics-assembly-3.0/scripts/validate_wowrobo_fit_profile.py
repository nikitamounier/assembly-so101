"""Matched, bounded native tests for an explicit legacy-mapped mating profile.

The motion is an explicitly guided fixture probe. Pair laws alone change between
baseline and mapped cases. This does not qualify free robot insertion or release.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import mujoco
import numpy as np

from assembly_policy.so101.wowrobo_balanced_push import BalancedPush
from assembly_policy.so101.wowrobo_contact_backend import load_backend, set_surface_mode
from scripts.diagnose_wowrobo_insertion import configure, reactions
from scripts.probe_wowrobo_aligned_force import SPECS

ROOT = Path(__file__).resolve().parents[1]
PIN_GUARD_M = 0.017220118


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def penetration(data):
    return max(0.0, -min((float(c.dist) for c in data.contact), default=0.0))


def run_case(args, source, output, alignment, saved, profile, force, contacts=True):
    # Deferred import allows independent review while the profile module is built.
    from assembly_policy.so101.wowrobo_mating_profile import write_profile_scene

    label = f"{profile}_{force:g}N" + ("_contacts_off" if not contacts else "")
    started = time.monotonic()
    scene = output / f"{label}.xml"
    provenance = write_profile_scene(
        source / "guide-scene.xml", scene, args.step, profile=profile
    )
    model = mujoco.MjModel.from_xml_path(str(scene))
    if model.nq != 1 or model.nv != 1:
        raise RuntimeError("Expected the frozen one-axis insertion fixture")
    piece, receivers = SPECS[args.step][2], SPECS[args.step][4]
    active = configure(model, piece, receivers, True)
    if not contacts:
        # Explicit <pair> contacts ignore masks, so disable contact globally.
        model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
    damping = force / 0.020
    model.dof_damping[:] = damping
    data = mujoco.MjData(model)
    data.qpos[:] = saved["qpos"]
    data.qvel[:] = saved["qvel"]
    mujoco.mj_forward(model, data)
    axis = np.asarray(alignment["axis_world"], dtype=float)
    axis /= np.linalg.norm(axis)
    target = float(alignment["authored_axial_travel_m"])
    push = BalancedPush(model, data, active, axis, damping=0)
    push.apply(data, force)
    mujoco.mj_forward(model, data)
    initial_reaction = reactions(model, data, active, axis)
    initial_q = float(data.qpos[0])
    peak_penetration = penetration(data)
    maximum_q = initial_q
    minimum_q = initial_q
    trace = [
        [0.0, initial_q, float(data.qvel[0]), peak_penetration, float(data.ncon), force]
    ]
    snapshots = [{"phase": "drive", "time_s": 0.0, "reaction": initial_reaction}]
    stop = "drive duration elapsed"
    guard_failures = []
    pin_crossed = args.step == "pin" and maximum_q > PIN_GUARD_M + 1e-12
    for step in range(round(args.seconds / model.opt.timestep)):
        push.apply(data, force)
        mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)
        q = float(data.qpos[0])
        depth = penetration(data)
        maximum_q = max(maximum_q, q)
        minimum_q = min(minimum_q, q)
        peak_penetration = max(peak_penetration, depth)
        trace.append(
            [float(data.time), q, float(data.qvel[0]), depth, float(data.ncon), force]
        )
        if step % 100 == 0:
            snapshots.append(
                {
                    "phase": "drive",
                    "time_s": float(data.time),
                    "reaction": reactions(model, data, active, axis),
                }
            )
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
            stop = "nonfinite state"
            guard_failures.append(stop)
            break
        if args.step == "pin" and q > PIN_GUARD_M + 1e-12:
            pin_crossed = True
            if contacts:
                stop = "known pin source-obstruction guard crossed"
                guard_failures.append(stop)
                break
        if depth > 0.0005:
            stop = "0.5mm native penetration guard"
            guard_failures.append(stop)
            break
        if q >= target:
            stop = "authored reference crossed; not seating proof"
            break
    push.apply(data, force)
    mujoco.mj_forward(model, data)
    driven_q = float(data.qpos[0])
    driven_v = float(data.qvel[0])
    drive_time = float(data.time)
    final_reaction = reactions(model, data, active, axis)
    np.savez_compressed(
        output / f"{label}-drive-state.npz", qpos=data.qpos, qvel=data.qvel
    )
    # Guided relaxation only: same fixed orientation/articulation and implicit
    # damping remain. Do not call this a free-body release or retention test.
    release_stop = "guided relaxation duration elapsed"
    for step in range(round(args.release_seconds / model.opt.timestep)):
        push.apply(data, 0)
        mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)
        q = float(data.qpos[0])
        depth = penetration(data)
        maximum_q = max(maximum_q, q)
        minimum_q = min(minimum_q, q)
        peak_penetration = max(peak_penetration, depth)
        trace.append(
            [float(data.time), q, float(data.qvel[0]), depth, float(data.ncon), 0.0]
        )
        if step % 100 == 0:
            snapshots.append(
                {
                    "phase": "guided_relaxation",
                    "time_s": float(data.time),
                    "reaction": reactions(model, data, active, axis),
                }
            )
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
            release_stop = "nonfinite state"
            guard_failures.append(release_stop)
            break
        if args.step == "pin" and q > PIN_GUARD_M + 1e-12:
            pin_crossed = True
            if contacts:
                release_stop = "known pin source-obstruction guard crossed"
                guard_failures.append(release_stop)
                break
        if depth > 0.0005:
            release_stop = "0.5mm native penetration guard"
            guard_failures.append(release_stop)
            break
    push.apply(data, 0)
    mujoco.mj_forward(model, data)
    result = {
        "step": args.step,
        "case": label,
        "profile": profile,
        "profile_provenance": provenance,
        "force_N": force,
        "contacts_enabled": contacts,
        "native_implicit_slide_damping_Ns_per_m": damping,
        "balanced_push_explicit_drag_Ns_per_m": 0,
        "nominal_contact_free_terminal_speed_m_per_s": 0.020,
        "initial_axial_mm": initial_q * 1000,
        "final_drive_axial_mm": driven_q * 1000,
        "additional_drive_axial_mm": (driven_q - initial_q) * 1000,
        "final_drive_velocity_m_per_s": driven_v,
        "reference_axial_mm": target * 1000,
        "drive_simulated_seconds": drive_time,
        "drive_stop": stop,
        "drive_initial_reaction": initial_reaction,
        "drive_final_reaction": final_reaction,
        "final_relaxed_axial_mm": float(data.qpos[0]) * 1000,
        "guided_relaxation_change_mm": float(data.qpos[0] - driven_q) * 1000,
        "guided_relaxation_stop": release_stop,
        "guided_relaxation_final_reaction": reactions(model, data, active, axis),
        "all_substep_maximum_axial_mm": maximum_q * 1000,
        "all_substep_minimum_axial_mm": minimum_q * 1000,
        "all_substep_peak_native_penetration_mm": peak_penetration * 1000,
        "pin_source_guard_mm": PIN_GUARD_M * 1000 if args.step == "pin" else None,
        "pin_guard_crossed": pin_crossed,
        "physical_case_guard_failures": guard_failures if contacts else [],
        "contacts_off_positive_control_only": not contacts,
        "native_pair_count": int(model.npair),
        "balanced_push": push.description(),
        "wall_seconds": time.monotonic() - started,
        "trace_columns": [
            "time_s",
            "axial_m",
            "velocity_m_per_s",
            "native_penetration_m",
            "contact_count",
            "drive_N",
        ],
        "reaction_snapshots": snapshots,
        "limits": "One-axis guided benchmark with nonlinear contact response. Rotation, lateral position, internal joints and helpers remain fixtures. Guided relaxation is not free-body release. Crossing the authored reference is not seating proof. Source triangle-path and blocked-wall tests remain separate acceptance gates.",
    }
    np.savez_compressed(
        output / f"{label}-trace.npz",
        trace=np.asarray(trace),
        qpos=data.qpos,
        qvel=data.qvel,
    )
    (output / f"{label}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k
                in [
                    "case",
                    "final_drive_axial_mm",
                    "additional_drive_axial_mm",
                    "final_relaxed_axial_mm",
                    "all_substep_peak_native_penetration_mm",
                    "all_substep_maximum_axial_mm",
                    "drive_stop",
                    "physical_case_guard_failures",
                    "wall_seconds",
                ]
            }
        ),
        flush=True,
    )
    return result


def run(args):
    source = ROOT / "outputs/linear-insertion-aligned-20261107" / args.step
    output = args.output or ROOT / "outputs/legacy-fit-port-20261107" / args.step
    output.mkdir(parents=True, exist_ok=True)
    load_backend(ROOT / "outputs/linear-insertion-updated-20261107/grids/manifest.json")
    set_surface_mode(1)
    alignment = json.loads((source / "alignment.json").read_text())
    saved = np.load(source / "final-state.npz")
    cases = []
    for profile in ["baseline", "legacy_mapped"]:
        for force in args.forces:
            cases.append(
                run_case(args, source, output, alignment, saved, profile, force)
            )
    cases.append(
        run_case(
            args, source, output, alignment, saved, "legacy_mapped", 0.5, contacts=False
        )
    )
    summary = {
        "step": args.step,
        "design": "Identical frozen actual-feature aligned stalled state, same geometry, grid, guide, sparse solver and source sampling. Change only explicit mating-pair law and force. Native implicit slide damping c=F/0.020 limits nominal contact-free speed; baseline and mapped use the same c at each force. This differs from the earlier fixed10Ns/m benchmark and avoids unstable explicit1000Ns/m drag at20N.",
        "script_sha256": sha256(__file__),
        "aligned_state_sha256": sha256(source / "final-state.npz"),
        "alignment_sha256": sha256(source / "alignment.json"),
        "guide_scene_sha256": sha256(source / "guide-scene.xml"),
        "cases": cases,
        "physical_success": False,
        "scope": "Bounded native counterfactuals and guided relaxation only. Source-overlap path and hard-wall controls required before profile qualification; no production default change.",
    }
    (output / "validation-matrix.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step", choices=SPECS, required=True)
    parser.add_argument("--seconds", type=float, default=2.5)
    parser.add_argument("--release-seconds", type=float, default=0.5)
    parser.add_argument("--forces", type=float, nargs="+", default=[0.5, 5.0, 20.0])
    parser.add_argument("--output", type=Path)
    run(parser.parse_args())
