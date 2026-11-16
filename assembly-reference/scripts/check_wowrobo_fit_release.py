"""Receiver-supported gravity release of a legacy-fit force-probe endpoint.

Only the incoming root is freed. Its preassembled internal shape stays fixed;
receiver/other target pieces remain fixtures and helper collisions are excluded.
This is a diagnostic, not production robot grasp or full retention admission.
"""

import argparse
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

from assembly_policy.so101.wowrobo_contact_backend import load_backend, set_surface_mode
from scripts.diagnose_wowrobo_insertion import configure
from scripts.probe_wowrobo_aligned_force import SPECS

ROOT = Path(__file__).resolve().parents[1]


def run(args):
    base = ROOT / "outputs/legacy-fit-port-20261107" / args.step
    label = f"{args.profile}_{args.force:g}N"
    case = json.loads((base / f"{label}.json").read_text())
    state = np.load(base / f"{label}-drive-state.npz")
    alignment = json.loads(
        (
            ROOT
            / "outputs/linear-insertion-aligned-20261107"
            / args.step
            / "alignment.json"
        ).read_text()
    )
    output = base / "free-release" / label
    output.mkdir(parents=True, exist_ok=True)
    load_backend(ROOT / "outputs/linear-insertion-updated-20261107/grids/manifest.json")
    set_surface_mode(1)
    source = base / f"{label}.xml"
    original = mujoco.MjModel.from_xml_path(str(source))
    original_data = mujoco.MjData(original)
    original_data.qpos[:] = state["qpos"]
    original_data.qvel[:] = state["qvel"]
    mujoco.mj_forward(original, original_data)
    piece = SPECS[args.step][2]
    root = ET.parse(source).getroot()
    body = root.find(f".//body[@name='{piece}']")
    if body is None:
        raise ValueError("Missing incoming root")
    root_id = original.body(piece).id
    body.set(
        "pos", " ".join(format(float(x), ".17g") for x in original_data.xpos[root_id])
    )
    body.set(
        "quat", " ".join(format(float(x), ".17g") for x in original_data.xquat[root_id])
    )
    for child in list(body):
        if child.tag in ("joint", "freejoint"):
            body.remove(child)
    ET.SubElement(body, "freejoint", name="release_free")
    root.find("option").set("gravity", "0 0 -9.81")
    scene = output / "scene.xml"
    ET.ElementTree(root).write(scene, encoding="unicode")
    model = mujoco.MjModel.from_xml_path(str(scene))
    all_targets = [
        "base_servo",
        "base_side_cover",
        "shoulder_connector",
        "assembly_pin",
        "upper_arm_module",
        "forearm_module",
    ]
    configure(model, piece, [p for p in all_targets if p != piece], True)
    # Include the real native table in the free-part diagnostic.
    for g in range(model.ngeom):
        if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE:
            model.geom_contype[g], model.geom_conaffinity[g] = 2, 1
            bid = int(model.geom_bodyid[g])
            model.body_contype[bid] |= 2
            model.body_conaffinity[bid] |= 1
    data = mujoco.MjData(model)
    axis = np.asarray(alignment["axis_world"])
    data.qvel[:3] = axis * float(state["qvel"][0])
    mujoco.mj_forward(model, data)
    position_error = max(
        float(
            np.max(
                np.abs(
                    data.geom_xpos[model.geom(original.geom(g).name).id]
                    - original_data.geom_xpos[g]
                )
            )
        )
        for g in range(original.ngeom)
        if original.geom(g).name
    )
    if position_error > 1e-12:
        raise RuntimeError("Release scene did not preserve actual endpoint")
    initial = data.body(piece).xpos.copy()
    trace = []
    peak = 0.0
    reason = "release duration elapsed"
    warning_count = 0
    for i in range(round(args.seconds / model.opt.timestep)):
        mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)
        depth = max(0.0, -min((float(c.dist) for c in data.contact), default=0))
        peak = max(peak, depth)
        if i % 10 == 0:
            trace.append(np.concatenate(([data.time, depth], data.qpos, data.qvel)))
        if not np.isfinite(data.qpos).all():
            reason = "nonfinite state"
            break
        if depth > 0.0005:
            reason = "0.5mm native penetration guard"
            break
    warning_count = int(sum(w.number for w in data.warning))
    travel = float(alignment["authored_axial_travel_m"])
    original_data.qpos[0] = travel
    mujoco.mj_forward(original, original_data)
    mating = alignment["mating_geom"]
    error = float(
        np.linalg.norm(data.geom(mating).xpos - original_data.geom(mating).xpos)
    )
    result = {
        "step": args.step,
        "profile": args.profile,
        "force_before_release_N": args.force,
        "source_drive_case": str(base / f"{label}.json"),
        "source_case_sha256": hashlib.sha256(
            (base / f"{label}.json").read_bytes()
        ).hexdigest(),
        "source_drive_q_mm": float(state["qpos"][0]) * 1000,
        "reference_q_mm": travel * 1000,
        "source_drive_crossed_reference": case["final_drive_axial_mm"] >= travel * 1000,
        "endpoint_geometry_error_m": position_error,
        "release_seconds": float(data.time),
        "stop": reason,
        "warning_count": warning_count,
        "peak_native_penetration_mm": peak * 1000,
        "root_displacement_mm": float(np.linalg.norm(data.body(piece).xpos - initial))
        * 1000,
        "mating_point_reference_error_mm": error * 1000,
        "final_free_velocity_norm": float(np.linalg.norm(data.qvel)),
        "physical_success": False,
        "limits": "Gravity-on free-root endpoint release, no drive or damping field. Incoming internal shape locked; all other target pieces/static table retained; helper collision excluded. No robot grasp, complete receiver dynamics or training admission asserted.",
    }
    np.savez_compressed(
        output / "trace.npz", trace=np.asarray(trace), qpos=data.qpos, qvel=data.qvel
    )
    (output / "release.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--step", choices=SPECS, required=True)
    p.add_argument(
        "--profile", choices=["baseline", "legacy_mapped"], default="legacy_mapped"
    )
    p.add_argument("--force", type=float, default=20)
    p.add_argument("--seconds", type=float, default=0.5)
    run(p.parse_args())
