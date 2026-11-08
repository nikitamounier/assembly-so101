"""Source-preserving review candidate for the two-horn connector aperture.

Original CAD stays immutable. This module exports a separately named manufacturing
candidate; it is not automatically selected by any training or physics builder.
Two measured facing planes are recessed 0.475 mm each: the 0.85 mm nominal
interference plus 0.10 mm total print clearance. Actual kit dimensions, material
and compliance still require measurement; this is not a claim of hardware truth.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PROFILE = "connector-two-horn-clearance-review-v1"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def beam_estimates(section_file, modulus_pa=(0.2e9, 0.5e9, 1e9, 1.973e9, 3e9)):
    """Approximate nonuniform cantilever quadrature; not calibrated FEA."""
    rows = json.loads(Path(section_file).read_text())["rows"]
    x = np.array([r["longitudinal_l_m"] for r in rows])
    root, tip = -0.033, -0.0075
    s = np.linspace(root, tip, 1001)
    estimates = []
    for modulus in modulus_pa:
        result = {"effective_modulus_pa": modulus}
        for name in ("lower", "upper"):
            inertia = np.interp(s, x, [r[name]["I_bend_m4"] for r in rows])
            compliance = float(np.trapezoid((tip - s) ** 2 / (modulus * inertia), s))
            result[name] = {
                "stiffness_N_per_m": 1 / compliance,
                "force_for_0p425mm_N": 0.000425 / compliance,
                "deflection_under_20N_m": 20 * compliance,
            }
        estimates.append(result)
    return {"unsupported_length_m": tip - root, "estimates": estimates}


def completed_connector_translation(model, data, centering_m=0.000425):
    """Candidate-only correction carried by the actual receiver shaft axis.

    Callers explicitly select completed prerequisite roots; never apply this to
    loose pickup pieces or twice to an already registered state.
    """
    joint = model.joint("target_servo_1_shaft").id
    axis = np.asarray(data.xaxis[joint], dtype=float)
    return -float(centering_m) * axis


def completed_connector_reference(model, data, centering_m=0.000425):
    """Target fork frame centered on both horns, including coherent shaft index.

    This is an engineering receiving reference, not an observed seated state.
    Original common-CAD connector root coincides with the base root at zero
    shaft index. Both its orientation and offset rotate about the actual shaft.
    """
    from scipy.spatial.transform import Rotation

    joint = model.joint("target_servo_1_shaft").id
    axis = np.asarray(data.xaxis[joint], dtype=float)
    anchor = np.asarray(data.xanchor[joint], dtype=float)
    angle = float(data.qpos[int(model.jnt_qposadr[joint])])
    rotation = Rotation.from_rotvec(axis * angle).as_matrix()
    base = data.body("base_servo")
    nominal = np.asarray(base.xpos) + completed_connector_translation(
        model, data, centering_m
    )
    position = anchor + rotation @ (nominal - anchor)
    orientation = rotation @ np.asarray(base.xmat).reshape(3, 3)
    return position, Rotation.from_matrix(orientation).as_quat(scalar_first=True)


def build_review_candidate(
    source_step, output, relief_mm=0.475, *, throat=True, case_post=True
):
    """Export measured horn/throat/post reliefs as a separate review candidate.

    The legacy plane-only variant is retained for reproducing its falsification.
    The default includes the source-measured lip and bottom-post corridors.
    """
    import cadquery as cq
    import trimesh

    if not 0 < relief_mm <= 0.5:
        raise ValueError(
            "Review relief must stay within the explicitly bounded0.5mm edit budget"
        )
    if case_post and not throat:
        raise ValueError(
            "The post corridor is a v4 extension of the complete horn throat"
        )
    source_step, output = Path(source_step), Path(output)
    if (output / "candidate-manifest.json").exists():
        raise FileExistsError(
            "Use a new output directory; frozen candidate artifacts are immutable"
        )
    output.mkdir(parents=True, exist_ok=True)
    source_hash = sha256(source_step)
    full = cq.importers.importStep(str(source_step)).val()
    solids = full.Solids()
    # Source printed_0 is the long fork; printed_1 is the separate socket.
    fork = max(solids, key=lambda s: s.BoundingBox().zlen)
    others = [s for s in solids if not s.isSame(fork)]
    angle = np.deg2rad(0.5)
    normal = np.array([0.0, -np.sin(angle), np.cos(angle)])
    selected = []
    tools = []
    for face in fork.Faces():
        outward = np.array(face.normalAt().toTuple())
        center = np.array(face.Center().toTuple())
        offset = float(center @ normal)
        lower = np.dot(outward, normal) > 0.99999 and abs(offset - 29.44497007) < 0.001
        upper = np.dot(outward, normal) < -0.99999 and abs(offset - 65.84497007) < 0.001
        if not (lower or upper):
            continue
        shifted = face.translate(cq.Vector(*(outward * 0.001)))
        tools.append(
            cq.Solid.extrudeLinear(
                shifted.outerWire(),
                shifted.innerWires(),
                cq.Vector(*(-outward * (relief_mm + 0.001))),
            )
        )
        selected.append(
            {
                "role": "lower_lip" if lower else "upper_ceiling",
                "source_plane_offset_mm": offset,
                "source_face_area_mm2": face.Area(),
                "outward_normal_source": outward.tolist(),
                "inward_recess_mm": relief_mm,
            }
        )
    if len(selected) != 2:
        raise ValueError(f"Expected exactly two measured mating faces, got {selected}")
    expected_area = {"lower_lip": 447.2101789810648, "upper_ceiling": 264.1908295160913}
    for record in selected:
        if abs(record["source_face_area_mm2"] - expected_area[record["role"]]) > 0.001:
            raise ValueError(
                "Mating face differs from measured source; refusing repeated or unreviewed relief"
            )
    throat_recipe = None
    if throat:
        # The flat face contains inner loops. Extruding that face alone leaves
        # the measured conical/45-degree throat walls intact (v1 falsification).
        # Restrict the replacement cutters to the actual swept horn footprint
        # inside the same two thin normal bands; protect bore/headstop above it.
        source_manifest = json.loads(
            (ROOT / "assets_local/wowrobo/components-v2/manifest.json").read_text()
        )
        shaft = np.array(source_manifest["servo_axes"]["1"]["position_m"]) * 1000
        tangent = np.cross(normal, [1, 0, 0])
        center_l = float(shaft @ tangent)
        radius, sweep = 9.65, 65.0
        face_planes = {r["role"]: r["source_plane_offset_mm"] for r in selected}
        tools = []
        for role, h in face_planes.items():
            h0 = h - relief_mm if role == "lower_lip" else h - 0.001
            plane = cq.Plane(
                origin=tuple(normal * h0), xDir=(1, 0, 0), normal=tuple(normal)
            )
            tools.append(
                cq.Workplane(plane)
                .center(float(shaft[0]), center_l + sweep / 2)
                .slot2D(sweep + 2 * radius, 2 * radius, angle=90)
                .extrude(relief_mm + 0.001)
                .val()
            )
        throat_recipe = {
            "capsule_radius_mm": radius,
            "source_horn_outer_radius_mm": 9.6,
            "radial_clearance_mm": 0.05,
            "sweep_mm": sweep,
            "source_u_mm": float(shaft[0]),
            "source_l_mm": center_l,
            "pin_bore_and_headstop_protected": True,
        }
    post_recipe = None
    if case_post:
        source_manifest = json.loads(
            (ROOT / "assets_local/wowrobo/components-v2/manifest.json").read_text()
        )
        shaft = np.array(source_manifest["servo_axes"]["1"]["position_m"]) * 1000
        tangent = np.cross(normal, [1, 0, 0])
        center_l = float(shaft @ tangent)
        h0 = 28.044965117 + 0.425 - 0.05
        h1 = 29.444970068622034 - relief_mm + 0.01
        plane = cq.Plane(
            origin=tuple(normal * h0), xDir=(1, 0, 0), normal=tuple(normal)
        )
        tools.append(
            cq.Workplane(plane)
            .center(float(shaft[0]), center_l + 65.0 / 2)
            .slot2D(65.0 + 2 * 3.05, 2 * 3.05, angle=90)
            .extrude(h1 - h0)
            .val()
        )
        post_recipe = {
            "source_post_radius_mm": 3.0,
            "clearance_radius_mm": 3.05,
            "normal_band_mm": [h0, h1],
            "sweep_mm": 65.0,
            "measured_original_source_intrusion_mm": 0.1613093153,
            "pin_headstop_protected": True,
        }
    revised = fork
    for tool in tools:
        revised = revised.cut(tool)
    if not revised.isValid() or len(revised.Solids()) != 1:
        raise RuntimeError("Derived fork is not one valid solid")
    candidate = cq.Compound.makeCompound([revised, *others])
    candidate.exportStep(str(output / "shoulder_connector_clearance_review.step"))
    revised.exportStep(
        str(output / "shoulder_connector_printed_0_clearance_review.step")
    )
    vertices, faces = revised.tessellate(0.08, 0.12)
    mesh = trimesh.Trimesh(
        np.array([v.toTuple() for v in vertices]) / 1000, faces, process=False
    )
    mesh.export(output / "shoulder_connector_printed_0.obj")
    printable = mesh.copy()
    printable.apply_scale(1000)
    printable.export(output / "shoulder_connector_printed_0_clearance_review.stl")
    mesh.export(output / "shoulder_connector_printed_0_clearance_review.glb")
    original_gap = (
        selected[1]["source_plane_offset_mm"] - selected[0]["source_plane_offset_mm"]
    )
    # Face order is not significant; record actual measured lower/upper planes.
    planes = {r["role"]: r["source_plane_offset_mm"] for r in selected}
    original_gap = planes["upper_ceiling"] - planes["lower_lip"]
    result = {
        "profile": "connector-two-horn-and-case-post-clearance-review-v4"
        if case_post
        else ("connector-two-horn-throat-clearance-review-v2" if throat else PROFILE),
        "throat_recipe": throat_recipe,
        "case_post_recipe": post_recipe,
        "source_step": str(source_step.resolve()),
        "source_step_sha256": source_hash,
        "source_step_unchanged": sha256(source_step) == source_hash,
        "source_cad_original_selected_by_default": True,
        "output_directory": str(output.resolve()),
        "derived_mesh": str((output / "shoulder_connector_printed_0.obj").resolve()),
        "derived_mesh_sha256": sha256(output / "shoulder_connector_printed_0.obj"),
        "measured_face_modifications": selected,
        "source_opening_mm": original_gap,
        "candidate_opening_mm": original_gap + 2 * relief_mm,
        "horn_outside_span_mm": 37.25000002,
        "candidate_total_nominal_print_clearance_mm": original_gap
        + 2 * relief_mm
        - 37.25000002,
        "fork_source_volume_mm3": fork.Volume(),
        "fork_candidate_volume_mm3": revised.Volume(),
        "removed_volume_mm3": fork.Volume() - revised.Volume(),
        "valid_single_fork_solid": True,
        "separate_socket_unchanged": True,
        "manufactured_dimensions_approved": False,
        "purpose": "Reviewable source-measured clearance candidate; no global contact softening or source CAD overwrite.",
        "qualification": "Requires exact removed-material bound, full path/negative controls, native insertion and gravity release. Hardware opening, horn stack span and fork load-deflection remain unmeasured.",
    }
    (output / "candidate-manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-step",
        type=Path,
        default=ROOT / "assets_local/wowrobo/components-v2/shoulder_connector.step",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    parser.add_argument("--relief-mm", type=float, default=0.475)
    parser.add_argument(
        "--variant",
        choices=["v1", "v2", "v4"],
        default="v4",
        help="v4 is the tested horn-and-post candidate; older variants reproduce falsified intermediate designs.",
    )
    parser.add_argument(
        "--throat",
        action="store_true",
        help="Include the measured inner chamfer/throat swept footprint (v2).",
    )
    parser.add_argument(
        "--case-post",
        action="store_true",
        help="Include the independently measured bottom-case-post corridor (v3).",
    )
    args = parser.parse_args()
    throat = args.throat or args.variant in ("v2", "v4")
    case_post = args.case_post or args.variant == "v4"
    if args.output is None:
        version = (
            "clearance-review-v3"
            if case_post
            else ("clearance-review-v2" if throat else "clearance-review-v1")
        )
        args.output = ROOT / "outputs/insertion-fix-20261107/connector" / version
    print(
        json.dumps(
            build_review_candidate(
                args.source_step,
                args.output,
                args.relief_mm,
                throat=throat,
                case_post=case_post,
            ),
            indent=2,
        )
    )
