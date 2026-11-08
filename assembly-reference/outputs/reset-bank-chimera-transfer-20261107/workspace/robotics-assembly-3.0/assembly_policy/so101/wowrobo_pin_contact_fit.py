"""Isolated source-preserving fine-grid contacts for the approved fit-sized pin.

The current 2.3 x 14.5 mm shaft and frozen 2.7 x 22 mm negative use the same
receiving case field. No hole, contact mask or analytic thread substitutes CAD.
This module is separate from the concurrent physical-bank builder.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT = ROOT / "outputs/insertion-fix-20261107/pin"
BASE = ROOT / "outputs/linear-insertion-updated-20261107/grids/manifest.json"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _native_mesh_frame(source):
    import mujoco

    root = ET.Element("mujoco")
    asset = ET.SubElement(root, "asset")
    ET.SubElement(asset, "mesh", name="pin", file=str(Path(source).resolve()))
    world = ET.SubElement(root, "worldbody")
    ET.SubElement(world, "geom", type="mesh", mesh="pin")
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    return model.mesh_pos[0].tolist(), model.mesh_quat[0].tolist()


def build_fields(output=DEFAULT, spacing=0.0001):
    from scripts.build_wowrobo_contact_grids import build_one

    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    original = json.loads(BASE.read_text())
    specs = []
    for name, source, label in [
        (
            "servo_1_case",
            Path(original["meshes"]["servo_1_case"]["source_file"]),
            "case",
        ),
        ("pin", ROOT / "assets_local/wowrobo/fit-pin-v2/pin.obj", "fit_pin"),
        ("pin", Path(original["meshes"]["pin"]["source_file"]), "old_negative_pin"),
    ]:
        target = output / "fields" / f"{label}.bin"
        target.parent.mkdir(exist_ok=True)
        spec = dict(original["meshes"][name])
        spec.update(
            source_file=str(source), grid_file=str(target), spacing=spacing, name=label
        )
        cache = target.with_suffix(".json")
        if cache.exists() and target.exists():
            row = json.loads(cache.read_text())
            if (
                row.get("source_sha256") == digest(source)
                and row.get("spacing") == spacing
            ):
                continue
        specs.append(spec)
    with ProcessPoolExecutor(max_workers=3) as pool:
        for future in as_completed([pool.submit(build_one, s) for s in specs]):
            row = future.result()
            Path(row["grid_file"]).with_suffix(".json").write_text(
                json.dumps(row, indent=2) + "\n"
            )
            print(
                json.dumps(
                    {
                        "built": row["name"],
                        "shape": row["shape"],
                        "seconds": row["elapsed_s"],
                        "error_max_m": row["sampled_error_max_m"],
                    }
                ),
                flush=True,
            )
    for kind, pinlabel in [("fit", "fit_pin"), ("negative", "old_negative_pin")]:
        grid = output / f"{kind}-grids"
        grid.mkdir(exist_ok=True)
        manifest = json.loads(BASE.read_text())
        manifest["grid_directory"] = str(grid)
        manifest["grid_dir"] = str(grid)
        manifest["status"] = "complete"
        manifest["profile"] = f"approved_fit_pin_same_case_{kind}_fine_{spacing:g}"
        for name, old in original["meshes"].items():
            label = (
                "case"
                if name == "servo_1_case"
                else pinlabel
                if name == "pin"
                else None
            )
            row = (
                json.loads((output / "fields" / f"{label}.json").read_text())
                if label
                else dict(old)
            )
            row["name"] = name
            if name == "pin":
                row["native_mesh_pos"], row["native_mesh_quat_wxyz"] = (
                    _native_mesh_frame(row["source_file"])
                )
            target = grid / f"grid_{old['grid_id']}.bin"
            source = Path(row["grid_file"])
            if target.exists():
                target.unlink()
            os.link(source, target)
            row["grid_file"] = str(target)
            manifest["meshes"][name] = row
        manifest["same_receiving_case_sha256"] = manifest["meshes"]["servo_1_case"][
            "grid_sha256"
        ]
        manifest["original_geometry_except_approved_pin"] = True
        (grid / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    result = {
        "profile": "wowrobo-approved-pin-source-grid-fit-v1",
        "spacing_m": spacing,
        "fit_manifest": str(output / "fit-grids/manifest.json"),
        "negative_manifest": str(output / "negative-grids/manifest.json"),
        "receiving_case_same_for_fit_and_old_negative": True,
        "hole_geometry_unchanged": True,
        "original_pin_only_historical_negative": True,
        "fit_pin_manifest": str(ROOT / "assets_local/wowrobo/fit-pin-v2/manifest.json"),
        "training_eligible": False,
    }
    (output / "field-profile.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def replace_pin_asset(source_scene, output_scene, *, pin_mesh=None):
    """Copy an MJCF and change only the approved pin's visual/native mesh file.

    Body/joint order, inertias and all receiving geometry are preserved. This
    does not create a physical profile or update a live bank.
    """
    source_scene, output_scene = Path(source_scene), Path(output_scene)
    pin_mesh = Path(pin_mesh or ROOT / "assets_local/wowrobo/fit-pin-v2/pin.obj")
    tree = ET.parse(source_scene)
    asset = tree.find("asset/mesh[@name='pin']")
    if asset is None:
        raise ValueError("Source scene has no pin mesh asset")
    asset.set("file", str(pin_mesh.resolve()))
    output_scene.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output_scene, encoding="unicode")
    return {
        "scene": str(output_scene.resolve()),
        "source_sha256": digest(source_scene),
        "scene_sha256": digest(output_scene),
        "pin_mesh": str(pin_mesh.resolve()),
        "pin_mesh_sha256": digest(pin_mesh),
        "other_geometry_changed": False,
    }


def installed_fit_pin_qpos(model, qpos, stage_id, *, registration=None, head_stop=True):
    """Register only an installed prerequisite pin in a complete qpos vector.

    Use for stages17 onward; held pin stages13-15 require fresh hand IK and are
    deliberately excluded. `head_stop=True` removes the authoring0.1mm gap.
    The receiver-relative registration follows the actual connector transform.
    """
    import mujoco
    import numpy as np

    from assembly_policy.so101.wowrobo_pin_registration import apply_pin_registration

    number = int(stage_id.split("_", 1)[0])
    if number < 17:
        raise ValueError("Installed-pin helper requires stage17 or later")
    if registration is None:
        registration = json.loads(
            (ROOT / "assets_local/wowrobo/fit-pin-v2/registration.json").read_text()
        )
    data = mujoco.MjData(model)
    values = np.asarray(qpos, dtype=float)
    if values.shape != (model.nq,):
        raise ValueError(f"Expected {model.nq} coordinates, got {values.shape}")
    data.qpos[:] = values
    mujoco.mj_forward(model, data)
    apply_pin_registration(model, data, registration, stage_id)
    if head_stop:
        receiver = model.body(registration["receiver_body"]).id
        rotation = data.xmat[receiver].reshape(3, 3) @ np.array(
            registration["receiver_local_rotation"]
        ).reshape(3, 3)
        address = model.joint("assembly_pin_free").qposadr[0]
        data.qpos[address : address + 3] += rotation[:, 2] * registration["head_gap_m"]
        mujoco.mj_forward(model, data)
    return data.qpos.copy()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=DEFAULT)
    p.add_argument("--spacing", type=float, default=0.0001)
    a = p.parse_args()
    print(json.dumps(build_fields(a.output, a.spacing), indent=2))
