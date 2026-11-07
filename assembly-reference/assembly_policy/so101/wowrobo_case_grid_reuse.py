"""Reuse the verified fine case field through measured rigid coordinate maps.

The four source cases have identical triangle indices and corresponding vertices
related by rigid transforms within OBJ export precision (<20nm). Visual meshes,
body inertias and all source CAD remain unchanged. Each SDF geom must receive the
recorded field-to-source transform: the manifest alone is insufficient.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import trimesh


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rigid_match(source, target):
    a = trimesh.load_mesh(source, process=False)
    b = trimesh.load_mesh(target, process=False)
    if not np.array_equal(a.faces, b.faces) or len(a.vertices) != len(b.vertices):
        raise ValueError("Source case triangle correspondence changed")
    x, y = np.asarray(a.vertices), np.asarray(b.vertices)
    u, _, vt = np.linalg.svd((x - x.mean(0)).T @ (y - y.mean(0)))
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = vt.T @ u.T
    translation = y.mean(0) - rotation @ x.mean(0)
    error = np.linalg.norm(x @ rotation.T + translation - y, axis=1)
    if error.max() > 2e-8:
        raise ValueError(f"Cases differ beyond export precision: {error.max()}")
    matrix = np.eye(4)
    matrix[:3, :3], matrix[:3, 3] = rotation, translation
    return matrix, float(error.max())


def build_profile(source_manifest, fine_case_manifest, output):
    source_manifest, fine_case_manifest, output = map(
        Path, (source_manifest, fine_case_manifest, output)
    )
    output.mkdir(parents=True, exist_ok=True)
    manifest = copy.deepcopy(json.loads(source_manifest.read_text()))
    fine = json.loads(fine_case_manifest.read_text())["meshes"]["servo_1_case"]
    if digest(fine["grid_file"]) != fine["grid_sha256"]:
        raise ValueError("Fine case field identity changed")
    for name, original in list(manifest["meshes"].items()):
        row = copy.deepcopy(original)
        field = Path(row["grid_file"])
        if name in {f"servo_{n}_case" for n in range(1, 5)}:
            matrix, error = rigid_match(fine["source_file"], original["source_file"])
            for key in (
                "spacing",
                "origin",
                "shape",
                "bounds",
                "grid_sha256",
                "grid_bytes",
                "sampled_error_max_m",
                "sampled_error_p99_m",
            ):
                row[key] = fine[key]
            row["field_source_file"] = fine["source_file"]
            row["field_source_sha256"] = fine["source_sha256"]
            row["field_to_source"] = matrix.tolist()
            row["source_rigid_match_max_error_m"] = error
            row["requires_sdf_geom_reframe"] = True
            field = Path(fine["grid_file"])
        target = output / f"grid_{row['grid_id']}.bin"
        if target.exists() or target.is_symlink():
            target.unlink()
        os.link(field, target)
        row["grid_file"] = str(target.resolve())
        manifest["meshes"][name] = row
    manifest.update(
        grid_directory=str(output.resolve()),
        grid_dir=str(output.resolve()),
        status="complete",
        profile="source-rigid-case-fine-field-reuse-v1",
        requires_sdf_geom_reframe=True,
        base_manifest_sha256=digest(source_manifest),
    )
    target = output / "manifest.json"
    target.write_text(json.dumps(manifest, indent=2) + "\n")
    return target


def reframe_scene(source_scene, manifest_path, output_scene):
    """Apply field-to-source maps once, before compiling the SDF scene."""
    source_scene, output_scene = Path(source_scene), Path(output_scene)
    root = ET.parse(source_scene).getroot()
    if root.find("custom/numeric[@name='wowrobo_case_fields_reframed']") is not None:
        raise ValueError("Refusing to apply case field transforms twice")
    rows = json.loads(Path(manifest_path).read_text())["meshes"]
    changed = []
    for name, row in rows.items():
        if not row.get("requires_sdf_geom_reframe"):
            continue
        geom = root.find(f".//geom[@name='{name}__contact']")
        if geom is None:
            raise ValueError(f"Missing case SDF geometry: {name}")
        if any(key in geom.attrib for key in ("euler", "axisangle", "xyaxes", "zaxis")):
            raise ValueError("Normalize XML geometry rotation to quaternion first")
        original = np.eye(4)
        rotation = np.empty(9)
        mujoco.mju_quat2Mat(
            rotation, np.fromstring(geom.get("quat", "1 0 0 0"), sep=" ")
        )
        original[:3, :3] = rotation.reshape(3, 3)
        original[:3, 3] = np.fromstring(geom.get("pos", "0 0 0"), sep=" ")
        result = original @ np.asarray(row["field_to_source"])
        quaternion = np.empty(4)
        mujoco.mju_mat2Quat(quaternion, result[:3, :3].ravel())
        geom.set("pos", " ".join(format(v, ".17g") for v in result[:3, 3]))
        geom.set("quat", " ".join(format(v, ".17g") for v in quaternion))
        changed.append(name)
    custom = root.find("custom")
    if custom is None:
        custom = ET.SubElement(root, "custom")
    ET.SubElement(custom, "numeric", name="wowrobo_case_fields_reframed", data="1")
    output_scene.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(output_scene, encoding="unicode")
    receipt = {
        "changed_sdf_geom_frames": changed,
        "visual_geometry_changed": False,
        "source_scene_sha256": digest(source_scene),
        "scene_sha256": digest(output_scene),
        "manifest_sha256": digest(manifest_path),
        "scope": "Coordinate reuse; source vertices match within20nm; native frame validation required",
    }
    output_scene.with_suffix(".case-frame.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )
    return receipt
