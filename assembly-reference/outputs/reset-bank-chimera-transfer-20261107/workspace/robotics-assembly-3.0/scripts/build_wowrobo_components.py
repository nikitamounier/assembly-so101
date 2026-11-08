"""Build photo-based WowRobo leader components from pinned public SO101 CAD.

CAD preparation only: cadquery==2.8.0, trimesh==5.2.0, numpy. No trainer changes.
STEP/STL exports use millimetres; GLB/OBJ/MJCF meshes use metres. Constituents are
kept separate, including servo cases and horns; a compound is not a rigid weld.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen

import cadquery as cq
import numpy as np
import trimesh
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.GeomAbs import GeomAbs_Cylinder
from OCP.STEPCAFControl import STEPCAFControl_Reader
from OCP.TCollection import TCollection_ExtendedString
from OCP.TDataStd import TDataStd_Name
from OCP.TDF import TDF_Label, TDF_LabelSequence
from OCP.TDocStd import TDocStd_Document
from OCP.TopLoc import TopLoc_Location
from OCP.XCAFDoc import XCAFDoc_DocumentTool

ROOT = Path(__file__).resolve().parents[1]
ASSEMBLY = "STEP/SO101/SO101 Assembly.step"
MOUNT = (
    "Optional/SO101_Wrist_Cam_Hex-Nut_Mount_32x32_UVC_Module/"
    "stl/SO-ARM101_camera_wrist_mount.step"
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def label_name(label):
    attr = TDataStd_Name()
    return (
        attr.Get().ToExtString()
        if label.FindAttribute(TDataStd_Name.GetID_s(), attr)
        else ""
    )


def definition(tool, label):
    result = TDF_Label()
    if tool.IsReference_s(label):
        tool.GetReferredShape_s(label, result)
        return result
    return label


def children(tool, label):
    seq = TDF_LabelSequence()
    tool.GetComponents_s(definition(tool, label), seq)
    return [seq.Value(i) for i in range(1, seq.Length() + 1)]


def load_assembly(path):
    doc = TDocStd_Document(TCollection_ExtendedString("MDTV-XCAF"))
    reader = STEPCAFControl_Reader()
    reader.SetNameMode(True)
    if int(reader.ReadFile(str(path))) != 1 or not reader.Transfer(doc):
        raise ValueError(f"Cannot read STEP assembly {path}")
    tool = XCAFDoc_DocumentTool.ShapeTool_s(doc.Main())
    seq = TDF_LabelSequence()
    tool.GetFreeShapes(seq)
    if seq.Length() != 1:
        raise ValueError("Expected one root assembly")
    return doc, tool, {label_name(l): l for l in children(tool, seq.Value(1))}


def leaves(tool, label, parent=None):
    parent = parent or TopLoc_Location()
    world = parent.Multiplied(tool.GetLocation_s(label))
    desc = children(tool, label)
    if desc:
        return [item for child in desc for item in leaves(tool, child, world)]
    shape = tool.GetShape_s(definition(tool, label))
    if shape.IsNull():
        return []
    return [(label_name(label), cq.Shape.cast(shape.Moved(world)))]


def compound(shapes):
    return cq.Compound.makeCompound(shapes)


def mesh(shape):
    vertices, faces = shape.tessellate(0.08, 0.12)
    result = trimesh.Trimesh(
        np.asarray([v.toTuple() for v in vertices]) * 0.001,
        np.asarray(faces),
        process=True,
    )
    if not np.isfinite(result.vertices).all() or not len(result.faces):
        raise ValueError("Invalid tessellation")
    return result


def horn_axis(shape):
    candidates = []
    for face in shape.Faces():
        surface = BRepAdaptor_Surface(face.wrapped)
        if surface.GetType() == GeomAbs_Cylinder:
            cylinder = surface.Cylinder()
            if 1 < cylinder.Radius() < 15:
                d = cylinder.Axis().Direction()
                candidates.append((cylinder.Radius(), np.array([d.X(), d.Y(), d.Z()])))
    if not candidates:
        raise ValueError("Servo horn has no cylindrical shaft axis")
    axis = max(candidates, key=lambda item: item[0])[1]
    center = np.array(shape.Center().toTuple())
    return {"position_m": (center * 0.001).tolist(), "axis": axis.tolist()}


def build(source, output):
    cfg = json.loads((ROOT / "configs/wowrobo_components.json").read_text())
    lock = json.loads((ROOT / "cad/wowrobo/source-lock.json").read_text())
    source.mkdir(parents=True, exist_ok=True)
    for item in lock["files"]:
        path = source / item["path"]
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            url = (
                f"https://raw.githubusercontent.com/TheRobotStudio/SO-ARM100/"
                f"{cfg['upstream_revision']}/" + quote(item["path"])
            )
            path.write_bytes(urlopen(url, timeout=120).read())
        if digest(path) != item["sha256"]:
            raise ValueError(f"Source hash mismatch: {item['path']}")
    _doc, tool, labels = load_assembly(source / ASSEMBLY)
    output.mkdir(parents=True, exist_ok=True)
    (output / "meshes").mkdir(exist_ok=True)
    manifest = {
        "profile": cfg["profile"],
        "upstream_revision": cfg["upstream_revision"],
        "source_sha256": digest(source / ASSEMBLY),
        "step_units": "mm",
        "stl_units": "mm",
        "mesh_units": "m",
        "target_piece_count": cfg["target_piece_count"],
        "structural_piece_count": cfg["structural_piece_count"],
        "moving_servo_parts": cfg["moving_servo_parts"],
        "pieces": {},
        "constituents": {},
        "servo_axes": {},
        "qualification": cfg["qualification"],
        "helper": cfg["helper"],
    }
    all_scene = trimesh.Scene()
    groups = {}
    for spec in cfg["pieces"]:
        name = spec["name"]
        if name == "assembly_pin":
            p = cfg["pin_mm"]
            # Smooth analytic STEP matches the two cylinders; STL/OBJ uses exactly128 facets.
            shape = (
                cq.Workplane("XY")
                .circle(p["grip_diameter"] / 2)
                .extrude(p["grip_length"])
                .union(
                    cq.Workplane("XY")
                    .workplane(offset=p["grip_length"] - p["overlap"])
                    .circle(p["shaft_diameter"] / 2)
                    .extrude(p["shaft_length"] + p["overlap"])
                )
            ).val()
            n = p["facets"]
            theta = np.arange(n) * 2 * np.pi / n
            rings = [
                (p["grip_diameter"] / 2, 0),
                (p["grip_diameter"] / 2, p["grip_length"]),
                (p["shaft_diameter"] / 2, p["grip_length"]),
                (p["shaft_diameter"] / 2, p["total_length"]),
            ]
            verts = (
                np.array(
                    [[r * np.cos(t), r * np.sin(t), z] for r, z in rings for t in theta]
                )
                * 0.001
            )
            faces = []
            for ring in range(3):
                for i in range(n):
                    a, b = ring * n + i, ring * n + (i + 1) % n
                    faces.extend([(a, b, b + n), (a, b + n, a + n)])
            verts = np.vstack([verts, [0, 0, 0], [0, 0, p["total_length"] * 0.001]])
            for i in range(n):
                faces.extend(
                    [
                        (4 * n, (i + 1) % n, i),
                        (4 * n + 1, 3 * n + i, 3 * n + (i + 1) % n),
                    ]
                )
            pin = trimesh.Trimesh(verts, faces, process=True)
            parts = [("pin", shape, "pin")]
            meshes = {"pin": pin}
            manifest["pin_mm"] = p
        else:
            parts = []
            for i, product in enumerate(spec["printed"]):
                sh = cq.Shape.cast(tool.GetShape_s(labels[product]))
                parts.append((f"{name}_printed_{i}", sh, "printed"))
            for servo in spec["servos"]:
                label = labels[f"ST3215 Servo v2:{servo}"]
                items = leaves(tool, label)
                horns = [(n, s) for n, s in items if "金属舵盘" in n]
                case = [s for n, s in items if "金属舵盘" not in n]
                drive = next(s for n, s in horns if "（驱动）" in n)
                manifest["servo_axes"][str(servo)] = horn_axis(drive)
                parts.append((f"servo_{servo}_case", compound(case), "servo_case"))
                parts.append(
                    (
                        f"servo_{servo}_horns",
                        compound([s for _, s in horns]),
                        "servo_horns",
                    )
                )
            meshes = {key: mesh(sh) for key, sh, _ in parts}
        shapes = [sh for _, sh, _ in parts]
        full = compound(shapes)
        cq.exporters.export(full, str(output / f"{name}.step"))
        group = trimesh.Scene()
        merged = trimesh.util.concatenate(list(meshes.values()))
        center = merged.bounds.mean(axis=0)
        for key, sh, kind in parts:
            m = meshes[key]
            m.visual.face_colors = {
                "printed": [73, 96, 119, 255],
                "servo_case": [37, 42, 48, 255],
                "servo_horns": [193, 200, 208, 255],
                "pin": [224, 155, 46, 255],
            }[kind]
            m.export(output / "meshes" / f"{key}.obj")
            group.add_geometry(m, node_name=key, geom_name=key)
            manifest["constituents"][key] = {
                "piece": name,
                "kind": kind,
                "mesh": f"meshes/{key}.obj",
                "bounds_m": m.bounds.tolist(),
                "faces": len(m.faces),
                "sha256": digest(output / "meshes" / f"{key}.obj"),
            }
        printable = merged.copy()
        printable.apply_scale(1000)
        printable.export(output / f"{name}.stl")
        group.export(output / f"{name}.glb")
        manifest["pieces"][name] = {
            "label": spec["label"],
            "review_position_m": spec["review_position_m"],
            "printed": spec["printed"],
            "servos": spec["servos"],
            "photos": spec["photos"],
            "constituents": [key for key, _, _ in parts],
            "bounds_m": merged.bounds.tolist(),
            "center_m": center.tolist(),
            "dimensions_mm": (merged.extents * 1000).tolist(),
            "step_sha256": digest(output / f"{name}.step"),
            "volume_mm3": sum(s.Volume() for s in full.Solids()),
            "step_valid": full.isValid(),
            "step_solids": len(full.Solids()),
        }
        groups[name] = group
        # Arrange the review tray; keep exact assembly frame in individual exports.
        offset = np.array(spec["review_position_m"])
        for key, m in group.geometry.items():
            m = m.copy()
            m.apply_translation(offset - center)
            all_scene.add_geometry(m, node_name=key, geom_name=key)
        print(name, np.round(merged.extents * 1000, 2), "mm", flush=True)
    mount_shape = cq.importers.importStep(str(source / MOUNT)).val()
    mount_mesh = mesh(mount_shape)
    mount_mesh.export(output / "meshes/camera_mount.obj")
    mount_shape.exportStep(str(output / "camera_mount.step"))
    manifest["camera_mount"] = {
        "source": MOUNT,
        "mesh": "meshes/camera_mount.obj",
        "bounds_m": mount_mesh.bounds.tolist(),
        "placement": "provisional",
    }
    # Preserve exact source and constituent identity; dimensions alone are insufficient.
    manifest["source_lock"] = lock
    all_scene.export(output / "components.glb")
    helper = output.parent / "helper"
    helper_lock = json.loads((ROOT / "cad/wowrobo/helper-lock.json").read_text())
    if helper_lock["revision"] != cfg["upstream_revision"]:
        raise ValueError("Helper and target revisions differ")
    for item in helper_lock["files"]:
        path = helper / item["path"]
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            if item["path"] == "revision.txt":
                path.write_text(cfg["upstream_revision"] + "\n")
            else:
                url = (
                    f"https://raw.githubusercontent.com/TheRobotStudio/SO-ARM100/"
                    f"{cfg['upstream_revision']}/Simulation/SO101/"
                    + quote(item["path"])
                )
                path.write_bytes(urlopen(url, timeout=120).read())
        if digest(path) != item["sha256"]:
            raise ValueError(f"Helper source hash mismatch: {item['path']}")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, default=ROOT / "assets_local/wowrobo/source")
    ap.add_argument(
        "--output", type=Path, default=ROOT / "assets_local/wowrobo/components-v2"
    )
    args = ap.parse_args()
    build(args.source, args.output)
