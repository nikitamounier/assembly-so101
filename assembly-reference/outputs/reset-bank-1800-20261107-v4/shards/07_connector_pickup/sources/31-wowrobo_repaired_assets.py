"""Export the frozen, derived insertion-fix CAD as isolated review components.

This imports geometry and provenance only. It does not install the guided
insertion contact law, move reference poses, or certify a training reset.
"""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_READY = ROOT / "outputs/insertion-fix-20261107/final-common-v1/FINAL_READY.json"
DEFAULT_BASE = ROOT / "assets_local/wowrobo/components-v2"
DEFAULT_OUTPUT = ROOT / "assets_local/wowrobo/components-repaired-v1"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def frozen_mesh_sources(ready_path=DEFAULT_READY, base_assets=DEFAULT_BASE):
    """Validate all16 source/field bindings, including final receiving relief."""
    ready_path, base_assets = Path(ready_path), Path(base_assets)
    ready = json.loads(ready_path.read_text())
    for path_key, hash_key in (
        ("scene", "scene_sha256"),
        ("source_scene", "source_scene_sha256"),
        ("manifest", "manifest_sha256"),
    ):
        if sha256(ready[path_key]) != ready[hash_key]:
            raise ValueError(f"Frozen CAD candidate changed: {path_key}")
    for name, item in ready["frozen_assets"].items():
        if sha256(item["frozen_asset"]) != item["sha256"]:
            raise ValueError(f"Frozen asset changed: {name}")
    source = Path(ready["source_scene"])
    xml = ET.parse(source).getroot()
    grid = json.loads(Path(ready["manifest"]).read_text())
    base = json.loads((base_assets / "manifest.json").read_text())
    sources = {}
    for name in base["constituents"]:
        node = xml.find(f"./asset/mesh[@name='{name}']")
        if node is None:
            raise ValueError(f"Frozen candidate missing constituent {name}")
        if node.get("scale", "1 1 1").split() != ["1", "1", "1"]:
            raise ValueError(f"Unexpected source mesh scaling: {name}")
        path = Path(node.get("file"))
        if not path.is_absolute():
            path = source.parent / path
        expected = grid["meshes"][name]["source_sha256"]
        if sha256(path) != expected:
            raise ValueError(f"Mesh does not match frozen field source: {name}")
        geom = xml.find(f".//geom[@name='{name}']")
        if geom is None or any(key in geom.attrib for key in ("pos", "quat", "euler")):
            raise ValueError(f"Unexpected constituent frame override: {name}")
        sources[name] = path
    if len(sources) != 16:
        raise ValueError("Expected exactly16 target mesh constituents")
    return ready, base, sources


def export_repaired_assets(
    output=DEFAULT_OUTPUT, *, ready_path=DEFAULT_READY, base_assets=DEFAULT_BASE
):
    """Create a new component collection accepted by PhotoResetSequence."""
    output = Path(output).resolve()
    ready, base, sources = frozen_mesh_sources(ready_path, base_assets)
    marker = output / "manifest.json"
    if marker.exists():
        prior = json.loads(marker.read_text())
        identity = prior.get("repaired_cad", {}).get("source_scene_sha256")
        if identity != ready["source_scene_sha256"]:
            raise ValueError("Refusing to replace another repaired asset identity")
    (output / "meshes").mkdir(parents=True, exist_ok=True)
    manifest = copy.deepcopy(base)
    manifest["profile"] = "wowrobo-components-repaired-20261107-v1"
    manifest["qualification"] = (
        "Derived CAD candidate with guided insertion evidence; reset dynamics and hardware unqualified"
    )
    manifest["pin_mm"] = dict(
        base["pin_mm"],
        shaft_diameter=2.3,
        shaft_length=14.0,
        total_length=42.0,
        grip_diameter=12.0,
        grip_length=28.0,
    )
    changed = []
    meshes = {}
    provenance = {}
    for name, source in sources.items():
        item = manifest["constituents"][name]
        digest = sha256(source)
        target = output / "meshes" / f"{name}.obj"
        if target.exists() and sha256(target) != digest:
            raise ValueError(f"Refusing to replace existing asset {target}")
        if not target.exists():
            shutil.copy2(source, target)
        mesh = trimesh.load_mesh(target, process=False)
        meshes[name] = mesh
        if digest != base["constituents"][name]["sha256"]:
            changed.append(name)
        item.update(
            mesh=f"meshes/{name}.obj",
            bounds_m=mesh.bounds.tolist(),
            faces=len(mesh.faces),
            sha256=digest,
        )
        provenance[name] = {
            "frozen_source": str(source),
            "sha256": digest,
            "changed_from_components_v2": name in changed,
        }
    for name, piece in manifest["pieces"].items():
        vertices = np.concatenate(
            [meshes[key].vertices for key in piece["constituents"]]
        )
        piece["bounds_m"] = [vertices.min(0).tolist(), vertices.max(0).tolist()]
        piece["dimensions_mm"] = (np.ptp(vertices, axis=0) * 1000).tolist()
        if any(key in changed for key in piece["constituents"]):
            # Original STEP-derived metrics do not describe the derived mesh.
            for key in ("step_sha256", "volume_mm3", "step_valid", "step_solids"):
                piece.pop(key, None)
            piece["derived_cad_candidate"] = True
    manifest["repaired_cad"] = {
        "ready_path": str(Path(ready_path).resolve()),
        "source_scene_sha256": ready["source_scene_sha256"],
        "scene_sha256": ready["scene_sha256"],
        "grid_manifest": ready["manifest"],
        "grid_manifest_sha256": ready["manifest_sha256"],
        "changed_constituents": changed,
        "constituents": provenance,
        "center_metadata": "Original review/frame centers preserved; not recomputed mesh centroids",
        "pin_registration": str(
            ROOT / "outputs/insertion-fix-20261107/pin/consistent-v3/registration.json"
        ),
        "physical_qualification": ready["mechanical_qualification"],
    }
    marker.write_text(json.dumps(manifest, indent=2) + "\n")
    from assembly_policy.so101.wowrobo_layout import InitialLayout

    layout = InitialLayout(output)
    (output / "scene.xml").write_text(ET.tostring(layout.xml_root, encoding="unicode"))
    report = validate_repaired_model(layout.model, ready["source_scene"], sources)
    report["changed_constituents"] = changed
    report["scene_sha256"] = sha256(output / "scene.xml")
    report["pin_mm"] = manifest["pin_mm"]
    (output / "geometry-comparison.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    return output, report


def validate_repaired_model(model, frozen_source_scene, sources):
    """Prove the component adapter preserves51-coordinate frames and geometry."""
    from assembly_policy.so101.wowrobo_layout import mesh_vertices

    frozen = mujoco.MjModel.from_xml_path(str(frozen_source_scene))
    if (model.nq, model.nv, model.nu) != (51, 46, frozen.nu):
        raise ValueError("Unexpected repaired model state layout")
    if [model.body(i).name for i in range(model.nbody)] != [
        frozen.body(i).name for i in range(frozen.nbody)
    ]:
        raise ValueError("Body order changed")
    if [model.joint(i).name for i in range(model.njnt)] != [
        frozen.joint(i).name for i in range(frozen.njnt)
    ]:
        raise ValueError("Joint order changed")
    errors = {}
    for name in (
        "jnt_qposadr",
        "jnt_dofadr",
        "jnt_bodyid",
        "jnt_pos",
        "jnt_axis",
        "body_pos",
        "body_quat",
        "body_parentid",
    ):
        error = float(np.max(np.abs(getattr(model, name) - getattr(frozen, name))))
        errors[name] = error
        if error > 1e-12:
            raise ValueError(f"Fixed model frame changed: {name}={error}")
    a, b = mujoco.MjData(model), mujoco.MjData(frozen)
    # A nontrivial common state validates composed hinge/free-root frames too.
    with np.load(
        ROOT / "assets_local/wowrobo/photo-resets-v3/stages/23_upper_near_train.npz"
    ) as z:
        mujoco.mj_setState(
            frozen, b, z["states"][0], mujoco.mjtState.mjSTATE_FULLPHYSICS
        )
    a.qpos[:] = b.qpos
    mujoco.mj_forward(model, a)
    mujoco.mj_forward(frozen, b)
    maximum = 0.0
    for name in sources:
        av = mesh_vertices(model, a, model.geom(name).id)
        bv = mesh_vertices(frozen, b, frozen.geom(name).id)
        if av.shape != bv.shape:
            raise ValueError(f"Source vertex shape differs: {name}")
        maximum = max(maximum, float(np.max(np.abs(av - bv))))
    if maximum > 1e-10:
        raise ValueError(f"Compiled source geometry differs by {maximum}m")
    helper_maximum = 0.0
    helper_count = 0
    if model.ngeom != frozen.ngeom:
        raise ValueError("Geometry ordering changed")
    for gid in range(model.ngeom):
        if model.geom_type[gid] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mesh_name = model.mesh(model.geom_dataid[gid]).name
        if mesh_name in sources:
            continue
        if mesh_name != frozen.mesh(frozen.geom_dataid[gid]).name:
            raise ValueError("Helper geometry ordering changed")
        av, bv = mesh_vertices(model, a, gid), mesh_vertices(frozen, b, gid)
        if av.shape != bv.shape:
            raise ValueError(f"Helper mesh changed: {mesh_name}")
        helper_maximum = max(helper_maximum, float(np.max(np.abs(av - bv))))
        helper_count += 1
    if helper_maximum > 1e-10:
        raise ValueError(f"Helper or camera world geometry changed: {helper_maximum}m")
    return {
        "nq": model.nq,
        "nv": model.nv,
        "nu": model.nu,
        "fixed_frame_max_errors": errors,
        "target_source_vertex_max_error_m": maximum,
        "helper_mesh_geoms_checked": helper_count,
        "helper_source_vertex_max_error_m": helper_maximum,
        "all16_frozen_mesh_bindings_verified": True,
        "helper_and_camera_geometry": "Original helper/reference meshes and body frames retained",
        "qualification": "Geometry/frame adapter only; no reset/force/retention qualification",
    }
