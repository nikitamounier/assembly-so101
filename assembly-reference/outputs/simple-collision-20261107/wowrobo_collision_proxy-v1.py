"""Opt-in compound-convex collision candidate; visual CAD is never simplified.

This builder does not modify the default factory, observation source or active
experiments. Generated assets and measured qualification belong to a release.
"""

import copy
import hashlib
import json
import multiprocessing
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial import ConvexHull


def _numbers(values):
    return " ".join(format(float(v), ".17g") for v in values)


def _write_obj(path, vertices, faces):
    with path.open("w") as stream:
        for point in vertices:
            stream.write("v " + _numbers(point) + "\n")
        for face in faces:
            stream.write("f " + " ".join(str(int(i) + 1) for i in face) + "\n")


def _parts(vertices, faces, directory, fine, threshold):
    settings = {
        "fine": fine,
        "threshold_m": threshold,
        "seed": 20261107,
        "resolution": 2000,
        "mcts_nodes": 20,
        "mcts_iterations": 150,
        "mcts_max_depth": 3,
        "real_metric": True,
        "weld_visual_seams": True,
        "preprocess_mode": "off",
        "decimate": False,
    }
    key = hashlib.sha256(
        vertices.tobytes()
        + faces.tobytes()
        + json.dumps(settings, sort_keys=True).encode()
    ).hexdigest()
    cache = directory / key
    receipt = cache / "receipt.json"
    if receipt.exists():
        return json.loads(receipt.read_text())
    cache.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    if fine:
        import coacd

        coacd.set_log_level("info")
        import trimesh

        source = trimesh.Trimesh(
            vertices=vertices, faces=faces, process=True, validate=True
        )
        if not source.is_watertight or not source.is_winding_consistent:
            raise ValueError(
                "Critical proxy source must be a closed oriented mesh after welding CAD seams"
            )
        hulls = coacd.run_coacd(
            coacd.Mesh(source.vertices, source.faces),
            preprocess_mode="off",
            threshold=threshold,
            real_metric=True,
            resolution=2000,
            mcts_nodes=20,
            mcts_iterations=150,
            mcts_max_depth=3,
            decimate=False,
            merge=True,
            seed=20261107,
        )
    else:
        hull = ConvexHull(vertices)
        hulls = [(vertices, hull.simplices)]
    paths = []
    for i, (v, f) in enumerate(hulls):
        path = cache / f"hull-{i:03d}.obj"
        # ConvexHull supplies an oriented manifold even when a mesh exporter
        # hands back inconsistent triangle winding. MuJoCo uses the hull.
        h = ConvexHull(v)
        v = np.asarray(v)
        triangles = h.simplices.copy()
        for j, face in enumerate(triangles):
            normal = np.cross(v[face[1]] - v[face[0]], v[face[2]] - v[face[0]])
            if normal @ h.equations[j, :3] < 0:
                triangles[j, [1, 2]] = triangles[j, [2, 1]]
        used, remap = np.unique(triangles, return_inverse=True)
        _write_obj(path, v[used], remap.reshape(-1, 3))
        paths.append(str(path.resolve()))
    result = {
        "key": key,
        "settings": settings,
        "paths": paths,
        "count": len(paths),
        "build_s": time.perf_counter() - started,
    }
    receipt.write_text(json.dumps(result, indent=2) + "\n")
    return result


def _geometry_input(model, gid, threshold):
    name = model.geom(gid).name
    visual_name = name.removesuffix("__contact")
    visual = model.geom(visual_name).id
    mid = int(model.geom_dataid[visual])
    va, vn = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
    fa, fn = model.mesh_faceadr[mid], model.mesh_facenum[mid]
    vertices = np.array(model.mesh_vert[va : va + vn], dtype=np.float64)
    faces = np.array(model.mesh_face[fa : fa + fn], dtype=np.int32)
    body_name = model.body(int(model.geom_bodyid[gid])).name
    fine = not name.startswith("physical_helper_") or any(
        word in body_name for word in ("gripper", "moving_jaw")
    )
    local_threshold = (
        threshold if "printed" in visual_name or visual_name == "pin" else 2 * threshold
    )
    rotation = np.empty(9)
    mujoco.mju_quat2Mat(rotation, model.mesh_quat[mid])
    rotation = rotation.reshape(3, 3)
    raw_vertices = np.round(vertices @ rotation.T + model.mesh_pos[mid], 8)
    return (
        name,
        visual_name,
        visual,
        mid,
        faces,
        fine,
        local_threshold,
        rotation,
        raw_vertices,
    )


def build_candidate(
    baseline,
    runtime_xml,
    output,
    *,
    threshold=0.0005,
    workers=1,
    compact=None,
    fallback=None,
):
    """Replace SDFs with convex compounds in the same body frames.

    Fine decomposition is used for all target parts and both finger assemblies.
    Other helper hardware uses a single convex envelope. The original visual
    meshes, body/joint tree, inertias, actuation, equality constraints and solver
    options are retained. Explicit contact laws expand over compound pieces.
    """
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if not (output / "source-model.mjb").exists():
        mujoco.mj_saveModel(baseline, str(output / "source-model.mjb"))
    root = ET.fromstring(runtime_xml)
    asset = root.find("asset")
    nodes = {
        g.get("name"): (b, g) for b in root.iter("body") for g in b.findall("geom")
    }
    inputs = {
        gid: _geometry_input(baseline, gid, threshold)
        for gid in range(baseline.ngeom)
        if baseline.geom_type[gid] == mujoco.mjtGeom.mjGEOM_SDF
    }
    overrides = {}
    if fallback is not None:
        for gid, data in inputs.items():
            result = fallback(data[0], data)
            if result is not None:
                overrides[gid] = result
    pending, unique = {}, {}
    pool = (
        ProcessPoolExecutor(
            max_workers=workers, mp_context=multiprocessing.get_context("fork")
        )
        if workers > 1
        else None
    )
    if pool is not None:
        for gid, data in inputs.items():
            (
                name,
                visual_name,
                visual,
                mid,
                faces,
                fine,
                local_threshold,
                rotation,
                raw_vertices,
            ) = data
            if not fine or gid in overrides:
                continue
            key = hashlib.sha256(
                raw_vertices.tobytes()
                + faces.tobytes()
                + repr(local_threshold).encode()
            ).hexdigest()
            if key not in unique:
                unique[key] = pool.submit(
                    _parts,
                    raw_vertices,
                    faces,
                    output / "meshes",
                    fine,
                    local_threshold,
                )
            pending[gid] = unique[key]
    mappings, records = {}, []
    for gid, data in inputs.items():
        (
            name,
            visual_name,
            visual,
            mid,
            faces,
            fine,
            local_threshold,
            rotation,
            raw_vertices,
        ) = data
        body, old = nodes[name]
        original_visual = nodes[visual_name][1]
        record = overrides.get(gid)
        if record is None:
            record = (
                pending[gid].result()
                if gid in pending
                else _parts(
                    raw_vertices, faces, output / "meshes", fine, local_threshold
                )
            )
        if compact is not None and fine:
            record = compact(name, record)
        frame_hash = hashlib.sha256(
            baseline.mesh_pos[mid].tobytes() + baseline.mesh_quat[mid].tobytes()
        ).hexdigest()[:12]
        posed = output / "posed" / (record["key"] + "-" + frame_hash)
        posed.mkdir(parents=True, exist_ok=True)
        posed_paths = []
        for path in record["paths"]:
            lines = Path(path).read_text().splitlines()
            v = np.array(
                [
                    [float(x) for x in line.split()[1:4]]
                    for line in lines
                    if line.startswith("v ")
                ]
            )
            f = np.array(
                [
                    [int(x) - 1 for x in line.split()[1:4]]
                    for line in lines
                    if line.startswith("f ")
                ]
            )
            if record.get("source_coordinate_frame") == "sdf-geom-local":
                source_rotation, visual_rotation = np.empty(9), np.empty(9)
                mujoco.mju_quat2Mat(source_rotation, baseline.geom_quat[gid])
                mujoco.mju_quat2Mat(visual_rotation, baseline.geom_quat[visual])
                point = (
                    v @ source_rotation.reshape(3, 3).T
                    + baseline.geom_pos[gid]
                    - baseline.geom_pos[visual]
                ) @ visual_rotation.reshape(3, 3)
            else:
                point = (v - baseline.mesh_pos[mid]) @ rotation
            target = posed / Path(path).name
            _write_obj(target, point, f)
            posed_paths.append(str(target))
        record = dict(
            record,
            paths=posed_paths,
            decomposition_frame="source-CAD",
            collision_frame="compiled-visual",
        )
        names = []
        for i, path in enumerate(record["paths"]):
            mesh_name = f"proxy_{gid}_{i}"
            ET.SubElement(asset, "mesh", name=mesh_name, file=path)
            attributes = {
                k: v
                for k, v in old.attrib.items()
                if k
                not in (
                    "mesh",
                    "type",
                    "pos",
                    "quat",
                    "euler",
                    "axisangle",
                    "xyaxes",
                    "zaxis",
                )
            }
            proxy_name = name + f"__proxy_{i}"
            attributes.update(
                name=proxy_name,
                type="mesh",
                mesh=mesh_name,
                pos=_numbers(baseline.geom_pos[visual]),
                quat=_numbers(baseline.geom_quat[visual]),
                mass="0",
            )
            ET.SubElement(body, "geom", attributes)
            names.append(proxy_name)
        # Keep semantic source names available, but these no longer collide.
        old.set("type", "mesh")
        old.set("mesh", original_visual.get("mesh"))
        old.set("contype", "0")
        old.set("conaffinity", "0")
        for child in list(old):
            old.remove(child)
        mappings[name] = names
        records.append(
            dict(
                name=name,
                visual=visual_name,
                body=body.get("name"),
                fine=fine,
                **record,
            )
        )
        print(
            json.dumps(
                {
                    "geom": name,
                    "fine": fine,
                    "hulls": record["count"],
                    "build_s": record.get("build_s"),
                }
            ),
            flush=True,
        )
    if pool is not None:
        pool.shutdown()
    contact = root.find("contact")
    pair_count = 0
    for pair in list(contact.findall("pair")):
        contact.remove(pair)
        for a in mappings.get(pair.get("geom1"), [pair.get("geom1")]):
            for b in mappings.get(pair.get("geom2"), [pair.get("geom2")]):
                new = copy.deepcopy(pair)
                new.attrib.pop("name", None)
                new.set("geom1", a)
                new.set("geom2", b)
                contact.append(new)
                pair_count += 1
    for node in list(asset.findall("mesh")):
        if node.find("plugin") is not None:
            asset.remove(node)
    extension = root.find("extension")
    if extension is not None:
        root.remove(extension)
    for numeric in root.findall("custom/numeric"):
        if "contact_guard" in numeric.get("name", ""):
            numeric.set("data", "0")
    xml = ET.tostring(root, encoding="unicode")
    path = output / "scene.xml"
    path.write_text(xml)
    candidate = mujoco.MjModel.from_xml_path(str(path))
    unchanged = [
        "body_parentid",
        "body_pos",
        "body_quat",
        "body_mass",
        "body_inertia",
        "body_ipos",
        "body_iquat",
        "jnt_type",
        "jnt_pos",
        "jnt_axis",
        "jnt_qposadr",
        "qpos0",
        "actuator_gainprm",
        "actuator_biasprm",
        "eq_data",
    ]
    for field in unchanged:
        np.testing.assert_array_equal(
            getattr(baseline, field), getattr(candidate, field), err_msg=field
        )
    assert (baseline.nq, baseline.nv, baseline.nu, baseline.neq) == (
        candidate.nq,
        candidate.nv,
        candidate.nu,
        candidate.neq,
    )
    assert not np.any(candidate.geom_type == mujoco.mjtGeom.mjGEOM_SDF)
    receipt = {
        "variant": "compound-convex-v1",
        "threshold_m": threshold,
        "build_workers": workers,
        "scene": str(path),
        "scene_sha256": hashlib.sha256(xml.encode()).hexdigest(),
        "source_xml_sha256": hashlib.sha256(runtime_xml.encode()).hexdigest(),
        "source_module_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "mappings": mappings,
        "records": records,
        "npair": pair_count,
        "nproxy": sum(len(v) for v in mappings.values()),
        "ngeom": candidate.ngeom,
        "unchanged_fields": unchanged,
        "observation_source_changed": False,
        "qualification": "EXPERIMENTAL: contact/retention/insertion validation required",
    }
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return candidate, receipt
