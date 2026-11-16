"""Build an isolated candidate from the actual frozen training runtime."""

import ctypes
import hashlib
import importlib.util
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
from assembly_policy.wowrobo.env import BatchedAssemblyEnv, EnvConfig

root, output = map(Path, sys.argv[1:3])
output.mkdir(parents=True, exist_ok=True)
bank = root / "robotics-assembly-3.0/assets_local/wowrobo/training-accepted-v5"
meta = json.loads((bank / "manifest.json").read_text())
env = None
path = output / "baseline-runtime.xml"
compiled = output / "candidate/source-model.mjb"
if compiled.exists() and path.exists():
    profile = json.loads(Path(meta["profile_path"]).read_text())
    spec = importlib.util.spec_from_file_location(
        "frozen_backend",
        root / "robotics-assembly-3.0/assembly_policy/so101/wowrobo_contact_backend.py",
    )
    backend = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(backend)
    backend.load_backend(profile["grid_manifest"])
    model = mujoco.MjModel.from_binary_path(str(compiled))
else:
    captured = []
    original_serialize = ET.tostring

    def capture_xml(*args, **kwargs):
        result = original_serialize(*args, **kwargs)
        captured.append(result)
        return result

    ET.tostring = capture_xml
    env = BatchedAssemblyEnv(
        meta["task_names"],
        np.zeros(1, np.int64),
        EnvConfig(
            profile_path=meta["profile_path"],
            review_model=meta["review_model"],
            nthread=1,
        ),
    )
    ET.tostring = original_serialize
    assert captured and isinstance(captured[-1], str)
    path.write_text(captured[-1])
    model = env.m
try:
    spec = importlib.util.spec_from_file_location(
        "proxy_builder", output / "wowrobo_collision_proxy.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    cs = importlib.util.spec_from_file_location(
        "proxy_compaction", output / "wowrobo_collision_compaction.py"
    )
    compaction = importlib.util.module_from_spec(cs)
    cs.loader.exec_module(compaction)
    source_data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, source_data)
    native_module = (
        backend
        if "backend" in globals()
        else sys.modules["_wowrobo_training_native_backend"]
    )
    native = native_module.surface_library()
    query = native.wowrobo_sdf_distance
    query.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),
    ]
    query.restype = ctypes.c_double
    batch = ctypes.CDLL(str(output / "distance-batch.so")).proxy_distance_batch
    batch.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),
        ctypes.c_size_t,
        np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),
    ]
    batch.restype = None
    callback = ctypes.cast(query, ctypes.c_void_p)

    def query_world(target, world):
        world = np.ascontiguousarray(world, dtype=np.float64)
        values = np.empty(len(world))
        batch(
            callback,
            model._address,
            source_data._address,
            target,
            world,
            len(world),
            values,
        )
        return values

    parity_target = model.geom("servo_1_case__contact").id
    parity_points = np.random.default_rng(91).normal(0, 0.04, (64, 3))
    scalar = np.array(
        [
            query(
                model._address,
                source_data._address,
                parity_target,
                np.ascontiguousarray(p),
            )
            for p in parity_points
        ]
    )
    np.testing.assert_array_equal(scalar, query_world(parity_target, parity_points))
    (output / "batch-query-parity.json").write_text(
        json.dumps(
            {
                "passed": True,
                "points": 64,
                "source_sha256": hashlib.sha256(
                    (output / "distance_batch.cc").read_bytes()
                ).hexdigest(),
            }
        )
        + "\n"
    )
    oracle_id = hashlib.sha256(path.read_bytes()).hexdigest()

    def distance_for(name):
        visual = model.geom(name.removesuffix("__contact")).id
        target = model.geom(name).id
        mid = model.geom_dataid[visual]
        rotation = np.empty(9)
        mujoco.mju_quat2Mat(rotation, model.mesh_quat[mid])
        rotation = rotation.reshape(3, 3)

        def distance(points):
            local = (points - model.mesh_pos[mid]) @ rotation
            world = (
                local @ source_data.geom_xmat[visual].reshape(3, 3).T
                + source_data.geom_xpos[visual]
            )
            return query_world(target, world)

        return distance

    def field_distance_for(name):
        target = model.geom(name).id

        def distance(points):
            world = (
                points @ source_data.geom_xmat[target].reshape(3, 3).T
                + source_data.geom_xpos[target]
            )
            return query_world(target, world)

        return distance

    def compact(name, record):
        distance = (
            field_distance_for(name)
            if record.get("source_coordinate_frame") == "sdf-geom-local"
            else distance_for(name)
        )
        result = compaction.compact_parts(
            record,
            output / "compacted",
            distance,
            oracle_id + name,
            target=64,
            added_material_guard=0.0005,
        )
        print(
            json.dumps(
                {"compacted": name, "before": record["count"], "after": result["count"]}
            ),
            flush=True,
        )
        return result

    manifest_path = Path(
        json.loads(Path(meta["profile_path"]).read_text())["grid_manifest"]
    )
    grid_manifest = json.loads(manifest_path.read_text())
    grid_dir = Path(
        grid_manifest.get(
            "grid_dir", grid_manifest.get("grid_directory", manifest_path.parent)
        )
    )
    if not grid_dir.is_absolute():
        grid_dir = manifest_path.parent / grid_dir

    def fallback(name, data):
        # These four vendor case meshes are open after welding. Sample their
        # existing native material field rather than inventing CAD end caps.
        if not (
            (name.startswith("servo_") and name.endswith("_case__contact"))
            or name == "shoulder_connector_printed_0__contact"
        ):
            return None
        visual_name = name.removesuffix("__contact")
        item = grid_manifest["meshes"][visual_name]
        grid_id = item.get("grid_id", item.get("id"))
        bounds = compaction.grid_zero_bounds(grid_dir / f"grid_{grid_id}.bin")
        result = compaction.voxel_parts(
            bounds,
            output / "native-case-boxes",
            field_distance_for(name),
            oracle_id + name + "|sdf-local-v1",
            pitch=0.00027
            if name == "shoulder_connector_printed_0__contact"
            else 0.0005,
        )
        result = dict(result, source_coordinate_frame="sdf-geom-local")
        print(
            json.dumps({"native_case_boxes": name, "boxes": result["count"]}),
            flush=True,
        )
        return result

    candidate, receipt = module.build_candidate(
        model,
        path.read_text(),
        output / "candidate",
        workers=4,
        compact=compact,
        fallback=fallback,
    )
    (output / "frozen-input.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(
        json.dumps(
            {
                "complete": True,
                "ngeom": candidate.ngeom,
                "npair": candidate.npair,
                "nproxy": receipt["nproxy"],
            }
        ),
        flush=True,
    )
finally:
    if env is not None:
        env.close()
