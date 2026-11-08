"""Construct primitive channels from the frozen material oracle, offline only."""

import ctypes
import hashlib
import importlib.util
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
from assembly_policy.so101.wowrobo_obs import GRIPS
from assembly_policy.wowrobo.obs import PointTables

root, output = map(Path, sys.argv[1:3])
output.mkdir(parents=True, exist_ok=True)
old = output.parent / "simple-collision-20261107"
base = output.parent / "bare-assembly-20261107"
meta = json.loads(
    (
        root
        / "robotics-assembly-3.0/assets_local/wowrobo/training-accepted-v6/manifest.json"
    ).read_text()
)
profile = json.loads(Path(meta["profile_path"]).read_text())


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


backend = load(
    "native_backend",
    root / "robotics-assembly-3.0/assembly_policy/so101/wowrobo_contact_backend.py",
)
backend.load_backend(profile["grid_manifest"])
source = mujoco.MjModel.from_binary_path(str(old / "candidate/source-model.mjb"))
data = mujoco.MjData(source)
mujoco.mj_kinematics(source, data)
channels = load("channels", output / "wowrobo_bare_channels.py")
compaction = load("bounds", old / "wowrobo_collision_compaction.py")
bare = load("bare", base / "wowrobo_bare.py")
tables = PointTables(meta["task_names"], review_model=meta["review_model"])
pads = {name: tables.grip[i].numpy() for i, name in enumerate(GRIPS)}
manifest_path = Path(profile["grid_manifest"])
manifest = json.loads(manifest_path.read_text())
grid_dir = Path(
    manifest.get("grid_dir", manifest.get("grid_directory", manifest_path.parent))
)
if not grid_dir.is_absolute():
    grid_dir = manifest_path.parent / grid_dir
native = backend.surface_library().wowrobo_sdf_distance
batch = ctypes.CDLL(str(old / "distance-batch.so")).proxy_distance_batch
array = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
batch.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
    array,
    ctypes.c_size_t,
    array,
]
batch.restype = None
callback = ctypes.cast(native, ctypes.c_void_p)
replacements = {}
records = []
cache = output / "cache"
cache.mkdir(exist_ok=True)
for gid in range(source.ngeom):
    if source.geom_type[gid] != mujoco.mjtGeom.mjGEOM_SDF:
        continue
    name = source.geom(gid).name
    visual = name.removesuffix("__contact")
    vid = source.geom(visual).id
    mid = source.geom_dataid[vid]
    robot = name.startswith("physical_helper_")
    body = source.body(int(source.geom_bodyid[gid])).name
    pad = (
        robot
        and body in pads
        and any(
            word in source.mesh(mid).name
            for word in ("wrist_roll_follower", "moving_jaw")
        )
    )
    if robot and not pad:
        continue
    item = manifest["meshes"][visual]
    grid_id = item.get("grid_id", item.get("id"))
    bounds = compaction.grid_zero_bounds(grid_dir / f"grid_{grid_id}.bin")
    pitch = 0.0005 if pad or visual == "pin" else 0.00075
    if pad:
        rotation = np.empty(9)
        mujoco.mju_quat2Mat(rotation, source.geom_quat[gid])
        rotation = rotation.reshape(3, 3)
        points = (pads[body] - source.geom_pos[gid]) @ rotation
        bounds = np.array(
            [
                np.maximum(bounds[0], points.min(0) - 0.001),
                np.minimum(bounds[1], points.max(0) + 0.001),
            ]
        )
    settings = {
        "name": name,
        "bounds": bounds.tolist(),
        "pitch": pitch,
        "corner_allowance": 0.00015,
        "source_model": meta["model_signature"],
        "code": hashlib.sha256(
            (output / "wowrobo_bare_channels.py").read_bytes()
        ).hexdigest(),
    }
    key = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
    path = cache / (key + ".json")
    if path.exists():
        record = json.loads(path.read_text())
    else:

        def query(points, gid=gid):
            world = np.ascontiguousarray(
                points @ data.geom_xmat[gid].reshape(3, 3).T + data.geom_xpos[gid]
            )
            values = np.empty(len(world))
            batch(
                callback, source._address, data._address, gid, world, len(world), values
            )
            return values

        boxes, stats = channels.field_cuboids(
            bounds, query, pitch=pitch, corner_allowance=0.00015
        )
        record = {"name": name, "boxes": boxes, "settings": settings, "stats": stats}
        path.write_text(json.dumps(record) + "\n")
    raw_boxes = record["boxes"]
    volume = np.array([np.prod(np.array(b["half_size"]) * 2) for b in raw_boxes])
    order = np.argsort(-volume)
    count = max(
        16 if pad else 32,
        int(np.searchsorted(np.cumsum(volume[order]), 0.90 * volume.sum())) + 1,
    )
    chosen = set(map(int, order[:count]))
    # Retain small rails/pad tips sampled by the unchanged CAD observation,
    # rather than choosing solely by volume. Removing boxes cannot seal voids.
    if pad:
        important = pads[body]
    else:
        body_index = tables.names.index(body)
        important = tables.points.numpy()[tables.which.numpy() == body_index]
        important = np.unique(np.round(important, 8), axis=0)
        if len(important) > 32:
            important = important[np.linspace(0, len(important) - 1, 32, dtype=int)]
    rotation = np.empty(9)
    mujoco.mju_quat2Mat(rotation, source.geom_quat[gid])
    rotation = rotation.reshape(3, 3)
    important = (important - source.geom_pos[gid]) @ rotation
    centers = np.array([b["position"] for b in raw_boxes])
    half = np.array([b["half_size"] for b in raw_boxes])
    for point in important:
        distance = np.linalg.norm(np.maximum(np.abs(point - centers) - half, 0), axis=1)
        chosen.add(int(np.argmin(distance)))
    selected_boxes = [raw_boxes[i] for i in sorted(chosen)]
    record = dict(
        record,
        boxes=selected_boxes,
        pruning={
            "original_boxes": len(raw_boxes),
            "retained_boxes": len(selected_boxes),
            "retained_voxel_volume_fraction": float(
                volume[list(chosen)].sum() / volume.sum()
            ),
            "method": "volume core plus original CAD feature/pad witnesses; deletion only",
        },
    )
    replacements[name] = record["boxes"]
    records.append(record)
    print(json.dumps(dict(name=name, **record["pruning"])), flush=True)
# Recover tiny functional surfaces from known TRAIN/native control witnesses.
patch_module = load("patches", output / "wowrobo_bare_patches.py")
patch_counts = {}
patch_keys = set()
patch_rejected = 0


def world_query(gid, world):
    world = np.ascontiguousarray(world, dtype=np.float64)
    values = np.empty(len(world))
    batch(callback, source._address, data._address, gid, world, len(world), values)
    return values


def add_contact(contact):
    global patch_rejected
    for side, name in enumerate(contact["geoms"]):
        gid = source.geom(name).id
        if source.geom_type[gid] != mujoco.mjtGeom.mjGEOM_SDF:
            continue
        body = source.body(int(source.geom_bodyid[gid])).name
        if name.startswith("physical_helper_") and body not in GRIPS:
            continue
        patch = patch_module.patch_from_contact(
            source, data, gid, contact, side, world_query
        )
        if patch is None:
            patch_rejected += 1
            continue
        key = (
            name,
            tuple(np.round(np.array(patch["position"]) / 0.0003).astype(int)),
            tuple(np.round(np.array(patch["quaternion"]) / 0.05).astype(int)),
        )
        if key in patch_keys:
            continue
        patch_keys.add(key)
        replacements.setdefault(name, []).append(patch)
        patch_counts[name] = patch_counts.get(name, 0) + 1


bank = root / "robotics-assembly-3.0/assets_local/wowrobo/training-accepted-v6"
near_states = {}
for task in meta["task_names"]:
    z = np.load(bank / (task + "_train.npz"))
    for stage in np.unique(z["stage"]):
        ids = [
            i
            for i in np.flatnonzero(z["stage"] == stage)
            if np.all(z["states"][i, -3:] == 1)
        ]
        if not ids:
            continue
        state = z["states"][ids[0]]
        if "near" in str(stage):
            near_states[task] = state.copy()
        mujoco.mj_setState(
            source,
            data,
            state[: mujoco.mj_stateSize(source, mujoco.mjtState.mjSTATE_FULLPHYSICS)],
            mujoco.mjtState.mjSTATE_FULLPHYSICS,
        )
        data.ctrl[:] = state[mujoco.mj_stateSize(source,mujoco.mjtState.mjSTATE_FULLPHYSICS)+source.neq:mujoco.mj_stateSize(source,mujoco.mjtState.mjSTATE_FULLPHYSICS)+source.neq+source.nu]
        mujoco.mj_forward(source, data)
        for con in data.contact:
            names = [source.geom(int(g)).name for g in con.geom]
            bodies = [
                source.body(int(source.geom_bodyid[int(g)])).name for g in con.geom
            ]
            if con.dist > 0.0001 or not any(b in GRIPS for b in bodies):
                continue
            add_contact(
                {
                    "geoms": names,
                    "pos": con.pos.tolist(),
                    "normal": con.frame[:3].tolist(),
                    "dist": float(con.dist),
                }
            )
# Six missed blocked controls are construction witnesses, not held-out tests.
old_checks = output / "functional-channels-check.json"
if old_checks.exists():
    for row in json.loads(old_checks.read_text())["insertion_controls"]:
        if not row["missed_block"]:
            continue
        state = near_states[row["task"]].copy()
        adr = int(source.joint(row["task"] + "_free").qposadr[0])
        state[1 + adr : 1 + adr + 3] += row["offset_world_m"]
        mujoco.mj_setState(
            source,
            data,
            state[: mujoco.mj_stateSize(source, mujoco.mjtState.mjSTATE_FULLPHYSICS)],
            mujoco.mjtState.mjSTATE_FULLPHYSICS,
        )
        mujoco.mj_kinematics(source, data)
        for contact in row["contacts"][0]:
            add_contact(contact)
print(
    json.dumps({"surface_patches": patch_counts, "rejected_patches": patch_rejected}),
    flush=True,
)
# The old bare builder is a frozen input, not modified or overwritten.
_, seed_receipt = bare.build(
    (old / "baseline-runtime.xml").read_text(),
    source,
    output / "seed.xml",
    finger_points=pads,
)
xml = ET.parse(output / "seed.xml").getroot()
mapping, groups = channels.apply_cuboids(xml, source, replacements)
scene = output / "scene.xml"
scene.write_text(ET.tostring(xml, encoding="unicode"))
model = mujoco.MjModel.from_xml_path(str(scene))
mujoco.mj_saveModel(model, str(output / "model.mjb"))
for attr in (
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
    "jnt_range",
    "actuator_gainprm",
    "actuator_biasprm",
    "eq_data",
    "qpos0",
):
    np.testing.assert_array_equal(
        getattr(model, attr), getattr(source, attr), err_msg=attr
    )
assert model.nplugin == 0 and not np.any(model.geom_type == mujoco.mjtGeom.mjGEOM_SDF)
receipt = dict(
    seed_receipt,
    variant="bare-open-channels-v2",
    scene_sha256=hashlib.sha256(scene.read_bytes()).hexdigest(),
    ngeom=model.ngeom,
    inter_part_collisions=True,
    self_collision_excluded_components=groups,
    mappings=mapping,
    records=records,
    construction="native-field occupied cuboids with center+8corner tests, no merges across unoccupied cells",
    contact_patches=patch_counts,
    qualification="EXPERIMENTAL: slide and retention probes pending",
)
(output / "scene.json").write_text(json.dumps(receipt, indent=2) + "\n")
# Compatibility paths for the pre-existing independent functional probe.
(output / "candidate").mkdir(exist_ok=True)
for name, target in [
    ("source-model.mjb", old / "candidate/source-model.mjb"),
    ("scene.xml", scene),
]:
    link = output / "candidate" / name
    if not link.exists():
        link.symlink_to(target)
(output / "candidate/receipt.json").write_text(json.dumps({"mappings": mapping}) + "\n")
print(
    json.dumps(
        {
            "complete": True,
            "ngeom": model.ngeom,
            "primitive_geoms": sum(map(len, mapping.values())),
            "scene_sha256": receipt["scene_sha256"],
        }
    ),
    flush=True,
)
