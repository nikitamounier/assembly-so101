"""Assert actual CAD visuals, mechanics, integration and contact-law contracts."""

import importlib.util
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

root, output = map(Path, sys.argv[1:3])
meta = json.loads(
    (
        root
        / "robotics-assembly-3.0/assets_local/wowrobo/training-accepted-v5/manifest.json"
    ).read_text()
)
profile = json.loads(Path(meta["profile_path"]).read_text())
spec = importlib.util.spec_from_file_location(
    "frozen_backend",
    root / "robotics-assembly-3.0/assembly_policy/so101/wowrobo_contact_backend.py",
)
backend = importlib.util.module_from_spec(spec)
spec.loader.exec_module(backend)
backend.load_backend(profile["grid_manifest"])
a = mujoco.MjModel.from_binary_path(str(output / "candidate/source-model.mjb"))
b = mujoco.MjModel.from_xml_path(str(output / "candidate/scene.xml"))
receipt = json.loads((output / "candidate/receipt.json").read_text())
fields = [
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
    "jnt_range",
    "qpos0",
    "dof_damping",
    "dof_frictionloss",
    "dof_armature",
    "actuator_trnid",
    "actuator_gainprm",
    "actuator_biasprm",
    "actuator_ctrlrange",
    "actuator_forcerange",
    "eq_data",
    "eq_solref",
    "eq_solimp",
    "eq_active0",
]
for name in fields:
    np.testing.assert_array_equal(getattr(a, name), getattr(b, name), err_msg=name)
options = []
for name in dir(a.opt):
    if name.startswith("_"):
        continue
    x, y = getattr(a.opt, name), getattr(b.opt, name)
    if callable(x):
        continue
    np.testing.assert_array_equal(x, y, err_msg="option." + name)
    options.append(name)
visuals = 0
for gid in range(a.ngeom):
    if (
        a.geom_type[gid] != mujoco.mjtGeom.mjGEOM_MESH
        or a.geom_group[gid] not in [0, 2]
        or a.geom_rgba[gid, 3] <= 0
    ):
        continue
    name = a.geom(gid).name
    j = b.geom(name).id
    for attr in ["geom_pos", "geom_quat", "geom_rgba", "geom_group", "geom_bodyid"]:
        np.testing.assert_array_equal(
            getattr(a, attr)[gid], getattr(b, attr)[j], err_msg=name + ":" + attr
        )
    x, y = int(a.geom_dataid[gid]), int(b.geom_dataid[j])
    for attr, start, count in [
        ("mesh_vert", "mesh_vertadr", "mesh_vertnum"),
        ("mesh_face", "mesh_faceadr", "mesh_facenum"),
    ]:
        vx = getattr(a, attr)[
            getattr(a, start)[x] : getattr(a, start)[x] + getattr(a, count)[x]
        ]
        vy = getattr(b, attr)[
            getattr(b, start)[y] : getattr(b, start)[y] + getattr(b, count)[y]
        ]
        np.testing.assert_array_equal(vx, vy, err_msg=name + ":" + attr)
    visuals += 1
reverse = {p: old for old, proxies in receipt["mappings"].items() for p in proxies}
base_pairs = {
    frozenset((a.geom(int(g1)).name, a.geom(int(g2)).name)): i
    for i, (g1, g2) in enumerate(zip(a.pair_geom1, a.pair_geom2, strict=True))
}
for i, (g1, g2) in enumerate(zip(b.pair_geom1, b.pair_geom2, strict=True)):
    names = [b.geom(int(g)).name for g in [g1, g2]]
    old = frozenset(reverse.get(n, n) for n in names)
    j = base_pairs[old]
    for attr in [
        "pair_dim",
        "pair_friction",
        "pair_solref",
        "pair_solimp",
        "pair_margin",
        "pair_gap",
    ]:
        np.testing.assert_array_equal(
            getattr(a, attr)[j], getattr(b, attr)[i], err_msg=str(old) + ":" + attr
        )
for name, proxies in receipt["mappings"].items():
    original = a.geom(name).id
    for proxy in proxies:
        g = b.geom(proxy).id
        for attr in [
            "geom_contype",
            "geom_conaffinity",
            "geom_condim",
            "geom_friction",
            "geom_solref",
            "geom_solimp",
        ]:
            np.testing.assert_array_equal(
                getattr(a, attr)[original],
                getattr(b, attr)[g],
                err_msg=name + ":" + attr,
            )
assert not np.any(b.geom_type == mujoco.mjtGeom.mjGEOM_SDF)
result = {
    "passed": True,
    "mechanics_fields": fields,
    "options": options,
    "visual_meshes_unchanged": visuals,
    "expanded_pairs_checked": b.npair,
    "collision_proxies_checked": len(reverse),
    "source_model_signature": meta["model_signature"],
}
(output / "contract-check.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result), flush=True)
