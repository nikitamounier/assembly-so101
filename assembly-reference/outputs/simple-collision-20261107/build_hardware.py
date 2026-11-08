"""Build the conservative fallback without rebaking any assets."""

import importlib.util
import json
import sys
from pathlib import Path

import mujoco

root, output = map(Path, sys.argv[1:3])
meta = json.loads((output / "frozen-input.json").read_text())
profile = json.loads(Path(meta["profile_path"]).read_text())
spec = importlib.util.spec_from_file_location(
    "backend",
    root / "robotics-assembly-3.0/assembly_policy/so101/wowrobo_contact_backend.py",
)
backend = importlib.util.module_from_spec(spec)
spec.loader.exec_module(backend)
backend.load_backend(profile["grid_manifest"])
m = mujoco.MjModel.from_binary_path(str(output / "candidate/source-model.mjb"))
spec = importlib.util.spec_from_file_location(
    "builder", output / "wowrobo_collision_proxy.py"
)
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)
path, receipt = builder.build_hardware_candidate(
    m,
    (output / "baseline-runtime.xml").read_text(),
    json.loads((output / "candidate/receipt.json").read_text()),
    output / "hardware",
)
c = mujoco.MjModel.from_xml_path(str(path))
import numpy as np

assert c.ngeom == m.ngeom and c.npair == m.npair
for name in (
    "body_pos",
    "body_quat",
    "body_mass",
    "body_inertia",
    "body_ipos",
    "body_iquat",
    "jnt_type",
    "jnt_range",
    "actuator_gainprm",
    "actuator_biasprm",
    "eq_data",
    "qpos0",
):
    np.testing.assert_array_equal(getattr(m, name), getattr(c, name), err_msg=name)
for name in receipt["preserved"]:
    x, y = m.geom(name).id, c.geom(name).id
    assert x == y
    for attr in (
        "geom_type",
        "geom_pos",
        "geom_quat",
        "geom_friction",
        "geom_solref",
        "geom_solimp",
        "geom_contype",
        "geom_conaffinity",
    ):
        np.testing.assert_array_equal(
            getattr(m, attr)[x], getattr(c, attr)[y], err_msg=name + ":" + attr
        )
    a, b = m.geom_dataid[x], c.geom_dataid[y]
    for attr, adr, num in [
        ("mesh_vert", "mesh_vertadr", "mesh_vertnum"),
        ("mesh_face", "mesh_faceadr", "mesh_facenum"),
    ]:
        np.testing.assert_array_equal(
            getattr(m, attr)[
                getattr(m, adr)[a] : getattr(m, adr)[a] + getattr(m, num)[a]
            ],
            getattr(c, attr)[
                getattr(c, adr)[b] : getattr(c, adr)[b] + getattr(c, num)[b]
            ],
            err_msg=name + ":" + attr,
        )
for name in (
    "pair_geom1",
    "pair_geom2",
    "pair_friction",
    "pair_solref",
    "pair_solimp",
    "pair_dim",
    "pair_margin",
    "pair_gap",
):
    np.testing.assert_array_equal(getattr(m, name), getattr(c, name), err_msg=name)
receipt["contract_passed"] = True
receipt["remaining_sdf_geoms"] = int(np.sum(c.geom_type == mujoco.mjtGeom.mjGEOM_SDF))
mujoco.mj_saveModel(c, str(output / "hardware/model.mjb"))
(output / "hardware/receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps(receipt), flush=True)
