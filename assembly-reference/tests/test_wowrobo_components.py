"""Photo-scope, units, pin topology, and articulation checks after CAD generation."""

import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
import pytest
import trimesh
from scripts.review_wowrobo_components import review_xml

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "assets_local/wowrobo/components-v2"
pytestmark = pytest.mark.skipif(
    not (FOLDER / "manifest.json").exists(), reason="Generate WowRobo CAD first"
)


@pytest.fixture(scope="module")
def bundle():
    return json.loads((FOLDER / "manifest.json").read_text())


def test_exact_photo_scope_and_constituent_integrity(bundle):
    assert list(bundle["pieces"]) == [
        "base_servo",
        "base_side_cover",
        "shoulder_connector",
        "upper_arm_module",
        "forearm_module",
        "assembly_pin",
    ]
    assert [p["servos"] for p in bundle["pieces"].values()] == [
        [1],
        [],
        [],
        [2],
        [3, 4],
        [],
    ]
    assert bundle["structural_piece_count"] == 5 and bundle["target_piece_count"] == 6
    assert bundle["pieces"]["base_servo"]["printed"] == ["Base_SO101 v12:1"]
    assert bundle["pieces"]["base_side_cover"]["printed"] == ["Base_08p v18:1"]
    assert len(bundle["pieces"]["shoulder_connector"]["printed"]) == 2
    assert bundle["step_units"] == "mm" and bundle["mesh_units"] == "m"
    for name, piece in bundle["pieces"].items():
        assert (
            hashlib.sha256((FOLDER / f"{name}.step").read_bytes()).hexdigest()
            == piece["step_sha256"]
        )
        for key in piece["constituents"]:
            item = bundle["constituents"][key]
            assert (
                hashlib.sha256((FOLDER / item["mesh"]).read_bytes()).hexdigest()
                == item["sha256"]
            )
    assert all(
        "gripper" not in name and "wrist_roll" not in name
        for name in bundle["constituents"]
    )


def test_pin_has_requested_dimensions_and_one_closed_surface(bundle):
    pin = trimesh.load_mesh(FOLDER / "assembly_pin.stl", process=True)
    assert bundle["stl_units"] == "mm"
    pin.apply_scale(0.001)
    np.testing.assert_allclose(pin.extents, [0.012, 0.012, 0.05], atol=1e-9)
    assert pin.is_watertight and pin.is_winding_consistent
    assert len(pin.split()) == 1
    np.testing.assert_allclose(
        np.unique(np.round(pin.vertices[:, 2], 9)), [0, 0.028, 0.05]
    )
    shaft = pin.vertices[np.isclose(pin.vertices[:, 2], 0.05)]
    assert np.count_nonzero(np.linalg.norm(shaft[:, :2], axis=1) > 0.001) == 128
    analytic = np.pi * (0.006**2 * 0.028 + 0.00135**2 * 0.022)
    np.testing.assert_allclose(
        pin.volume, analytic * np.sin(2 * np.pi / 128) / (2 * np.pi / 128), rtol=1e-7
    )
    assert bundle["pin_mm"]["grip_length"] == 28


def test_preassembled_servos_keep_four_internal_pivots():
    import xml.etree.ElementTree as ET

    model = mujoco.MjModel.from_xml_string(
        ET.tostring(review_xml(FOLDER), encoding="unicode")
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    assert model.nu == 12 and model.ncam == 2
    assert sum(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE) == 6
    assert (
        sum(
            (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) or "").startswith(
                "target_servo_"
            )
            for i in range(model.njnt)
        )
        == 4
    )
    for servo in (1, 2, 3, 4):
        moving = f"servo_{servo}_" + ("case" if servo in (2, 3) else "horns")
        fixed = f"servo_{servo}_" + ("horns" if servo in (2, 3) else "case")
        mi, fi = model.geom(moving).id, model.geom(fixed).id
        before_m = data.geom_xmat[mi].copy()
        before_f = data.geom_xmat[fi].copy()
        joint = model.joint(f"target_servo_{servo}_shaft")
        data.qpos[joint.qposadr[0]] = 0.35
        mujoco.mj_forward(model, data)
        assert np.linalg.norm(data.geom_xmat[mi] - before_m) > 0.1
        np.testing.assert_allclose(data.geom_xmat[fi], before_f, atol=1e-12)
        data.qpos[joint.qposadr[0]] = 0
        mujoco.mj_forward(model, data)
    for side in ["left", "right"]:
        camera = model.camera(side + "_wrist_camera").id
        before = data.cam_xmat[camera].copy()
        data.qpos[model.joint(side + "_wrist_roll").qposadr[0]] = 0.4
        mujoco.mj_forward(model, data)
        assert np.linalg.norm(data.cam_xmat[camera] - before) > 0.1
    # A visual-only review must not accidentally advertise convex mesh hulls as mating physics.
    assert np.all(model.geom_contype == 0) and np.all(model.geom_conaffinity == 0)
