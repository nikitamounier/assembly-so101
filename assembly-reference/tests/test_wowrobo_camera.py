"""Regression for the photo-corrected camera mount and its optical frame."""

import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import pytest
from scripts.review_wowrobo_components import review_xml

ASSETS = Path(__file__).resolve().parents[1] / "assets_local/wowrobo/components-v2"
pytestmark = pytest.mark.skipif(
    not (ASSETS.parent / "camera-reference/so101_new_calib_camera.xml").exists(),
    reason="Fetch camera reference first",
)


@pytest.fixture(scope="module")
def camera_model():
    root = review_xml(ASSETS)
    return root, mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))


def test_registered_mount_matches_reference_in_same_wrist_frame(camera_model):
    root, m = camera_model
    ref = ET.parse(
        ASSETS.parent / "camera-reference/so101_new_calib_camera.xml"
    ).getroot()
    body = ref.find('.//body[@name="wrist_camera_mount"]')
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    for side in ["left", "right"]:
        ours = root.find(
            f'.//body[@name="{side}_gripper"]/body[@name="{side}_camera_rig"]'
        )
        assert ours is not None
        for field in ["pos", "quat"]:
            np.testing.assert_array_equal(
                np.fromstring(ours.get(field), sep=" "),
                np.fromstring(body.get(field), sep=" "),
            )
        b = m.body(side + "_gripper").id
        c = m.camera(side + "_wrist_camera").id
        R = d.xmat[b].reshape(3, 3)
        eye = R.T @ (d.cam_xpos[c] - d.xpos[b])
        axis = R.T @ (-d.cam_xmat[c].reshape(3, 3)[:, 2])
        # Board/lens are on the raised side of the gripper, not beneath it.
        assert eye[1] > 0.05 and axis[2] < -0.8
        toward = m.site(side + "_gripperframe").pos - eye
        toward /= np.linalg.norm(toward)
        assert axis @ toward > 0.95


def test_camera_is_rigid_to_wrist_not_moving_jaw(camera_model):
    _, m = camera_model
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    for side in ["left", "right"]:
        c = m.camera(side + "_wrist_camera").id
        before = d.cam_xpos[c].copy()
        R = d.cam_xmat[c].copy()
        d.qpos[m.joint(side + "_gripper").qposadr[0]] = 0.8
        mujoco.mj_forward(m, d)
        np.testing.assert_allclose(d.cam_xpos[c], before, atol=1e-12)
        np.testing.assert_allclose(d.cam_xmat[c], R, atol=1e-12)
        d.qpos[m.joint(side + "_wrist_roll").qposadr[0]] = 0.4
        mujoco.mj_forward(m, d)
        assert np.linalg.norm(d.cam_xmat[c] - R) > 0.1


def test_camera_mesh_scales_and_lens_clearance(camera_model):
    root, m = camera_model
    for side in ["left", "right"]:
        mount = root.find(f'asset/mesh[@name="{side}_wrist_camera_mount_so101_v1"]')
        camera = root.find(f'asset/mesh[@name="{side}_wrist_camera_so101_v1"]')
        np.testing.assert_allclose(
            np.fromstring(mount.get("scale"), sep=" "), [0.001] * 3
        )
        np.testing.assert_allclose(np.fromstring(camera.get("scale"), sep=" "), [1] * 3)
        optics = root.find(f'.//camera[@name="{side}_wrist_camera"]')
        assert np.fromstring(optics.get("pos"), sep=" ")[2] > 0.0263
    assert m.ncam == 2
