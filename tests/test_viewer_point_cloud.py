"""The viewer exports the current assembly's exact ordered encoder input."""

import hashlib

import mujoco
import numpy as np
import pytest

from assembly_policy.so101.wowrobo_obs import GRIPS, WOW_BASE, WOW_TASKS
from assembly_policy.viewer.point_cloud import PolicyPointCloud


@pytest.fixture(scope="module")
def observation():
    # Independent, portable current-assembly fixture, including an articulated
    # case. No legacy assets or native plugins are required to verify encoding.
    bodies = [
        f'<body name="{n}" pos="{i * 0.04} 0 0"><freejoint/><geom name="{n}_surface" type="mesh" mesh="surface" group="2"/></body>'
        for i, n in enumerate((WOW_BASE, *WOW_TASKS))
    ]
    bodies[4] = bodies[4].replace(
        "</body>",
        '<body name="case_pivot"><joint name="case_hinge" axis="0 1 0"/><geom type="mesh" mesh="surface" pos=".02 0 0" group="2"/></body></body>',
    )
    for i, name in enumerate(GRIPS):
        side = name.split("_")[0]
        site = f'<site name="{side}_gripperframe"/>' if name.endswith("_gripper") else ""
        bodies.append(
            f'<body name="{name}" pos="0 {i * 0.04} .1"><freejoint/><geom name="{name}_surface" type="mesh" mesh="surface" group="2"/>{site}</body>'
        )
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><asset><mesh name="surface" vertex="0 0 0 .01 0 0 0 .01 0 0 0 .01"/></asset><worldbody>'
        + "".join(bodies)
        + "</worldbody></mujoco>"
    )
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)
    config = dict(
        tasks={
            task: dict(
                reference_qpos=model.qpos0.tolist(),
                receiver=WOW_BASE,
                geom=task + "_surface",
                mesh_center_local=[0, 0, 0],
                center_receiver_local=[0, 0, 0],
                rotation_receiver_local=np.eye(3).tolist(),
                feature_count=64,
                feature_geoms=[WOW_BASE + "_surface"],
                feature_distance_m=0.02,
            )
            for task in WOW_TASKS
        }
    )
    provider = PolicyPointCloud(model, "wowrobo-layout", point_config=config)
    yield model, data, provider


def entry(task):
    return dict(id="selected-state", task="20_upper_transport", stage=dict(task=task))


def test_exact_ordered_tensor_without_legacy_model(observation, monkeypatch):
    import assembly_policy.so101.obs as legacy

    monkeypatch.setattr(legacy, "model", lambda: pytest.fail("Loaded old assembly"))
    model, data, provider = observation
    before = data.qpos.copy()
    cloud = provider.snapshot(data, entry("upper_arm_module"))
    tables = provider.tables
    task = WOW_TASKS.index("upper_arm_module")
    ids = tables.which[task].numpy()
    world = np.array(
        [
            data.body(tables.names[i]).xpos + data.body(tables.names[i]).xmat.reshape(3, 3) @ p
            for p, i in zip(tables.points[task].numpy(), ids)
        ]
    )
    grip = np.concatenate(
        [
            data.body(name).xpos + tables.grip[i].numpy() @ data.body(name).xmat.reshape(3, 3).T
            for i, name in enumerate(GRIPS)
        ]
    )
    expected = 10 * (
        (np.concatenate([world, grip]) - data.body(GRIPS[0]).xpos) @ data.body(GRIPS[0]).xmat.reshape(3, 3)
    )
    actual = np.array(cloud["flat"], np.float32).reshape(448, 3)
    np.testing.assert_allclose(actual, expected, atol=1e-6)
    frame = cloud["frame_pose_world"]
    overlay = (
        actual / cloud["scale"] @ np.array(frame["rotation_row_major"]).reshape(3, 3).T + frame["position"]
    )
    np.testing.assert_allclose(overlay, np.concatenate([world, grip]), atol=1e-7)
    np.testing.assert_array_equal(cloud["points"], actual)
    np.testing.assert_array_equal(data.qpos, before)
    assert cloud["state_id"] == "selected-state" and cloud["preview"]
    assert len(cloud["bodies"]) == 448
    assert [g["count"] for g in cloud["groups"]] == [128, 256, 64]
    assert cloud["tensor_sha256"] == hashlib.sha256(actual.tobytes()).hexdigest()
    repeated = provider.snapshot(data, entry("upper_arm_module"))
    assert repeated == cloud


def test_articulated_points_follow_own_body(observation):
    model, data, provider = observation
    cloud = provider.snapshot(data, entry("upper_arm_module"))
    before = np.array(cloud["points"])
    address = model.joint("case_hinge").qposadr[0]
    data.qpos[address] = 0.6
    mujoco.mj_kinematics(model, data)
    after = np.array(provider.snapshot(data, entry("upper_arm_module"))["points"])
    pivot = np.array(cloud["bodies"]) == "case_pivot"
    assert pivot.any() and np.max(abs(after[pivot] - before[pivot])) > 0.01
    np.testing.assert_array_equal(after[~pivot], before[~pivot])
    data.qpos[address] = 0
    mujoco.mj_kinematics(model, data)


def test_setup_has_no_invented_task_or_stale_points(observation):
    _, data, provider = observation
    assert provider.snapshot(data, entry("forearm_module"))["available"]
    setup = provider.snapshot(data, entry("initial_scene"))
    assert not setup["available"] and "points" not in setup
    assert not PolicyPointCloud(provider.model, "so101").snapshot(data, entry("motor"))["available"]


def test_contract_rejects_changed_reference_frames(observation):
    from assembly_policy.so101.wowrobo_obs import PointTables

    model, _, provider = observation
    config = dict(provider.point_config, kinematic_fingerprint="different")
    with pytest.raises(ValueError, match="reference frames differ"):
        PointTables(model, config=config)


def test_all_samples_remain_on_source_mesh_surfaces(observation):
    import igl

    model, data, provider = observation
    provider.snapshot(data, entry("forearm_module"))
    tables = provider.tables
    for k in range(len(WOW_TASKS)):
        points = np.array(
            [
                data.body(tables.names[i]).xpos + data.body(tables.names[i]).xmat.reshape(3, 3) @ p
                for p, i in zip(tables.points[k].numpy(), tables.which[k].numpy())
            ]
        )
        for gid in set(tables.provenance[k]):
            chosen = np.asarray(tables.provenance[k]) == gid
            native = (points[chosen] - data.geom_xpos[gid]) @ data.geom_xmat[gid].reshape(3, 3)
            mid = model.geom_dataid[gid]
            va, vn = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
            fa, fn = model.mesh_faceadr[mid], model.mesh_facenum[mid]
            squared = igl.point_mesh_squared_distance(
                np.ascontiguousarray(native),
                np.asarray(model.mesh_vert[va : va + vn], float),
                np.asarray(model.mesh_face[fa : fa + fn], np.int64),
            )[0]
            assert np.sqrt(squared).max() < 1e-7
        budget = tables.receiver_budget[WOW_TASKS[k]]
        assert budget["feature"] + budget["nearby"] + budget["context"] == 256
