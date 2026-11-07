"""Observation semantics must not silently drift under old policy weights."""

import mujoco
import numpy as np
import pytest

from assembly_policy.so101.obs import (
    MATING_BODIES,
    PointTables,
    require_observation_signature,
    surface_budget,
)
from assembly_policy.so101.scene import ASSEMBLY_TASKS, OBS_BODIES, model


@pytest.fixture(scope="module")
def tables():
    return PointTables(ASSEMBLY_TASKS)


def test_only_active_and_local_receivers(tables):
    expected = [
        ("base",),
        ("base", "motor"),
        ("base", "motor", "horn"),
        ("base", "motor", "horn", "screw"),
        ("base", "motor", "horn", "screw", "retainer"),
    ]
    for k, task in enumerate(ASSEMBLY_TASKS):
        assert MATING_BODIES[task] == expected[k]
        ids = tables.which[k].numpy()
        assert all(OBS_BODIES[i] == task for i in ids[:128])
        assert {OBS_BODIES[i] for i in ids[128:]} == set(MATING_BODIES[task]) | (
            set(("driver", "driver_trigger_saddle", "driver_forward", "driver_reverse", "driver_bit"))
            if task == "screw"
            else set()
        )
        assert len(ids) == 384
        assert sum(OBS_BODIES[i].startswith("driver") for i in ids) == (128 if task == "screw" else 0)


def test_distal_finger_samples_exclude_palm(tables):
    m = model()
    d = mujoco.MjData(m)
    mujoco.mj_kinematics(m, d)
    for i, name in enumerate(
        s + "_" + b for s in ("left", "right") for b in ("gripper", "moving_jaw_so101_v1")
    ):
        body = d.body(name)
        site = d.site(name.split("_")[0] + "_gripperframe")
        world = tables.grip[i].numpy() @ body.xmat.reshape(3, 3).T + body.xpos
        local = (world - site.xpos) @ site.xmat.reshape(3, 3)
        assert (local[:, 0] > -0.025001).all()
        assert len(np.unique(tables.grip[i].numpy(), axis=0)) == 16


def test_base_coverage_spans_whole_receiving_body(tables):
    m = model()
    d = mujoco.MjData(m)
    mujoco.mj_kinematics(m, d)
    body = d.body("base")
    vertices = []
    for gid in range(m.ngeom):
        if m.geom_bodyid[gid] != body.id or m.geom_group[gid] != 2:
            continue
        mid = m.geom_dataid[gid]
        start = m.mesh_vertadr[mid]
        count = m.mesh_vertnum[mid]
        world = m.mesh_vert[start : start + count] @ d.geom_xmat[gid].reshape(3, 3).T + d.geom_xpos[gid]
        vertices.extend((world - body.xpos) @ body.xmat.reshape(3, 3))
    sampled = tables.points[0, 128:].numpy()
    assert len(sampled) == 256
    assert (np.ptp(sampled, axis=0) > 0.9 * np.ptp(vertices, axis=0)).all()


def test_old_observations_rejected(tables):
    with pytest.raises(ValueError, match="observation schema"):
        require_observation_signature({}, tables)
    require_observation_signature({"observation_signature": tables.signature}, tables)


def test_tiny_fastener_does_not_get_equal_body_quota(tables):
    for task in ("retainer", "yoke"):
        counts = tables.mating_counts[task]
        assert sum(counts.values()) == 256
        assert 4 <= counts["screw"] <= 8
        assert counts["base"] > counts["screw"] * 5
        assert all(n >= 4 for n in counts.values())


def test_area_budget_preserves_total_and_small_components():
    counts = surface_budget([100, 10, 0.01], 256)
    assert counts.sum() == 256 and counts[-1] == 4
    assert counts[0] > counts[1] > counts[2]


def test_lower_coverage_preset_matches_tool_to_mating(tables):
    low = PointTables(ASSEMBLY_TASKS, n_fix=128)
    assert low.n_pts == 320 and tables.n_pts == 448
    assert all(sum(c.values()) == (64 if t == "screw" else 128) for t, c in low.mating_counts.items())
    assert sum(OBS_BODIES[i].startswith("driver") for i in low.which[2].tolist()) == 64
    assert low.signature != tables.signature
