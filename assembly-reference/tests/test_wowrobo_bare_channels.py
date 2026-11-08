"""The coarse primitive partition must not seal a sampled insertion bore."""

import numpy as np
from assembly_policy.so101.wowrobo_bare_channels import field_cuboids


def contains(boxes, points):
    return np.any(
        [
            np.all(
                np.abs(points - b["position"]) <= np.array(b["half_size"]) + 1e-12,
                axis=1,
            )
            for b in boxes
        ],
        axis=0,
    )


def test_solid_cuboid_compacts_without_material_loss():
    bounds = np.array([[-0.01, -0.02, -0.03], [0.01, 0.02, 0.03]])
    boxes, receipt = field_cuboids(
        bounds,
        lambda p: np.max(np.abs(p) - bounds[1], axis=1),
        pitch=0.002,
        corner_allowance=0,
    )
    assert len(boxes) == 1
    np.testing.assert_allclose(boxes[0]["half_size"], bounds[1])
    assert receipt["material_cells"] == 10 * 20 * 30


def test_bore_and_material_survive_without_merging_across_hole():
    bounds = np.array([[-0.01] * 3, [0.01] * 3])

    def field(p):
        return np.maximum(
            np.max(np.abs(p) - 0.01, axis=1), 0.003 - np.linalg.norm(p[:, :2], axis=1)
        )

    boxes, _ = field_cuboids(bounds, field, pitch=0.001, corner_allowance=0.0001)
    axis = np.column_stack([np.zeros(21), np.zeros(21), np.linspace(-0.01, 0.01, 21)])
    assert not contains(boxes, axis).any()
    assert contains(boxes, np.array([[0.008, 0, 0]])).all()
    assert len(boxes) < 200


def test_open_slide_channel_remains_open():
    bounds = np.array([[-0.01] * 3, [0.01] * 3])

    def field(p):
        block = np.max(np.abs(p) - 0.01, axis=1)
        channel = np.minimum(0.003 - np.abs(p[:, 0]), p[:, 1] + 0.002)
        return np.maximum(block, channel)

    boxes, _ = field_cuboids(bounds, field, pitch=0.001, corner_allowance=0)
    channel = np.column_stack(
        [np.zeros(21), np.full(21, 0.005), np.linspace(-0.01, 0.01, 21)]
    )
    assert not contains(boxes, channel).any()
    assert contains(boxes, np.array([[0.008, 0, 0]])).all()
