"""Collision patches contained in the original small bearing face."""

import numpy as np


def horn_bearing_patches(meta):
    """Four collision patches contained in the original flat washer face."""
    from .fastener import AXIS

    axis = AXIS - np.asarray(meta["horn"]["origin"])
    floor = meta["horn"]["bounds"][1][2] - 0.0015
    result = []
    for i in range(4):
        angle = i * np.pi / 2
        result.append(
            dict(
                name=f"horn_bearing_{i}",
                type="box",
                pos=[axis[0] + 0.00215 * np.cos(angle), axis[1] + 0.00215 * np.sin(angle), floor + 0.0001],
                quat=[np.cos(angle / 2), 0, 0, np.sin(angle / 2)],
                size=[0.00025, 0.00045, 0.0001],
                contype=0,
                conaffinity=0,
                density=0,
                group=3,
                rgba=[0.1, 0.7, 0.8, 0.2],
            )
        )
    return result
