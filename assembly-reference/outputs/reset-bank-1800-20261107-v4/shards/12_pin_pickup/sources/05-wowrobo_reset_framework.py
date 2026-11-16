"""Object-frame grasp geometry and quota-independent reset curricula.

A grasp is a measured site transform, not a symmetric-jaw approximation or an
admission certificate. Every generated pose still needs native physical checks.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np


def relative_grasp(object_position, object_rotation, site_position, site_rotation):
    r = np.asarray(object_rotation, dtype=float).reshape(3, 3)
    return {
        "position": (r.T @ (np.asarray(site_position) - object_position)).tolist(),
        "rotation": (r.T @ np.asarray(site_rotation).reshape(3, 3)).tolist(),
    }


def world_grasp(object_position, object_rotation, grasp):
    r = np.asarray(object_rotation, dtype=float).reshape(3, 3)
    return (
        np.asarray(object_position) + r @ np.asarray(grasp["position"]),
        r @ np.asarray(grasp["rotation"]),
    )


def difficulty_for_slot(sampling, seed, split, rng):
    """Retries retain their slot's curriculum band for arbitrary bank sizes."""
    quota = int(sampling[f"{split}_per_stage"])
    if quota <= 0 or split not in ("train", "eval"):
        raise ValueError("Positive quota and train/eval split required")
    slot = int(seed) % quota
    if split == "eval":
        return (slot + rng.random()) / quota
    bands = sampling.get("train_difficulty", [])
    counts = [int(b["count_per_stage"]) for b in bands]
    if not bands or sum(counts) != quota or min(counts) < 0:
        raise ValueError("Difficulty band counts must sum to the training quota")
    for band, count in zip(bands, counts):
        if slot < count:
            low, high = band["interval"]
            if not 0 <= low < high <= 1:
                raise ValueError("Difficulty intervals must lie in [0,1]")
            return float(rng.uniform(low, high))
        slot -= count
    raise AssertionError("Unallocated slot")


def resize_sampling(sampling, train=60, evaluation=12):
    """Largest-remainder allocation keeps the configured difficulty proportions."""
    if min(train, evaluation) <= 0:
        raise ValueError("Both splits need positive quotas")
    result = json.loads(json.dumps(sampling))
    bands = result["train_difficulty"]
    weights = np.array([b["probability"] for b in bands], dtype=float)
    if not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
        raise ValueError("Invalid difficulty weights")
    exact = train * weights / weights.sum()
    counts = np.floor(exact).astype(int)
    for index in np.argsort(-(exact - counts), kind="stable")[: train - counts.sum()]:
        counts[index] += 1
    for band, count in zip(bands, counts):
        band["count_per_stage"] = int(count)
    result.update(train_per_stage=train, eval_per_stage=evaluation)
    return result


def grasp_id(record):
    # Measured angular opening and offset encode the real asymmetric SO101 jaw.
    content = {k: record[k] for k in ("anchor", "transform", "calibration")}
    return (
        record["anchor"]
        + ":"
        + hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()[:16]
    )


def check_library(library):
    if library.get("schema_version") != 1:
        raise ValueError("Unsupported grasp library")
    for key, record in library["grasps"].items():
        if key != grasp_id(record):
            raise ValueError("Grasp identity mismatch")
        r = np.asarray(record["transform"]["rotation"])
        p = np.asarray(record["transform"]["position"])
        if (
            r.shape != (3, 3)
            or p.shape != (3,)
            or not np.isfinite(p).all()
            or not np.allclose(r.T @ r, np.eye(3), atol=1e-7)
            or not np.isclose(np.linalg.det(r), 1, atol=1e-7)
        ):
            raise ValueError("Invalid object-relative grasp transform")
        c = record["calibration"]
        if c["span_site_axis"] != 2 or not np.isfinite(c["opening_radians"]):
            raise ValueError("Expected native SO101 rotational jaw calibration")
    for bindings in library["bindings"].values():
        for key in bindings.values():
            if key not in library["grasps"]:
                raise ValueError("Missing bound grasp")
    return library
