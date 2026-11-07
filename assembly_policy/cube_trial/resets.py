"""Portable physics identity and complete integration-state cube bank validation."""

import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from .core import ASSETS, STATE_SPEC, build_xml, model
from .core import signature as local_signature

TASK_NAMES = ("cube_handoff",)


def signature():
    xml = build_xml().replace(str(ASSETS), "${ASSETS}")
    return hashlib.sha256((mujoco.__version__ + "\n" + xml).encode()).hexdigest()


def check_bank(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    portable = manifest.get("portable_model_signature")
    if portable is not None:
        if portable != signature():
            raise ValueError("Cube portable model signature differs")
    elif manifest.get("model_signature") != local_signature():
        raise ValueError("Cube bank requires portable identity on this host")
    if manifest.get("state_spec") != "mjSTATE_INTEGRATION":
        raise ValueError("Cube bank must preserve complete integration/user state")
    nstate = mujoco.mj_stateSize(model(), STATE_SPEC)
    for split in ["train", "eval"]:
        with np.load(directory / f"cube_handoff_{split}.npz") as bank:
            states = bank["states"]
            if states.ndim != 2 or states.shape[1] != nstate or not np.isfinite(states).all():
                raise ValueError("Invalid cube integration-state bank")
    if "reset_context_sha256" in manifest:
        path = directory / "reset-context.npz"
        if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["reset_context_sha256"]:
            raise ValueError("Reset context checksum differs")
        with np.load(path) as context:
            n = len(context["state_sha256"])
            if context["histories"].shape != (n, 5, 50) or context["previous_actions"].shape != (n, 12):
                raise ValueError("Invalid reset context shapes")
            if len(set(context["state_sha256"])) != n:
                raise ValueError("Duplicate reset context state")
            for name in ("histories", "previous_actions", "right_grasp_age_s"):
                if not np.isfinite(context[name]).all():
                    raise ValueError("Nonfinite reset context")
    return manifest


def directory_signature(directory):
    digest = hashlib.sha256()
    directory = Path(directory)
    for name in ["manifest.json", "cube_handoff_train.npz", "cube_handoff_eval.npz"]:
        digest.update(name.encode())
        digest.update((directory / name).read_bytes())
    return digest.hexdigest()


def pooled_signature(arrays):
    digest = hashlib.sha256()
    for a in arrays:
        digest.update(str((a.shape, a.dtype)).encode())
        digest.update(np.ascontiguousarray(a).tobytes())
    return digest.hexdigest()
