"""Load the versioned, self-contained viewer collection shipped with the repo."""

import hashlib
import json
from pathlib import Path

import mujoco


def load_snapshot(folder):
    """Verify every runtime asset before loading the saved inspection scene.

    The source model signature identifies the unchanged saved states and their
    reward/observation contracts. Portable XML has its own file checksum because
    only mesh paths were rewritten; no geometry or states were regenerated.
    """
    folder = Path(folder).resolve()
    manifest = json.loads((folder / "snapshot.json").read_text())
    if manifest.get("schema") != 1:
        raise ValueError("Unsupported viewer snapshot schema")

    def member(name):
        path = (folder / name).resolve()
        if not path.is_relative_to(folder):
            raise ValueError(f"Viewer snapshot path leaves its directory: {name}")
        return path

    for name, expected in manifest["files"].items():
        path = member(name)
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Viewer runtime asset is missing or changed: {name}")
    for key in ("scene", "point_config", "reward_config"):
        if manifest[key] not in manifest["files"]:
            raise ValueError(f"Unverified viewer snapshot file: {manifest[key]}")
    bank = member(manifest["bank"])
    bank_manifest = json.loads((bank / "manifest.json").read_text())
    signature = manifest["model_signature"]
    point_config = json.loads(member(manifest["point_config"]).read_text())
    reward_config = json.loads(member(manifest["reward_config"]).read_text())
    if any(config["model_signature"] != signature for config in (bank_manifest, point_config, reward_config)):
        raise ValueError("Viewer scene, states and observation/reward contracts differ")
    return dict(
        roots=[bank],
        scene="wowrobo-layout",
        model=mujoco.MjModel.from_xml_path(str(member(manifest["scene"]))),
        model_signature=signature,
        point_config=point_config,
        reward_config=reward_config,
    ), manifest
