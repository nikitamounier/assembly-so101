"""Verify a returned Linux reset bank and create a Mac display-only collection.

The untouched Linux-qualified bank and receipts remain in --output. Only a new
review-copy's paths and model-path signature fields are changed; no outcomes or
physical qualification identities are relabeled.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tarfile
import xml.etree.ElementTree as ET
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive", required=True, type=Path)
    p.add_argument("--sha256", required=True)
    p.add_argument(
        "--output",
        required=True,
        type=Path,
        help="New directory for untouched returned artifacts",
    )
    p.add_argument(
        "--display",
        required=True,
        type=Path,
        help="New local display collection directory",
    )
    p.add_argument(
        "--workspace", type=Path, default=Path(__file__).resolve().parents[3]
    )
    a = p.parse_args()
    archive, output, display = (
        a.archive.resolve(),
        a.output.resolve(),
        a.display.resolve(),
    )
    workspace = a.workspace.resolve()
    if output.exists() or display.exists():
        raise ValueError(
            "Return and display directories are versioned; choose unused paths"
        )
    if sha(archive) != a.sha256:
        raise ValueError("Downloaded archive SHA256 differs from return receipt")
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar.getmembers():
            if not member.isfile() and not member.isdir():
                raise ValueError(
                    "Only regular files/directories are allowed in the return archive"
                )
            if Path(member.name).is_absolute() or ".." in Path(member.name).parts:
                raise ValueError(
                    "Return archive path is outside its versioned destination"
                )
        output.mkdir(parents=True)
        tar.extractall(output, filter="data")
    result = json.loads((output / "return-manifest.json").read_text())
    if result["accepted"] != 1800 or result["qualified_platform"] != "Linux":
        raise ValueError("Not a completed1800-state Linux-qualified return")
    for relative, expected in result["files"].items():
        path = (output / relative).resolve()
        if (
            not path.is_relative_to(output)
            or not path.is_file()
            or sha(path) != expected
        ):
            raise ValueError(f"Returned file changed: {relative}")
    parity = json.loads((output / "provenance/native-array-parity.json").read_text())
    if not parity.get("pass_", False):
        raise ValueError("Linux physical-model parity did not pass")
    transport = json.loads((output / "provenance/transport-manifest.json").read_text())
    checked_assets = 0
    for relative, record in transport["files"].items():
        if Path(relative).suffix.lower() not in {".obj", ".stl", ".bin"}:
            continue
        path = (workspace / relative).resolve()
        if not path.is_relative_to(workspace):
            raise ValueError(f"Runtime asset path escapes workspace: {relative}")
        if not path.is_file() or sha(path) != record["source_sha256"]:
            raise ValueError(f"Local runtime source asset differs: {relative}")
        checked_assets += 1
    if not checked_assets:
        raise ValueError("No native source asset hashes were checked")
    remote = result["remote_root"]
    shutil.copytree(output / "review", display)
    # Keep raw qualification receipts (including model SHA and remote paths)
    # exactly untouched. They remain authoritative in --output.
    # Only the XML file paths need relocation for rendering. Per-row physical
    # profile paths and validation-model identities remain verbatim Linux
    # provenance, including nested metadata in stages/manifest.json.
    scene_path = display / "scene.xml"
    scene_path.write_text(scene_path.read_text().replace(remote, str(workspace)))
    scene = display / "scene.xml"
    signature = sha(scene)
    original_signatures = {}
    for filename in [
        "manifest.json",
        "stages/manifest.json",
        "wowrobo_point_features.json",
        "wowrobo_reward_regions.json",
    ]:
        path = display / filename
        if not path.exists():
            continue
        value = json.loads(path.read_text())
        original_signatures[filename] = value.get("model_signature")
        if "model_signature" in value:
            value["model_signature"] = signature
        if filename == "manifest.json":
            value.update(
                render_only=True,
                qualification_platform="Linux",
                qualification_provenance=str(output),
                local_native_admission_repeated=False,
            )
        write(path, value)
    for element in ET.parse(scene).iter():
        file = element.get("file")
        if file:
            path = Path(file) if Path(file).is_absolute() else scene.parent / file
            if not path.is_file():
                raise ValueError(f"Local display model asset missing: {path}")
    # The existing viewer's contracts must bind to the actual Mac-compiled
    # geometry. Never regenerate fingerprints to conceal a mismatch.
    sys.path.insert(0, str(workspace / "assembly-so101"))
    import mujoco
    import numpy as np
    from assembly_policy.so101.wowrobo_obs import PointTables

    model = mujoco.MjModel.from_xml_string(scene.read_text())
    point_config = json.loads((display / "wowrobo_point_features.json").read_text())
    points = PointTables(model, config=point_config)
    manifest = json.loads((display / "stages/manifest.json").read_text())
    variation_count = 0
    for stage in manifest["task_names"]:
        for split in ("train", "eval"):
            metadata = manifest["variation_by_task"][stage][split]
            variants = [row for row in metadata if row["kind"] == "variation"]
            expected = 60 if split == "train" else 12
            if len(variants) != expected or not all(
                row["training_eligible"] for row in variants
            ):
                raise ValueError(f"Display quota/admission mismatch: {stage}/{split}")
            with np.load(display / "stages" / f"{stage}_{split}.npz") as saved:
                if (
                    len(saved["states"]) != len(metadata)
                    or not np.isfinite(saved["states"]).all()
                ):
                    raise ValueError(
                        f"Display state metadata mismatch: {stage}/{split}"
                    )
            variation_count += len(variants)
    if variation_count != 1800:
        raise ValueError("Display must contain exactly1800admitted variations")
    receipt = {
        "status": "verified",
        "accepted_variations": variation_count,
        "untouched_linux_result": str(output),
        "display": str(display),
        "local_display_model_signature": signature,
        "source_model_signatures": original_signatures,
        "source_assets_checked": checked_assets,
        "point_count": points.n_pts,
        "local_native_admission_repeated": False,
        "qualification": "Linux native overlap and held-grip retention; Mac display only",
    }
    write(display / "local-relocation.json", receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
