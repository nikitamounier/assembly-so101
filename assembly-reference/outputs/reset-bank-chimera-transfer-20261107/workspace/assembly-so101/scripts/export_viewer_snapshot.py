"""Freeze a WowRobo review layout and its exact referenced assets for sharing."""

import argparse
import hashlib
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assembly-root", type=Path, required=True)
    parser.add_argument("--layout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source_root = args.assembly_root.resolve()
    layout = args.layout.resolve()
    repo = Path(__file__).resolve().parents[1]
    out = args.output.resolve()
    if out.exists():
        parser.error("Output already exists; use a new versioned snapshot directory")
    out.mkdir(parents=True)
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()

    def write(p, obj):
        p.write_text(json.dumps(obj, indent=2) + "\n")

    def portable(x):
        if isinstance(x, dict):
            return {k: portable(v) for k, v in x.items()}
        if isinstance(x, list):
            return [portable(v) for v in x]
        if isinstance(x, str):
            return (
                x.replace(str(source_root) + "/outputs/cover-support-grasp-20261107/", "evidence/")
                .replace(str(source_root) + "/outputs/cover-support-fix-20261107/", "evidence/")
                .replace(str(source_root) + "/", "assembly-reference/")
            )
        return x

    xml = ET.parse(layout / "scene.xml")
    assets = {}
    for e in xml.iter():
        if "file" not in e.attrib:
            continue
        p = Path(e.get("file"))
        rel = p.relative_to(source_root / "assets_local/wowrobo")
        dest = out / "meshes" / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            shutil.copy2(p, dest)
        e.set("file", dest.relative_to(out).as_posix())
        assets[dest.relative_to(out).as_posix()] = dict(source=rel.as_posix(), sha256=sha(p))
    ET.indent(xml)
    xml.write(out / "scene.xml", encoding="unicode")
    (out / "stages").mkdir(exist_ok=True)
    for p in (layout / "stages").iterdir():
        if p.suffix == ".npz":
            shutil.copy2(p, out / "stages" / p.name)
        elif p.name == "manifest.json":
            write(out / "stages" / p.name, portable(json.loads(p.read_text())))
    for source, target in [
        ("configs/wowrobo_point_features.json", "point-features.json"),
        ("configs/wowrobo_reward_regions.json", "reward-regions.json"),
    ]:
        write(out / target, portable(json.loads((repo / source).read_text())))
    lic = out / "licenses"
    lic.mkdir(exist_ok=True)
    for name in [
        "NOTICE",
        "SO-ARM100.LICENSE",
        "ICL-so101.LICENSE",
        "source-lock.json",
        "helper-lock.json",
        "camera-source-lock.json",
    ]:
        shutil.copy2(source_root / "cad/wowrobo" / name, lic / name)
    # Include the two audit receipts referenced by the displayed validation metadata.
    evidence = out / "evidence"
    evidence.mkdir(exist_ok=True)
    for rel in [
        "outputs/cover-support-grasp-20261107/best-precontact.json",
        "outputs/cover-support-fix-20261107/nominal-audits.json",
    ]:
        p = source_root / rel
        if p.is_file():
            write(evidence / p.name, portable(json.loads(p.read_text())))
    bank_manifest = json.loads((out / "stages/manifest.json").read_text())
    import numpy as np

    state_count = 0
    for archive in (out / "stages").glob("*.npz"):
        with np.load(archive, allow_pickle=False) as states:
            state_count += len(states["states"])
    nominal_count = len(bank_manifest["task_names"])
    source_manifest = json.loads((layout / "manifest.json").read_text())
    write(
        out / "snapshot.json",
        dict(
            schema=1,
            name="WowRobo repaired CAD · cover support · 100-state viewer",
            scene="scene.xml",
            bank="stages",
            point_config="point-features.json",
            reward_config="reward-regions.json",
            model_signature=source_manifest["model_signature"],
            source_collection=layout.name,
            source_scene_sha256=sha(layout / "scene.xml"),
            state_count=state_count,
            nominal_count=nominal_count,
            variation_count=state_count - nominal_count,
            assets=assets,
            files={
                p.relative_to(out).as_posix(): sha(p)
                for p in sorted(out.rglob("*"))
                if p.is_file() and p.name != "snapshot.json"
            },
        ),
    )
    print(
        "Bundled files",
        len(json.loads((out / "snapshot.json").read_text())["files"]),
        "bytes",
        sum(p.stat().st_size for p in out.rglob("*") if p.is_file()),
    )


if __name__ == "__main__":
    main()
