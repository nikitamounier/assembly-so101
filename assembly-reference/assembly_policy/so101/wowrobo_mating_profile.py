"""Explicit, isolated legacy contact-law mappings for WowRobo mating pairs."""

import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/wowrobo_mating_profile.json"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_profile_scene(source_scene, out_scene, step, profile="legacy_mapped"):
    """Apply only this step's pairs; explicit MuJoCo pairs bypass body masks.

    No meshes, body transforms, joints, inertias or solver options are changed.
    Unmapped surfaces retain the baseline's firm contacts. The baseline output
    is a byte-identical copy and remains available for matched comparisons.
    """
    if profile not in ("baseline", "legacy_mapped"):
        raise ValueError("Unknown mating profile")
    config = json.loads(CONFIG.read_text())
    if step not in config["steps"]:
        raise ValueError("Unknown insertion step")
    source_scene, out_scene = Path(source_scene).resolve(), Path(out_scene).resolve()
    if source_scene == out_scene:
        raise ValueError("Write the candidate separately to preserve the baseline scene")
    out_scene.parent.mkdir(parents=True, exist_ok=True)
    source = source_scene.read_bytes()
    tree = ET.fromstring(source)
    geoms = {g.get("name") for g in tree.iter("geom")}
    mapped = []
    if profile == "legacy_mapped":
        contact = tree.find("contact")
        if contact is None:
            contact = ET.SubElement(tree, "contact")
        for i, mapping in enumerate(config["steps"][step]):
            if not {mapping["geom1"], mapping["geom2"]} <= geoms:
                raise ValueError(f"Missing collision geometry for {mapping}")
            law = config["classes"][mapping["contact_class"]]
            params = {k: v for k, v in law.items() if k != "source_pair"}
            attrs = {
                "name": f"wowfit_{step}_{i}",
                "geom1": mapping["geom1"],
                "geom2": mapping["geom2"],
                **{
                    key: " ".join(format(float(x), ".17g") for x in value)
                    if isinstance(value, list)
                    else str(value)
                    for key, value in params.items()
                },
            }
            ET.SubElement(contact, "pair", attrs)
            mapped.append({**mapping, "legacy_law": law})
        ET.ElementTree(tree).write(out_scene, encoding="unicode")
    else:
        out_scene.write_bytes(source)
    result = {
        "step": step,
        "profile": profile,
        "configuration_version": config["version"],
        "source_scene": str(source_scene),
        "scene": str(out_scene),
        "source_scene_sha256": hashlib.sha256(source).hexdigest(),
        "scene_sha256": digest(out_scene),
        "configuration_sha256": digest(CONFIG),
        "legacy_source": config["legacy_source"],
        "pairs": mapped,
        "geometry_changed": False,
        "note": "Explicit pairs override masks; use mjDSBL_CONTACT for a contacts-off control. This is an uncalibrated candidate profile, not automatic training admission.",
    }
    out_scene.with_suffix(".profile.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    return result
