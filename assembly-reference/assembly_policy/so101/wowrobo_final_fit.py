"""Isolated compliance candidate for the measured final-module pocket fit.

Vendor geometry and every unmapped contact remain unchanged. This combines
the inherited fork/case law with the inherited motor/base compliance width.
It is a bounded fixture candidate, not calibrated material or training admission.
"""

import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

PAIR = frozenset(("servo_3_case__contact", "upper_arm_module_printed_0__contact"))
PARAMETERS = {
    "condim": "3",
    "friction": ".3 .3 .005 .0001 .0001",
    "solref": ".001 1",
    "solimp": ".1 .99 .0004 .8 2",
    "margin": "0",
}


def replace_final_pocket_asset(source_scene, derived_mesh, output_scene):
    """Select an explicit derived printed pocket, preserving all other assets.

    The caller must use the matching source-distance field and pass the
    measured local-relief, wall-thickness and original-material controls.
    This does not change the contact law or overwrite vendor CAD.
    """
    source_scene, derived_mesh, output_scene = map(
        Path, (source_scene, derived_mesh, output_scene)
    )
    source = source_scene.read_bytes()
    root = ET.fromstring(source)
    mesh = root.find("asset/mesh[@name='upper_arm_module_printed_0']")
    if mesh is None:
        raise ValueError("Source scene has no upper-arm printed pocket asset")
    previous = mesh.get("file")
    mesh.set("file", str(derived_mesh.resolve()))
    output_scene.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(output_scene, encoding="unicode")
    return {
        "variant": "explicit-final-local-pocket-clearance-candidate-v1",
        "source_scene_sha256": hashlib.sha256(source).hexdigest(),
        "scene_sha256": hashlib.sha256(output_scene.read_bytes()).hexdigest(),
        "original_mesh": previous,
        "derived_mesh": str(derived_mesh.resolve()),
        "derived_mesh_sha256": hashlib.sha256(derived_mesh.read_bytes()).hexdigest(),
        "changed_asset_only": "upper_arm_module_printed_0",
        "contact_law_changed": False,
    }


def write_final_fit_scene(source_scene, output_scene):
    """Change only the explicit final case/pocket pair's compliance law.

    The reviewed straight path has <=0.178 mm sampled source interference;
    the 0.2 mm inherited transition stiffens too early. A 0.4 mm transition
    permits that bounded fit while retaining full source material exclusion.
    The separately checked 1 mm displaced wall must remain blocked.
    """
    source_scene, output_scene = Path(source_scene), Path(output_scene)
    source = source_scene.read_bytes()
    root = ET.fromstring(source)
    names = {g.get("name") for g in root.iter("geom")}
    if not PAIR <= names:
        raise ValueError("Final case/pocket collision geometry is missing")
    contact = root.find("contact")
    if contact is None:
        contact = ET.SubElement(root, "contact")
    matches = [p for p in contact.findall("pair")
               if frozenset((p.get("geom1"), p.get("geom2"))) == PAIR]
    if len(matches) > 1:
        raise ValueError("Duplicate final case/pocket contact pairs")
    pair = matches[0] if matches else ET.SubElement(
        contact, "pair", name="wowrobo_final_local_fit",
        geom1="servo_3_case__contact", geom2="upper_arm_module_printed_0__contact",
    )
    before = dict(pair.attrib)
    pair.attrib.update(PARAMETERS)
    output_scene.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(output_scene, encoding="unicode")
    receipt = {
        "profile": "wowrobo-final-local-fit-candidate-v1",
        "source_scene_sha256": hashlib.sha256(source).hexdigest(),
        "scene_sha256": hashlib.sha256(output_scene.read_bytes()).hexdigest(),
        "before": before, "after": dict(pair.attrib),
        "geometry_changed": False,
        "unmapped_contacts_changed": False,
        "basis": "Actual pocket feature alignment; 51-pose/4096-sample source sweep max0.1773mm; inherited fork/case law with motor/base0.4mm compliance transition.",
        "limits": "Uncalibrated local case/pocket compliance candidate. Requires <=0.5mm native/source penetration guards, actual wall negative control and unloaded release. No automatic production admission.",
    }
    output_scene.with_suffix(".final-fit.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )
    return receipt
