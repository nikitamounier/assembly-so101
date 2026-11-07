"""Canonical physical configuration for native training and native evaluation."""

import ctypes
import hashlib
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/so101_scene.json"


def configuration():
    return json.loads(CONFIG.read_text())


@lru_cache(maxsize=1)
def load_contact_search():
    """Install the version-pinned native search adapter before any model steps."""
    if mujoco.__version__ != "4.0.0":
        raise RuntimeError("Shared SO101 contact adapter requires MuJoCo 4.0.0")
    source = ROOT / "plugin/so101_contact_search.c"
    library = ROOT / "plugin/build/libso101_contact_search.so"
    if not library.exists() or source.stat().st_mtime > library.stat().st_mtime:
        subprocess.run(
            ["bash", str(ROOT / "plugin/build_contact_search.sh")],
            env=dict(os.environ, PYTHON=sys.executable),
            check=True,
            capture_output=True,
            text=True,
        )
    native = ctypes.CDLL(str(library), mode=ctypes.RTLD_GLOBAL)
    if native.so101_install_contact_search() != 4:
        raise RuntimeError("Unexpected shared contact adapter version")
    return native


def environment_identity():
    from .resets import signature

    return dict(
        model_signature=signature(),
        profile=configuration()["profile"],
        configuration_sha256=hashlib.sha256(CONFIG.read_bytes()).hexdigest(),
        mujoco_version=mujoco.__version__,
    )


def numbers(values):
    return " ".join(format(float(v), ".17g") for v in np.asarray(values).ravel())


def axes(position, target, up=(0.0, 0.0, 1.0)):
    z = np.asarray(position) - np.asarray(target)
    z /= np.linalg.norm(z)
    x = np.cross(up, z)
    x /= np.linalg.norm(x)
    return np.concatenate((x, np.cross(z, x)))


def apply_xml(root, assembled=False):
    cfg = configuration()
    world = root.find("worldbody")
    camera = cfg["main_camera"]
    from .gripper import configure

    configure(root, cfg["gripper"], ROOT / "assets_local")
    search = cfg["contact_search"]
    fine = cfg["motor_base_search"]
    grasp = cfg["grasp_search"]
    horn = cfg["horn_search"]
    assembly = cfg["assembly_search"]
    for values in (search, fine, grasp, horn, assembly):
        if not 1 <= values["initpoints"] <= mujoco.mjMAXCONPAIR or values["iterations"] < 1:
            raise ValueError("Invalid shared SDF contact-search budget")
    # The allocator needs50even when plane/mesh colliders use only4search starts.
    root.find("option").set("sdf_initpoints", str(mujoco.mjMAXCONPAIR))
    root.find("option").set("sdf_iterations", str(search["iterations"]))
    custom = ET.SubElement(root, "custom")
    for name, values in (
        ("so101_sdf_search", search),
        ("so101_motor_base_search", fine),
        ("so101_grasp_search", grasp),
        ("so101_horn_search", horn),
        ("so101_assembly_search", assembly),
    ):
        ET.SubElement(
            custom, "numeric", name=name, data=numbers([values["initpoints"], values["iterations"]])
        )
    if cfg.get("motor_base_contact_guard", False):
        ET.SubElement(custom, "numeric", name="so101_material_contact_guard", data="1")
    if cfg.get("motor_horn_contact_guard", False):
        ET.SubElement(custom, "numeric", name="so101_horn_contact_guard", data="1")
    for name, value in cfg["solver"].items():
        root.find("option").set(name, str(value))
    ET.SubElement(root.find("option"), "flag", sleep="disable")
    for pair in cfg.get("contact_exclusions", []):
        ET.SubElement(root.find("contact"), "exclude", **pair)
    for geom in cfg.get("contact_surfaces", []):
        attrs = dict(geom)
        body = world.find(f".//body[@name='{attrs.pop('body')}']")
        ET.SubElement(
            body, "geom", **{k: numbers(v) if isinstance(v, list) else str(v) for k, v in attrs.items()}
        )
    for pair in cfg.get("contact_pairs", []):
        ET.SubElement(
            root.find("contact"),
            "pair",
            **{k: numbers(v) if isinstance(v, list) else str(v) for k, v in pair.items()},
        )
    ET.SubElement(
        world,
        "camera",
        name="workspace_camera",
        pos=numbers(camera["position"]),
        xyaxes=numbers(axes(camera["position"], camera["look_at"])),
        fovy=str(camera["fovy_degrees"]),
    )
    for side, mount in cfg["mounts"].items():
        body = world.find(f".//body[@name='{side}_base']")
        body.set("pos", numbers(mount["position"]))
        body.set("quat", numbers(mount["quaternion"]))
        gripper = world.find(f".//body[@name='{side}_gripper']")
        site = gripper.find(f"site[@name='{side}_gripperframe']")
        pos = np.fromstring(site.get("pos"), sep=" ")
        rotation = np.empty(9)
        mujoco.mju_quat2Mat(rotation, np.fromstring(site.get("quat"), sep=" "))
        rotation = rotation.reshape(3, 3)
        eye = pos - 0.08 * rotation[:, 0] + 0.08 * rotation[:, 2] - 0.05 * rotation[:, 1]
        target = pos + 0.015 * rotation[:, 0]
        ET.SubElement(
            gripper,
            "camera",
            name=side + "_wrist_view",
            pos=numbers(eye),
            xyaxes=numbers(axes(eye, target, rotation[:, 2])),
            fovy="75",
        )
    for actuator in root.find("actuator"):
        cap = cfg["effort_caps"]["gripper" if actuator.get("name").endswith("_gripper") else "arm"]
        actuator.set("forcerange", numbers([-cap, cap]))
    if not assembled:
        for name, values in cfg["initial_joints"].items():
            if len(values) != 7:
                continue
            body = world.find(f".//freejoint[@name='{name}']/..")
            body.set("pos", numbers(values[:3]))
            body.set("quat", numbers(values[3:]))
    else:
        base = cfg["initial_joints"]["base_free"]
        quat = np.asarray(base[3:])
        rotation = np.empty(9)
        mujoco.mju_quat2Mat(rotation, quat)
        rotation = rotation.reshape(3, 3)
        anchor = np.asarray(cfg["assembly_plan"]["assembled_base_position"])
        old = np.fromstring(world.find(".//body[@name='base']").get("pos"), sep=" ")
        for name in ("base", "motor", "horn", "screw", "retainer", "yoke"):
            body = world.find(f".//body[@name='{name}']")
            pos = np.fromstring(body.get("pos"), sep=" ")
            local = np.fromstring(body.get("quat", "1 0 0 0"), sep=" ")
            composed = np.empty(4)
            mujoco.mju_mulQuat(composed, quat, local)
            body.set("pos", numbers(anchor + rotation @ (pos - old)))
            body.set("quat", numbers(composed))


def apply_model(model):
    for part, values in configuration().get("free_body_damping", {}).items():
        dof = int(model.joint(part + "_free").dofadr[0])
        model.dof_damping[dof : dof + 3] = values["linear"]
        model.dof_damping[dof + 3 : dof + 6] = values["angular"]


def default_bank_directory():
    return ROOT / configuration()["bank_directory"]


def initialization_identity():
    cfg = configuration()
    return dict(
        initial_joints=cfg["initial_joints"],
        free_body_damping=cfg.get("free_body_damping", {}),
        reset_distribution=cfg["reset_distribution"],
    )


def assert_matches_reference(model):
    """Check compiled physics against the versioned shared-factory reference."""
    from .robot import home

    prefix = ROOT / configuration()["reference_prefix"]
    expected = json.loads(prefix.with_suffix(".json").read_text())
    counts = {key: int(getattr(model, key)) for key in expected["counts"]}
    options = {key: getattr(model.opt, key) for key in expected["options"]}
    if expected["model_signature"] != environment_identity()["model_signature"]:
        raise ValueError("Workcell reference belongs to a different environment")
    if counts != expected["counts"] or any(options[k] != expected["options"][k] for k in options):
        raise ValueError("Workcell reference counts/options differ")
    data = mujoco.MjData(model)
    home(model, data)
    differences = {}
    with np.load(prefix.with_suffix(".npz")) as reference:
        for name in reference.files:
            actual = data.qpos if name == "home" else np.asarray(getattr(model, name))
            wanted = reference[name]
            if actual.shape != wanted.shape or not np.allclose(actual, wanted, rtol=0, atol=1e-13):
                raise ValueError(f"Workcell reference differs: {name}")
            differences[name] = float(np.max(np.abs(actual.astype(float) - wanted.astype(float)), initial=0))
    return dict(counts=counts, options=options, max_absolute_difference=differences, passed=True)
