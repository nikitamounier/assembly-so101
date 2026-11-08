"""Canonical current-assembly native factory for training (never viewer XML)."""

import ast
import hashlib
import importlib.util
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROFILE = (
    ROOT / "assets_local/wowrobo/physical-resets-repaired-cover-support-v2/profile.json"
)
_CACHE = {}


def load_native(profile_path=DEFAULT_PROFILE):
    path = Path(profile_path).resolve()
    profile = json.loads(path.read_text())
    source = Path(profile["model_path"])
    if hashlib.sha256(source.read_bytes()).hexdigest() != profile["model_sha256"]:
        raise ValueError("Current assembly physical model identity changed")
    key = str(path)
    if key not in _CACHE:
        name = "_wowrobo_training_native_backend"
        backend = sys.modules.get(name)
        if backend is None:
            spec = importlib.util.spec_from_file_location(
                name, ROOT / "assembly_policy/so101/wowrobo_contact_backend.py"
            )
            backend = importlib.util.module_from_spec(spec)
            sys.modules[name] = backend
            spec.loader.exec_module(backend)
        backend.load_backend(profile["grid_manifest"])
        # Mode 2 is the exhaustive diagnostic oracle; training uses the exact
        # surface tree to avoid scanning every source sample at each substep.
        backend.set_surface_mode(1)
        tree = ET.parse(source).getroot()
        # Restore 80 physics substeps per 40 ms policy action. Apply to the
        # training XML so the changed timestep is included in its identity;
        # saved reset/admission models retain their original provenance.
        tree.find("option").set("timestep", ".0005")
        size = tree.find("size")
        if size is None:
            size = ET.SubElement(tree, "size")
        # Deep diagnostic contacts can exceed MuJoCo's default 64 MiB arena.
        # Capacity only: contact laws and solver settings remain unchanged.
        size.set("memory", "256M")
        if profile.get("training_episode_friction") == "native-userdata-v1":
            size.set("nuserdata", "3")
        sensors = tree.find("sensor")
        if sensors is None:
            sensors = ET.SubElement(tree, "sensor")
        ET.SubElement(
            sensors,
            "contact",
            name="training_min_contact_distance",
            data="dist",
            num="1",
            reduce="mindist",
        )
        contact = tree.find("contact")
        if contact is None:
            contact = ET.SubElement(tree, "contact")
        existing = {
            frozenset((p.get("geom1"), p.get("geom2"))) for p in contact.findall("pair")
        }
        mechanics_source = ROOT / "assembly_policy/so101/wowrobo_physical_resets.py"
        constants = {}
        for node in ast.parse(mechanics_source.read_text()).body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id in (
                        "PASSIVE_BRACKET_STOPS",
                        "PASSIVE_BRACKET_STOP_LAW",
                    ):
                        constants[target.id] = ast.literal_eval(node.value)
        for motor, bracket in constants["PASSIVE_BRACKET_STOPS"]:
            if frozenset((motor, bracket)) not in existing:
                ET.SubElement(
                    contact,
                    "pair",
                    geom1=motor,
                    geom2=bracket,
                    **constants["PASSIVE_BRACKET_STOP_LAW"],
                )
        limit_helper = None
        limit_source = b""
        if profile.get("hinge_limit_fit"):
            if profile.get("training_hinge_limits") != "native-userdata-v1":
                raise ValueError(
                    "Hinge limits require versioned per-reset userdata payloads"
                )
            if profile.get("training_episode_friction") != "native-userdata-v1":
                raise ValueError(
                    "Hinge userdata requires the versioned three-friction prefix"
                )
            fit_path = Path(profile["hinge_limit_fit"])
            fit_bytes = fit_path.read_bytes()
            if (
                hashlib.sha256(fit_bytes).hexdigest()
                != profile["hinge_limit_fit_sha256"]
            ):
                raise ValueError("Measured hinge-stop provenance changed")
            helper_path = ROOT / "assembly_policy/so101/wowrobo_hinge_limits.py"
            helper_spec = importlib.util.spec_from_file_location(
                "_native_hinge_limits", helper_path
            )
            limit_helper = importlib.util.module_from_spec(helper_spec)
            sys.modules[helper_spec.name] = limit_helper
            helper_spec.loader.exec_module(limit_helper)
            modified, _ = limit_helper.configure_xml(
                ET.tostring(tree, encoding="unicode"), json.loads(fit_bytes)
            )
            tree = ET.fromstring(modified)
            limit_source = (
                fit_bytes
                + helper_path.read_bytes()
                + helper_path.with_suffix(".cc").read_bytes()
            )
        if limit_helper is not None:
            limit_source += json.dumps(
                {
                    "noise_amplitude_rad": profile.get(
                        "hinge_limit_noise_rad", limit_helper.DEFAULT_NOISE_RAD
                    ),
                    "userdata_offset": 3,
                },
                sort_keys=True,
            ).encode()
        xml = ET.tostring(tree, encoding="unicode")
        m = mujoco.MjModel.from_xml_string(xml)
        m.opt.jacobian = mujoco.mjtJacobian.mjJAC_SPARSE
        friction_source = b""
        if profile.get("training_episode_friction") == "native-userdata-v1":
            hook_path = ROOT / "assembly_policy/so101/wowrobo_rollout_friction.py"
            hook_spec = importlib.util.spec_from_file_location(
                "_wowrobo_native_episode_friction", hook_path
            )
            hook = importlib.util.module_from_spec(hook_spec)
            sys.modules[hook_spec.name] = hook
            hook_spec.loader.exec_module(hook)
            hook.install(m, ROOT / "outputs/native-rollout-friction")
            friction_source = (
                hook_path.read_bytes()
                + hook_path.with_suffix(".cc").read_bytes()
                + hook_path.with_name("wowrobo_friction.py").read_bytes()
            )
        if limit_helper is not None:
            limit_helper.install(
                m,
                ROOT / "outputs/native-hinge-limits",
                offset=3,
                amplitude=profile.get(
                    "hinge_limit_noise_rad", limit_helper.DEFAULT_NOISE_RAD
                ),
            )
        if m.opt.gravity[2] >= 0 or m.npair == 0 or not any(m.geom_contype):
            raise ValueError("Training requires gravity and enabled source contacts")
        identity = hashlib.sha256(
            xml.encode()
            + mujoco.__version__.encode()
            + Path(__file__).read_bytes()
            + Path(profile["grid_manifest"]).read_bytes()
            + mechanics_source.read_bytes()
            + friction_source
            + limit_source
        ).hexdigest()
        _CACHE[key] = (m, profile, identity)
    return _CACHE[key]
