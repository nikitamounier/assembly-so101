"""Measured passive stops and small per-reset native hinge-limit variations."""

import ctypes
import fcntl
import hashlib
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

JOINTS = ("target_servo_2_shaft", "target_servo_3_shaft")
DEFAULT_NOISE_RAD = float(np.deg2rad(0.05))
_LIBS = []


def configure_xml(xml, fit, *, retain_servo_travel_caps=False):
    """Remove only the three measured stop pairs; retain other CAD contacts."""
    root = ET.fromstring(xml)
    if root.find("compiler") is None or root.find("compiler").get("angle") != "radian":
        raise ValueError("Measured hinge limits require a radian XML compiler")
    contact = root.find("contact")
    size = root.find("size")
    if size is None:
        size = ET.SubElement(root, "size")
    size.set("nuserdata", str(max(7, int(size.get("nuserdata", "0")))))
    removed = []
    for pair in list(contact):
        if pair.tag == "pair" and pair.get("name", "").startswith(
            "passive_motor_stop_"
        ):
            removed.append(dict(pair.attrib))
            contact.remove(pair)
    if len(removed) != 3:
        raise ValueError("Expected exactly three explicit passive motor stop pairs")
    for joint in JOINTS:
        node = next(n for n in root.iter("joint") if n.get("name") == joint)
        row = fit["joints"][joint]
        bounds = (
            row["effective_range_rad"] if retain_servo_travel_caps else row["range_rad"]
        )
        if any(v is None for v in bounds) or not bounds[0] < bounds[1]:
            raise ValueError("Both measured stop boundaries are required")
        node.set("limited", "true")
        node.set("range", " ".join(map(str, bounds)))
        node.set("solreflimit", ".002 1")
        node.set("solimplimit", ".95 .99 .002")
    # Explicit pairs bypass body exclusions, hence remove them before excluding.
    for a, b in [
        ("upper_arm_module", "target_servo_2_pivot"),
        ("forearm_module", "target_servo_3_pivot"),
    ]:
        if not any(
            {p.get("body1"), p.get("body2")} == {a, b}
            for p in contact.findall("exclude")
        ):
            ET.SubElement(contact, "exclude", body1=a, body2=b)
    return ET.tostring(root, encoding="unicode"), removed


def sample_noise(rng, ranges, qpos, amplitude=DEFAULT_NOISE_RAD):
    """Keep one draw per endpoint for the episode; never invalidate its reset."""
    ranges, qpos = np.asarray(ranges, float), np.asarray(qpos, float)
    if ranges.shape != (2, 2) or qpos.shape != (2,) or not 0 <= amplitude < 0.01:
        raise ValueError(
            "Expected two finite hinge intervals and a small noise amplitude"
        )
    if not np.isfinite(ranges).all() or not np.isfinite(qpos).all():
        raise ValueError("Nonfinite hinge configuration")
    if not np.all(ranges[:, 0] < ranges[:, 1]):
        raise ValueError("Inverted hinge interval")
    out = np.empty((2, 2))
    for k in range(2):
        lower_max = min(amplitude, qpos[k] - ranges[k, 0])
        upper_min = max(-amplitude, qpos[k] - ranges[k, 1])
        if lower_max < -amplitude or upper_min > amplitude:
            raise ValueError("Reset lies outside its noisy hinge-limit support")
        out[k] = [rng.uniform(-amplitude, lower_max), rng.uniform(upper_min, amplitude)]
    return out.ravel()


def install(model, directory, offset=3, amplitude=DEFAULT_NOISE_RAD):
    """Install after friction hook; pass four saved noise offsets in mjData.userdata."""
    source = Path(__file__).with_suffix(".cc")
    package = Path(mujoco.__file__).parent
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(
        source.read_bytes() + mujoco.__version__.encode()
    ).hexdigest()[:16]
    output = directory / ("hinge-" + digest + ".so")
    with (directory / ".build.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not output.exists():
            native = next(
                package.glob(
                    "libmujoco*.dylib" if sys.platform == "darwin" else "libmujoco.so*"
                )
            )
            link = ["-dynamiclib"] if sys.platform == "darwin" else ["-shared", "-fPIC"]
            subprocess.run(
                [
                    "c++",
                    "-std=c++17",
                    *link,
                    "-O2",
                    "-Wall",
                    "-Werror",
                    "-I" + str(package / "include"),
                    str(source),
                    str(native),
                    "-Wl,-rpath," + str(package),
                    "-o",
                    str(output) + ".tmp",
                ],
                check=True,
            )
            Path(str(output) + ".tmp").replace(output)
    lib = ctypes.CDLL(str(output))
    lib.wowrobo_install_hinge_limits.argtypes = [
        ctypes.c_void_p,
        np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS"),
        ctypes.c_int,
        ctypes.c_double,
    ]
    ids = np.array([model.joint(n).id for n in JOINTS], np.int32)
    if not lib.wowrobo_install_hinge_limits(model._address, ids, offset, amplitude):
        raise ValueError(
            "Hinge hook requires limited hinges, four userdata slots, and one installation"
        )
    _LIBS.append(lib)
    return lib


def augment_reset_states(
    states, model, rng, amplitude=DEFAULT_NOISE_RAD, nominal=False
):
    """Append four saved limit offsets to the existing warmstart/friction schema."""
    states = np.asarray(states, float)
    expected = (
        mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_FULLPHYSICS)
        + model.neq
        + model.nu
        + model.nv
        + 3
    )
    if states.ndim != 2 or states.shape[1] != expected:
        raise ValueError("Expected fullphysics/eq/ctrl/warmstart/3friction schema")
    ranges = np.array([model.joint(j).range for j in JOINTS])
    addresses = np.array([int(model.joint(j).qposadr[0]) + 1 for j in JOINTS])
    noise = (
        np.zeros((len(states), 4))
        if nominal
        else np.array(
            [sample_noise(rng, ranges, row[addresses], amplitude) for row in states]
        ).reshape(-1, 4)
    )
    if nominal and np.any(
        (states[:, addresses] < ranges[:, 0]) | (states[:, addresses] > ranges[:, 1])
    ):
        raise ValueError("Nominal reset outside measured hinge limits")
    return np.concatenate([states, noise], axis=1)
