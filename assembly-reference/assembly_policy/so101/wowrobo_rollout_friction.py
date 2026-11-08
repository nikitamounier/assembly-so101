"""Compiled friction hook; no Python callback in threaded rollout substeps."""

import ctypes
import fcntl
import hashlib
import subprocess
import sys
from pathlib import Path

import mujoco
import numpy as np

try:
    from .wowrobo_friction import CATEGORIES, infer_geom_roles
except ImportError:
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_native_episode_friction_roles",
        Path(__file__).with_name("wowrobo_friction.py"),
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    CATEGORIES, infer_geom_roles = module.CATEGORIES, module.infer_geom_roles

LIBRARIES = []


def build_library(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).with_suffix(".cc")
    digest = hashlib.sha256(
        source.read_bytes() + mujoco.__version__.encode()
    ).hexdigest()[:16]
    output = directory / ("friction-" + digest + ".so")
    with (directory / ".build.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not output.exists():
            package = Path(mujoco.__file__).parent
            library = next(
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
                    str(library),
                    "-Wl,-rpath," + str(package),
                    "-o",
                    str(output) + ".tmp",
                ],
                check=True,
            )
            Path(str(output) + ".tmp").replace(output)
    return output


def install(model, directory, offset=0):
    lib = ctypes.CDLL(str(build_library(directory)))
    lib.wowrobo_install_friction.argtypes = [
        ctypes.c_void_p,
        np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS"),
        ctypes.c_int,
    ]
    roles = infer_geom_roles(model)
    codes = {"finger": 1, "part": 2, "table": 3}
    array = np.asarray(
        [codes.get(roles.get(i), 0) for i in range(model.ngeom)], np.int32
    )
    if not lib.wowrobo_install_friction(model._address, array, offset):
        raise RuntimeError(
            "Friction hook requires one model, three userdata values, and an unused control callback"
        )
    LIBRARIES.append(lib)
    return lib
