"""Bridge to the one canonical physical factory; no duplicate physics builder."""

import importlib.util
import os
import sys
from pathlib import Path

ROOT = Path(
    os.environ.get("WOWROBO_CANONICAL_ROOT", Path(__file__).resolve().parents[3] / "robotics-assembly-3.0")
).resolve()
DEFAULT_PROFILE = ROOT / "assets_local/wowrobo/physical-resets-repaired-cover-support-v2/profile.json"
DEFAULT_REVIEW = ROOT / "assets_local/wowrobo/randomization-bank-repaired-cover-support-v2/scene.xml"
DEFAULT_BANK = ROOT / "assets_local/wowrobo/training-pilot-v1"


def load_runtime(profile=DEFAULT_PROFILE):
    name = "_canonical_wowrobo_training_runtime"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            name, ROOT / "assembly_policy/so101/wowrobo_training_runtime.py"
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name].load_native(profile)
