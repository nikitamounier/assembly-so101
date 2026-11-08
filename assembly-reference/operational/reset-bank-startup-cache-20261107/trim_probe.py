"""Allocated-node constructor/cache/RSS probe; does not edit model or states."""

from __future__ import annotations

import argparse
import ctypes
import gc
import hashlib
import json
import resource
import sys
import time
from pathlib import Path


def rss_kib():
    status = Path("/proc/self/status")
    if not status.exists():
        return None
    for line in status.read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1])
    return None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project", required=True, type=Path)
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    sys.path.insert(0, str(a.project.resolve()))
    from assembly_policy.so101.wowrobo_randomization import WowRoboRandomizer

    before = rss_kib()
    started = time.monotonic()
    sampler = WowRoboRandomizer(config=a.config)
    initialized = rss_kib()
    baseline = sampler.restore("21_upper_far_side")
    state_before = hashlib.sha256(
        baseline.qpos.tobytes() + baseline.qvel.tobytes() + baseline.ctrl.tobytes()
    ).hexdigest()
    collected = gc.collect()
    libc = ctypes.CDLL(None)
    trim = libc.malloc_trim if hasattr(libc, "malloc_trim") else None
    trim_result = trim(0) if trim is not None else None
    after = rss_kib()
    state_after = hashlib.sha256(
        baseline.qpos.tobytes() + baseline.qvel.tobytes() + baseline.ctrl.tobytes()
    ).hexdigest()
    assert state_before == state_after
    result = {
        "initialization_seconds": time.monotonic() - started,
        "rss_before_kib": before,
        "rss_after_constructor_kib": initialized,
        "rss_after_gc_trim_kib": after,
        "max_rss_platform_units": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "gc_objects": collected,
        "malloc_trim_result": trim_result,
        "state_unchanged": True,
    }
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
