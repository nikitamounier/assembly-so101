"""Compare old/new contact callbacks on identical ordinary manifolds/transforms."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import warp as wp


def install(physics, baseline_source, output):
    spec = importlib.util.spec_from_file_location(
        "assembly_policy.so101._contact_baseline_oracle",
        Path(baseline_source) / "wowrobo_warp.py",
    )
    old = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = old
    spec.loader.exec_module(old)
    candidate = physics.model.callback.contactfilter
    completed = []
    fields = (
        "dist",
        "pos",
        "frame",
        "includemargin",
        "friction",
        "solref",
        "solreffriction",
        "solimp",
        "dim",
        "geom",
        "worldid",
        "type",
        "geomcollisionid",
    )

    def table(d):
        n = int(d.nacon.numpy()[0])
        values = np.concatenate(
            [getattr(d.contact, k).numpy()[:n].reshape(n, -1) for k in fields], axis=1
        )
        return values[np.lexsort(values.T[::-1])]

    def audit(m, d):
        if completed:
            return candidate(m, d)
        arrays = {
            name: getattr(d.contact, name)
            for name in dir(d.contact)
            if isinstance(getattr(d.contact, name), wp.array)
            and getattr(d.contact, name).size
        }
        original = {name: array.numpy().copy() for name, array in arrays.items()}
        count = d.nacon.numpy().copy()
        old.WowRoboWarp._contacts(physics, m, d)
        expected = table(d)
        for name, array in arrays.items():
            array.assign(original[name])
        d.nacon.assign(count)
        candidate(m, d)
        actual = table(d)
        proof = {
            "contacts": len(actual),
            "fields": list(fields),
            "same_shape": expected.shape == actual.shape,
            "bit_equal": bool(np.array_equal(expected, actual)),
            "ordinary_contacts": int(count[0]),
            "max_error": float(abs(expected - actual).max())
            if expected.shape == actual.shape
            else None,
            "worlds": d.nworld,
            "ordinary_pairs": int(d.ncollision.numpy()[0]),
            "capacity": d.naconmax,
        }
        Path(output).write_text(json.dumps(proof, indent=2) + "\n")
        assert proof["bit_equal"], proof
        completed.append(True)

    physics.model.callback.contactfilter = audit
