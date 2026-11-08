"""Contact-positive exhaustive oracle and thread-local scratch qualification."""

import importlib.util
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import mujoco
import numpy as np

root, experiment = map(Path, sys.argv[1:3])
canonical = root / "robotics-assembly-3.0"
name = "_wowrobo_training_native_backend"
spec = importlib.util.spec_from_file_location(
    name, canonical / "assembly_policy/so101/wowrobo_contact_backend.py"
)
backend = importlib.util.module_from_spec(spec)
sys.modules[name] = backend
spec.loader.exec_module(backend)
backend.SOURCE = experiment / "candidate"
load = backend.load_backend
backend.load_backend = lambda manifest: load(manifest, experiment / "candidate/native")
from assembly_policy.wowrobo.env import BatchedAssemblyEnv, EnvConfig

bank = canonical / "assets_local/wowrobo/training-accepted-v5"
meta = json.loads((bank / "manifest.json").read_text())
env = BatchedAssemblyEnv(
    meta["task_names"],
    np.zeros(1, np.int64),
    EnvConfig(
        profile_path=meta["profile_path"], review_model=meta["review_model"], nthread=1
    ),
)
m = env.m
state_spec = mujoco.mjtState.mjSTATE_INTEGRATION
cases = []
for ti, task in enumerate(meta["task_names"]):
    with np.load(bank / (task + "_train.npz")) as z:
        for stage in np.unique(z["stage"]):
            ids = np.flatnonzero(z["stage"] == stage)
            proto = [i for i in ids if np.all(z["states"][i, -3:] == 1)]
            if not proto:
                continue
            env.set_states(np.array([0]), z["states"][proto[0]][None], np.array([ti]))
            packed = np.empty(mujoco.mj_stateSize(m, state_spec))
            mujoco.mj_getState(m, env.data[0], packed, state_spec)
            for perturbation in range(3):
                cases.append((task, str(stage), perturbation, packed.copy()))


def capture(case, mode):
    _, _, perturbation, state = case
    backend.set_surface_mode(mode)
    d = mujoco.MjData(m)
    mujoco.mj_setState(m, d, state, state_spec)
    if perturbation:
        rng = np.random.default_rng(193 + perturbation)
        # Contact-positive material-boundary probes; no admission claim.
        for j in range(m.njnt):
            address = m.jnt_qposadr[j]
            if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                d.qpos[address : address + 3] += rng.uniform(-0.0003, 0.0003, 3)
                q = d.qpos[address + 3 : address + 7]
                q += rng.uniform(-0.001, 0.001, 4)
                q /= np.linalg.norm(q)
    mujoco.mj_forward(m, d)
    contacts = np.array(
        [np.r_[c.dist, c.pos, c.frame, c.geom, c.friction] for c in d.contact]
    )
    forces = np.zeros((d.ncon, 6))
    for i in range(d.ncon):
        mujoco.mj_contactForce(m, d, i, forces[i])
    return [
        contacts,
        forces,
        d.qacc.copy(),
        d.qfrc_constraint.copy(),
        d.sensordata.copy(),
    ]


try:
    expected = [capture(case, 2) for case in cases]
    # Model is immutable, each worker owns mjData and native scratch/cache.
    with ThreadPoolExecutor(max_workers=2) as pool:
        actual = list(pool.map(lambda case: capture(case, 1), cases))
    count = 0
    for case, x, y in zip(cases, expected, actual, strict=True):
        for left, right in zip(x, y, strict=True):
            np.testing.assert_array_equal(left, right, err_msg=str(case[:3]))
            assert left.tobytes() == right.tobytes(), str(case[:3])
            count += 1
    result = {
        "exact_parity": True,
        "cases": len(cases),
        "arrays_compared": count,
        "max_contacts": max(len(x[0]) for x in expected),
        "model_signature": meta["model_signature"],
        "scope": "Exhaustive serial oracle vs two-thread candidate: original poses plus free-body translation/quaternion perturbations",
    }
    (experiment / "oracle-parity.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
finally:
    env.close()
