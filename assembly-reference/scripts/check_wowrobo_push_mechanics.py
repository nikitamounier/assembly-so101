"""Check that exterior pushes and Cartesian drag add no artificial rotation."""

import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from assembly_policy.so101.wowrobo_balanced_push import BalancedPush
from assembly_policy.so101.wowrobo_contact_backend import load_backend
from scripts.probe_wowrobo_aligned_force import SPECS, descendants

ROOT = Path(__file__).resolve().parents[1]


def main():
    base = ROOT / "outputs/linear-insertion-aligned-20261107"
    source = ROOT / "outputs/linear-insertion-updated-20261107"
    load_backend(source / "grids/manifest.json")
    rows = []
    for step, spec in SPECS.items():
        config_path = base / step / "alignment.json"
        config = json.loads(config_path.read_text())
        model = mujoco.MjModel.from_xml_path(str(source / "scene.xml"))
        data = mujoco.MjData(model)
        data.qpos[:] = config["start_qpos"]
        model.geom_contype[:] = 0
        model.geom_conaffinity[:] = 0
        model.body_contype[:] = 0
        model.body_conaffinity[:] = 0
        model.opt.gravity[:] = 0
        model.dof_damping[:] = 0
        mujoco.mj_forward(model, data)
        active = descendants(model, spec[2])
        push = BalancedPush(model, data, active, config["axis_world"])
        rotations = data.xmat.copy()
        # Free root and all active internal hinges: the alignment guide is absent.
        for _ in range(500):
            push.apply(data, 0.5)
            mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)
        rotation_error = max(
            float(np.max(np.abs(data.xmat[b] - rotations[b]))) for b in active
        )
        jp = np.zeros((3, model.nv))
        jr = np.zeros_like(jp)
        axis = np.array(config["axis_world"])
        lateral_speed = 0.0
        for body in active:
            mujoco.mj_jacBodyCom(model, data, jp, jr, body)
            velocity = jp @ data.qvel
            lateral_speed = max(
                lateral_speed,
                float(np.linalg.norm(velocity - (velocity @ axis) * axis)),
            )
        assert rotation_error < 1e-7
        assert lateral_speed < 1e-8
        rows.append(
            {
                "step": step,
                "alignment_sha256": hashlib.sha256(
                    config_path.read_bytes()
                ).hexdigest(),
                "max_rotation_matrix_error": rotation_error,
                "max_lateral_velocity_m_per_s": lateral_speed,
                **push.description(),
            }
        )
    result = {
        "passed": True,
        "test": "500 free-root contact-off steps, constant 0.5N, no guide; all original active hinges free",
        "scope": "Applied-load and drag mechanics only, not collision fit or seating",
        "steps": rows,
    }
    (base / "free-force-validation.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
