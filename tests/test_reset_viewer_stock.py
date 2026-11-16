"""Stock-model banks preserve integration state and scene-specific labels."""

import json
import sys

import mujoco
import numpy as np

from assembly_policy.viewer.banks import ResetCatalog
from assembly_policy.viewer.server import ResetApplication


def test_scene_manifest_supports_all_nine_families(tmp_path):
    names = [f"Cube family {i}" for i in range(9)]
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "model_signature": "cube",
                "task_names": ["cube_handoff"],
                "family_names": names,
                "state_spec": "mjSTATE_INTEGRATION",
            }
        )
    )
    np.savez(
        tmp_path / "cube_handoff_train.npz", states=np.zeros((9, 5)), family=np.arange(9), seed=np.arange(9)
    )
    catalog = ResetCatalog([tmp_path], "cube", 5, state_spec=mujoco.mjtState.mjSTATE_INTEGRATION)
    assert [entry["family_name"] for entry in catalog.entries] == names
    assert {entry["task"] for entry in catalog.entries} == {"cube_handoff"}
    incompatible = ResetCatalog([tmp_path], "cube", 5, state_spec=mujoco.mjtState.mjSTATE_FULLPHYSICS)
    assert not incompatible.entries and not incompatible.banks[0]["compatible"]


def test_injected_stock_model_restores_full_integration_state(tmp_path):
    model = mujoco.MjModel.from_xml_string("""<mujoco><worldbody><body>
        <freejoint/><geom type="box" size=".02 .02 .02"/>
        </body></worldbody></mujoco>""")
    data = mujoco.MjData(model)
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    data.time = 1.25
    data.qvel[:] = np.arange(model.nv) * 0.1
    data.qacc_warmstart[:] = np.arange(model.nv) * 0.2
    data.xfrc_applied[1] = np.arange(6) * 0.3
    state = np.empty(mujoco.mj_stateSize(model, spec))
    mujoco.mj_getState(model, data, state, spec)
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "model_signature": "stock",
                "task_names": ["cube_handoff"],
                "family_names": ["Cube"],
                "state_spec": int(spec),
            }
        )
    )
    np.savez(tmp_path / "cube_handoff_train.npz", states=state[None])
    (tmp_path / "qualification.json").write_text(
        json.dumps([{"split": "train", "row": 0, "holder": "left", "native_contacts": 2}])
    )
    viewer = ResetApplication(
        [tmp_path], sys.executable, 0, scene="cube-trial", model=model, model_signature="stock"
    )
    try:
        restored = np.empty_like(state)
        mujoco.mj_getState(viewer.model, viewer.data, restored, spec)
        np.testing.assert_array_equal(restored, state)
        assert viewer.snapshot["entry"]["family_name"] == "Cube"
        assert viewer.snapshot["qualification"]["native_contacts"] == 2
    finally:
        viewer.temp.cleanup()
