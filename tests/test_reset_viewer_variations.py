"""Variation receipts remain correctly associated with stage/split/row."""

import json

import numpy as np

from assembly_policy.viewer.banks import ResetCatalog


def test_variation_metadata_does_not_cross_stage_or_split(tmp_path):
    progress = {"accepted_train": 1, "target_train": 16000}
    manifest = {
        "model_signature": "same",
        "task_names": ["08_connector_aligned", "20_upper_transport"],
        "family_names": ["Review candidate"],
        "reset_distribution": "photo_variations",
        "progress": progress,
        "variation_by_task": {
            "08_connector_aligned": {
                "train": [
                    {"kind": "nominal"},
                    {
                        "kind": "variation",
                        "difficulty": 0.3,
                        "training_eligible": False,
                        "metadata": {"noise": {"lateral_m": [0.001, 0.002]}},
                    },
                ],
                "eval": [{"kind": "variation", "difficulty": 0.8}],
            },
            "20_upper_transport": {
                "train": [
                    {"kind": "variation", "candidate_rejected": True, "rejection_reasons": ["pin overlap"]}
                ]
            },
        },
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    for task, split, count in [
        ("08_connector_aligned", "train", 2),
        ("08_connector_aligned", "eval", 1),
        ("20_upper_transport", "train", 1),
    ]:
        np.savez(tmp_path / f"{task}_{split}.npz", states=np.zeros((count, 5)), seed=np.arange(count))
    catalog = ResetCatalog([tmp_path], "same", 5)
    rows = {(e["task"], e["split"], e["row"]): e for e in catalog.entries}
    assert rows["08_connector_aligned", "train", 0]["review_kind"] == "nominal"
    assert rows["08_connector_aligned", "train", 1]["variation"]["difficulty"] == 0.3
    entry = rows["08_connector_aligned", "train", 1]
    assert "metadata" not in entry["variation"]
    assert catalog.variations[entry["id"]]["metadata"]["noise"]["lateral_m"] == [0.001, 0.002]
    assert rows["08_connector_aligned", "eval", 0]["variation"]["difficulty"] == 0.8
    assert rows["20_upper_transport", "train", 0]["variation"]["candidate_rejected"]
    assert catalog.banks[0]["progress"] == progress
    assert catalog.state(rows["08_connector_aligned", "train", 1]["id"])[1].shape == (5,)


def test_refresh_discovers_appended_states_and_preserves_selection(tmp_path):
    import sys

    import mujoco

    from assembly_policy.viewer.server import ResetApplication

    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><body><freejoint/><geom type="box" size=".02 .02 .02"/></body></worldbody></mujoco>'
    )
    data = mujoco.MjData(model)
    spec = mujoco.mjtState.mjSTATE_FULLPHYSICS
    state = np.empty(mujoco.mj_stateSize(model, spec))
    mujoco.mj_getState(model, data, state, spec)
    manifest = {
        "model_signature": "same",
        "task_names": ["08_connector_aligned"],
        "family_names": ["Candidate"],
        "state_spec": int(spec),
        "reset_distribution": "photo_variations",
        "progress": {"generated_count": 1},
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    np.savez(tmp_path / "08_connector_aligned_train.npz", states=state[None, :])
    viewer = ResetApplication(
        [tmp_path], sys.executable, 0, scene="wowrobo-layout", model=model, model_signature="same"
    )
    try:
        selected = viewer.snapshot["entry"]["id"]
        np.savez(tmp_path / "08_connector_aligned_train.npz", states=np.stack([state, state]))
        manifest["progress"]["generated_count"] = 2
        (tmp_path / "manifest.json").write_text(json.dumps(manifest))
        viewer.refresh_catalog()
        assert len(viewer.catalog.entries) == 2
        assert viewer.snapshot["entry"]["id"] == selected
        assert viewer.catalog.banks[0]["progress"]["generated_count"] == 2
        np.testing.assert_array_equal(viewer.snapshot["qpos"], data.qpos)
    finally:
        viewer.temp.cleanup()
