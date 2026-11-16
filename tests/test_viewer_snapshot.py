"""A moved viewer snapshot retains its state contract and detects bad assets."""

import hashlib
import json

import mujoco
import numpy as np
import pytest

from assembly_policy.viewer.banks import ResetCatalog
from assembly_policy.viewer.snapshot import load_snapshot


def fixture_snapshot(folder):
    folder.mkdir()
    xml = '<mujoco><worldbody><body><freejoint/><geom type="box" size=".02 .02 .02"/></body></worldbody></mujoco>'
    (folder / "scene.xml").write_text(xml)
    (folder / "stages").mkdir()
    contract = dict(model_signature="source-model")
    bank = dict(**contract, task_names=["setup"], family_names=["Reference"])
    (folder / "stages/manifest.json").write_text(json.dumps(bank))
    for name in ("point-features.json", "reward-regions.json"):
        (folder / name).write_text(json.dumps(contract))
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    data.qpos[0] = 0.25
    state = np.empty(mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_FULLPHYSICS))
    mujoco.mj_getState(model, data, state, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    np.savez_compressed(folder / "stages/setup_train.npz", states=state[None])
    manifest = dict(
        schema=1,
        scene="scene.xml",
        bank="stages",
        point_config="point-features.json",
        reward_config="reward-regions.json",
        model_signature="source-model",
        state_count=1,
        files={
            p.relative_to(folder).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in folder.rglob("*")
            if p.is_file()
        },
    )
    (folder / "snapshot.json").write_text(json.dumps(manifest))
    return state


def test_moved_snapshot_has_no_source_directory_dependency(tmp_path):
    original = tmp_path / "original"
    state = fixture_snapshot(original)
    moved = tmp_path / "friend clone with spaces"
    original.rename(moved)
    options, manifest = load_snapshot(moved)
    catalog = ResetCatalog(options["roots"], options["model_signature"], len(state))
    assert len(catalog.entries) == manifest["state_count"] == 1
    np.testing.assert_array_equal(catalog.state(catalog.entries[0]["id"])[1], state)


@pytest.mark.parametrize("action", ["change", "remove"])
def test_snapshot_rejects_changed_or_missing_runtime_file(tmp_path, action):
    folder = tmp_path / "snapshot"
    fixture_snapshot(folder)
    path = folder / "stages/setup_train.npz"
    if action == "change":
        path.write_bytes(b"incomplete bank")
    else:
        path.unlink()
    with pytest.raises(ValueError, match="missing or changed"):
        load_snapshot(folder)


def test_snapshot_rejects_paths_outside_its_root(tmp_path):
    folder = tmp_path / "snapshot"
    fixture_snapshot(folder)
    path = folder / "snapshot.json"
    manifest = json.loads(path.read_text())
    manifest["files"]["../outside"] = "anything"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="leaves its directory"):
        load_snapshot(folder)
