"""Reset inspection preserves native state and renderer geometry."""

import asyncio
import json
import sys

import mujoco
import numpy as np
import pytest
from aiohttp.test_utils import TestClient, TestServer

from assembly_policy.so101.env import STATE_SPEC
from assembly_policy.so101.resets import signature
from assembly_policy.so101.robot import home
from assembly_policy.so101.scene import model
from assembly_policy.viewer.banks import ResetCatalog
from assembly_policy.viewer.scene import display_xml, scene_poses, visual_scene
from assembly_policy.viewer.server import ResetApplication


def bank(tmp_path, states, *, model_signature="test"):
    directory = tmp_path / "bank"
    directory.mkdir(exist_ok=True)
    (directory / "manifest.json").write_text(json.dumps(dict(model_signature=model_signature)))
    np.savez(directory / "motor_train.npz", states=states, seed=[42, 43], family=[0, 2])
    family = directory / "family_eval"
    family.mkdir(exist_ok=True)
    np.savez(family / "horn_eval.npz", states=states, seed=[142, 143], family=[1, 4])
    return directory


def test_catalog_discovers_family_holdouts_and_rejects_stale_models(tmp_path):
    directory = bank(tmp_path, np.zeros((2, 5)))
    catalog = ResetCatalog([tmp_path, directory], "test", 5)
    assert len(catalog.banks) == 1 and len(catalog.entries) == 4
    assert {entry["family_name"] for entry in catalog.entries} == {
        "Loose",
        "Pickup",
        "In hand",
        "Near insertion",
    }
    assert any(entry["collection"].startswith("family_eval/") for entry in catalog.entries)
    stale = ResetCatalog([directory], "different", 5)
    assert not stale.entries and not stale.banks[0]["compatible"]


def test_catalog_does_not_accept_paths_as_state_ids_and_returns_copies(tmp_path):
    directory = bank(tmp_path, np.zeros((2, 5)))
    catalog = ResetCatalog([directory], "test", 5)
    entry_id = catalog.entries[0]["id"]
    _, state = catalog.state(entry_id)
    state[:] = 7
    assert not catalog.state(entry_id)[1].any()
    with pytest.raises(KeyError):
        catalog.state("../../secrets")


@pytest.mark.parametrize("states", [np.zeros((2, 4)), np.full((2, 5), np.nan)])
def test_catalog_reports_invalid_arrays(tmp_path, states):
    catalog = ResetCatalog([bank(tmp_path, states)], "test", 5)
    assert not catalog.entries and len(catalog.banks[0]["errors"]) == 2


@pytest.fixture(scope="module")
def native_scene():
    native = model()
    data = mujoco.MjData(native)
    home(native, data)
    state = np.empty(mujoco.mj_stateSize(native, STATE_SPEC))
    mujoco.mj_getState(native, data, state, STATE_SPEC)
    return native, data, state


def test_display_mesh_world_vertices_match_native_model(native_scene):
    native, data, _ = native_scene
    mujoco.mj_kinematics(native, data)
    scene = visual_scene(native)
    poses = np.asarray(scene_poses(native, data, scene))
    display = mujoco.MjModel.from_xml_string(display_xml(scene))
    shown = mujoco.MjData(display)
    shown.mocap_pos[:] = poses[:, :3]
    shown.mocap_quat[:] = poses[:, 3:]
    mujoco.mj_forward(display, shown)
    assert display.nplugin == 0 and display.njnt == 0
    for i, geom in enumerate(scene["geoms"]):
        if "mesh" not in geom:
            continue
        native_vertices = np.asarray(scene["meshes"][geom["mesh"]]["vertices"]).reshape(-1, 3)
        expected = native_vertices @ data.geom_xmat[geom["id"]].reshape(3, 3).T + data.geom_xpos[geom["id"]]
        mesh_id = display.geom_dataid[i]
        offset, count = display.mesh_vertadr[mesh_id], display.mesh_vertnum[mesh_id]
        vertices = display.mesh_vert[offset : offset + count]
        actual = vertices @ shown.geom_xmat[i].reshape(3, 3).T + shown.geom_xpos[i]
        np.testing.assert_allclose(actual, expected, atol=2e-7)


def test_selection_preserves_full_saved_state_and_bank_bytes(tmp_path, native_scene):
    native, _, state = native_scene
    directory = bank(tmp_path, np.stack([state, state]), model_signature=signature())
    path = directory / "motor_train.npz"
    before = path.read_bytes()
    app = ResetApplication([directory], sys.executable, 8765)
    try:
        snapshot = app.select(next(entry["id"] for entry in app.catalog.entries if entry["task"] == "motor"))
        restored = np.empty_like(state)
        mujoco.mj_getState(app.model, app.data, restored, STATE_SPEC)
        np.testing.assert_array_equal(restored, state)
        assert snapshot["time"] == state[0] and snapshot["version"] == 2
        assert path.read_bytes() == before
        assert native.nq == len(snapshot["qpos"])
    finally:
        app.temp.cleanup()


def test_http_selection_is_validated_and_cross_origin_control_is_refused(tmp_path, native_scene):
    _, _, state = native_scene
    directory = bank(tmp_path, np.stack([state, state]), model_signature=signature())

    async def check():
        viewer = ResetApplication([directory], sys.executable, 8765)
        async with TestClient(TestServer(viewer.app())) as client:
            response = await client.get("/api/catalog")
            assert response.headers["Cross-Origin-Embedder-Policy"] == "require-corp"
            catalog = await response.json()
            entry_id = catalog["entries"][0]["id"]
            response = await client.post("/api/select", json={"id": entry_id})
            assert (await response.json())["entry"]["id"] == entry_id
            assert (await client.post("/api/select", json={"id": "../file"})).status == 400
            assert (
                await client.post(
                    "/api/select", json={"id": entry_id}, headers={"Origin": "https://other.example"}
                )
            ).status == 403

    asyncio.run(check())


def test_official_webviewer_assets_state_stream_and_cleanup(tmp_path, native_scene):
    pytest.importorskip("mujoco.studio")
    _, _, state = native_scene
    directory = bank(tmp_path, np.stack([state, state]), model_signature=signature())

    async def check():
        viewer = ResetApplication([directory], sys.executable, 0)
        async with TestClient(TestServer(viewer.app())) as client:
            viewer.port = client.server.port
            response = await client.post("/api/studio")
            assert response.status == 200, await response.text()
            assert (await client.get("/mujoco/")).status == 200
            font = await client.get("/mujoco/assets/AtkinsonHyperlegibleNext[wght].ttf")
            assert font.status == 200 and len(await font.read()) > 1000
            binary = await client.get("/model")
            assert binary.status == 200 and len(await binary.read()) > 1000
            async with client.ws_connect("/state") as stream:
                message = await stream.receive(timeout=10)
                assert message.type.name in ("BINARY", "TEXT")
            process = viewer.studio_process
            assert process.poll() is None
            assert (await client.post(
                "/api/studio/stop", headers={"Origin": "https://other.example"}
            )).status == 403
            assert process.poll() is None
            assert (await client.post("/api/studio/stop")).status == 200
            assert process.poll() is not None
            assert (await client.get("/mujoco/")).status == 503
            assert (await client.post("/api/studio/stop")).status == 200
            response = await client.post("/api/studio")
            assert response.status == 200, await response.text()
            assert viewer.studio_process is not process
            assert (await client.get("/mujoco/")).status == 200
        assert viewer.studio_process.poll() is not None

    asyncio.run(check())
