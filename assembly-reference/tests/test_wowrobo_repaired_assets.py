"""Repaired geometry must match the frozen insertion candidate, not old CAD."""

import json

import numpy as np
import pytest
import trimesh

from assembly_policy.so101.wowrobo_repaired_assets import (
    DEFAULT_OUTPUT,
    DEFAULT_READY,
    export_repaired_assets,
    frozen_mesh_sources,
    sha256,
)


@pytest.fixture(scope="module")
def repaired():
    if not DEFAULT_READY.exists():
        pytest.skip("Requires the local frozen repaired-CAD candidate")
    return export_repaired_assets(DEFAULT_OUTPUT)


def test_adapter_imports_all_frozen_mesh_bindings_and_exact_frames(repaired):
    path, report = repaired
    _, _, sources = frozen_mesh_sources()
    manifest = json.loads((path / "manifest.json").read_text())
    assert len(sources) == len(manifest["constituents"]) == 16
    assert set(report["changed_constituents"]) == {
        "shoulder_connector_printed_0",
        "shoulder_connector_printed_1",
        "upper_arm_module_printed_0",
        "pin",
    }
    for name, source in sources.items():
        assert sha256(path / manifest["constituents"][name]["mesh"]) == sha256(source)
    assert (report["nq"], report["nv"], report["nu"]) == (51, 46, 12)
    assert all(value == 0 for value in report["fixed_frame_max_errors"].values())
    assert report["target_source_vertex_max_error_m"] == 0
    assert report["helper_mesh_geoms_checked"] == 38
    assert report["helper_source_vertex_max_error_m"] == 0


def test_repaired_pin_metadata_and_exported_mesh_agree(repaired):
    path, _ = repaired
    manifest = json.loads((path / "manifest.json").read_text())
    pin = manifest["pin_mm"]
    assert pin["total_length"] == 42 and pin["shaft_length"] == 14
    assert pin["grip_length"] == 28 and pin["shaft_diameter"] == 2.3
    mesh = trimesh.load_mesh(
        path / manifest["constituents"]["pin"]["mesh"], process=False
    )
    np.testing.assert_allclose(
        mesh.bounds, [[-0.006, -0.006, 0], [0.006, 0.006, 0.042]], atol=1e-12
    )
    radius = np.linalg.norm(mesh.vertices[mesh.vertices[:, 2] > 0.02801, :2], axis=1)
    # The frozen OBJ rounds metre coordinates at1e-8; verify its intended
    # diameter within that source precision, preserving the exact source bytes.
    np.testing.assert_allclose(radius[radius > 0], 0.00115, atol=1e-8, rtol=0)
    assert "step_sha256" not in manifest["pieces"]["assembly_pin"]


def test_modified_frozen_source_is_rejected_before_export(tmp_path):
    if not DEFAULT_READY.exists():
        pytest.skip("Requires the local frozen repaired-CAD candidate")
    ready = json.loads(DEFAULT_READY.read_text())
    changed = tmp_path / "changed.xml"
    changed.write_text("<mujoco/>")
    ready["source_scene"] = str(changed)
    ready_file = tmp_path / "READY.json"
    ready_file.write_text(json.dumps(ready))
    with pytest.raises(ValueError, match="Frozen CAD candidate changed: source_scene"):
        export_repaired_assets(tmp_path / "export", ready_path=ready_file)
    assert not (tmp_path / "export").exists()
