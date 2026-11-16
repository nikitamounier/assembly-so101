"""The viewer must expose every real entry and reject misleading source mappings."""
import json

import numpy as np
import pytest

from assembly_policy.so101.sgs_viewer import bank_rows, digest_json, discover, iter_json_array


def test_streams_every_provenance_row_across_chunk_boundaries(tmp_path):
    rows = [dict(seed=i, note="quoted \\\" bracket ] and unicode é", state=list(range(i))) for i in range(12)]
    path = tmp_path / "rows.json"
    path.write_text(json.dumps(rows))
    assert list(iter_json_array(path, chunk_size=7)) == rows
    path.write_text("[]")
    assert list(iter_json_array(path, chunk_size=1)) == []


@pytest.mark.parametrize("content", ['[{"seed":1}', '[{} ,]', '[{}] garbage', '{}', '[{}{}]'])
def test_rejects_truncated_or_malformed_provenance(tmp_path, content):
    path = tmp_path / "rows.json"
    path.write_text(content)
    with pytest.raises(ValueError):
        list(iter_json_array(path, chunk_size=3))


def make_bank(tmp_path, prepared=False):
    families = ["loose", "near"] if prepared else ["loose"]
    manifest = dict(tasks={"motor": {}}, families=families,
                    reset_distribution="subtask_loose_plus_prepared" if prepared else "all_loose")
    if prepared:
        manifest.update(independent_family_evaluation="family_eval", generation_plan={})
        (tmp_path / "family_eval").mkdir()
    rows = []
    for split, offset in [("train", 10), ("eval", 20)]:
        states = np.arange(len(families)*3, dtype=float).reshape(-1, 3)+offset
        path = tmp_path / ("family_eval" if prepared and split == "eval" else "") / f"motor_{split}.npz"
        np.savez(path, states=states, family=np.arange(len(families)), seed=np.arange(len(families))+offset)
        if prepared and split == "eval":
            np.savez(tmp_path / "motor_eval.npz", states=states[:1], family=[0], seed=[offset])
        for i, family in enumerate(families):
            rows.append(dict(state=states[i].tolist(), controls=[0]*12, task="motor", family=family,
                             split=split, seed=offset+i))
    if prepared:
        (tmp_path / "motor_provenance.json").write_text(json.dumps(rows))
    return dict(path=tmp_path, manifest=manifest), rows


def test_all_baseline_rows_and_splits_are_indexed(tmp_path):
    bank, _ = make_bank(tmp_path)
    rows = list(bank_rows(bank))
    assert len(rows) == 2
    assert {row[0]["split"] for row in rows} == {"train", "eval"}
    assert all(row[0]["controls"] is None for row in rows)


def test_all_families_indexed_and_eval_alias_counted_once(tmp_path, monkeypatch):
    from assembly_policy.so101 import subtask_curriculum
    monkeypatch.setattr(subtask_curriculum, "validate_row", lambda *args: None)
    bank, expected = make_bank(tmp_path, prepared=True)
    actual = list(bank_rows(bank))
    assert [row[0] for row in actual] == expected
    assert len(actual) == 4
    assert len([row for row in actual if row[1].startswith("family_eval/")]) == 2
    path = tmp_path / "motor_provenance.json"
    expected[-1]["state"][0] += .001
    path.write_text(json.dumps(expected))
    with pytest.raises(ValueError, match="differs from published"):
        list(bank_rows(bank))


def test_missing_provenance_and_modified_alias_rejected(tmp_path, monkeypatch):
    from assembly_policy.so101 import subtask_curriculum
    monkeypatch.setattr(subtask_curriculum, "validate_row", lambda *args: None)
    bank, rows = make_bank(tmp_path, prepared=True)
    path = tmp_path / "motor_provenance.json"
    path.write_text(json.dumps(rows[:-1]))
    with pytest.raises(ValueError, match="missing 1 published"):
        list(bank_rows(bank))
    path.write_text(json.dumps(rows))
    np.savez(tmp_path / "motor_eval.npz", states=[[99, 99, 99]], family=[0], seed=[20])
    with pytest.raises(ValueError, match="alias differs"):
        list(bank_rows(bank))


def test_discovery_separates_current_and_historical_sources(tmp_path):
    validator = {"native": "current"}
    for name, model, validation in [("banks_current", "current", validator), ("banks_old", "old", validator),
                                     ("banks_stale", "current", {"native": "old"})]:
        folder = tmp_path / name
        folder.mkdir()
        (folder / "manifest.json").write_text(json.dumps(dict(model_signature=model, validator_identity=validation,
                                                              reset_distribution="subtask_loose_plus_prepared")))
    banks, excluded = discover(tmp_path, {"model_signature": "current"}, validator)
    assert [bank["path"].name for bank in banks] == ["banks_current"]
    assert {row["name"] for row in excluded} == {"banks_old", "banks_stale"}


def test_content_identity_changes_with_saved_pose_or_camera():
    value = dict(pose=[1, 2, 3], camera="top")
    assert digest_json(value) == digest_json(dict(camera="top", pose=[1, 2, 3]))
    assert digest_json(value) != digest_json(dict(value, camera="side"))
    assert digest_json(value) != digest_json(dict(value, pose=[1, 2, 3.001]))
