"""Keep full-sequence and prerequisite-assembled loose collections distinct."""

import json

import numpy as np
import pytest

from assembly_policy.so101.staged_resets import collection_rows, publish
from assembly_policy.viewer.banks import ResetCatalog


def collection(root, name, distribution, *, preferred=False):
    directory = root / name
    directory.mkdir()
    (directory / "manifest.json").write_text(
        json.dumps(
            dict(
                model_signature="test",
                reset_distribution=distribution,
                default_viewer_collection=preferred,
            )
        )
    )
    np.savez(directory / "screw_train.npz", states=np.zeros((1, 5)), seed=[123], family=[0])
    return directory


def test_staged_loose_names_completed_prefix_and_is_preferred(tmp_path):
    collection(tmp_path, "all_loose", "all_loose")
    collection(tmp_path, "staged", "subtask_loose", preferred=True)
    catalog = ResetCatalog([tmp_path], "test", 5)
    staged = next(entry for entry in catalog.entries if entry["distribution"] == "subtask_loose")
    loose = next(entry for entry in catalog.entries if entry["distribution"] == "all_loose")
    assert staged["predecessors"] == ["motor", "horn"]
    assert staged["setup"] == "Prerequisites assembled"
    assert loose["predecessors"] == [] and loose["setup"] == "All parts loose"
    assert catalog.default_bank == staged["bank"]
    assert next(bank for bank in catalog.banks if bank["id"] == loose["bank"])["label"].startswith(
        "All parts loose"
    )


def test_incomplete_staged_bank_is_never_published(tmp_path):
    folder = tmp_path / "build"
    folder.mkdir()
    output = tmp_path / "published"
    with pytest.raises(ValueError, match="Incomplete staged loose"):
        publish(folder, output, dict(train_count=1, eval_count=1))
    assert not output.exists()


def test_unpublished_builds_are_not_discovered(tmp_path):
    collection(tmp_path, ".staged.building", "subtask_loose", preferred=True)
    collection(tmp_path, "staged.publishing", "subtask_loose", preferred=True)
    catalog = ResetCatalog([tmp_path], "test", 5)
    assert catalog.entries == [] and catalog.default_bank is None


def test_sharded_rows_preserve_saved_values_and_do_not_double_count_root(tmp_path):
    (tmp_path / "motor_loose.jsonl").write_text(json.dumps(dict(seed=999)) + "\n")
    for index in range(2):
        directory = tmp_path / f"shard_{index:03d}"
        directory.mkdir()
        (directory / "motor_loose.jsonl").write_text(json.dumps(dict(seed=index, state=[index, 1.5])) + "\n")
    rows = collection_rows(tmp_path, "motor", dict(shards=2))
    assert rows == [dict(seed=0, state=[0, 1.5]), dict(seed=1, state=[1, 1.5])]
