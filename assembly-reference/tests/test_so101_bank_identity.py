"""Evaluation data is part of a frozen training experiment's identity."""
import hashlib
import json

import pytest

from assembly_policy.so101.resets import TASK_NAMES, directory_signature


def make_bank(path, staged):
    manifest={'independent_family_evaluation':'family_eval'} if staged else {}
    (path/'manifest.json').write_text(json.dumps(manifest))
    names=['manifest.json']
    for task in TASK_NAMES:
        for split in ('train','eval'):
            name=f'{task}_{split}.npz'
            (path/name).write_bytes(name.encode())
            names.append(name)
    if staged:
        (path/'family_eval').mkdir()
        for task in TASK_NAMES:
            (path/'family_eval'/f'{task}_eval.npz').write_bytes(task.encode())
    return names


def test_staged_identity_changes_when_held_out_family_data_changes(tmp_path):
    make_bank(tmp_path,True)
    before=directory_signature(tmp_path)
    (tmp_path/'family_eval/horn_eval.npz').write_bytes(b'changed evaluation starts')
    assert directory_signature(tmp_path)!=before


def test_declared_family_evaluation_cannot_be_missing(tmp_path):
    make_bank(tmp_path,True)
    (tmp_path/'family_eval/screw_eval.npz').unlink()
    with pytest.raises(FileNotFoundError):
        directory_signature(tmp_path)


def test_legacy_loose_bank_identity_is_preserved(tmp_path):
    names=make_bank(tmp_path,False)
    previous=hashlib.sha256()
    for name in names:
        previous.update(name.encode())
        previous.update((tmp_path/name).read_bytes())
    assert directory_signature(tmp_path)==previous.hexdigest()


def test_evaluation_directory_cannot_escape_bank(tmp_path):
    make_bank(tmp_path,False)
    (tmp_path/'manifest.json').write_text(json.dumps({'independent_family_evaluation':'../other'}))
    with pytest.raises(ValueError,match='evaluation directory'):
        directory_signature(tmp_path)


def test_undeclared_family_evaluation_cannot_escape_identity(tmp_path):
    make_bank(tmp_path,True)
    (tmp_path/'manifest.json').write_text('{}')
    with pytest.raises(ValueError,match='declared'):
        directory_signature(tmp_path)
