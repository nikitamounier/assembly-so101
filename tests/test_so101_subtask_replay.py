"""Revalidation must execute both replays and preserve the saved starting pose."""

from types import SimpleNamespace

import numpy as np
import pytest

from scripts.replay_so101_subtask_build import replay_row, source_rows


def fixture(family="partial"):
    builder = SimpleNamespace(
        env=SimpleNamespace(nstate=3, substeps=80, ecfg=SimpleNamespace(control_dt=0.04)),
        m=SimpleNamespace(opt=SimpleNamespace(timestep=0.0005)),
    )
    row = dict(
        state=[0.0, 1.0, 2.0],
        controls=[0.0] * 12,
        family=family,
        qualification={"old": True},
        halfstep_qualification={"old": True},
    )
    return builder, row


def test_partial_gets_two_fresh_replays_without_preparation():
    builder, row = fixture()
    calls = []

    def qualify(item, warmup):
        assert warmup is False
        calls.append(builder.m.opt.timestep)
        return item, {"timestep": builder.m.opt.timestep}

    builder.qualify = qualify
    result, _ = replay_row(builder, row)
    assert calls == [0.0005, 0.00025]
    assert result["qualification"] == {"timestep": 0.0005}
    assert result["halfstep_qualification"] == {"timestep": 0.00025}
    assert row["qualification"] == {"old": True}
    np.testing.assert_array_equal(result["state"], row["state"])
    assert builder.env.substeps == 80 and builder.m.opt.timestep == 0.0005


def test_fine_rejection_is_not_promoted():
    builder, row = fixture()
    builder.qualify = lambda item, warmup: (
        (item, {}) if builder.m.opt.timestep == 0.0005 else (None, {"reason": "grasp_lost"})
    )
    result, receipt = replay_row(builder, row)
    assert result is None and receipt["reason"] == "halfstep_grasp_lost"
    assert builder.env.substeps == 80 and builder.m.opt.timestep == 0.0005


def test_replay_cannot_repair_a_starting_pose():
    builder, row = fixture("loose")

    def qualify(item, warmup):
        item["state"][1] += 0.001
        return item, {}

    builder.qualify = qualify
    with pytest.raises(RuntimeError, match="changed the saved"):
        replay_row(builder, row)


def test_fine_exception_restores_production_timestep():
    builder, row = fixture()

    def qualify(item, warmup):
        if builder.m.opt.timestep == 0.00025:
            raise FloatingPointError("unstable")
        return item, {}

    builder.qualify = qualify
    with pytest.raises(FloatingPointError):
        replay_row(builder, row)
    assert builder.env.substeps == 80 and builder.m.opt.timestep == 0.0005


def test_replay_reads_only_the_recorded_source_bytes(tmp_path):
    import hashlib

    path = tmp_path / "motor_loose.jsonl"
    content = b'{"seed":1}\n'
    path.write_bytes(content)
    digest = hashlib.sha256(content).hexdigest()
    assert source_rows(path, digest) == [{"seed": 1}]
    path.write_bytes(b'{"seed":2}\n')
    with pytest.raises(ValueError, match="source changed"):
        source_rows(path, digest)
    path.unlink()
    with pytest.raises(ValueError, match="source changed"):
        source_rows(path, digest)
    assert source_rows(path, None) == []
