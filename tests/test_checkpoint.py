"""Verify atomic SGS/model pairing and explicit failure for incomplete resume."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from assembly_policy.rl.nets import ActorCritic
from assembly_policy.rl.sampler import ResetSampler
from assembly_policy.train import Trainer


def trainer(path, seed=1):
    torch.manual_seed(seed)
    tr = Trainer.__new__(Trainer)
    tr.run = path
    path.mkdir(exist_ok=True)
    tr.dev = torch.device("cpu")
    tr.model = ActorCritic(2, proprio_dim=4, act_dim=2)
    tr.opt = torch.optim.Adam(tr.model.parameters(), lr=1e-4)
    tr.sampler = ResetSampler(3, history=4, seed=seed)
    tr.backend = None
    tr.a = SimpleNamespace(init_policy=None, no_eval=True)
    tr.args = {}
    tr.lr = 1e-4
    tr.it = 7
    tr.transitions = 123
    return tr


def saved(path):
    tr = trainer(path)
    tr.sampler.update([0, 0, 1], [0, 1, 1])
    tr.save()
    return tr


def test_embedded_sampler_wins_over_unrelated_sidecar(tmp_path):
    original = saved(tmp_path)
    torch.save(ResetSampler(7).state_dict(), tmp_path / "sampler.pt")
    restored = trainer(tmp_path, seed=2)
    restored._maybe_resume()
    assert restored.it == 7 and restored.transitions == 123
    assert np.array_equal(restored.sampler.cnt, original.sampler.cnt)
    assert np.array_equal(restored.sampler.sample(30), original.sampler.sample(30))
    assert all(torch.equal(v, restored.model.state_dict()[k]) for k, v in original.model.state_dict().items())


def test_missing_embedded_and_sidecar_sampler_is_explicit(tmp_path):
    saved(tmp_path)
    p = tmp_path / "checkpoint.pt"
    d = torch.load(p, weights_only=False)
    d.pop("sampler")
    torch.save(d, p)
    (tmp_path / "sampler.pt").unlink()
    with pytest.raises(ValueError, match="missing SGS"):
        trainer(tmp_path, seed=2)._maybe_resume()


def test_mismatched_sampler_history_is_explicit(tmp_path):
    saved(tmp_path)
    p = tmp_path / "checkpoint.pt"
    d = torch.load(p, weights_only=False)
    d["sampler"]["buf"] = np.zeros((3, 5), dtype=np.uint8)
    torch.save(d, p)
    with pytest.raises(ValueError, match="SGS state shape"):
        trainer(tmp_path, seed=2)._maybe_resume()


def test_complete_legacy_checkpoint_still_resumes(tmp_path):
    original = saved(tmp_path)
    p = tmp_path / "checkpoint.pt"
    d = torch.load(p, weights_only=False)
    d.pop("sampler")
    torch.save(d, p)
    restored = trainer(tmp_path, seed=2)
    restored._maybe_resume()
    assert np.array_equal(restored.sampler.cnt, original.sampler.cnt)
    assert restored.it == original.it


def test_so101_resume_rejects_changed_observation_contract(tmp_path):
    import dataclasses

    from assembly_policy.so101.env import EnvConfig

    def configured():
        tr = trainer(tmp_path)
        tr.ecfg = EnvConfig(nthread=1)
        tr.backend = SimpleNamespace(resolve_tasks=lambda x: x.split("+"), EnvConfig=EnvConfig)
        tr.tasks = ["motor"]
        tr.pts = SimpleNamespace(signature="mating-v1")
        tr.args = dict(
            model_signature="model",
            bank_signature="bank",
            physics_backend="native",
            physics_signature="model",
            tasks="motor",
            observation_signature="mating-v1",
            env={k: v for k, v in dataclasses.asdict(tr.ecfg).items() if k != "nthread"},
        )
        return tr

    original = configured()
    original.save()
    configured()._maybe_resume()
    p = tmp_path / "checkpoint.pt"
    checkpoint = torch.load(p, weights_only=False)
    checkpoint["args"]["observation_signature"] = "old-uniform-layout"
    torch.save(checkpoint, p)
    with pytest.raises(ValueError, match="observation schema"):
        configured()._maybe_resume()
    checkpoint["args"].pop("observation_signature")
    torch.save(checkpoint, p)
    with pytest.raises(ValueError, match="observation schema"):
        configured()._maybe_resume()


@pytest.mark.parametrize("stored_identity", ["different", None])
def test_so101_resume_rejects_changed_or_unbound_evaluation_bank(tmp_path, stored_identity):
    import dataclasses

    from assembly_policy.so101.env import EnvConfig

    tr = trainer(tmp_path)
    tr.ecfg = EnvConfig(nthread=1)
    tr.backend = SimpleNamespace(resolve_tasks=lambda x: x.split("+"), EnvConfig=EnvConfig)
    tr.tasks = ["motor"]
    tr.pts = SimpleNamespace(signature="mating-v1")
    tr.args = dict(
        model_signature="model",
        bank_signature="training-array",
        bank_directory_signature="full-bank",
        physics_backend="native",
        physics_signature="model",
        tasks="motor",
        observation_signature="mating-v1",
        env={k: v for k, v in dataclasses.asdict(tr.ecfg).items() if k != "nthread"},
    )
    tr.save()
    path = tmp_path / "checkpoint.pt"
    checkpoint = torch.load(path, weights_only=False)
    if stored_identity is None:
        checkpoint["args"].pop("bank_directory_signature")
    else:
        checkpoint["args"]["bank_directory_signature"] = stored_identity
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match="reset bank"):
        tr._maybe_resume()
