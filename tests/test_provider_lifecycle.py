"""A partially created provider pool must not leave paid resources behind."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from assembly_policy.compute import providers


def test_sail_partial_creation_cleans_up_already_created_hosts(tmp_path, monkeypatch):
    calls = []

    class Image:
        def apt_install(self, *_):
            return self

        def add_local_file(self, *_):
            return self

        def run_commands(self, *_):
            return self

        def build(self):
            return self

    class Box:
        def terminate(self):
            calls.append("terminated")

    def create(**kwargs):
        calls.append(kwargs)
        if len(calls) == 2:
            raise RuntimeError("capacity unavailable")
        return Box()

    fake = SimpleNamespace(
        App=SimpleNamespace(find=lambda **_: "app"),
        Image=SimpleNamespace(debian=lambda *_: Image()),
        AutoSleep=SimpleNamespace(never=lambda: "never"),
        Sailbox=SimpleNamespace(create=create),
    )
    monkeypatch.setitem(sys.modules, "sail", fake)
    monkeypatch.setattr(providers, "build_bundle", lambda *args: {})
    args = SimpleNamespace(hosts=2, memory_mib=16384, seconds=30, sail_size="m", region=None)
    with pytest.raises(RuntimeError, match="capacity unavailable"):
        providers.sail_pool(args, {})
    assert calls[-1] == "terminated"
    assert calls[0]["visibility"] == "private"
    assert calls[0]["max_lifetime_seconds"] == 630


def test_chimera_launcher_refuses_login_node_before_submission(monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    args = SimpleNamespace(run_dir=Path("/tmp/run"))
    with pytest.raises(RuntimeError, match="allocated CPU step"):
        providers.chimera_pool(args, {})
