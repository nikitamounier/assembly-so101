"""Prevent stale native plugins after header edits, copied checkouts or parallel starts."""

import shutil
from concurrent.futures import ThreadPoolExecutor

import pytest

from assembly_policy.so101 import native_build


@pytest.fixture
def checkout(tmp_path, monkeypatch):
    root = tmp_path / "checkout"
    for name in native_build.SOURCES:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name)
    monkeypatch.setattr(native_build, "ROOT", root)
    calls = []

    def compile_plugins(*args, **kwargs):
        calls.append(args)
        for name in native_build.LIBRARIES:
            (root / "plugin/build" / name).write_bytes(b"compiled plugin")

    monkeypatch.setattr(native_build.subprocess, "run", compile_plugins)
    return root, calls


def test_parallel_cold_starts_compile_once(checkout):
    _, calls = checkout
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda _: native_build.ensure_plugins(), range(8)))
    assert len(calls) == 1


@pytest.mark.parametrize(
    "name",
    [
        "plugin/so101_grid_bounds.h",
        "plugin/so101_surface_tree.h",
        "plugin/build.sh",
        "plugin/build_contact_search.sh",
        "scripts/so101_build_plugins.sh",
    ],
)
def test_content_change_rebuilds_even_with_preserved_mtime(checkout, name):
    import os

    root, calls = checkout
    native_build.ensure_plugins()
    path = root / name
    original = path.stat()
    path.write_text(path.read_text() + "\nchanged")
    os.utime(path, ns=(original.st_atime_ns, original.st_mtime_ns))
    native_build.ensure_plugins()
    assert len(calls) == 2


def test_missing_or_modified_binary_rebuilds(checkout):
    root, calls = checkout
    native_build.ensure_plugins()
    library = root / "plugin/build" / native_build.LIBRARIES[0]
    library.write_bytes(b"stale or replaced library")
    native_build.ensure_plugins()
    library.unlink()
    native_build.ensure_plugins()
    assert len(calls) == 3


def test_copied_checkout_rebuilds_its_own_libraries(checkout, tmp_path, monkeypatch):
    root, calls = checkout
    native_build.ensure_plugins()
    copied = tmp_path / "copied"
    shutil.copytree(root, copied)
    monkeypatch.setattr(native_build, "ROOT", copied)
    native_build.ensure_plugins()
    assert len(calls) == 2
