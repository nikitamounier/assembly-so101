"""Worker images contain frozen code/runtime, never run data or credentials."""

import json
import tarfile

from assembly_policy.compute.bundle import build_bundle


def test_bundle_excludes_credentials_runs_and_generated_banks(tmp_path):
    root = tmp_path / "source"
    files = {
        "assets_local/parts/manifest.json": "{}",
        "assembly_policy/__init__.py": "",
        "plugin/adapter.c": "int x;",
        "configs/scene.json": "{}",
        "pyproject.toml": "",
        "requirements-worker.txt": "",
        ".env": "secret",
        "runs/secret.json": "secret",
        "data/reset.npz": "bank",
        "plugin/build/native.so": "binary",
        "assembly_policy/key.pem": "secret",
    }
    for name, value in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value)
    output = tmp_path / "worker.tar.gz"
    first = build_bundle(output, root)
    second = build_bundle(tmp_path / "second.tar.gz", root)
    assert first["sha256"] == second["sha256"]
    with tarfile.open(output) as archive:
        names = archive.getnames()
        assert ".env" not in names and not any(
            name.startswith(("runs/", "data/", "plugin/build/")) for name in names
        )
        assert not any(name.endswith(".pem") for name in names)
        manifest = json.load(archive.extractfile("worker-manifest.json"))
        assert "plugin/adapter.c" in manifest
