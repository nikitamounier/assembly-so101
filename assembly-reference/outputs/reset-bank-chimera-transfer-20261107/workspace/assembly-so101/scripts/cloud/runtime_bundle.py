"""Create or install the hash-bound native SO101 runtime; original CAD is excluded."""

import argparse
import hashlib
import json
import shutil
import tarfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def install(archive, destination, config):
    expected = json.loads(config.read_text())
    if digest(archive) != expected["sha256"]:
        raise ValueError("Runtime archive SHA-256 mismatch")
    destination = destination.resolve()
    with tarfile.open(archive, "r:gz") as tar:
        members = tar.getmembers()
        names = [m.name for m in members]
        if len(names) != len(set(names)) or set(names) != set(expected["files"]) | {"runtime-manifest.json"}:
            raise ValueError("Unexpected runtime archive members")
        manifest = json.load(tar.extractfile("runtime-manifest.json"))
        if manifest != expected["files"]:
            raise ValueError("Runtime archive manifest mismatch")
        for m in members:
            path = PurePosixPath(m.name)
            if not m.isfile() or path.is_absolute() or ".." in path.parts:
                raise ValueError("Unsafe runtime member")
            if m.name == "runtime-manifest.json":
                continue
            target = destination / m.name
            if not target.resolve().is_relative_to(destination):
                raise ValueError("Runtime path escapes checkout")
            if target.exists():
                if digest(target) != manifest[m.name]["sha256"]:
                    raise FileExistsError(f"Existing runtime file differs: {target}")
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(target.suffix + ".installing")
            try:
                with temporary.open("xb") as out, tar.extractfile(m) as source:
                    shutil.copyfileobj(source, out)
                if (
                    temporary.stat().st_size != manifest[m.name]["bytes"]
                    or digest(temporary) != manifest[m.name]["sha256"]
                ):
                    raise ValueError(f"Runtime member failed integrity: {m.name}")
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)
    print("Native runtime installed and verified", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    q = sub.add_parser("install")
    q.add_argument("archive", type=Path)
    q.add_argument("--destination", type=Path, default=ROOT)
    q.add_argument("--config", type=Path, default=ROOT / "configs/runtime_bundle.json")
    a = p.parse_args()
    install(a.archive, a.destination, a.config)
