"""Export every current-model bank entry for read-only browser inspection.

Run through scripts/so101_sgs_viewer.py. This exporter never steps physics,
repairs states, changes bank eligibility, or writes to its input banks.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
WEB = ROOT / "tools/sgs_viewer"
VIEWS = {"side": (135, -22, .9), "top": (90, -89.8, .9), "detail": (135, -35, .24)}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, separators=(",", ":"), allow_nan=False) + "\n")
    temporary.replace(path)


def iter_json_array(path, chunk_size=1024 * 1024):
    """Stream a provenance array, retaining at most one row plus a read chunk."""
    decoder = json.JSONDecoder()
    with Path(path).open() as stream:
        buffer, eof = "", False

        def fill():
            nonlocal buffer, eof
            block = stream.read(chunk_size)
            buffer += block
            eof = not block

        def whitespace():
            nonlocal buffer
            buffer = buffer.lstrip()
            while not buffer and not eof:
                fill()
                buffer = buffer.lstrip()

        whitespace()
        if not buffer.startswith("["):
            raise ValueError(f"{path}: expected a JSON array")
        buffer = buffer[1:]
        whitespace()
        if buffer.startswith("]"):
            buffer = buffer[1:]
        else:
            while True:
                while True:
                    try:
                        row, end = decoder.raw_decode(buffer)
                        break
                    except json.JSONDecodeError:
                        if eof:
                            raise ValueError(f"{path}: incomplete or malformed provenance") from None
                        fill()
                if not isinstance(row, dict):
                    raise ValueError(f"{path}: expected a provenance object")
                buffer = buffer[end:]
                yield row
                whitespace()
                if buffer.startswith("]"):
                    buffer = buffer[1:]
                    break
                if not buffer.startswith(","):
                    raise ValueError(f"{path}: expected comma or closing bracket")
                buffer = buffer[1:]
                whitespace()
        whitespace()
        if buffer:
            raise ValueError(f"{path}: data after provenance array")


def archive_rows(bank, task, split, families):
    relative = f"{task}_{split}.npz"
    if split == "eval" and bank["manifest"].get("independent_family_evaluation"):
        if bank["manifest"]["independent_family_evaluation"] != "family_eval":
            raise ValueError("Unsupported family evaluation location")
        relative = f"family_eval/{relative}"
    path = bank["path"] / relative
    with np.load(path, allow_pickle=False) as archive:
        states = archive["states"]
        seeds = archive["seed"]
        ids = archive["family"]
    if states.ndim != 2 or len(states) != len(seeds) or len(states) != len(ids):
        raise ValueError(f"{path}: inconsistent state metadata")
    rows = {}
    for index, (state, family, seed) in enumerate(zip(states, ids, seeds)):
        if not 0 <= int(family) < len(families) or int(family) != family:
            raise ValueError(f"{path}: unknown family")
        key = (split, families[int(family)], int(seed))
        if key in rows:
            raise ValueError(f"{path}: duplicate entry key {key}")
        if not np.isfinite(state).all():
            raise ValueError(f"{path}: nonfinite state")
        rows[key] = (index, state)
    # The loose-only EVAL archive is an alias, not additional SGS entries.
    if relative.startswith("family_eval/"):
        with np.load(bank["path"] / f"{task}_eval.npz", allow_pickle=False) as archive:
            loose = [(key, value) for key, value in rows.items() if key[1] == "loose"]
            if len(loose) != len(archive["states"]):
                raise ValueError("Loose EVAL alias count differs from family EVAL")
            for index, (key, (_, state)) in enumerate(loose):
                if (not np.array_equal(state, archive["states"][index])
                        or key[2] != int(archive["seed"][index]) or int(archive["family"][index]) != families.index("loose")):
                    raise ValueError("Loose EVAL alias differs from family EVAL")
    return relative, rows


def bank_rows(bank):
    """Yield ALL entries; cross-check provenance against the published arrays."""
    from .subtask_curriculum import validate_row

    manifest = bank["manifest"]
    families = manifest["families"]
    for task in manifest["tasks"]:
        archives, remaining = {}, {}
        for split in ("train", "eval"):
            relative, rows = archive_rows(bank, task, split, families)
            archives[split] = relative
            remaining.update(rows)
        provenance = bank["path"] / f"{task}_provenance.json"
        if provenance.exists():
            for row in iter_json_array(provenance):
                key = (row["split"], row["family"], int(row["seed"]))
                if key not in remaining:
                    raise ValueError(f"{provenance}: duplicate or unpublished row {key}")
                index, state = remaining.pop(key)
                if not np.array_equal(state, np.asarray(row["state"], dtype=float)):
                    raise ValueError(f"{provenance}: row differs from published NPZ")
                validate_row(row, task, row["family"], manifest["generation_plan"])
                yield row, archives[row["split"]], index
            if remaining:
                raise ValueError(f"{provenance}: missing {len(remaining)} published entries")
        else:
            if manifest.get("reset_distribution") != "all_loose":
                raise ValueError(f"{bank['path']}: prepared bank lacks qualification provenance")
            for (split, family, seed), (index, state) in remaining.items():
                yield dict(state=state.tolist(), controls=None, task=task, family=family,
                           seed=seed, split=split), archives[split], index


def discover(directory, environment, validator):
    banks, excluded = [], []
    for path in sorted(Path(directory).glob("banks*")):
        source = path / "manifest.json"
        if not source.is_file():
            continue
        manifest = json.loads(source.read_text())
        reason = None
        if manifest.get("model_signature") != environment["model_signature"]:
            reason = "Historical CAD/model identity; not rendered with the current geometry"
        elif manifest.get("validator_identity") not in (None, validator):
            reason = "Validator changed; replay/requalification required"
        elif manifest.get("reset_distribution") not in ("all_loose", "subtask_loose_plus_prepared"):
            reason = "Unsupported bank distribution"
        if reason:
            excluded.append(dict(name=path.name, reason=reason, model_signature=manifest.get("model_signature")))
        else:
            banks.append(dict(path=path, manifest=manifest))
    banks.sort(key=lambda bank: bank["manifest"].get("reset_distribution") == "all_loose")
    return banks, excluded


def export(directory, output, width=800, height=560):
    import gzip
    import mujoco
    from PIL import Image
    from . import scene
    from .curriculum import STATE_SPEC
    from .resets import validation_identity, directory_signature
    from .workcell import environment_identity
    from teleop.visual_scene import scene_payload

    started = time.monotonic()
    directory, output = Path(directory).resolve(), Path(output).resolve()
    if output == directory or directory in output.parents:
        raise ValueError("Export must be outside the input bank directory")
    environment, validator = environment_identity(), validation_identity()
    banks, excluded = discover(directory, environment, validator)
    if not banks:
        raise ValueError("No supported current-model banks found")
    # Hash all inputs before reading, and recheck after export to detect concurrent edits.
    sources = {str(path): sha(path) for bank in banks for path in bank["path"].rglob("*")
               if path.is_file() and path.suffix in (".json", ".npz")}
    output.mkdir(parents=True, exist_ok=True)
    for name in ("index.html", "style.css", "app.mjs", "core.mjs", "inspect.mjs"):
        shutil.copy2(WEB / name, output / name)
    vendor = output / "vendor"
    vendor.mkdir(exist_ok=True)
    for name in ("three.module.js", "three.core.js", "LICENSE", "PROVENANCE.json"):
        shutil.copy2(ROOT / "teleop/web/vendor" / name, vendor / name)
    shutil.copy2(ROOT / "teleop/web/immersive.mjs", output / "native-scene.mjs")
    model = scene.model(contact_monitor=True)
    data = mujoco.MjData(model)
    geometry = scene_payload(model, environment["model_signature"])
    geometry_hash = hashlib.sha256(gzip.decompress(geometry)).hexdigest()
    geometry_name = f"scene-{geometry_hash[:24]}.json.gz"
    (output / geometry_name).write_bytes(geometry)
    settings = dict(width=width, height=height, views=VIEWS, lookat=[0, .09, .11],
                    geometry=geometry_hash, exporter=sha(__file__), mujoco=mujoco.__version__)
    render_key = digest_json(settings)
    entries, summaries = [], []
    renderer = mujoco.Renderer(model, height=height, width=width)
    rendered, reused, pose_checks = 0, 0, 0
    try:
        for bank in banks:
            bank_id = directory_signature(bank["path"])
            counts = Counter()
            for row, archive, index in bank_rows(bank):
                state = np.asarray(row["state"], dtype=float)
                if state.shape != (mujoco.mj_stateSize(model, STATE_SPEC),):
                    raise ValueError("Saved state has the wrong model dimensions")
                identity = dict(bank=bank["path"].name, model=environment["model_signature"],
                                task=row["task"], family=row["family"], split=row["split"],
                                seed=row["seed"], state=row["state"], controls=row["controls"])
                state_id = digest_json(identity)
                relative = f"states/{state_id}"
                folder = output / relative
                folder.mkdir(parents=True, exist_ok=True)
                image_prefix = f"{relative}/{render_key[:24]}"
                images = {view: f"{image_prefix}-{view}.jpg" for view in VIEWS}
                thumb = f"{image_prefix}-thumb.jpg"
                mujoco.mj_resetData(model, data)
                mujoco.mj_setState(model, data, state, STATE_SPEC)
                if row["controls"] is not None:
                    data.ctrl[:] = row["controls"]
                mujoco.mj_forward(model, data)
                restored = np.empty_like(state)
                mujoco.mj_getState(model, data, restored, STATE_SPEC)
                if not np.array_equal(state, restored):
                    raise ValueError("Static rendering changed the saved physics state")
                poses = np.concatenate((data.xpos, data.xquat), axis=1)
                if not np.isfinite(poses).all():
                    raise ValueError("Nonfinite native body poses")
                pose_checks += 1
                for view, (azimuth, elevation, distance) in VIEWS.items():
                    path = output / images[view]
                    cached = False
                    if path.exists():
                        try:
                            with Image.open(path) as image:
                                cached = image.size == (width, height)
                                image.verify()
                        except (OSError, ValueError):
                            cached = False
                    if cached:
                        reused += 1
                        continue
                    camera = mujoco.MjvCamera()
                    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
                    camera.lookat[:] = data.body(row["task"]).xpos if view == "detail" else settings["lookat"]
                    camera.distance = distance
                    camera.azimuth, camera.elevation = azimuth, elevation
                    renderer.update_scene(data, camera=camera)
                    image = Image.fromarray(renderer.render())
                    image.save(path, quality=88)
                    if view == "side":
                        image.resize((320, round(height * 320 / width))).save(output / thumb, quality=78)
                    rendered += 1
                if not (output / thumb).exists():
                    with Image.open(output / images["side"]) as image:
                        image.resize((320, round(height * 320 / width))).save(output / thumb, quality=78)
                qualification = row.get("qualification", {})
                status = "qualified" if qualification.get("passed") else "loose baseline"
                metadata = {key: value for key, value in row.items() if key != "state"}
                detail_name = f"{relative}/detail-{digest_json(row)[:24]}.json"
                write_json(output / detail_name, dict(id=state_id, state=row["state"], metadata=metadata,
                           body_poses=poses.tolist(), active_body=int(model.body(row["task"]).id),
                           scope="Native saved snapshot; no physics stepping"))
                entry = dict(id=state_id, bank=bank["path"].name, task=row["task"], family=row["family"],
                             split=row["split"], seed=row["seed"], side=row.get("side"), status=status,
                             material_mm=qualification.get("max_sampled_material_overlap_m"),
                             depth_mm=qualification.get("peak_control_sampled_depth_m"),
                             thumb=thumb, images=images, detail=detail_name, archive=archive, archive_index=index)
                for metric in ("material_mm", "depth_mm"):
                    if entry[metric] is not None:
                        entry[metric] *= 1000
                entries.append(entry)
                counts[(row["task"], row["family"], row["split"])] += 1
                if len(entries) % 10 == 0:
                    progress = dict(entries=len(entries), bank=bank["path"].name, elapsed_s=round(time.monotonic()-started, 2))
                    write_json(output / "progress.json", progress)
                    print(json.dumps(progress), flush=True)
            summaries.append(dict(name=bank["path"].name, identity=bank_id, entries=sum(counts.values()),
                                  distribution=bank["manifest"]["reset_distribution"],
                                  coverage=[dict(task=k[0], family=k[1], split=k[2], count=v) for k, v in counts.items()]))
    finally:
        renderer.close()
    if len({entry["id"] for entry in entries}) != len(entries):
        raise ValueError("Duplicate state IDs")
    if any(sha(path) != expected for path, expected in sources.items()):
        raise RuntimeError("Bank changed during export; rerun before publishing the manifest")
    if environment_identity() != environment or validation_identity() != validator:
        raise RuntimeError("Model/validator changed during export")
    receipt = dict(environment=environment, validator_identity=validator, sources=sources,
                   static_state_parity_checks=pose_checks, entries=len(entries),
                   rendered_images=rendered, reused_images=reused, elapsed_s=round(time.monotonic()-started, 2),
                   render_settings=settings, scope="Static native snapshots; not trajectories or new qualification")
    write_json(output / "render_receipt.json", receipt)
    manifest = dict(version=1, environment=environment, geometry=geometry_name, banks=summaries, excluded=excluded,
                    entries=entries, receipt="render_receipt.json", created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    write_json(output / "manifest.json", manifest)
    print(json.dumps(dict(entries=len(entries), images=len(entries)*len(VIEWS), index=str(output / "index.html"),
                          elapsed_s=receipt["elapsed_s"])), flush=True)
    return manifest


class ViewerHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        # The small index can change after an incremental export; assets use content keys.
        if self.path.split("?")[0] in ("/manifest.json", "/progress.json", "/render_receipt.json"):
            self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        super().end_headers()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--banks", type=Path, default=ROOT / "data/so101")
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts_local/sgs-viewer")
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--height", type=int, default=560)
    parser.add_argument("--serve", action="store_true", help="Serve an existing export without loading MuJoCo")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()
    if args.serve:
        from functools import partial
        if not (args.out / "manifest.json").is_file():
            parser.error("Export first; no manifest exists at --out")
        server = ThreadingHTTPServer((args.host, args.port), partial(ViewerHandler, directory=str(args.out.resolve())))
        print(f"SGS viewer: http://{args.host}:{server.server_port}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()
    else:
        # Avoid oversubscribing the independent training job while making previews.
        os.environ.setdefault("MUJOCO_GL", "egl" if os.name == "posix" and os.uname().sysname == "Linux" else "glfw")
        export(args.banks, args.out, args.width, args.height)


if __name__ == "__main__":
    main()
