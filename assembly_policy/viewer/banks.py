"""Discover saved reset collections without modifying or regenerating them."""

import hashlib
import json
from collections import OrderedDict
from pathlib import Path
from zipfile import BadZipFile

import numpy as np

from assembly_policy.so101.env import TASK_NAMES

FAMILIES = ("Loose", "Pickup", "In hand", "Partial", "Near insertion")


class ResetCatalog:
    def __init__(self, roots, model_signature, state_size):
        self.banks = []
        self.entries = []
        self.sources = {}
        self._cache = OrderedDict()
        seen = set()
        for root in roots:
            root = Path(root).expanduser().resolve()
            manifests = (
                [root / "manifest.json"]
                if (root / "manifest.json").is_file()
                else sorted(root.rglob("manifest.json"))
            )
            for path in manifests:
                if path in seen:
                    continue
                seen.add(path)
                self._read(path, model_signature, state_size)
        self.by_id = {entry["id"]: entry for entry in self.entries}

    def _read(self, path, signature, state_size):
        try:
            manifest = json.loads(path.read_text())
        except (ValueError, OSError):
            return
        if not isinstance(manifest, dict) or "model_signature" not in manifest:
            return
        directory = path.parent
        bank_id = hashlib.sha256(str(directory).encode()).hexdigest()[:12]
        bank = dict(
            id=bank_id,
            name=directory.name,
            count=0,
            compatible=manifest["model_signature"] == signature,
            errors=[],
        )
        self.banks.append(bank)
        bank["distribution"] = manifest.get("reset_distribution", "prepared")
        if not bank["compatible"]:
            bank["errors"].append("Model signature differs from the current workcell")
            return
        for archive in sorted(directory.rglob("*.npz")):
            if archive.is_symlink():
                continue
            stem = archive.stem
            if not any(stem == f"{task}_{split}" for task in TASK_NAMES for split in ("train", "eval")):
                continue
            task, split = stem.rsplit("_", 1)
            try:
                with np.load(archive, allow_pickle=False) as data:
                    states = data["states"]
                    if states.ndim != 2 or states.shape[1] != state_size or not np.isfinite(states).all():
                        raise ValueError("Invalid state dimensions or nonfinite values")
                    seeds = data["seed"] if "seed" in data else np.arange(len(states))
                    families = data["family"] if "family" in data else np.zeros(len(states), dtype=int)
                    if seeds.shape != (len(states),) or families.shape != (len(states),):
                        raise ValueError("Invalid seed or family dimensions")
                    relative = str(archive.relative_to(directory))
                    source_id = hashlib.sha256((bank_id + relative).encode()).hexdigest()[:16]
                    self.sources[source_id] = archive
                    for row in range(len(states)):
                        family = int(families[row])
                        self.entries.append(
                            dict(
                                id=f"{source_id}-{row}",
                                bank=bank_id,
                                task=task,
                                split=split,
                                family=family,
                                family_name=FAMILIES[family]
                                if 0 <= family < len(FAMILIES)
                                else f"Family {family}",
                                seed=int(seeds[row]),
                                row=row,
                                collection=relative.removesuffix(".npz"),
                            )
                        )
                    bank["count"] += len(states)
            except (ValueError, KeyError, OSError, BadZipFile, EOFError) as exc:
                bank["errors"].append(f"{archive.name}: {exc}")

    def state(self, entry_id):
        entry = self.by_id[entry_id]
        source_id = entry_id.rsplit("-", 1)[0]
        if source_id not in self._cache:
            with np.load(self.sources[source_id], allow_pickle=False) as data:
                self._cache[source_id] = data["states"].copy()
            if len(self._cache) > 4:
                self._cache.popitem(last=False)
        self._cache.move_to_end(source_id)
        return entry, self._cache[source_id][entry["row"]].copy()
