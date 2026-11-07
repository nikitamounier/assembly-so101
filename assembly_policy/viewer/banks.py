"""Discover saved reset collections without modifying or regenerating them."""

import hashlib
import json
from collections import OrderedDict
from pathlib import Path
from zipfile import BadZipFile

import numpy as np

TASK_NAMES = ("motor", "horn", "screw", "retainer", "yoke")
FAMILIES = ("Loose", "Pickup", "In hand", "Partial", "Near insertion")


class ResetCatalog:
    def __init__(self, roots, model_signature, state_size, *, state_spec=None):
        self.banks = []
        self.entries = []
        self.sources = {}
        self.qualifications = {}
        self.variations = {}
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
                if any(part.endswith((".building", ".publishing")) for part in path.parts):
                    continue
                if path in seen:
                    continue
                seen.add(path)
                self._read(path, model_signature, state_size, state_spec)
        self.by_id = {entry["id"]: entry for entry in self.entries}
        preferred = [
            bank for bank in self.banks if bank.get("default") and bank["compatible"] and bank["count"]
        ]
        self.default_bank = preferred[-1]["id"] if preferred else None

    def _read(self, path, signature, state_size, state_spec):
        try:
            manifest = json.loads(path.read_text())
        except (ValueError, OSError):
            return
        if not isinstance(manifest, dict) or "model_signature" not in manifest:
            return
        task_names = manifest.get("task_names", TASK_NAMES)
        family_names = manifest.get("family_names", FAMILIES)
        if not isinstance(task_names, (list, tuple)) or not all(isinstance(t, str) for t in task_names):
            return
        if not isinstance(family_names, (list, tuple)) or not all(isinstance(f, str) for f in family_names):
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
        staged = bank["distribution"].startswith("subtask")
        all_loose = bank["distribution"] == "all_loose"
        bank["label"] = manifest.get(
            "display_name", "All parts loose · full sequence" if all_loose else directory.name
        )
        bank["default"] = bool(manifest.get("default_viewer_collection", False))
        bank["task_names"] = list(task_names)
        bank["family_names"] = list(family_names)
        bank["progress"] = manifest.get("progress")
        if not bank["compatible"]:
            bank["errors"].append("Model signature differs from the current workcell")
            return
        if state_spec is not None and "state_spec" in manifest:
            import mujoco

            saved_spec = manifest["state_spec"]
            if isinstance(saved_spec, str):
                saved_spec = getattr(mujoco.mjtState, saved_spec, None)
            if saved_spec != int(state_spec):
                bank["compatible"] = False
                bank["errors"].append("Saved state specification differs from the current scene")
                return
        qualifications = {}
        qualification_path = directory / "qualification.json"
        if qualification_path.is_file():
            try:
                receipts = json.loads(qualification_path.read_text())
                if isinstance(receipts, list):
                    qualifications = {
                        (receipt["split"], receipt["row"]): receipt
                        for receipt in receipts
                        if isinstance(receipt, dict) and "split" in receipt and "row" in receipt
                    }
            except (ValueError, OSError, TypeError):
                bank["errors"].append("Could not read reset qualification metadata")
        for archive in sorted(directory.rglob("*.npz")):
            if archive.is_symlink():
                continue
            stem = archive.stem
            if not any(stem == f"{task}_{split}" for task in task_names for split in ("train", "eval")):
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
                        entry_id = f"{source_id}-{row}"
                        variations = manifest.get("variation_by_task", {}).get(task, {}).get(split, [])
                        variation = variations[row] if isinstance(variations, list) and row < len(variations) else None
                        if variation is not None and not isinstance(variation, dict):
                            raise ValueError("Invalid per-row variation metadata")
                        summary = None
                        if variation is not None:
                            self.variations[entry_id] = variation
                            summary = {
                                key: variation[key]
                                for key in ("kind", "difficulty", "friction_multipliers", "connector_swivel_degrees",
                                            "training_eligible", "candidate_rejected", "rejected", "rejection_reasons", "qualification")
                                if key in variation
                            }
                        if (split, row) in qualifications:
                            self.qualifications[entry_id] = qualifications[split, row]
                        self.entries.append(
                            dict(
                                id=entry_id,
                                bank=bank_id,
                                task=task,
                                focus=manifest.get("focus_by_task", {}).get(task),
                                stage=manifest.get("stage_by_task", {}).get(task),
                                variation=summary,
                                review_kind=(variation or {}).get("kind", "nominal" if bank["distribution"] == "photo_review" else "variation"),
                                split=split,
                                family=family,
                                family_name=family_names[family]
                                if 0 <= family < len(family_names)
                                else f"Family {family}",
                                seed=int(seeds[row]),
                                row=row,
                                collection=relative.removesuffix(".npz"),
                                distribution=bank["distribution"],
                                setup="Prerequisites assembled"
                                if staged
                                else ("All parts loose" if all_loose else None),
                                predecessors=list(task_names[: task_names.index(task)]) if staged else [],
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
