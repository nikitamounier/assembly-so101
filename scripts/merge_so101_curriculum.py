"""Merge disjoint task shards after verifying their states and qualification receipts."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

from assembly_policy.so101.curriculum import DEPTH, FAMILIES, validation_identity
from assembly_policy.so101.resets import check_bank, directory_signature
from assembly_policy.so101.scene import ASSEMBLY_TASKS
from assembly_policy.so101.workcell import default_bank_directory, environment_identity


def merge(sources, parent, output, additional_checks=()):
    if output.exists():
        raise ValueError("Use a new output directory")
    check_bank(parent)
    parent_hash = directory_signature(parent)
    identity = environment_identity()
    validator = validation_identity()
    owned = {}
    reports = []
    generator_sources = {}
    all_seen = {}
    all_state_hashes = {}
    extra_groups = set()
    extra_passed = {}
    extra_reports = []
    excluded = []
    for source in additional_checks:
        check_bank(source)
        report = json.loads((source / "qualification.json").read_text())
        if (
            report["environment"] != identity
            or report.get("validator_identity") != validator
            or report["parent_directory_signature"] != parent_hash
        ):
            raise ValueError("Additional replay uses a different environment or validator")
        for failure in report["failures"]:
            extra_groups.add((failure["task"], failure["family"]))
        for file in source.glob("*_provenance.json"):
            for row in json.loads(file.read_text()):
                receipt = row["qualification"]
                if not receipt.get("passed") or receipt.get("validator_identity") != validator:
                    raise ValueError("Invalid additional qualification receipt")
                extra_groups.add((row["task"], row["family"]))
                key = (row["task"], row["family"], row["seed"], row["split"])
                extra_passed[key] = np.asarray(row["state"], float)
        extra_reports.append(dict(source=source.name, report=report))
    for source in sources:
        check_bank(source)
        report = json.loads((source / "qualification.json").read_text())
        if report["environment"] != identity or report["parent_directory_signature"] != parent_hash:
            raise ValueError("Shard uses a different physical environment or parent bank")
        if report.get("validator_identity") != validator or report.get("diagnostic_timestep_s") != 0.0005:
            raise ValueError("Shard needs fresh replay at the canonical timestep")
        code = (source / "generator_source.py").read_bytes()
        if hashlib.sha256(code).hexdigest() != report["generator_sha256"]:
            raise ValueError("Generating source does not match its receipt")
        generator_sources[report["generator_sha256"]] = code
        for file in source.glob("*_provenance.json"):
            task = file.name.removesuffix("_provenance.json")
            if task not in ASSEMBLY_TASKS:
                raise ValueError("Unknown task shard")
            rows = json.loads(file.read_text())
            retained = []
            for row in rows:
                key = (task, row["family"], row["seed"], row["split"])
                if (task, row["family"]) in extra_groups:
                    if key not in extra_passed:
                        excluded.append(
                            dict(
                                task=task,
                                family=row["family"],
                                seed=row["seed"],
                                split=row["split"],
                                reason="additional_replay_missing_or_failed",
                            )
                        )
                        continue
                    if not np.array_equal(np.asarray(row["state"], float), extra_passed[key]):
                        raise ValueError("Additional replay refers to a different saved state")
                retained.append(row)
            rows = retained
            seen = all_seen.setdefault(task, {})
            state_hashes = all_state_hashes.setdefault(task, {})
            for row in rows:
                receipt = row["qualification"]
                if (
                    receipt.get("validator_identity") != validator
                    or receipt.get("diagnostic_timestep_s") != 0.0005
                ):
                    raise ValueError("Stale or diagnostic-only qualification receipt")
                if (
                    not receipt.get("passed")
                    or receipt["max_sampled_material_overlap_m"] > DEPTH
                    or receipt["peak_control_sampled_depth_m"] > DEPTH
                ):
                    raise ValueError("An unqualified state reached a shard")
                if receipt["predecessor_error_m"] >= 0.001 or (
                    row["family"] != "pickup" and receipt["max_grasp_slip_m"] > 0.002
                ):
                    raise ValueError("A state exceeds the predecessor/grasp limits")
                split = row["split"]
                key = (row["family"], row["seed"])
                if key in seen:
                    raise ValueError("Repeated seed or TRAIN/EVAL overlap")
                seen[key] = split
                s = np.asarray(row["state"], float)
                if not np.isfinite(s).all():
                    raise ValueError("Nonfinite saved state")
                digest = hashlib.sha256(s[1:].tobytes()).hexdigest()
                if digest in state_hashes:
                    raise ValueError("Duplicate physical states")
                state_hashes[digest] = split
                path = (
                    source / f"{task}_train.npz"
                    if split == "train"
                    else source / "family_eval" / f"{task}_eval.npz"
                )
                with np.load(path) as z:
                    indices = np.flatnonzero(
                        (z["seed"] == row["seed"]) & (z["family"] == FAMILIES.index(row["family"]))
                    )
                    if len(indices) != 1 or not np.array_equal(z["states"][indices[0]], s):
                        raise ValueError("Saved array does not match provenance")
            owned.setdefault(task, []).extend(rows)
        reports.append(dict(source=source.name, report=report))
    if set(owned) != set(ASSEMBLY_TASKS):
        raise ValueError("Expected one completed shard for every task")
    for task, rows in owned.items():
        for family in FAMILIES[1:]:
            for split in ("train", "eval"):
                if not any(r["family"] == family and r["split"] == split for r in rows):
                    raise ValueError(f"Missing qualified coverage: {task}/{family}/{split}")
    temporary = output.with_name("." + output.name + ".merging")
    if temporary.exists():
        raise ValueError("An incomplete merge exists")
    shutil.copytree(parent, temporary)
    (temporary / "family_eval").mkdir()
    coverage = {}
    available = {0}
    new_states = 0
    for task, rows in owned.items():
        coverage[task] = {}
        for split in ("train", "eval"):
            selected = [r for r in rows if r["split"] == split]
            with np.load(parent / f"{task}_{split}.npz") as z:
                states = [z["states"]]
                family = [z["family"]]
                seed = [z["seed"]]
            if selected:
                states.append(np.array([r["state"] for r in selected]))
                family.append(np.array([FAMILIES.index(r["family"]) for r in selected]))
                seed.append(np.array([r["seed"] for r in selected]))
            target = (
                temporary / f"{task}_train.npz"
                if split == "train"
                else temporary / "family_eval" / f"{task}_eval.npz"
            )
            np.savez_compressed(
                target,
                states=np.concatenate(states),
                family=np.concatenate(family),
                seed=np.concatenate(seed),
            )
            new_states += len(selected)
            for f in FAMILIES:
                count = int(sum(len(a[a == FAMILIES.index(f)]) for a in family))
                coverage[task].setdefault(f, {})[split] = count
                if count:
                    available.add(FAMILIES.index(f))
        (temporary / f"{task}_provenance.json").write_text(json.dumps(rows, indent=2) + "\n")
    manifest = json.loads((temporary / "manifest.json").read_text())
    manifest.update(
        validator_identity=validator,
        parent_directory_signature=parent_hash,
        reset_distribution="loose_plus_qualified_prepared",
        families=[FAMILIES[i] for i in sorted(available)],
        requested_families=list(FAMILIES),
        coverage=coverage,
        noise_bounds=reports[0]["report"]["noise_bounds"],
        qualification_report="qualification.json",
        independent_family_evaluation="family_eval",
        coverage_note="Accepted noisy prepared starts plus original loose states. Original loose evaluation preserved; inspect zero-coverage families.",
    )
    for task in ASSEMBLY_TASKS:
        counts = {str(FAMILIES.index(f)): v["train"] for f, v in coverage[task].items() if v["train"]}
        manifest["tasks"][task]["train"]["family_counts"] = counts
        manifest["tasks"][task]["train"]["entries"] = sum(counts.values())
        if (temporary / f"{task}_eval.npz").read_bytes() != (parent / f"{task}_eval.npz").read_bytes():
            raise RuntimeError("Loose evaluation changed")
    for digest, code in generator_sources.items():
        (temporary / f"generator_{digest[:12]}.py").write_bytes(code)
    (temporary / "merge_source.py").write_bytes(Path(__file__).read_bytes())
    report = dict(
        environment=identity,
        validator_identity=validator,
        parent_directory_signature=parent_hash,
        coverage=coverage,
        added_states=new_states,
        additional_checks=extra_reports,
        excluded_by_additional_checks=excluded,
        checks=dict(
            provenance_matches_arrays=True,
            no_duplicate_states=True,
            independent_seed_sets=True,
            original_loose_evaluation_unchanged=True,
        ),
        shards=reports,
    )
    (temporary / "qualification.json").write_text(json.dumps(report, indent=2) + "\n")
    (temporary / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if (
        directory_signature(parent) != parent_hash
        or environment_identity() != identity
        or validation_identity() != validator
    ):
        raise RuntimeError("Canonical environment or parent changed")
    temporary.rename(output)
    print(json.dumps(dict(output=str(output), added_states=new_states, coverage=coverage), indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parent", type=Path, default=default_bank_directory())
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--additional-check",
        type=Path,
        action="append",
        default=[],
        help="Require passing extra replay for every state in its audited task/family groups",
    )
    ap.add_argument("sources", type=Path, nargs="+")
    a = ap.parse_args()
    merge(a.sources, a.parent, a.out, a.additional_check)
