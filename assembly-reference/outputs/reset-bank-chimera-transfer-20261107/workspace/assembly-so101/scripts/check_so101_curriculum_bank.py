"""Replay proposed starts under the current validator and publish only survivors."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

from assembly_policy.so101.curriculum import FAMILIES, NOISE, Builder, validation_identity
from assembly_policy.so101.resets import check_bank, directory_signature
from assembly_policy.so101.scene import ASSEMBLY_TASKS
from assembly_policy.so101.workcell import default_bank_directory, environment_identity


def proposals(paths):
    seen = {}
    for path in paths:
        if path.is_file():
            value = json.loads(path.read_text())
            rows = value if isinstance(value, list) else [value]
        else:
            rows = []
            for file in path.glob("*_provenance.json"):
                rows.extend(json.loads(file.read_text()))
            if (path / "accepted.jsonl").exists():
                rows.extend(
                    json.loads(line) for line in (path / "accepted.jsonl").read_text().splitlines() if line
                )
        for row in rows:
            if (
                row["task"] not in ASSEMBLY_TASKS
                or row["family"] not in FAMILIES[1:]
                or row.get("split") not in ("train", "eval")
            ):
                raise ValueError("Invalid task/family/split provenance")
            key = (row["task"], row["family"], row["seed"], row["split"])
            digest = hashlib.sha256(np.asarray(row["state"], float).tobytes()).hexdigest()
            if key in seen:
                if seen[key][0] != digest:
                    raise ValueError("Same seed identifies conflicting states")
                continue
            seen[key] = (digest, row | dict(proposal_source=str(path)))
    return [pair[1] for pair in seen.values()]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("sources", type=Path, nargs="+")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--parent", type=Path, default=default_bank_directory())
    ap.add_argument("--tasks", default="+".join(ASSEMBLY_TASKS))
    ap.add_argument("--timestep", type=float, default=0.0005)
    args = ap.parse_args()
    tasks = args.tasks.split("+")
    if set(tasks) - set(ASSEMBLY_TASKS):
        raise ValueError("Unknown task")
    if args.out.exists():
        raise ValueError("Choose a new output")
    check_bank(args.parent)
    parent_hash = directory_signature(args.parent)
    rows = [r for r in proposals(args.sources) if r["task"] in tasks]
    temporary = args.out.with_name("." + args.out.name + ".building")
    if temporary.exists():
        raise ValueError("Incomplete output exists")
    shutil.copytree(args.parent, temporary)
    (temporary / "family_eval").mkdir()
    builder = Builder()
    builder.m.opt.timestep = args.timestep
    builder.env.substeps = round(builder.env.ecfg.control_dt / args.timestep)
    source = Path(__file__).parents[1] / "assembly_policy/so101/curriculum.py"
    code = source.read_bytes()
    (temporary / "generator_source.py").write_bytes(code)
    (temporary / "revalidator_source.py").write_bytes(Path(__file__).read_bytes())
    report = dict(
        environment=builder.identity,
        validator_identity=builder.validation_identity,
        parent_directory_signature=parent_hash,
        generator_sha256=hashlib.sha256(code).hexdigest(),
        diagnostic_timestep_s=args.timestep,
        noise_bounds=NOISE,
        coverage={},
        failures=[],
        scope="Fresh production-action replay of saved proposals; prepared starts do not prove an assembly path",
    )
    accepted = []
    try:
        for row in rows:
            item = row.copy()
            item["state"] = np.asarray(row["state"], float)
            item["controls"] = np.asarray(row["controls"], float)
            if item["state"].shape != (builder.env.nstate,) or item["controls"].shape != (12,):
                raise ValueError("Wrong state/control shape")
            try:
                result, receipt = builder.qualify(item, warmup=False)
            except (ValueError, FloatingPointError) as error:
                result = None
                receipt = dict(reason=str(error))
            if result is None:
                report["failures"].append(
                    dict(
                        task=row["task"],
                        family=row["family"],
                        seed=row["seed"],
                        split=row["split"],
                        **receipt,
                    )
                )
            else:
                result["prior_qualification"] = row.get("qualification")
                result["qualification"] = receipt
                accepted.append(result)
                with (temporary / "accepted.jsonl").open("a") as stream:
                    stream.write(
                        json.dumps(
                            {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in result.items()}
                        )
                        + "\n"
                    )
            print(
                json.dumps(
                    dict(
                        task=row["task"],
                        family=row["family"],
                        seed=row["seed"],
                        split=row["split"],
                        result=receipt.get("reason", "pass"),
                    )
                ),
                flush=True,
            )
        for task in tasks:
            selected = [r for r in accepted if r["task"] == task]
            report["coverage"][task] = {
                family: {
                    split: sum(r["family"] == family and r["split"] == split for r in selected)
                    for split in ("train", "eval")
                }
                for family in FAMILIES[1:]
            }
            for split in ("train", "eval"):
                group = [r for r in selected if r["split"] == split]
                with np.load(args.parent / f"{task}_{split}.npz") as old:
                    states = [old["states"]]
                    families = [old["family"]]
                    seeds = [old["seed"]]
                if group:
                    states.append(np.stack([r["state"] for r in group]))
                    families.append(np.array([FAMILIES.index(r["family"]) for r in group]))
                    seeds.append(np.array([r["seed"] for r in group]))
                target = (
                    temporary / f"{task}_train.npz"
                    if split == "train"
                    else temporary / "family_eval" / f"{task}_eval.npz"
                )
                np.savez_compressed(
                    target,
                    states=np.concatenate(states),
                    family=np.concatenate(families),
                    seed=np.concatenate(seeds),
                )
            (temporary / f"{task}_provenance.json").write_text(
                json.dumps(
                    [
                        {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in r.items()}
                        for r in selected
                    ],
                    indent=2,
                )
                + "\n"
            )
        (temporary / "qualification.json").write_text(json.dumps(report, indent=2) + "\n")
        manifest = json.loads((temporary / "manifest.json").read_text())
        available = {0}
        for task in tasks:
            with np.load(temporary / f"{task}_train.npz") as z:
                ids, counts = np.unique(z["family"], return_counts=True)
                available.update(int(x) for x in ids)
                manifest["tasks"][task]["train"]["family_counts"] = {
                    str(int(i)): int(n) for i, n in zip(ids, counts)
                }
                manifest["tasks"][task]["train"]["entries"] = len(z["states"])
        manifest.update(
            families=[FAMILIES[i] for i in sorted(available)],
            reset_distribution="loose_plus_qualified_prepared",
            qualification_report="qualification.json",
            validator_identity=builder.validation_identity,
            independent_family_evaluation="family_eval",
            noise_bounds=NOISE,
        )
        (temporary / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        if environment_identity() != builder.identity or validation_identity() != builder.validation_identity:
            raise RuntimeError(
                "Physical or validator source changed during replay; retained unpublished receipts"
            )
        if directory_signature(args.parent) != parent_hash:
            raise RuntimeError("Parent bank changed")
    finally:
        builder.close()
    temporary.rename(args.out)
    print(json.dumps(report["coverage"], indent=2), flush=True)


if __name__ == "__main__":
    main()
