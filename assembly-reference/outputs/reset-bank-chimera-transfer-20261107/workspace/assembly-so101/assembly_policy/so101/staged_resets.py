"""Build a qualified loose-only subtask collection for scene inspection."""

import argparse
import hashlib
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import mujoco
import numpy as np

from . import subtask_curriculum
from .curriculum import candidate_digest
from .env import STATE_SPEC
from .resets import validation_identity
from .scene import ASSEMBLY_TASKS
from .subtask_curriculum import build_lock, generate_group, read_rows, validate_row, write_json
from .workcell import default_bank_directory, environment_identity


def collection_rows(folder, task, plan):
    if "shards" not in plan:
        return read_rows(folder / f"{task}_loose.jsonl")
    return [
        row
        for shard in range(plan["shards"])
        for row in read_rows(folder / f"shard_{shard:03d}" / f"{task}_loose.jsonl")
    ]


def publish(folder, output, plan):
    """Publish only a complete loose cohort; this is not a five-family training bank."""
    rows = {}
    digests, seeds = set(), set()
    for task in ASSEMBLY_TASKS:
        rows[task] = collection_rows(folder, task, plan)
        for split in ("train", "eval"):
            group = [row for row in rows[task] if row["split"] == split]
            if len(group) != plan[split + "_count"]:
                raise ValueError(f"Incomplete staged loose group: {task}/{split}")
            for row in group:
                validate_row(row, task, "loose", plan)
                digest = candidate_digest(row)
                if digest in digests or row["seed"] in seeds:
                    raise ValueError("Duplicate staged reset or seed")
                digests.add(digest)
                seeds.add(row["seed"])
    if environment_identity() != plan["environment"] or validation_identity() != plan["validator_identity"]:
        raise RuntimeError("Environment changed during staged reset generation")
    if output.exists():
        raise FileExistsError(output)
    temporary = output.with_name(output.name + ".publishing")
    if temporary.exists():
        raise FileExistsError(f"Inspect incomplete publication: {temporary}")
    temporary.mkdir()
    for task, records in rows.items():
        for split in ("train", "eval"):
            group = [row for row in records if row["split"] == split]
            np.savez_compressed(
                temporary / f"{task}_{split}.npz",
                states=np.asarray([row["state"] for row in group]),
                controls=np.asarray([row["controls"] for row in group]),
                family=np.zeros(len(group), dtype=np.int64),
                seed=np.asarray([row["seed"] for row in group], dtype=np.int64),
            )
        write_json(temporary / f"{task}_provenance.json", records)
    write_json(
        temporary / "manifest.json",
        dict(
            model_signature=plan["environment"]["model_signature"],
            validator_identity=plan["validator_identity"],
            state_spec=int(STATE_SPEC),
            mujoco_version=mujoco.__version__,
            reset_distribution="subtask_loose",
            display_name="Staged loose · prerequisites assembled",
            families=["loose"],
            default_viewer_collection=True,
            scope="Loose subtask reset inspection; active part on table, completed prerequisites dynamic. Not a complete five-family training bank.",
            tasks={
                task: {split: dict(entries=plan[split + "_count"]) for split in ("train", "eval")}
                for task in ASSEMBLY_TASKS
            },
            generation_plan=plan,
        ),
    )
    if environment_identity() != plan["environment"] or validation_identity() != plan["validator_identity"]:
        raise RuntimeError("Environment changed before publication; bank remains unpublished")
    temporary.rename(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=default_bank_directory().with_name(default_bank_directory().name + "_staged_loose"),
    )
    parser.add_argument("--count", type=int, default=8, help="Independent TRAIN and EVAL starts per task")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--shard-size", type=int, default=2)
    parser.add_argument(
        "--adopt-build", type=Path, help="Reuse accepted states from a stopped, identical build"
    )
    parser.add_argument("--attempts", type=int, default=128)
    args = parser.parse_args()
    if min(args.count, args.workers, args.attempts, args.shard_size) < 1 or args.attempts < args.count:
        parser.error("Counts must be positive and attempts must cover the requested count")
    if args.out.exists():
        parser.error("Output already exists; choose a new versioned directory")
    folder = args.out.with_name("." + args.out.name + ".building")
    folder.mkdir(parents=True, exist_ok=True)
    shards = (args.count + args.shard_size - 1) // args.shard_size
    if 70000000 + (shards - 1) * 10000000 >= 1000000000:
        parser.error("Too many shards for independent seed ranges; increase --shard-size")
    plan = dict(
        environment=environment_identity(),
        validator_identity=validation_identity(),
        train_count=args.count,
        eval_count=args.count,
        attempts=args.attempts,
        seed_offset=70000000,
        generation_sha256=hashlib.sha256(Path(subtask_curriculum.__file__).read_bytes()).hexdigest(),
        shards=shards,
        shard_size=args.shard_size,
    )
    adopted = {}
    if args.adopt_build:
        with build_lock(args.adopt_build):
            source_plan = json.loads((args.adopt_build / "plan.json").read_text())
            if any(source_plan[key] != plan[key] for key in ("environment", "validator_identity")):
                raise ValueError("Cannot adopt states from a different environment or validator")
            for task in ASSEMBLY_TASKS:
                adopted[task] = collection_rows(args.adopt_build, task, source_plan)
                for row in adopted[task]:
                    validate_row(row, task, "loose", plan)
                if any(
                    sum(row["split"] == split for row in adopted[task]) > args.count
                    for split in ("train", "eval")
                ):
                    raise ValueError("Requested count would discard accepted states")
    with build_lock(folder):
        path = folder / "plan.json"
        if path.exists() and json.loads(path.read_text()) != plan:
            raise ValueError("Existing staged build identity differs; choose a new output")
        write_json(path, plan)
        jobs = []
        for shard in range(shards):
            directory = folder / f"shard_{shard:03d}"
            directory.mkdir(exist_ok=True)
            count = min(args.shard_size, args.count - shard * args.shard_size)
            shard_plan = plan | dict(
                train_count=count, eval_count=count, seed_offset=70000000 + shard * 10000000
            )
            for task in ASSEMBLY_TASKS:
                target = directory / f"{task}_loose.jsonl"
                if task in adopted and not target.exists():
                    rows = [
                        row
                        for split in ("train", "eval")
                        for row in [r for r in adopted[task] if r["split"] == split][
                            shard * args.shard_size : shard * args.shard_size + count
                        ]
                    ]
                    target.write_text("".join(json.dumps(row) + "\n" for row in rows))
                jobs.append((task, "loose", shard_plan, str(directory), []))
        with ProcessPoolExecutor(
            max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")
        ) as pool:
            futures = [pool.submit(generate_group, job) for job in jobs]
            for future in as_completed(futures):
                result = future.result()
                print(json.dumps(result), flush=True)
                if not result.get("complete"):
                    raise RuntimeError("Staged reset group incomplete; receipts retained")
        publish(folder, args.out, plan)
        print(
            json.dumps(dict(published=str(args.out), states=len(ASSEMBLY_TASKS) * args.count * 2)), flush=True
        )


if __name__ == "__main__":
    main()
