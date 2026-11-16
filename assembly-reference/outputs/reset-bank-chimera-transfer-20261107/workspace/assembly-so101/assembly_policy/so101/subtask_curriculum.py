"""Build a balanced, resumable subtask bank; publish only complete qualified sets.

Each worker owns one task/family. Full-sequence banks remain separate. State
receipts are streamed to disk so an interrupted build resumes without padding.
"""

import argparse
import hashlib
import json
import multiprocessing
import shutil
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path

import mujoco
import numpy as np

from assembly_policy.so101.limits import PENETRATION_LIMIT_M

from .bank_paths import default_training_bank_directory
from .curriculum import FAMILIES, Builder, candidate_digest, set_pose, surface
from .env import STATE_SPEC
from .resets import validation_identity
from .scene import ASSEMBLY_TASKS
from .workcell import CONFIG, configuration, environment_identity


def serial(row):
    return {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in row.items()}


def read_rows(path):
    if not path.exists():
        return []
    # An incomplete final write is rejected, never silently accepted as a state.
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def yoke_gap(builder):
    source = builder.receiver("yoke")
    m = builder.m
    d = mujoco.MjData(m)
    mujoco.mj_setState(m, d, source["state"], STATE_SPEC)
    mujoco.mj_kinematics(m, d)
    base = d.body("base")
    R = base.xmat.reshape(3, 3)
    goal = base.xpos + R @ (builder.targets["yoke"] - builder.targets["base"])
    set_pose(m, d, "yoke", goal, R @ builder.target_rotations["yoke"])
    mujoco.mj_kinematics(m, d)
    axis = R @ np.asarray(configuration()["assembly_plan"]["extractions"]["yoke"])
    axis /= np.linalg.norm(axis)

    def bounds(name):
        b = d.body(name)
        return (surface(name)[0] @ b.xmat.reshape(3, 3).T + b.xpos) @ axis

    return float(bounds("motor").max() - bounds("yoke").min() - 0.0013)


def propose(builder, task, family, seed, attempt, split, templates, variant_hint=None):
    if family == "loose":
        return builder.qualify(builder.loose_candidate(task, seed), warmup=False)
    if family == "partial" and task == "screw":
        return builder.fixture_partial(task, seed, attempt % 3)
    if task == "yoke" and family in ("near", "partial"):
        item = builder.candidate(
            task,
            family,
            seed,
            32,
            side_override="right",
            axis_gap=yoke_gap(builder) if family == "partial" else None,
            preserve_part_pose=family == "partial",
        )
        item, receipt = builder.calibrate_yoke_grasp(
            item, pad_target_m=-0.000001 if family == "near" else -0.000005
        )
        return builder.qualify(item) if item is not None else (None, receipt)
    if family == "inhand" and templates:
        return builder.noisy_grasp(
            templates[attempt % len(templates)],
            seed,
            split,
            arm_noise=0.02,
            workcell_xy=0.008,
            workcell_yaw=8.0,
        )
    measured_mating = bool(templates) and (
        (task in ("screw", "yoke") and family in ("near", "partial"))
        or (task in ("retainer", "horn") and family == "near")
    )
    if measured_mating and attempt % 4 != 3:
        item, _ = builder.noisy_grasp(
            templates[attempt % len(templates)],
            seed + 100000000,
            split,
            validate=False,
            arm_noise=0.02,
            workcell_xy=0.008,
            workcell_yaw=8.0,
        )
        if task == "yoke" and family == "partial":
            item, receipt = builder.alternate_grasp(item, -0.175, seed + 200000000)
            if item is None:
                return None, receipt
        return builder.place_grasp(
            item,
            family,
            seed,
            "left" if attempt % 2 == 0 else "right",
            axis_gap=yoke_gap(builder) if task == "yoke" and family == "partial" else None,
        )
    preferred = {"motor": (56, 44, 8, 4, 0), "retainer": (9, 5, 1, 21, 17, 13), "horn": (0, 4, 8, 12, 16)}
    variants = preferred.get(task, tuple(range(24)))
    # Direct fallbacks occur every fourth attempt; advance their own index.
    direct_index = attempt // 4 if measured_mating else attempt
    variant = (
        variants[(direct_index if measured_mating else attempt // 8) % len(variants)]
        if family == "partial"
        else direct_index % 48
    )
    if variant_hint is not None:
        variant = int(variant_hint)
    return builder.qualify(builder.candidate(task, family, seed, variant))


def calibrate(builder, task, family, folder, plan):
    """Discover a reusable grasp from native preparation without external data."""
    for path in sorted(folder.glob(f"{task}_*_calibrated.json")):
        record = json.loads(path.read_text())
        if record["validator_identity"] == plan["validator_identity"]:
            return record["template"]
    for attempt in range(64):
        seed = 3000000000 + 1000000 * ASSEMBLY_TASKS.index(task) + 100000 * FAMILIES.index(family) + attempt
        try:
            if task in ("screw", "yoke"):
                variant = 52 if task == "screw" and attempt < 16 else attempt % 48
                item = builder.candidate(task, "pickup", seed, variant)
                item, receipt = builder.qualify(item)
                if item is not None:
                    item, receipt = builder.lift_pickup(item, seed + 50000)
            else:
                item = builder.candidate(task, "inhand", seed, attempt % 48)
                item, receipt = builder.qualify(item)
            if item is not None:
                item.update(split="train", qualification=receipt)
                template = serial(item)
                write_json(
                    folder / f"{task}_{family}_calibrated.json",
                    dict(validator_identity=plan["validator_identity"], template=template),
                )
                return template
        except (ValueError, FloatingPointError) as error:
            receipt = dict(reason=str(error))
        with (folder / f"{task}_{family}_calibration_failures.jsonl").open("a") as stream:
            stream.write(json.dumps(dict(seed=seed, **receipt)) + "\n")
    raise ValueError(f"No qualified grasp calibration for {task}/{family}; inspect native candidates")


def validate_row(row, task, family, plan):
    """Reject mislabeled, stale, nonfinite or out-of-gate persisted receipts."""
    if row.get("task") != task or row.get("family") != family or row.get("split") not in ("train", "eval"):
        raise ValueError("Incorrect task/family/split metadata")
    if row.get("predecessors") != list(ASSEMBLY_TASKS[: ASSEMBLY_TASKS.index(task)]):
        raise ValueError("Incorrect prerequisite set")
    state = np.asarray(row["state"], float)
    controls = np.asarray(row["controls"], float)
    if (
        state.ndim != 1
        or controls.shape != (12,)
        or not np.isfinite(state).all()
        or not np.isfinite(controls).all()
    ):
        raise ValueError("Invalid saved state/control array")
    expected_sides = Builder.ready_sides(row)
    ready = row.get("ready_arm", {})
    if ready.get("sides") != expected_sides or not ready.get("preparation", {}).get("ok"):
        raise ValueError("Missing nearby ready-arm preparation metadata")
    receipts = [(row.get("qualification", {}), 0.0005)]
    if family == "partial":
        receipts.append((row.get("halfstep_qualification", {}), 0.00025))
    for receipt, timestep in receipts:
        if (
            not receipt.get("passed")
            or receipt.get("validator_identity") != plan["validator_identity"]
            or receipt.get("diagnostic_timestep_s") != timestep
        ):
            raise ValueError("Stale or missing qualification receipt")
        if receipt.get("candidate_digest") != candidate_digest(row):
            raise ValueError("Qualification does not match saved state/controls")
        checks = receipt.get("ready_arm_checks", [])
        minimum_checks = 1 + round((1.28 if family == "pickup" else 1.0) / 0.04)
        if len(checks) < minimum_checks or any(
            not check.get("ok") or check.get("sides") != expected_sides for check in checks
        ):
            raise ValueError("Missing or failed nearby ready-arm saved/replay checks")
        duration = receipt.get("replay_seconds", 0.0)
        if not np.isfinite(duration) or duration < (1.28 if family == "pickup" else 1.0) - 1e-9:
            raise ValueError("Qualification replay is too short")
        expected = (
            "table"
            if family == "loose"
            else (
                "fixture_supported" if row.get("support_mode") == "fixture_supported" else "bilateral_grasp"
            )
        )
        if receipt.get("support_mode") != expected:
            raise ValueError("Incorrect support mode")
        if expected == "bilateral_grasp" and receipt.get("tail_bilateral_samples", 0) < 4:
            raise ValueError("Missing loaded bilateral grasp")
        limits = {
            "max_sampled_material_overlap_m": (PENETRATION_LIMIT_M, False),
            "peak_control_sampled_depth_m": (PENETRATION_LIMIT_M, False),
            "predecessor_error_m": (0.001, True),
        }
        if family not in ("loose", "pickup"):
            limits.update(max_grasp_slip_m=(0.002, False), max_grasp_rotation_slip_rad=(0.1, False))
        if receipt.get("support_mode") == "fixture_supported":
            limits.update(max_grasp_slip_m=(0.00025, False), max_grasp_rotation_slip_rad=(0.01, False))
        for name, (limit, strict) in limits.items():
            value = receipt.get(name, float("nan"))
            if not np.isfinite(value) or value < 0 or (value >= limit if strict else value > limit):
                raise ValueError(f"Qualification exceeds {name} gate")


def generate_group(job):
    task, family, plan, folder, library = job
    folder = Path(folder)
    templates = [r for r in library if r["family"] == "inhand"]
    accepted_file = folder / f"{task}_{family}.jsonl"
    failure_file = folder / f"{task}_{family}_failures.jsonl"
    status_file = folder / f"{task}_{family}_status.json"
    rows = read_rows(accepted_file)
    failures = read_rows(failure_file)
    counts = Counter(r["split"] for r in rows)
    seen = {int(r["seed"]) for r in rows + failures}
    if (folder / "STOP").exists():
        status = dict(
            task=task, family=family, train=counts["train"], eval=counts["eval"], complete=False, stopped=True
        )
        write_json(status_file, status)
        return status
    if counts == {"train": plan["train_count"], "eval": plan["eval_count"]}:
        for row in rows:
            validate_row(row, task, family, plan)
        status = dict(task=task, family=family, train=counts["train"], eval=counts["eval"], complete=True)
        write_json(status_file, status)
        return status
    builder = Builder()
    try:
        if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != plan["generation_sha256"]:
            raise RuntimeError("Generator changed during build")
        if (
            builder.identity != plan["environment"]
            or builder.validation_identity != plan["validator_identity"]
        ):
            raise RuntimeError("Environment/validator changed; use a new build")
        if (
            not templates
            and task in ("screw", "yoke", "retainer", "horn")
            and (family in ("inhand", "near") or (task == "yoke" and family == "partial"))
        ):
            templates = [calibrate(builder, task, family, folder, plan)]
        variants = sorted(
            {
                int(r["variant"])
                for r in library + rows
                if r["family"] == family and r.get("split") == "train" and "variant" in r
            }
        )
        digests = {hashlib.sha256(np.asarray(r["state"], float)[1:].tobytes()).hexdigest() for r in rows}
        for split, wanted, offset in (
            ("train", plan["train_count"], 1000000000),
            ("eval", plan["eval_count"], 2000000000),
        ):
            for attempt in range(plan["attempts"]):
                if counts[split] >= wanted:
                    break
                if (folder / "STOP").exists():
                    status = dict(
                        task=task,
                        family=family,
                        train=counts["train"],
                        eval=counts["eval"],
                        complete=False,
                        stopped=True,
                    )
                    write_json(status_file, status)
                    return status
                if shutil.disk_usage(folder).free < 128 * 1024 * 1024:
                    raise RuntimeError(
                        "Less than 128 MiB free; checkpoint retained, free disk before resuming"
                    )
                seed = (
                    offset
                    + plan["seed_offset"]
                    + ASSEMBLY_TASKS.index(task) * 1000000
                    + FAMILIES.index(family) * 100000
                    + attempt
                )
                if seed in seen:
                    continue
                try:
                    hint = variants[attempt % len(variants)] if variants and attempt % 5 else None
                    result, receipt = propose(builder, task, family, seed, attempt, split, templates, hint)
                    # The saved pose must survive a fresh production reset, not only preparation.
                    if result is not None:
                        result, receipt = builder.prepare_ready(result)
                    if result is not None:
                        result, receipt = builder.qualify(result, warmup=False)
                    if result is not None and family == "partial":
                        builder.m.opt.timestep = 0.00025
                        builder.env.substeps = 160
                        try:
                            extra, extra_receipt = builder.qualify(result, warmup=False)
                        finally:
                            builder.m.opt.timestep = 0.0005
                            builder.env.substeps = 80
                        if extra is None:
                            result = None
                            receipt = dict(reason="halfstep_" + extra_receipt["reason"], detail=extra_receipt)
                        else:
                            result["halfstep_qualification"] = extra_receipt
                except (ValueError, FloatingPointError) as error:
                    result = None
                    receipt = dict(reason=str(error))
                if result is not None:
                    digest = hashlib.sha256(np.asarray(result["state"])[1:].tobytes()).hexdigest()
                    if digest in digests:
                        result = None
                        receipt = dict(reason="duplicate_state")
                if (
                    validation_identity() != plan["validator_identity"]
                    or hashlib.sha256(CONFIG.read_bytes()).hexdigest()
                    != plan["environment"]["configuration_sha256"]
                ):
                    raise RuntimeError("Validator changed during generation")
                if result is None:
                    with failure_file.open("a") as stream:
                        stream.write(
                            json.dumps(dict(task=task, family=family, split=split, seed=seed, **receipt))
                            + "\n"
                        )
                else:
                    digests.add(digest)
                    result.update(split=split, qualification=receipt)
                    validate_row(result, task, family, plan)
                    with accepted_file.open("a") as stream:
                        stream.write(json.dumps(serial(result)) + "\n")
                    counts[split] += 1
                    if split == "train" and "variant" in result and result["variant"] not in variants:
                        variants.append(int(result["variant"]))
                status = dict(
                    task=task,
                    family=family,
                    train=counts["train"],
                    eval=counts["eval"],
                    attempt=attempt,
                    split=split,
                    last_result=receipt.get("reason", "pass"),
                )
                write_json(status_file, status)
                print(json.dumps(status), flush=True)
        complete = counts["train"] == plan["train_count"] and counts["eval"] == plan["eval_count"]
        status = dict(task=task, family=family, train=counts["train"], eval=counts["eval"], complete=complete)
        write_json(status_file, status)
        return status
    finally:
        builder.close()


def publish(folder, output, plan):
    if (folder / "STOP").exists():
        raise ValueError("Build explicitly stopped")
    all_rows = {}
    coverage = {}
    digests = set()
    seeds = set()
    for task in ASSEMBLY_TASKS:
        rows = []
        coverage[task] = {}
        for family in FAMILIES:
            group = read_rows(folder / f"{task}_{family}.jsonl")
            counts = Counter(r["split"] for r in group)
            if counts != {"train": plan["train_count"], "eval": plan["eval_count"]}:
                raise ValueError(f"Incomplete balanced group: {task}/{family}: {dict(counts)}")
            coverage[task][family] = dict(counts)
            for row in group:
                validate_row(row, task, family, plan)
                state = np.asarray(row["state"], float)
                digest = hashlib.sha256(state[1:].tobytes()).hexdigest()
                if digest in digests or row["seed"] in seeds:
                    raise ValueError("Duplicate state or train/eval seed")
                digests.add(digest)
                seeds.add(row["seed"])
            rows.extend(group)
        all_rows[task] = rows
    if environment_identity() != plan["environment"] or validation_identity() != plan["validator_identity"]:
        raise RuntimeError("Environment/validator changed before publication")
    temporary = output.with_name("." + output.name + ".publishing")
    if output.exists():
        raise ValueError("Output already exists")
    if temporary.exists():
        marker = temporary / "publication_plan.json"
        if not marker.exists() or json.loads(marker.read_text()) != plan:
            raise ValueError("Unrecognized incomplete publication; inspect it before retrying")
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    write_json(temporary / "publication_plan.json", plan)
    (temporary / "family_eval").mkdir()
    tasks = {}
    for task, rows in all_rows.items():
        tasks[task] = {}
        for split in ("train", "eval"):
            selected = [r for r in rows if r["split"] == split]

            def save(path, items):
                np.savez_compressed(
                    path,
                    states=np.asarray([r["state"] for r in items]),
                    family=np.asarray([FAMILIES.index(r["family"]) for r in items]),
                    seed=np.asarray([r["seed"] for r in items]),
                )

            if split == "train":
                save(temporary / f"{task}_train.npz", selected)
            else:
                save(temporary / "family_eval" / f"{task}_eval.npz", selected)
                save(temporary / f"{task}_eval.npz", [r for r in selected if r["family"] == "loose"])
            counted = selected if split == "train" else [r for r in selected if r["family"] == "loose"]
            tasks[task][split] = dict(
                entries=len(counted),
                family_counts={
                    str(i): sum(r["family"] == f for r in counted)
                    for i, f in enumerate(FAMILIES)
                    if any(r["family"] == f for r in counted)
                },
            )
        (temporary / f"{task}_provenance.json").write_text(json.dumps(rows, separators=(",", ":")) + "\n")
    manifest = dict(
        model_signature=plan["environment"]["model_signature"],
        validator_identity=plan["validator_identity"],
        reset_distribution="subtask_loose_plus_prepared",
        families=list(FAMILIES),
        tasks=tasks,
        coverage=coverage,
        independent_family_evaluation="family_eval",
        evaluation_distribution="subtask loose; completed prerequisites",
        generation_plan=plan,
    )
    write_json(temporary / "manifest.json", manifest)
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != plan["generation_sha256"]:
        raise RuntimeError("Generator changed before publication")
    shutil.copy2(Path(__file__), temporary / "generation_source.py")
    if (
        (folder / "STOP").exists()
        or environment_identity() != plan["environment"]
        or validation_identity() != plan["validator_identity"]
    ):
        raise RuntimeError("Stop or source change during publication; bank remains unpublished")
    temporary.rename(output)
    return coverage


@contextmanager
def build_lock(folder):
    """One writer per build; stop requests use their separate marker."""
    import fcntl

    with (folder / ".writer.lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("This build is already running; inspect status or request --stop") from None
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=default_training_bank_directory())
    ap.add_argument(
        "--stop", action="store_true", help="Request a graceful stop of this build at candidate boundaries"
    )
    ap.add_argument("--resume-after-stop", action="store_true")
    ap.add_argument("--train-count", type=int, default=256)
    ap.add_argument("--eval-count", type=int, default=64)
    ap.add_argument("--attempts", type=int, default=12000)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed-offset", type=int, default=0)
    ap.add_argument("--grasp-source", type=Path, action="append", default=[])
    ap.add_argument(
        "--adopt-build",
        type=Path,
        action="append",
        default=[],
        help="Reuse qualified rows only when physical and validator identities match",
    )
    ap.add_argument("--tasks", default="+".join(ASSEMBLY_TASKS))
    ap.add_argument("--families", default="+".join(FAMILIES))
    args = ap.parse_args()
    folder = args.out.with_name("." + args.out.name + ".building")
    if args.stop:
        if not (folder / "plan.json").exists():
            raise ValueError("No existing build to stop")
        (folder / "STOP").write_text("Explicit stop requested; resume only with --resume-after-stop\n")
        print("Stop requested; wait for active workers to finish their current candidate", flush=True)
        return
    if (folder / "STOP").exists() and not args.resume_after_stop:
        raise ValueError(
            "Build explicitly stopped; use --resume-after-stop after verifying its source identity"
        )
    if args.out == default_training_bank_directory() and (args.train_count, args.eval_count) != (256, 64):
        raise ValueError(
            "The default training bank requires 256/64 counts; choose --out for a pilot or custom bank"
        )
    if min(args.train_count, args.eval_count, args.attempts, args.workers) < 1:
        raise ValueError("Counts must be positive")
    if args.attempts < max(args.train_count, args.eval_count):
        raise ValueError("Attempt budget is smaller than the requested state count")
    if args.attempts >= 100000 or not 0 <= args.seed_offset < 1000000000 or args.seed_offset % 10000000:
        raise ValueError("Use a seed-offset multiple of 10000000 below 1e9; fewer than 100000 attempts")
    tasks = args.tasks.split("+")
    families = args.families.split("+")
    if (
        set(tasks) - set(ASSEMBLY_TASKS)
        or set(families) - set(FAMILIES)
        or len(set(tasks)) != len(tasks)
        or len(set(families)) != len(families)
    ):
        raise ValueError("Unknown or duplicate task/family")
    if args.out.exists():
        raise ValueError("Output already exists")
    folder = args.out.with_name("." + args.out.name + ".building")
    folder.mkdir(parents=True, exist_ok=True)
    with build_lock(folder):
        run_build(args, tasks, families, folder)


def run_build(args, tasks, families, folder):
    templates = {task: [] for task in ASSEMBLY_TASKS}
    sources = {}
    for path in args.grasp_source:
        files = sorted(path.glob("*_provenance.json")) if path.is_dir() else [path]
        for file in files:
            sources[str(file.resolve())] = hashlib.sha256(file.read_bytes()).hexdigest()
            value = json.loads(file.read_text())
            rows = value if isinstance(value, list) else [value]
            for row in rows:
                if row.get("split") == "train":
                    templates[row["task"]].append(row)
    adopted = {}
    for source in args.adopt_build:
        adopted[str(source.resolve())] = hashlib.sha256((source / "plan.json").read_bytes()).hexdigest()
    plan = dict(
        adopted_builds=adopted,
        environment=environment_identity(),
        validator_identity=validation_identity(),
        train_count=args.train_count,
        eval_count=args.eval_count,
        attempts=args.attempts,
        seed_offset=args.seed_offset,
        grasp_sources=sources,
        generation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope="Balanced independent subtask starts; full-sequence evaluation remains separate",
    )
    path = folder / "plan.json"
    if path.exists() and json.loads(path.read_text()) != plan:
        raise ValueError("Existing build has different provenance; choose a new output")
    write_json(path, plan)
    if args.resume_after_stop:
        (folder / "STOP").unlink(missing_ok=True)
    for source in args.adopt_build:
        old = json.loads((source / "plan.json").read_text())
        if (
            old["environment"] != plan["environment"]
            or old["validator_identity"] != plan["validator_identity"]
        ):
            raise ValueError("Cannot adopt stale physical/validator receipts")
        for task in tasks:
            for family in families:
                target = folder / f"{task}_{family}.jsonl"
                existing = read_rows(target)
                seen = {r["seed"] for r in existing}
                counts = Counter(r["split"] for r in existing)
                for row in read_rows(source / target.name):
                    limit = plan[row["split"] + "_count"]
                    if row["seed"] in seen or counts[row["split"]] >= limit:
                        continue
                    validate_row(row, task, family, plan)
                    row["adopted_from"] = str(source)
                    with target.open("a") as stream:
                        stream.write(json.dumps(row) + "\n")
                    seen.add(row["seed"])
                    counts[row["split"]] += 1
    jobs = [(task, family, plan, str(folder), templates[task]) for task in tasks for family in families]
    write_json(folder / "build_status.json", dict(status="running", groups=len(jobs)))
    try:
        with ProcessPoolExecutor(
            max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")
        ) as pool:
            futures = {pool.submit(generate_group, job): job[:2] for job in jobs}
            for future in as_completed(futures):
                print(json.dumps(future.result()), flush=True)
        if (folder / "STOP").exists():
            write_json(folder / "build_status.json", dict(status="stopped"))
            return
        if set(tasks) == set(ASSEMBLY_TASKS) and set(families) == set(FAMILIES):
            coverage = publish(folder, args.out, plan)
            write_json(folder / "build_status.json", dict(status="complete", published=str(args.out)))
            print(json.dumps(dict(published=str(args.out), coverage=coverage)), flush=True)
        else:
            write_json(folder / "build_status.json", dict(status="selected_groups_finished"))
    except BaseException as error:
        write_json(folder / "build_status.json", dict(status="failed", reason=str(error)))
        raise


if __name__ == "__main__":
    main()
