"""Partition an unchanged native generator without changing its quota identity."""

from __future__ import annotations

import argparse
import ast
import builtins
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path

DEFAULT_ROOT = Path(
    "/large_storage/goodarzilab/nikitamounier/assembly/reset-bank-19200-20261107-v1"
)
CAPS = (20, 40, 80, 160)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


class NamespacedHashlib:
    """Only exact randomizer RNG keys acquire the new namespace."""

    def __init__(self, namespace, stages):
        if not namespace or "\0" in namespace:
            raise ValueError("Invalid RNG namespace")
        self.prefix = namespace.encode() + b"\0"
        self.pattern = re.compile(
            rb"(?:train|eval):(?:"
            + b"|".join(re.escape(s.encode()) for s in stages)
            + rb"):(?:0|[1-9][0-9]*)\Z"
        )

    def sha256(self, data=b"", *args, **kwargs):
        if isinstance(data, bytes) and self.pattern.fullmatch(data):
            data = self.prefix + data
        return hashlib.sha256(data, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(hashlib, name)


def inspect_generator(path):
    source = Path(path).read_bytes()
    tree = ast.parse(source)
    calls = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "range"
    ]
    expressions = {ast.unparse(n) for n in calls}
    expected = {"range(target[split])", "range(args.max_attempts if args.full else 1)"}
    if len(calls) != 2 or expressions != expected:
        raise ValueError(f"Unexpected generator range calls: {expressions}")
    return hashlib.sha256(source).hexdigest()


def partition_range(part, parts, publish_only=False):
    def selected(*args):
        if len(args) == 1 and args[0] in (640, 128):
            return (
                builtins.range(0)
                if publish_only
                else builtins.range(part, args[0], parts)
            )
        return builtins.range(*args)

    return selected


def assignment(root, index):
    tasks = json.loads((root / "assignment.json").read_text())["tasks"]
    matches = [t for t in tasks if t["index"] == index]
    if len(matches) != 1:
        raise ValueError("Task index is missing or repeated")
    task = matches[0]
    if task["parts"] != 64 or not 0 <= task["part"] < 64:
        raise ValueError("Expected 64 partitions")
    return task


def context(root):
    settings = json.loads((root / "settings.json").read_text())
    if (settings["train"], settings["eval"], settings["parts"]) != (640, 128, 64):
        raise ValueError("Expected immutable 640/128 quotas and 64 partitions")
    plan = Path(settings.get("plan", root / "plan"))
    config = json.loads((plan / "config.json").read_text())
    if (
        config["sampling"]["train_per_stage"],
        config["sampling"]["eval_per_stage"],
    ) != (640, 128):
        raise ValueError("Plan quota mismatch")
    return settings, plan, config


def native(args, settings, plan, config, task):
    project = Path(settings["project"]).resolve()
    generator = project / "scripts/build_wowrobo_randomized_bank.py"
    source_sha = inspect_generator(generator)
    os.chdir(project)
    sys.path.insert(0, str(project))
    import assembly_policy.so101.wowrobo_randomization as randomization

    proxy = NamespacedHashlib(settings["rng_namespace"], config["stages"])
    randomization.hashlib = proxy  # Module-local only; never modify shared hashlib.
    original_sample = randomization.WowRoboRandomizer.sample

    def sample(self, stage_id, seed, split="train", **kwargs):
        data, metadata = original_sample(self, stage_id, seed, split=split, **kwargs)
        key = f"{split}:{stage_id}:{int(seed)}".encode()
        metadata["rng_namespace"] = settings["rng_namespace"]
        metadata["effective_seed"] = int.from_bytes(
            proxy.sha256(key).digest()[:8], "little"
        )
        return data, metadata

    randomization.WowRoboRandomizer.sample = sample
    spec = importlib.util.spec_from_file_location(
        "partition_native_generator", generator
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.range = partition_range(task["part"], task["parts"], args.publish_only)
    output = (
        (plan / "shards" / task["stage"])
        if args.publish_only
        else (plan / "partitions" / task["stage"] / str(task["part"]))
    )
    output.mkdir(parents=True, exist_ok=True)
    write_json(
        output / "partition-provenance.json",
        {
            "generator_sha256": source_sha,
            "worker_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "config_sha256": hashlib.sha256(
                (plan / "config.json").read_bytes()
            ).hexdigest(),
            "rng_namespace": settings["rng_namespace"],
            "task": task,
            "attempt_cap": args.attempt_cap,
            "publish_only": args.publish_only,
            "source_unmodified": True,
            "slot_selection": "part::64",
        },
    )
    sys.argv = [
        str(generator),
        "--config",
        str(plan / "config.json"),
        "--output",
        str(output),
        "--stages",
        task["stage"],
        "--validation",
        "physical",
        "--physical-profile",
        str(settings["physical"]),
        "--settle-seconds",
        "0.08",
        "--hold-seconds",
        "0.2",
        "--full",
        "--max-attempts",
        str(args.attempt_cap),
    ]
    return module.main()


def verify_partition(output, task, namespace):
    run = json.loads((output / "run.json").read_text())
    expected = {
        (split, slot)
        for split, count in [("train", 640), ("eval", 128)]
        for slot in range(task["part"], count, task["parts"])
    }
    accepted = {}
    identities, inputs = set(), set()
    for path in sorted((output / "attempts").glob("*.json")):
        row = json.loads(path.read_text())
        key = (row["split"], row["slot"])
        if (
            row["stage"] != task["stage"]
            or key not in expected
            or row["identity"] != run["identity"]
        ):
            raise ValueError(f"Foreign receipt {path}")
        if row["metadata"].get("rng_namespace") != namespace:
            raise ValueError(f"Missing RNG namespace {path}")
        if not row["training_eligible"]:
            continue
        if (
            key in accepted
            or row["sample_identity"] in identities
            or row["input_sample_identity"] in inputs
        ):
            raise ValueError("Repeated accepted slot or input/qualified identity")
        accepted[key] = row
        identities.add(row["sample_identity"])
        inputs.add(row["input_sample_identity"])
    if set(accepted) != expected:
        return None
    if (
        sum(k[0] == "train" for k in accepted),
        sum(k[0] == "eval" for k in accepted),
    ) != (10, 2):
        raise ValueError("Wrong accepted partition quota")
    # A successful native invocation already revalidates all archive hashes.
    return {
        "identity": run["identity"],
        "accepted_train": 10,
        "accepted_eval": 2,
        "sample_identities": sorted(identities),
        "input_sample_identities": sorted(inputs),
    }


def self_test():
    stage = "21_upper_far_side"
    proxy = NamespacedHashlib("wowrobo19200-v1", [stage])
    for split in ("train", "eval"):
        for seed in (0, 1, 127, 639, 102399):
            key = f"{split}:{stage}:{seed}".encode()
            assert (
                proxy.sha256(key).digest()
                == hashlib.sha256(b"wowrobo19200-v1\0" + key).digest()
            )
    for data in (
        b"",
        b"<xml/>",
        b"nottrain:21_upper_far_side:1",
        b"train:foreign:1",
        b"train:21_upper_far_side:-1",
        b"train:21_upper_far_side:01",
        b"train:21_upper_far_side:1\n",
    ):
        assert proxy.sha256(data).digest() == hashlib.sha256(data).digest()
    h = proxy.sha256()
    h.update(b"ordinary file hash")
    assert h.digest() == hashlib.sha256(b"ordinary file hash").digest()
    assert proxy.sha512(b"x").digest() == hashlib.sha512(b"x").digest()
    for quota in (640, 128):
        selections = [set(partition_range(p, 64)(quota)) for p in range(64)]
        assert set.union(*selections) == set(range(quota))
        assert sum(map(len, selections)) == quota
        assert all(len(s) == quota // 64 for s in selections)
        assert not list(partition_range(0, 64, True)(quota))
    for cap in CAPS:
        assert list(partition_range(7, 64)(cap)) == list(range(cap))
    return {
        "passed": True,
        "tests": [
            "strict RNG hash routing",
            "ordinary hash delegation",
            "disjoint complete 64-way slots",
            "unchanged attempt ranges",
            "publication skips slots only",
        ],
        "worker_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--index", type=int)
    parser.add_argument("--lease-fd", type=int)
    parser.add_argument("--publish-only", action="store_true")
    parser.add_argument("--stage")
    parser.add_argument("--native", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--attempt-cap", type=int, choices=CAPS, default=20)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(), sort_keys=True))
        return 0
    settings, plan, config = context(args.root)
    if args.publish_only:
        if args.stage not in config["stages"]:
            raise ValueError("Unknown publication stage")
        task = {"stage": args.stage, "part": 0, "parts": 64}
    else:
        if args.index is None or args.lease_fd is None:
            parser.error("--index and inherited --lease-fd are required")
        os.fstat(args.lease_fd)  # Fail closed if parent did not pass a live lease.
        task = assignment(args.root, args.index)
    if args.native:
        return native(args, settings, plan, config, task)
    if args.publish_only:
        return native(args, settings, plan, config, task)
    output = plan / "partitions" / task["stage"] / str(task["part"])
    output.mkdir(parents=True, exist_ok=True)
    for cap in CAPS:
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--root",
            str(args.root),
            "--index",
            str(args.index),
            "--lease-fd",
            str(args.lease_fd),
            "--native",
            "--attempt-cap",
            str(cap),
        ]
        status = subprocess.run(
            command, check=False, pass_fds=(args.lease_fd,)
        ).returncode
        if status not in (0, 2):
            write_json(
                output / "worker.json",
                {"status": "error", "native_returncode": status, "attempt_cap": cap},
            )
            return status if status > 0 else 1
        proof = verify_partition(output, task, settings["rng_namespace"])
        if proof is not None:
            proof.update(
                status="complete",
                task=task,
                rng_namespace=settings["rng_namespace"],
                attempt_cap=cap,
                native_returncode=status,
            )
            write_json(output / "worker.json", proof)
            write_json(args.root / "done" / f"{args.index}.json", proof)
            return 0
    write_json(
        output / "worker.json",
        {"status": "exhausted", "task": task, "attempt_cap": CAPS[-1]},
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
