"""Launch-only fencing for an authorized reset-bank stage farm.

No physics/generator source is modified. Run pause/resume/supervise in the
original coordinator's allocation/node. Stage workers must claim their inherited
.farm-worker.lock while holding control_lock(root), after checking the fresh
farm_open/coordinator_paused fence.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from contextlib import ExitStack, contextmanager
from pathlib import Path


def atomic(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def assignment_path(root):
    return root / "launch/farm-assignment.json"


@contextmanager
def control_lock(root):
    with (root / "launch/.farm-control.lock").open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield


def proc_info(pid):
    folder = Path("/proc") / str(pid)
    stat = (folder / "stat").read_text().rsplit(")", 1)[1].split()
    return {
        "pid": int(pid),
        "ppid": int(stat[1]),
        "state": stat[0],
        "start_ticks": int(stat[19]),
        "command": [
            x.decode(errors="replace")
            for x in (folder / "cmdline").read_bytes().split(b"\0")
            if x
        ],
    }


def option(command, name):
    if name not in command:
        return None
    index = command.index(name) + 1
    return command[index] if index < len(command) else None


def owner(root, prior=None):
    allocated = json.loads((root / "allocated.json").read_text())
    if str(os.environ.get("SLURM_JOB_ID")) != str(allocated["job_id"]):
        raise RuntimeError("Coordinator control requires its original Slurm allocation")
    if os.uname().nodename != allocated["host"]:
        raise RuntimeError("Coordinator control must run on its actual node")
    info = proc_info(allocated["pid"])
    if not any(x.endswith("/launch/driver.py") for x in info["command"]) or option(
        info["command"], "--root"
    ) != str(root):
        raise RuntimeError("PID does not own this exact isolated reset driver")
    if prior and (
        prior["coordinator_pid"] != info["pid"]
        or prior["coordinator_start_ticks"] != info["start_ticks"]
        or prior["coordinator_node"] != os.uname().nodename
        or str(prior["coordinator_job"]) != str(allocated["job_id"])
    ):
        raise RuntimeError(
            "Coordinator ownership changed; run launch gate before controlling it"
        )
    return allocated, info


def plan_info(root):
    project = root / "robotics-assembly-3.0"
    release = json.loads((root / "release.json").read_text())
    plan = project / release["plan_relative"]
    spec = json.loads((plan / "plan.json").read_text())
    return plan, spec


def complete(plan, spec, stage):
    path = plan / "shards" / stage / "qualified/manifest.json"
    if not path.exists():
        return False
    value = json.loads(path.read_text())
    progress = value.get("progress", {})
    return (
        value.get("complete") is True
        and value.get("task_names") == [stage]
        and progress.get("accepted_train") == spec["train_per_stage"]
        and progress.get("accepted_eval") == spec["eval_per_stage"]
    )


def stage_leases(plan, stages):
    result = {}
    for stage in stages:
        folder = plan / "shards" / stage
        path = folder / ".farm-worker.lock"
        metadata = folder / "farm-worker.json"
        record = json.loads(metadata.read_text()) if metadata.exists() else None
        held = False
        if path.exists():
            with path.open("a") as stream:
                try:
                    fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    held = True
                else:
                    fcntl.flock(stream, fcntl.LOCK_UN)
        if record and record.get("stage") != stage:
            raise RuntimeError(f"Stage lease metadata mismatch:{stage}")
        result[stage] = {"lock_held": held, "worker": record}
    return result


def prepare(root):
    with control_lock(root):
        if assignment_path(root).exists():
            prior = json.loads(assignment_path(root).read_text())
            owner(root, prior)
            return prior
        allocated, info = owner(root)
        os.kill(info["pid"], signal.SIGSTOP)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            states = [
                (Path("/proc") / str(info["pid"]) / "task" / task.name / "stat")
                .read_text()
                .rsplit(")", 1)[1]
                .split()[0]
                for task in (Path("/proc") / str(info["pid"]) / "task").iterdir()
            ]
            if states and all(state in {"T", "t"} for state in states):
                break
            time.sleep(0.05)
        else:
            raise RuntimeError("Coordinator did not reach all-thread stopped state")
        plan, spec = plan_info(root)
        leaves = []
        for folder in Path("/proc").iterdir():
            if not folder.name.isdigit():
                continue
            try:
                child = proc_info(int(folder.name))
            except (FileNotFoundError, ProcessLookupError, PermissionError):
                continue
            command = child["command"]
            if child["ppid"] != info["pid"] or not any(
                x.endswith("/scripts/build_wowrobo_randomized_bank.py") for x in command
            ):
                continue
            stage = option(command, "--stages")
            if (
                stage not in spec["stage_ids"]
                or option(command, "--output") != str(plan / "shards" / stage)
                or option(command, "--config") != str(plan / "config.json")
            ):
                raise RuntimeError(
                    "Direct child does not own an expected isolated stage"
                )
            leaves.append(dict(child, stage=stage, node=os.uname().nodename))
        active = [child["stage"] for child in leaves]
        if len(active) != len(set(active)):
            raise RuntimeError("Two original workers already own one stage")
        completed = [
            stage for stage in spec["stage_ids"] if complete(plan, spec, stage)
        ]
        assigned = [
            stage
            for stage in spec["stage_ids"]
            if stage not in active and stage not in completed
        ]
        value = {
            "schema_version": 1,
            "root": str(root),
            "plan": str(plan),
            "farm_id": hashlib.sha256(
                f"{root}:{info['pid']}:{info['start_ticks']}".encode()
            ).hexdigest()[:20],
            "coordinator_pid": info["pid"],
            "coordinator_start_ticks": info["start_ticks"],
            "coordinator_node": os.uname().nodename,
            "coordinator_job": allocated["job_id"],
            "coordinator_paused": True,
            "farm_open": True,
            "active_stages": active,
            "active_workers": leaves,
            "completed_stages": completed,
            "assigned_stages": assigned,
            "all_stage_ids": spec["stage_ids"],
            "prepared_unix": time.time(),
        }
        atomic(assignment_path(root), value)
        return value


def status(root):
    value = json.loads(assignment_path(root).read_text())
    plan, spec = plan_info(root)
    completed = [stage for stage in spec["stage_ids"] if complete(plan, spec, stage)]
    leases = stage_leases(plan, value["assigned_stages"])
    return {
        "farm_open": value["farm_open"],
        "coordinator_paused": value["coordinator_paused"],
        "coordinator_pid": value["coordinator_pid"],
        "coordinator_node": value["coordinator_node"],
        "complete_count": len(completed),
        "completed_stages": completed,
        "all_complete": len(completed) == len(spec["stage_ids"]),
        "active_external_stages": [
            stage for stage, lease in leases.items() if lease["lock_held"]
        ],
        "leases": leases,
    }


def resume(root):
    with control_lock(root):
        value = json.loads(assignment_path(root).read_text())
        _, info = owner(root, value)
        current = status(root)
        if not current["all_complete"]:
            raise RuntimeError(
                "Cannot resume original queue before all25 stages are complete"
            )
        if current["active_external_stages"]:
            raise RuntimeError(
                "Cannot resume while inherited external stage leases remain held"
            )
        # Holding control_lock serializes this fence with every external claim.
        value.update(
            farm_open=False, coordinator_paused=False, resumed_unix=time.time()
        )
        atomic(assignment_path(root), value)
        os.kill(info["pid"], signal.SIGCONT)
        return {
            "status": "original_coordinator_resumed",
            "pid": info["pid"],
            "all_complete": True,
        }


def gate_coordinator(root, pid):
    """Call in launch driver before importing/running the canonical framework."""
    with control_lock(root):
        path = (
            partition_path(root)
            if partition_path(root).exists()
            else assignment_path(root)
        )
        if not path.exists():
            return {"status": "no_farm"}
        value = json.loads(path.read_text())
        if not value.get("open", value.get("farm_open", False)):
            return {"status": "farm_closed"}
        allocated, info = owner(root)
        if info["pid"] != pid or str(allocated["job_id"]) != str(
            value["coordinator_job"]
        ):
            raise RuntimeError(
                "Launch gate does not own the authorized coordinator job"
            )
        value.update(
            coordinator_pid=pid,
            coordinator_start_ticks=info["start_ticks"],
            coordinator_node=os.uname().nodename,
            coordinator_paused=True,
            adopted_unix=time.time(),
        )
        atomic(path, value)
        os.kill(pid, signal.SIGSTOP)
        return {"status": "coordinator_gated", "pid": pid}


def supervise(root, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = status(root)
        atomic(root / "launch/farm-status.json", dict(value, checked_unix=time.time()))
        if value["all_complete"] and not value["active_external_stages"]:
            return resume(root)
        time.sleep(10)
    value = status(root)
    atomic(
        root / "launch/farm-status.json",
        dict(
            value,
            status="bounded_wait_expired_coordinator_stays_paused",
            checked_unix=time.time(),
        ),
    )
    raise RuntimeError(
        "Farm wait expired; coordinator remains paused and farm fence remains open"
    )


def partition_path(root):
    return root / "launch/partition-assignment.json"


def accepted_slots(folder, stage, part=None, parts=4):
    slots = set()
    for path in (folder / "attempts").glob("*.json"):
        row = json.loads(path.read_text())
        if row["stage"] != stage or row["split"] not in ("train", "eval"):
            raise RuntimeError(f"Invalid stage receipt:{path}")
        limit = 60 if row["split"] == "train" else 12
        if not 0 <= row["slot"] < limit or (
            part is not None and row["slot"] % parts != part
        ):
            raise RuntimeError(f"Invalid partition receipt:{path}")
        if row.get("training_eligible"):
            slots.add((row["split"], row["slot"]))
    return slots


def wanted_slots(part, parts):
    return {
        (split, slot)
        for split, count in [("train", 60), ("eval", 12)]
        for slot in range(part, count, parts)
    }


def prepare_partitions(root, parts):
    if parts != 4:
        raise ValueError("This farm is explicitly four partitions per60/12 stage")
    prepare(root)
    with control_lock(root):
        if partition_path(root).exists():
            return json.loads(partition_path(root).read_text())
        prior = json.loads(assignment_path(root).read_text())
        _, info = owner(root, prior)
        os.kill(info["pid"], signal.SIGSTOP)
        plan, spec = plan_info(root)
        if any(
            lease["lock_held"]
            for lease in stage_leases(plan, prior["assigned_stages"]).values()
        ):
            raise RuntimeError("Plain stage farm still has an active lease")
        children = []
        for folder in Path("/proc").iterdir():
            if not folder.name.isdigit():
                continue
            try:
                child = proc_info(int(folder.name))
            except (FileNotFoundError, ProcessLookupError, PermissionError):
                continue
            command = child["command"]
            if child["ppid"] != info["pid"] or not any(
                x.endswith("/scripts/build_wowrobo_randomized_bank.py") for x in command
            ):
                continue
            stage = option(command, "--stages")
            if (
                stage not in spec["stage_ids"]
                or option(command, "--output") != str(plan / "shards" / stage)
                or option(command, "--config") != str(plan / "config.json")
            ):
                raise RuntimeError("Refusing to stop an unproven original stage child")
            if child["state"] not in {"Z", "X"}:
                os.kill(child["pid"], signal.SIGTERM)
                children.append(dict(child, stage=stage))
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            alive = []
            for child in children:
                try:
                    current = proc_info(child["pid"])
                except (FileNotFoundError, ProcessLookupError):
                    continue
                if current["start_ticks"] == child["start_ticks"] and current[
                    "state"
                ] not in {"Z", "X"}:
                    alive.append(child["pid"])
            if not alive:
                break
            time.sleep(0.1)
        else:
            raise RuntimeError(f"Original stage workers did not terminate:{alive}")
        items = []
        completed = []
        for stage in spec["stage_ids"]:
            if complete(plan, spec, stage):
                completed.append(stage)
                continue
            have = accepted_slots(plan / "shards" / stage, stage)
            for part in range(parts):
                if not wanted_slots(part, parts) <= have:
                    items.append({"stage": stage, "part": part, "parts": parts})
        value = dict(
            prior,
            open=True,
            active_stages=[],
            active_workers=[],
            original_workers_stopped=children,
            completed_stages=completed,
            items=items,
            parts=parts,
            prepared_partitions_unix=time.time(),
        )
        atomic(partition_path(root), value)
        prior.update(farm_open=False, mode="partitions", coordinator_paused=True)
        atomic(assignment_path(root), prior)
        return value


def partition_leases(plan, assignment):
    result = {}
    for item in assignment["items"]:
        stage, part = item["stage"], item["part"]
        folder = plan / "partitions" / stage / str(part)
        lease = folder / ".worker.lock"
        held = False
        if lease.exists():
            with lease.open("a") as stream:
                try:
                    fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    held = True
                else:
                    fcntl.flock(stream, fcntl.LOCK_UN)
        metadata = folder / "partition-worker.json"
        worker = json.loads(metadata.read_text()) if metadata.exists() else None
        if worker and (
            worker["stage"] != stage
            or worker["part"] != part
            or worker["parts"] != item["parts"]
        ):
            raise RuntimeError("Partition worker ownership metadata mismatch")
        result[f"{stage}/{part}"] = {"lock_held": held, "worker": worker}
    return result


def copy_verified(source, destination):
    if not source.is_file():
        raise RuntimeError(f"Missing referenced partition archive:{source}")
    fingerprint = hashlib.sha256(source.read_bytes()).hexdigest()
    if destination.exists():
        if hashlib.sha256(destination.read_bytes()).hexdigest() != fingerprint:
            raise RuntimeError(f"Conflicting preserved master artifact:{destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".farm-copy-tmp")
    temporary.write_bytes(source.read_bytes())
    if hashlib.sha256(temporary.read_bytes()).hexdigest() != fingerprint:
        raise RuntimeError("Partition copy hash mismatch")
    temporary.replace(destination)


def merge_parts(plan, stage, parts):
    master = plan / "shards" / stage
    identities = {
        json.loads(path.read_text())["identity"]
        for path in (master / "attempts").glob("*.json")
    }
    for part in range(parts):
        for path in (plan / "partitions" / stage / str(part) / "attempts").glob(
            "*.json"
        ):
            identities.add(json.loads(path.read_text())["identity"])
    if len(identities) != 1:
        raise RuntimeError(
            f"Stage identity differs across partitions before merge:{stage}"
        )
    for part in range(parts):
        source = plan / "partitions" / stage / str(part)
        for path in sorted((source / "attempts").glob("*.json")):
            row = json.loads(path.read_text())
            if row["stage"] != stage or row["slot"] % parts != part:
                raise RuntimeError(f"Cross-partition receipt:{path}")
            identities.add(row["identity"])
            for key in ("archive", "physical_input_archive", "qualified_archive"):
                if not row.get(key):
                    continue
                relative = Path(row[key])
                if relative.is_absolute() or ".." in relative.parts:
                    raise RuntimeError("Receipt path escapes partition")
                copy_verified(source / relative, master / relative)
            copy_verified(path, master / "attempts" / path.name)
    for path in (master / "attempts").glob("*.json"):
        identities.add(json.loads(path.read_text())["identity"])
    if len(identities) != 1:
        raise RuntimeError(f"Stage identity differs across partitions:{stage}")
    if len(accepted_slots(master, stage)) != 72:
        raise RuntimeError(f"Partition merge did not provide exactly72 slots:{stage}")


def supervise_partitions(root, timeout):
    plan, spec = plan_info(root)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        assignment = json.loads(partition_path(root).read_text())
        if not assignment["open"]:
            return {"status": "partition_farm_already_closed"}
        # Only publish complete-stage unions. Hold each inherited part lease
        # across merge+publication, excluding late duplicate worker claims.
        for stage in spec["stage_ids"]:
            if complete(plan, spec, stage):
                continue
            with ExitStack() as locks:
                lease_fds = []
                with control_lock(root):
                    locked = True
                    for part in range(assignment["parts"]):
                        folder = plan / "partitions" / stage / str(part)
                        folder.mkdir(parents=True, exist_ok=True)
                        stream = locks.enter_context(
                            (folder / ".worker.lock").open("a")
                        )
                        try:
                            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        except BlockingIOError:
                            locked = False
                            break
                        os.set_inheritable(stream.fileno(), True)
                        lease_fds.append(stream.fileno())
                if not locked:
                    continue
                have = accepted_slots(plan / "shards" / stage, stage)
                have |= set().union(
                    *(
                        accepted_slots(
                            plan / "partitions" / stage / str(part),
                            stage,
                            part,
                            assignment["parts"],
                        )
                        for part in range(assignment["parts"])
                    )
                )
                if len(have) != 72:
                    continue
                merge_parts(plan, stage, assignment["parts"])
                with (root / "launch" / f"publish-{stage}.log").open("a") as log:
                    subprocess.run(
                        [
                            sys.executable,
                            str(root / "launch/partition_worker.py"),
                            "--root",
                            str(root),
                            "--publish-only",
                            "--stage",
                            stage,
                            "--part",
                            "0",
                            "--parts",
                            str(assignment["parts"]),
                        ],
                        env=dict(
                            os.environ, PYTHONPATH=str(root / "robotics-assembly-3.0")
                        ),
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        pass_fds=tuple(lease_fds),
                        check=True,
                    )
                if not complete(plan, spec, stage):
                    raise RuntimeError(
                        f"Canonical final publication did not seal72 slots:{stage}"
                    )
        with control_lock(root):
            assignment = json.loads(partition_path(root).read_text())
            completed = [
                stage for stage in spec["stage_ids"] if complete(plan, spec, stage)
            ]
            leases = partition_leases(plan, assignment)
            held = [item for item, lease in leases.items() if lease["lock_held"]]
            atomic(
                root / "launch/partition-status.json",
                {
                    "completed_stages": completed,
                    "complete_count": len(completed),
                    "active_partitions": held,
                    "checked_unix": time.time(),
                },
            )
            if len(completed) == len(spec["stage_ids"]) and not held:
                _, info = owner(root, assignment)
                assignment.update(
                    open=False, coordinator_paused=False, resumed_unix=time.time()
                )
                atomic(partition_path(root), assignment)
                prior = json.loads(assignment_path(root).read_text())
                prior.update(farm_open=False, coordinator_paused=False)
                atomic(assignment_path(root), prior)
                os.kill(info["pid"], signal.SIGCONT)
                return {
                    "status": "all25_stage_banks_complete_original_resumed",
                    "accepted": 1800,
                    "coordinator_pid": info["pid"],
                }
        time.sleep(10)
    raise RuntimeError(
        "Partition farm wait expired; original remains paused; no unsafe resume"
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "action",
        choices=[
            "prepare",
            "status",
            "resume",
            "supervise",
            "gate-coordinator",
            "prepare-partitions",
            "supervise-partitions",
        ],
    )
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--pid", type=int)
    p.add_argument("--parts", type=int, default=4)
    p.add_argument("--timeout", type=float, default=7200)
    a = p.parse_args()
    root = a.root.resolve()
    if a.action == "prepare-partitions":
        result = prepare_partitions(root, a.parts)
    elif a.action == "supervise-partitions":
        result = supervise_partitions(root, a.timeout)
    elif a.action == "prepare":
        result = prepare(root)
    elif a.action == "status":
        result = status(root)
    elif a.action == "resume":
        result = resume(root)
    elif a.action == "gate-coordinator":
        result = gate_coordinator(root, a.pid or os.getppid())
    else:
        result = supervise(root, a.timeout)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
