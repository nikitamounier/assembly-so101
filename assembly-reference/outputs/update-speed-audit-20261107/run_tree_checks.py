"""Sequential uninstrumented full panel, then exhaustive two-thread oracle."""

import json
import shlex
import subprocess
from pathlib import Path

base = Path(__file__).resolve().parent
cfg = json.loads(
    (
        base.parents[1] / "operational/wowrobo-chimera-20261107/modal-live/target.json"
    ).read_text()
)
root = cfg["remote"]
exp = root + "/tree-optimization-fork-20261107"
commands = [
    [
        cfg["python"],
        exp + "/qualify_tree.py",
        root,
        exp,
        v,
        "--panel",
        "all",
        "--no-profile",
    ]
    for v in ["baseline", "candidate"]
]
commands.append([cfg["python"], exp + "/oracle_tree.py", root, exp])
for command in commands:
    remote = (
        [
            "srun",
            "--jobid=3598717",
            "--overlap",
            "--exact",
            "--ntasks=1",
            "--cpus-per-task=2",
            "--mem=12G",
            "--cpu-bind=cores",
        ]
        + shlex.split(cfg["env"])
        + command
    )
    subprocess.run(["ssh", "chimera-login", shlex.join(remote)], check=True)
