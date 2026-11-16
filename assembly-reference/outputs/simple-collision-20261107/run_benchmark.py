"""Use one pinned host for both benchmark variants, separate from decomposition."""

import json
import shlex
import subprocess
import sys
from pathlib import Path

base = Path(__file__).resolve().parent
cfg = json.loads(
    (
        base.parents[1] / "operational/wowrobo-chimera-20261107/modal-live/target.json"
    ).read_text()
)
output = "/large_storage/goodarzilab/nikitamounier/assembly/simple-collision-20261107"
cpus = (
    max(2, int(sys.argv[sys.argv.index("--nthread") + 1]))
    if "--nthread" in sys.argv
    else 2
)
command = (
    [
        "srun",
        "--jobid=3598717",
        "--overlap",
        "--exact",
        "--ntasks=1",
        "--cpus-per-task=" + str(cpus),
        "--mem=24G" if cpus > 2 else "--mem=12G",
        "--cpu-bind=cores",
    ]
    + shlex.split(cfg["env"])
    + [cfg["python"], output + "/benchmark.py", cfg["remote"], output, sys.argv[1]]
    + sys.argv[2:]
)
steps = subprocess.check_output(
    ["ssh", "chimera-login", "squeue --steps -h -j 3598717 -o %i"], text=True
).split()
assert all(s.endswith((".batch", ".extern")) for s in steps), steps
subprocess.run(["ssh", "chimera-login", shlex.join(command)], check=True)
