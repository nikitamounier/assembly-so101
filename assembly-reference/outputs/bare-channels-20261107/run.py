"""Use only an idle retained fallback, never compete with its main learner."""

import json
import shlex
import subprocess
import sys
from pathlib import Path

folder = Path(__file__).resolve().parent
config = json.loads(
    (
        folder.parents[1]
        / "operational/wowrobo-chimera-20261107/modal-live/target.json"
    ).read_text()
)
output = "/large_storage/goodarzilab/nikitamounier/assembly/bare-channels-20261107"
steps = subprocess.check_output(
    ["ssh", "chimera-login", "squeue --steps -h -j 3606264 -o %i"], text=True
).split()
assert steps and all(s.endswith((".batch", ".extern")) for s in steps), steps
script = sys.argv[1]
args = [config["remote"], output] + sys.argv[2:]
command = (
    [
        "srun",
        "--jobid=3606264",
        "--overlap",
        "--exact",
        "--ntasks=1",
        "--cpus-per-task=4" if script == "build.py" else "--cpus-per-task=2",
        "--mem=16G",
        "--cpu-bind=cores",
    ]
    + shlex.split(config["env"])
    + [config["python"], output + "/" + script]
    + args
)
subprocess.run(["ssh", "chimera-login", shlex.join(command)], check=True)
