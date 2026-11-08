"""Launch sequential isolated CPU probes inside an existing retained lease."""

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
root = cfg["remote"]
exp = root + "/tree-optimization-fork-20261107"
setup = (
    "from pathlib import Path; import shutil; root=Path("
    + repr(root)
    + "); exp=Path("
    + repr(exp)
    + "); names=['assemblenist_sdf.cc','so101_grid_bounds.h','so101_surface_tree.h','so101_contact_search.c','build.sh','build_contact_search.sh']; [(shutil.copy2(root/'assembly-so101/plugin'/n,exp/'baseline/plugin'/n)) for n in names]; [(shutil.copy2(root/'assembly-so101/plugin'/n,exp/'candidate/plugin'/n)) for n in names if n!='so101_surface_tree.h']"
)
commands = [[cfg["python"], "-c", setup]]
for variant in sys.argv[2:] or ["baseline", "candidate"]:
    commands.append(
        shlex.split(cfg["env"])
        + [
            cfg["python"],
            exp + "/qualify_tree.py",
            root,
            exp,
            variant,
            "--panel",
            sys.argv[1],
        ]
    )
for command in commands:
    remote = [
        "srun",
        "--jobid=3598717",
        "--overlap",
        "--exact",
        "--ntasks=1",
        "--cpus-per-task=2",
        "--mem=12G",
        "--cpu-bind=cores",
    ] + command
    subprocess.run(["ssh", "chimera-login", shlex.join(remote)], check=True)
