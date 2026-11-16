import os
import pathlib
import shlex
import subprocess
import sys

out = pathlib.Path(__file__).resolve().parent
job = os.environ.get("WARP_JOB", "3598717")
cpu = os.environ.get("WARP_CPU", "2")
mem = os.environ.get("WARP_MEM", "24G")
minutes = os.environ.get("WARP_TIME", "00:30:00")
py = "/large_storage/goodarzilab/nikitamounier/assembly/cube-trial-20261107-v6/.venv/bin/python"
cmd = shlex.join(
    [
        "srun",
        "--jobid=" + job,
        "--overlap",
        "--exact",
        "-n1",
        "-c" + cpu,
        "--mem=" + mem,
        "--time=" + minutes,
        py,
        "-c",
        sys.stdin.read(),
    ]
)
p = subprocess.run(
    ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "chimera-login", cmd],
    check=False,
    capture_output=True,
    text=True,
    timeout=1900,
)
(out / (sys.argv[1] + ".stdout")).write_text(p.stdout)
(out / (sys.argv[1] + ".stderr")).write_text(p.stderr)
print(p.stdout[-10000:])
print(p.stderr[-1500:])
print("returncode", p.returncode)

sys.exit(p.returncode)
