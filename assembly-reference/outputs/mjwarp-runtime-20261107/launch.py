"""Deploy only to a private experiment directory and run in a retained idle GPU."""

import argparse
import shlex
import subprocess
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--job", required=True)
p.add_argument("--steps", type=int, default=4)
p.add_argument("--worlds", type=int, default=2)
p.add_argument("--controls", type=int, default=2)
p.add_argument("--ppo", action="store_true")
a = p.parse_args()
here = Path(__file__).resolve().parent
repo = here.parents[1]
files = {
    n: (repo / "assembly_policy/so101" / n).read_text()
    for n in ("wowrobo_warp.py", "wowrobo_warp_runtime.py")
}
files["qualify.py"] = (here / "qualify.py").read_text()
files["canonical/scripts/train_wowrobo_warp.py"] = (
    repo / "scripts/train_wowrobo_warp.py"
).read_text()
for name in ("wowrobo_warp.py", "wowrobo_warp_runtime.py", "wowrobo_friction.py"):
    files["canonical/assembly_policy/so101/" + name] = (
        repo / "assembly_policy/so101" / name
    ).read_text()
root = "/large_storage/goodarzilab/nikitamounier/assembly"
target = root + "/mjwarp-runtime-20261107"
release = root + "/wowrobo-diagnostic-20261107T205745Z-minimum24-accepted-v4/native-v13"
py = root + "/mjwarp-h100-20261107/.venv/bin/python"
label = f"w{a.worlds}-s{a.steps}-c{a.controls}"
if a.ppo:
    label = "ppo-w" + str(a.worlds)
code = """
import json, os, subprocess, threading, time
from pathlib import Path
"""
code += "target=Path(" + repr(target) + ")\nfiles=" + repr(files) + "\n"
code += "release=Path(" + repr(release) + ")\n"
code += """
app=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True)
if app.strip(): raise RuntimeError('GPU busy; refusing test')
def watch():
 while True:
  time.sleep(2)
  app=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True)
  if any(x.strip() and x.strip()!=str(child.pid) for x in app.splitlines()):
   child.terminate();return
target.mkdir(exist_ok=True)
for n,s in files.items():
 (target/n).parent.mkdir(parents=True,exist_ok=True)
 (target/n).write_text(s)
env=os.environ.copy()
env.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
 WOWROBO_WARP_GRID_SOURCE=str(release/'assembly-so101/plugin/assemblenist_sdf.cc'),
 WOWROBO_WARP_FRICTION_SOURCE=str(release/'robotics-assembly-3.0/assembly_policy/so101/wowrobo_friction.py'),
 WARP_CACHE_PATH=str(target/'warp-cache'))
"""
cmd = [
    py,
    "-u",
    target + "/qualify.py",
    "--release",
    release,
    "--bank",
    release + "/training-accepted-v13",
    "--out",
    target + "/" + label,
    "--steps",
    str(a.steps),
    "--worlds",
    str(a.worlds),
    "--controls",
    str(a.controls),
]
if a.ppo:
    train_args = [
        target + "/canonical/scripts/train_wowrobo_warp.py",
        "--trainer-root",
        release + "/assembly-so101",
        "--scratch",
        target + "/ppo-scratch",
        "--",
        "--run-dir",
        target + "/" + label,
        "--bank-dir",
        release + "/training-accepted-v13",
        "--wowrobo-debug-resets",
        "--envs",
        str(a.worlds),
        "--iters",
        "1",
        "--steps",
        "2",
        "--epochs",
        "1",
        "--minibatches",
        "1",
        "--micro-batch",
        "4",
        "--nthread",
        "1",
        "--no-eval",
        "--bank-per-family",
        "0",
        "--device",
        "cuda",
    ]
    bootstrap = (
        "import sys,os,runpy;sys.path.append("
        + repr(root + "/source-85c006a/.venv/lib/python3.12/site-packages")
        + ");os.environ['WOWROBO_CANONICAL_ROOT']="
        + repr(str(Path(release).parent / "robotics-assembly-3.0"))
        + ";sys.argv="
        + repr(train_args)
        + ";runpy.run_path(sys.argv[0],run_name='__main__')"
    )
    cmd = [py, "-u", "-c", bootstrap]
code += "cmd=" + repr(cmd) + "\n"
code += "log=target/" + repr(label + ".log") + "\n"
code += """
with log.open('w') as f:
 child=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,env=env)
 threading.Thread(target=watch,daemon=True).start()
 try: status=child.wait(timeout=1500)
 except subprocess.TimeoutExpired: child.terminate();child.wait(timeout=30);status=124
print(log.read_text()[-16000:])
raise SystemExit(status)
"""
remote = shlex.join(
    [
        "srun",
        "--jobid=" + a.job,
        "--overlap",
        "--exact",
        "-n1",
        "-c2",
        "--mem=24G",
        "--time=00:28:00",
        py,
        "-",
    ]
)
result = subprocess.run(
    ["ssh", "-o", "BatchMode=yes", "chimera-login", remote],
    text=True,
    input=code,
    capture_output=True,
    timeout=1700,
    check=False,
)
(here / (label + ".stdout")).write_text(result.stdout)
(here / (label + ".stderr")).write_text(result.stderr)
print(result.stdout[-10000:])
print(result.stderr[-2000:])
raise SystemExit(result.returncode)
