"""Launch a pinned, guarded profiling process in an already retained GPU."""

import argparse
import json
import shlex
import subprocess
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--job", default="3606264")
p.add_argument("--worlds", type=int, default=16)
p.add_argument("--steps", type=int, default=80)
p.add_argument("--repeats", type=int, default=2)
p.add_argument(
    "--variant", default="baseline", choices=["baseline", "candidate", "lazy"]
)
p.add_argument("--profile", action="store_true")
p.add_argument("--nsys", action="store_true")
p.add_argument("--tolerance", type=float)
p.add_argument("--ls-tolerance", type=float)
p.add_argument("--contact-cap", type=int, default=512)
p.add_argument("--constraint-cap", type=int, default=2048)
p.add_argument("--noslip-block", type=int)
p.add_argument("--audit-noslip", action="store_true")
p.add_argument("--audit-contacts", action="store_true")
p.add_argument("--tag", default="")
p.add_argument("--all-stages", action="store_true")
p.add_argument("--environment", action="store_true")
p.add_argument("--controls", type=int, default=3)
p.add_argument("--diagnostics", action="store_true")
a = p.parse_args()
here = Path(__file__).resolve().parent
repo = here.parents[1]
source = (
    here / "baseline" if a.variant == "baseline" else repo / "assembly_policy/so101"
)
if a.variant == "lazy":
    source = here / "lazy-rejected"
files = {
    a.variant + "/" + n: (source / n).read_text()
    for n in ("wowrobo_warp.py", "wowrobo_warp_runtime.py")
}
files["profile_runtime.py"] = (here / "profile_runtime.py").read_text()
files["contact_audit.py"] = (here / "contact_audit.py").read_text()
for n in ("wowrobo_warp.py", "wowrobo_warp_runtime.py"):
    files["baseline/" + n] = (here / "baseline" / n).read_text()
root = "/large_storage/goodarzilab/nikitamounier/assembly"
target = root + "/mjwarp-profile-20261107"
release = root + "/wowrobo-diagnostic-20261107T205745Z-minimum24-accepted-v4/native-v13"
py = root + "/mjwarp-h100-20261107/.venv/bin/python"
label = f"{a.variant}-w{a.worlds}-s{a.steps}-tol{a.tolerance}-ls{a.ls_tolerance}-cap{a.contact_cap}-j{a.constraint_cap}"
if a.profile:
    label += "-profile"
if a.noslip_block is not None:
    label += "-nb" + str(a.noslip_block)
if a.audit_noslip:
    label += "-audit"
if a.audit_contacts:
    label += "-contactaudit"
if a.nsys:
    label += "-nsys"
if a.tag:
    label += "-" + a.tag
if a.all_stages:
    label += "-allstages"
if a.environment:
    label += "-environment"
if a.diagnostics:
    label += "-diagnostics"
cmd = [
    py,
    "-u",
    target + "/profile_runtime.py",
    "--release",
    release,
    "--bank",
    release + "/training-accepted-v13",
    "--out",
    target + "/" + label,
    "--source",
    target + "/" + a.variant,
    "--worlds",
    str(a.worlds),
    "--steps",
    str(a.steps),
    "--repeats",
    str(a.repeats),
    "--contact-cap",
    str(a.contact_cap),
    "--constraint-cap",
    str(a.constraint_cap),
]
if a.profile:
    cmd += ["--profile"]
if a.noslip_block is not None:
    cmd += ["--noslip-block", str(a.noslip_block)]
if a.audit_noslip:
    cmd += ["--audit-noslip"]
if a.audit_contacts:
    cmd += ["--audit-contacts"]
if a.all_stages:
    cmd += ["--all-stages"]
if a.diagnostics:
    cmd += ["--diagnostics"]
if a.environment:
    files["qualify.py"] = (
        here.parent / "mjwarp-runtime-20261107/qualify.py"
    ).read_text()
    for name in ("wowrobo_warp.py", "wowrobo_warp_runtime.py"):
        files[name] = (source / name).read_text()
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
        "--worlds",
        str(a.worlds),
        "--steps",
        str(a.steps),
        "--controls",
        str(a.controls),
    ]
if a.tolerance is not None:
    cmd += ["--tolerance", str(a.tolerance)]
if a.ls_tolerance is not None:
    cmd += ["--ls-tolerance", str(a.ls_tolerance)]
if a.nsys:
    cmd += ["--nsys"]
    cmd = [
        "/usr/local/cuda/bin/nsys",
        "profile",
        "--trace=cuda,nvtx",
        "--sample=none",
        "--cpuctxsw=none",
        "--cuda-graph-trace=node",
        "--capture-range=cudaProfilerApi",
        "--capture-range-end=stop",
        "--force-overwrite=true",
        "-o",
        target + "/" + label + "/nsys",
        *cmd,
    ]
code = "import os,subprocess,threading,time,json\nfrom pathlib import Path\n"
code += (
    "target=Path("
    + repr(target)
    + ")\nrelease=Path("
    + repr(release)
    + ")\nfiles="
    + repr(files)
    + "\ncmd="
    + repr(cmd)
    + "\nlabel="
    + repr(label)
    + "\n"
)
code += """
if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
 raise RuntimeError('GPU busy; refusing profile')
target.mkdir(exist_ok=True)
for name,value in files.items():
 f=target/name;f.parent.mkdir(exist_ok=True,parents=True);f.write_text(value)
out=target/label;out.mkdir(exist_ok=True)
env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
 WOWROBO_WARP_GRID_SOURCE=str(release/'assembly-so101/plugin/assemblenist_sdf.cc'),
 WOWROBO_WARP_FRICTION_SOURCE=str(release/'robotics-assembly-3.0/assembly_policy/so101/wowrobo_friction.py'),
 WARP_CACHE_PATH='/large_storage/goodarzilab/nikitamounier/assembly/mjwarp-runtime-20261107/warp-cache')
log=out/'run.log'
with log.open('w') as f:
 child=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,env=env,start_new_session=True)
 def owned_target(pid):
  # Nsight starts its target in another session. Require our UID plus both
  # exact script and output-directory arguments before accepting that target.
  proc=Path('/proc')/str(pid)
  try:
   argv=(proc/'cmdline').read_bytes().split(bytes([0]))
   return proc.stat().st_uid==os.getuid() and bytes(str(target/'profile_runtime.py'),'utf8') in argv and bytes(str(out),'utf8') in argv
  except OSError:return False
 def watch():
  while child.poll() is None:
   time.sleep(2)
   # profiler creates a child; permit only our process session.
   text=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True)
   for line in text.splitlines():
    try:
     if os.getsid(int(line.strip())) != child.pid and not owned_target(int(line.strip())):
      os.killpg(child.pid,15);return
    except (ValueError,ProcessLookupError):pass
 threading.Thread(target=watch,daemon=True).start()
 try:status=child.wait(timeout=1600)
 except subprocess.TimeoutExpired:os.killpg(child.pid,15);child.wait(timeout=30);status=124
print(log.read_text()[-15000:])
if (out/'result.json').exists():
 print('RESULT_JSON_BEGIN');print((out/'result.json').read_text());print('RESULT_JSON_END')
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
        "--mem=48G",
        "--time=00:29:00",
        py,
        "-",
    ]
)
r = subprocess.run(
    ["ssh", "-o", "BatchMode=yes", "chimera-login", remote],
    capture_output=True,
    text=True,
    input=code,
    check=False,
    timeout=1750,
)
(here / (label + ".stdout")).write_text(r.stdout)
(here / (label + ".stderr")).write_text(r.stderr)
if "RESULT_JSON_BEGIN" in r.stdout:
    data = r.stdout.split("RESULT_JSON_BEGIN")[1].split("RESULT_JSON_END")[0].strip()
    (here / (label + ".json")).write_text(data + "\n")
    result = json.loads(data)
    print(
        json.dumps(
            {
                "label": label,
                "status": result["status"],
                "worlds": result["worlds"],
                "wall_s": [x.get("wall_s", x.get("gpu_s")) for x in result["records"]],
                "graph_ms": result.get("graph_ms"),
                "top_kernels": result.get("kernels", [])[:5],
            }
        )
    )
else:
    print(r.stdout[-5000:])
print(r.stderr[-2000:])
raise SystemExit(r.returncode)
