"""Fresh upper force proof on explicitly frozen common candidate."""

from pathlib import Path
import argparse
import json
import mujoco
import numpy as np
from assembly_policy.so101.wowrobo_contact_backend import load_backend, set_surface_mode
from assembly_policy.so101.wowrobo_assembly_fit import coherent_step_alignment
from assembly_policy.so101.wowrobo_insertion_guide import build_guide
from scripts.validate_wowrobo_fit_profile import run_case

ROOT = Path(__file__).resolve().parents[3]
p = argparse.ArgumentParser()
p.add_argument(
    "--candidate",
    type=Path,
    default=ROOT / "outputs/insertion-fix-20261107/final-common-v1",
)
p.add_argument(
    "--output",
    type=Path,
    default=Path(__file__).resolve().parent / "canonical-common-v1",
)
args = p.parse_args()
candidate = args.candidate.resolve()
out = args.output.resolve()
out.mkdir(parents=True, exist_ok=True)
load_backend(candidate / "grids/manifest.json")
set_surface_mode(1)
model = mujoco.MjModel.from_xml_path(str(candidate / "scene.xml"))
raw = json.loads(
    (
        ROOT / "outputs/linear-insertion-aligned-20261107/upper/alignment.json"
    ).read_text()
)
reg = json.loads(
    (
        ROOT / "outputs/insertion-fix-20261107/pin/consistent-v3/registration.json"
    ).read_text()
)
a = coherent_step_alignment(model, raw, "upper", pin_registration=reg)
(out / "alignment.json").write_text(json.dumps(a, indent=2) + "\n")
build_guide(
    candidate / "scene.xml",
    out / "alignment.json",
    out / "guide-scene.xml",
    "upper_arm_module",
)
r = run_case(
    argparse.Namespace(step="upper", seconds=8.0, release_seconds=0.5),
    out,
    out,
    a,
    {"qpos": np.zeros(1), "qvel": np.zeros(1)},
    "legacy_mapped",
    20.0,
)
# Transfer the actual driven endpoint, preceding drive-off relaxation, to full51.
g = mujoco.MjModel.from_xml_path(str(out / "legacy_mapped_20N.xml"))
d = mujoco.MjData(g)
saved = np.load(out / "legacy_mapped_20N-drive-state.npz")
d.qpos[:] = saved["qpos"]
d.qvel[:] = saved["qvel"]
mujoco.mj_forward(g, d)
fd = mujoco.MjData(model)
fd.qpos[:] = a["start_qpos"]
adr = int(model.joint("upper_arm_module_free").qposadr[0])
fd.qpos[adr : adr + 3] = d.body("upper_arm_module").xpos
fd.qpos[adr + 3 : adr + 7] = d.body("upper_arm_module").xquat
mujoco.mj_forward(model, fd)
err = max(
    float(np.max(abs(fd.geom(g.geom(i).name).xpos - d.geom_xpos[i])))
    for i in range(g.ngeom)
    if g.geom(i).name
)
assert err < 1e-12, err
np.savez_compressed(out / "actual-endpoint-full51.npz", qpos=fd.qpos)
rows = np.load(out / "legacy_mapped_20N-trace.npz")["trace"]
np.savez_compressed(
    out / "replay-trace.npz", time=rows[:, 0], qpos=rows[:, 1:2], qvel=rows[:, 2:3]
)
summary = {
    "candidate": str(candidate),
    "fullstroke": bool(saved["qpos"][0] >= a["authored_axial_travel_m"]),
    "full51_transfer_error_m": err,
    "full51_endpoint": str(out / "actual-endpoint-full51.npz"),
    "hinge_rad": float(fd.qpos[model.joint("target_servo_2_shaft").qposadr[0]]),
    "scope": "Actual native force-driven endpoint; gripper-held rigid shape, internal hinge locked. Free gravity release is separate.",
    "drive": r,
}
(out / "receipt.json").write_text(json.dumps(summary, indent=2) + "\n")
spec = {
    "model": str(out / "legacy_mapped_20N.xml"),
    "grid_manifest": str(candidate / "grids/manifest.json"),
    "trace": str(out / "replay-trace.npz"),
    "title": "Upper arm | native force insertion",
    "method": "20 N balanced push | legacy mapped law | accelerated SDF | sparse solver",
    "result": f"{r['final_drive_axial_mm']:.3f} / 25.000 mm; native drive {r['drive_simulated_seconds']:.3f} s",
    "limitation": "Lateral/rotation/internal hinge held by virtual gripper; receivers fixed. Isolated derived CAD.",
    "lookat": [0.025, 0, 0.11],
    "distance": 0.48,
    "azimuth": 125,
    "elevation": -24,
    "force": {
        "piece": "upper_arm_module",
        "axis_world": a["axis_world"],
        "maximum_N": 20,
        "ramp_s": 0,
        "off_s": r["drive_simulated_seconds"],
    },
    "output": str(out / "insertion.mp4"),
}
(out / "render-spec.json").write_text(json.dumps(spec, indent=2) + "\n")
print(
    json.dumps({k: v for k, v in summary.items() if k != "drive"}, indent=2), flush=True
)
