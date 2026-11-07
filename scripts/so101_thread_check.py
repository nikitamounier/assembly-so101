"""Mechanical bench check: native helical contacts under torque versus axial push.

The bench guides lateral alignment with slide/hinge joints. The actual workcell
has a free screw and no guide, weld, equality, or automatic mating constraint.
"""

import json
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from assembly_policy.so101.fastener import AXIS, PITCH
from assembly_policy.so101.scene import ROOT, build_xml, load_plugin, metadata, numbers


def thread_model(push_only=False):
    """Build the guided bench in motor-local coordinates, independent of layout."""
    load_plugin()
    xml = ET.fromstring(build_xml(True))
    w = xml.find("worldbody")
    xml.remove(xml.find("contact"))
    for e in list(w):
        if e.get("name") not in ("motor", "screw"):
            w.remove(e)
    motor = w.find("body[@name='motor']")
    motor.remove(motor.find("freejoint"))
    motor.set("pos", "0 0 0")
    motor.set("quat", "1 0 0 0")
    screw = w.find("body[@name='screw']")
    screw.remove(screw.find("freejoint"))
    pos = AXIS - np.asarray(metadata()["motor"]["origin"]) + [0, 0, 0.0045]
    screw.set("pos", numbers(pos))
    screw.set("quat", "1 0 0 0")
    ET.SubElement(screw, "joint", name="depth", type="slide", axis="0 0 1", damping=".3")
    ET.SubElement(screw, "joint", name="turn", type="hinge", axis="0 0 1", armature=".000002")
    for e in list(xml.find("actuator")):
        xml.find("actuator").remove(e)
    for e in list(xml.find("extension")):
        if e.get("plugin") == "so101.contact_driver":
            xml.find("extension").remove(e)
    # Slow controlled rotation; no prescribed axial motion.
    ET.SubElement(xml.find("actuator"), "velocity", joint="turn", kv=".0003", forcerange="-.002 .002")
    m = mujoco.MjModel.from_xml_string(ET.tostring(xml, encoding="unicode"))
    d = mujoco.MjData(m)
    # Replacing free joints with guides must not lower collision accuracy.
    # In the real workcell this pair uses the free-body assembly budget.
    m.numeric("so101_sdf_search").data[:] = m.numeric("so101_assembly_search").data
    d.ctrl[0] = 0 if push_only else -2 * np.pi
    d.qfrc_applied[0] = -1.0 if push_only else -0.05
    return m, d


def bench(push_only=False, timestep=None):
    m, d = thread_model(push_only)
    if timestep is not None:
        if not np.isfinite(timestep) or not 0 < timestep <= m.opt.timestep:
            raise ValueError("Expected the workcell timestep or a finer one")
        m.opt.timestep = timestep
    trace = []
    for k in range(round(2 / m.opt.timestep)):
        mujoco.mj_step(m, d)
        if k % round(0.01 / m.opt.timestep) == 0:
            trace.append([d.time, *d.qpos])
    a = np.asarray(trace)
    turns = (a[-1, 2] - a[30, 2]) / (2 * np.pi)
    advance = a[-1, 1] - a[30, 1]
    return dict(
        push_only=push_only,
        reference_frame="motor-local",
        timestep_s=float(m.opt.timestep),
        contact_search=m.numeric("so101_sdf_search").data.astype(int).tolist(),
        turns=float(turns),
        advance_m=float(advance),
        expected_advance_m=float(turns * PITCH),
        trace=trace,
    )


if __name__ == "__main__":
    result = {"rotation": bench(), "axial": bench(True)}
    out = ROOT / "artifacts_local/thread_check.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print({k: {a: b for a, b in v.items() if a != "trace"} for k, v in result.items()}, flush=True)
