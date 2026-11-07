"""Source-preserving upper-fit mechanisms and bounded compliant-tool probes."""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

from assembly_policy.so101.wowrobo_balanced_push import BalancedPush
from assembly_policy.so101.wowrobo_contact_backend import load_backend
from scripts.diagnose_wowrobo_insertion import configure

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/insertion-fix-20261107/upper"


def nums(x):
    return " ".join(format(float(v), ".17g") for v in np.asarray(x).ravel())


def build_compliant(source, out, alignment, force=20, lateral_k=1000, rotation_k=1):
    """Re-express locked source shape as an exactly equivalent rigid COM body.

    Six native tool DOFs retain force/contact self-alignment. Native damping at
    the whole COM avoids both explicit-drag instability and offset-origin torque.
    """
    source = Path(source)
    out = Path(out)
    m = mujoco.MjModel.from_xml_path(str(source))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    active = {m.body("upper_arm_module").id, m.body("target_servo_2_pivot").id}
    mass = sum(float(m.body_mass[b]) for b in active)
    com = sum(m.body_mass[b] * d.xipos[b] for b in active) / mass
    inertia = np.zeros((3, 3))
    for b in active:
        r = d.xipos[b] - com
        rot = d.ximat[b].reshape(3, 3)
        inertia += rot @ np.diag(m.body_inertia[b]) @ rot.T + m.body_mass[b] * (
            (r @ r) * np.eye(3) - np.outer(r, r)
        )
    frame = np.array(alignment["feature_frame"]["socket_frame_world"])
    local_inertia = frame.T @ inertia @ frame
    root = ET.parse(source).getroot()
    world = root.find("worldbody")
    body = world.find("body[@name='upper_arm_module']")
    bid = m.body("upper_arm_module").id
    for child in list(body):
        if child.tag in ("joint", "freejoint"):
            body.remove(child)
    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, frame.ravel())
    wrapper = ET.Element("body", name="upper_fit_tool", pos=nums(com), quat=nums(quat))
    ET.SubElement(
        wrapper,
        "inertial",
        mass=nums([mass]),
        pos="0 0 0",
        fullinertia=nums(
            [
                local_inertia[0, 0],
                local_inertia[1, 1],
                local_inertia[2, 2],
                local_inertia[0, 1],
                local_inertia[0, 2],
                local_inertia[1, 2],
            ]
        ),
    )
    for k in range(3):
        ET.SubElement(
            wrapper,
            "joint",
            name=f"upper_tool_slide_{k}",
            type="slide",
            axis=nums(np.eye(3)[k]),
            damping=nums([2 * np.sqrt(lateral_k * mass) if k < 2 else force / 0.02]),
            stiffness=nums([lateral_k if k < 2 else 0]),
            armature="0",
            frictionloss="0",
            limited="false",
        )
    ET.SubElement(
        wrapper,
        "joint",
        name="upper_tool_ball",
        type="ball",
        damping=nums(
            [2 * np.sqrt(rotation_k * np.linalg.eigvalsh(local_inertia).max())]
        ),
        stiffness=nums([rotation_k]),
        armature="0",
        frictionloss="0",
        limited="false",
    )
    q = np.empty(4)
    mujoco.mju_mat2Quat(q, (frame.T @ d.xmat[bid].reshape(3, 3)).ravel())
    body.set("pos", nums(frame.T @ (d.xpos[bid] - com)))
    body.set("quat", nums(q))
    for b in body.iter("body"):
        inertial = b.find("inertial")
        if inertial is not None:
            b.remove(inertial)
        ET.SubElement(b, "inertial", mass="0", pos="0 0 0", diaginertia="0 0 0")
    world.remove(body)
    wrapper.append(body)
    world.append(wrapper)
    out.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(out, encoding="unicode")
    new = mujoco.MjModel.from_xml_path(str(out))
    nd = mujoco.MjData(new)
    mujoco.mj_forward(new, nd)
    geom_error = max(
        float(np.max(abs(nd.geom(m.geom(g).name).xpos - d.geom_xpos[g])))
        for g in range(m.ngeom)
        if m.geom(g).name
    )
    if geom_error > 1e-12:
        raise RuntimeError(f"COM re-expression moved source geometry {geom_error}")
    new_mass = sum(new.body_mass)
    old_mass = sum(m.body_mass)
    if abs(new_mass - old_mass) > 1e-12:
        raise RuntimeError("Whole model mass changed")
    return {
        "scene": str(out),
        "whole_mass_kg": mass,
        "whole_com_world_m": com.tolist(),
        "whole_inertia_world": inertia.tolist(),
        "whole_inertia_tool_frame": local_inertia.tolist(),
        "geometry_error_m": geom_error,
        "lateral_stiffness_N_per_m": lateral_k,
        "rotation_stiffness_Nm_per_rad": rotation_k,
        "note": "Explicit compliant virtualgripper;6DOFforce/contact,selfalignmentnotposeclamp. Fixedsubassembly articulation retained for firsttest. Originalmass/COM/inertia exactlyconsolidated;noCADchange.",
    }


class WholePush(BalancedPush):
    def __init__(self, m, d, axis):
        self.model = m
        self.axis = np.asarray(axis)
        self.damping = 0.0
        bid = m.body("upper_fit_tool").id
        rb = d.xmat[bid].reshape(3, 3)
        points = []
        for gid in range(m.ngeom):
            if m.geom(gid).name not in [
                "upper_arm_module_printed_0",
                "servo_2_case",
                "servo_2_horns",
            ]:
                continue
            mid = m.geom_dataid[gid]
            v = m.mesh_vert[
                m.mesh_vertadr[mid] : m.mesh_vertadr[mid] + m.mesh_vertnum[mid]
            ].astype(float)
            world = v @ d.geom_xmat[gid].reshape(3, 3).T + d.geom_xpos[gid]
            points.append((world - d.xpos[bid]) @ rb)
        vertices = np.unique(np.vstack(points), axis=0)
        hull = ConvexHull(vertices)
        self.bodies = [
            {
                "id": bid,
                "name": "upper_fit_tool",
                "mass_fraction": 1.0,
                "vertices": vertices,
                "faces": hull.simplices,
                "normals": hull.equations[:, :3],
                "offsets": hull.equations[:, 3],
                "com": np.zeros(3),
            }
        ]
        self.jp = np.zeros((3, m.nv))
        self.jr = np.zeros((3, m.nv))
        self.max_drive_moment = 0.0
        self.max_line_error = 0.0


def run(args):
    load_backend(ROOT / "outputs/linear-insertion-updated-20261107/grids/manifest.json")
    alignment = json.loads(
        (
            ROOT / "outputs/linear-insertion-aligned-20261107/upper/alignment.json"
        ).read_text()
    )
    OUT.mkdir(parents=True, exist_ok=True)
    source = ROOT / "outputs/legacy-fit-port-20261107/upper/legacy_mapped_20N.xml"
    scene = OUT / f"compliant-k{args.lateral_k:g}-r{args.rotation_k:g}.xml"
    meta = build_compliant(
        source, scene, alignment, args.force, args.lateral_k, args.rotation_k
    )
    m = mujoco.MjModel.from_xml_path(str(scene))
    d = mujoco.MjData(m)
    configure(
        m,
        "upper_fit_tool",
        ["base_servo", "base_side_cover", "shoulder_connector", "assembly_pin"],
        True,
    )
    mujoco.mj_forward(m, d)
    axis = np.array(alignment["axis_world"])
    held = m.geom("servo_2_case").id
    p0 = d.geom_xpos[held].copy()
    r0 = d.geom_xmat[held].reshape(3, 3).copy()
    push = WholePush(m, d, axis)
    rows = []
    peak = 0.0
    stop = "duration"
    maxlat = 0.0
    maxangle = 0.0
    for i in range(round(args.seconds / m.opt.timestep)):
        push.apply(d, args.force * min(1.0, d.time / 0.8))
        mujoco.mj_step(m, d)
        mujoco.mj_forward(m, d)
        delta = d.geom_xpos[held] - p0
        ax = float(delta @ axis)
        lat = float(np.linalg.norm(delta - ax * axis))
        angle = float(
            np.linalg.norm(
                Rotation.from_matrix(d.geom_xmat[held].reshape(3, 3) @ r0.T).as_rotvec()
            )
        )
        depth = max(0.0, -min((float(c.dist) for c in d.contact), default=0))
        peak = max(peak, depth)
        maxlat = max(maxlat, lat)
        maxangle = max(maxangle, angle)
        if i % 10 == 0:
            rows.append(
                [float(d.time), ax, lat, angle, depth, int(d.ncon), *d.qpos, *d.qvel]
            )
        if depth > 0.0005:
            stop = "0.5mm source contact guard"
            break
        if lat > 0.002:
            stop = "2mm lateral tool travel guard"
            break
        if angle > np.deg2rad(5):
            stop = "5degree orientation tool travel guard"
            break
        if ax >= 0.025:
            stop = "reference crossed"
            break
    result = {
        "mechanism": "6DOF compliant virtualgripper, native implicit COM damping",
        "force_N": args.force,
        "build": meta,
        "final_axial_mm": ax * 1000,
        "max_lateral_mm": maxlat * 1000,
        "max_rotation_deg": float(np.degrees(maxangle)),
        "peak_native_penetration_mm": peak * 1000,
        "stop": stop,
        "simulated_seconds": float(d.time),
        "physical_success": False,
        "push": push.description(),
    }
    prefix = OUT / f"compliant-k{args.lateral_k:g}-r{args.rotation_k:g}"
    prefix.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
    np.savez_compressed(
        str(prefix) + "-state.npz", qpos=d.qpos, qvel=d.qvel, trace=np.array(rows)
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--force", type=float, default=20)
    p.add_argument("--lateral-k", type=float, default=1000)
    p.add_argument("--rotation-k", type=float, default=1)
    p.add_argument("--seconds", type=float, default=4)
    run(p.parse_args())
