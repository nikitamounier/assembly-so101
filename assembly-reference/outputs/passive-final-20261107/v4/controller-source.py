"""Force-held motor diagnostic: free bracket articulation, gravity and native contacts.

Receiver is explicitly fixed; this is not a bimanual grasp/training admission test.
Only initial configuration is authored. All subsequent hinge angles are simulated.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

from .wowrobo_contact_backend import load_backend, set_surface_mode
from .wowrobo_mating_profile import write_profile_scene

ROOT = Path(__file__).resolve().parents[2]
FROZEN = ROOT / "outputs/insertion-fix-20261107/final-common-v1"
ALIGN = ROOT / "outputs/insertion-fix-20261107/final/canonical/alignment-prepared.json"


def nums(a):
    return " ".join(format(float(x), ".17g") for x in np.asarray(a).ravel())


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def build_fixture(out, own_contact=True):
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    load_backend(FROZEN / "grids/manifest.json")
    set_surface_mode(1)
    src = FROZEN / "scene.xml"
    m = mujoco.MjModel.from_xml_path(str(src))
    d = mujoco.MjData(m)
    a = json.loads(ALIGN.read_text())
    d.qpos[:] = a["start_qpos"]
    mujoco.mj_forward(m, d)
    goal = mujoco.MjData(m)
    goal.qpos[:] = a["end_qpos"]
    mujoco.mj_forward(m, goal)
    r = ET.parse(src).getroot()
    r.find("option").set("gravity", "0 0 -9.81")
    # Keep final free root and its two real hinges. Shaft4 only carries the distal
    # horn, so freeze that irrelevant passive horn at its initial pose.
    keep = {"forearm_module_free", "target_servo_3_shaft"}
    for body in r.iter("body"):
        bid = m.body(body.get("name")).id
        par = int(m.body_parentid[bid])
        pr = d.xmat[par].reshape(3, 3)
        q = np.empty(4)
        mujoco.mju_mat2Quat(q, (pr.T @ d.xmat[bid].reshape(3, 3)).ravel())
        for key in ("quat", "euler", "axisangle", "xyaxes", "zaxis"):
            body.attrib.pop(key, None)
        body.set("pos", nums(pr.T @ (d.xpos[bid] - d.xpos[par])))
        body.set("quat", nums(q))
        for j in list(body):
            if j.tag in ("joint", "freejoint") and j.get("name") not in keep:
                body.remove(j)
    wb = r.find("worldbody")
    for b in list(wb):
        if b.tag == "body" and b.get("name") in ("left_base", "right_base"):
            wb.remove(b)
        if b.get("name", "").endswith("witness"):
            wb.remove(b)
    for body in r.iter("body"):
        for g in list(body.findall("geom")):
            if g.get("name", "").endswith(("_grip_center", "_grip_tangent")):
                body.remove(g)
    for tag in ("actuator", "tendon", "equality", "keyframe", "sensor"):
        for node in r.findall(tag):
            r.remove(node)
    # Incoming-vs-receiver only. Fixed prerequisites do not generate meaningless
    # fixed-fixed contacts. Explicit own-pairs below bypass parent filtering.
    for b in r.iter("body"):
        active = b.get("name") in (
            "forearm_module",
            "target_servo_3_pivot",
            "target_servo_4_pivot",
        )
        for g in b.findall("geom"):
            if g.get("type") == "sdf":
                g.set("contype", "1" if active else "2")
                g.set("conaffinity", "2" if active else "1")
    table = wb.find("geom[@name='review_table']")
    table.set("contype", "2")
    table.set("conaffinity", "1")
    base = out / "fixture-base.xml"
    ET.ElementTree(r).write(base, encoding="unicode")
    profile = out / "scene.xml"
    write_profile_scene(base, profile, "final")
    r = ET.parse(profile).getroot()
    con = r.find("contact")
    if own_contact:
        for i in (0, 1):
            ET.SubElement(
                con,
                "pair",
                name=f"own_motor_stop_{i}",
                geom1="servo_3_case__contact",
                geom2=f"forearm_module_printed_{i}__contact",
                condim="4",
                friction=".7 .005 .0001 .0001 .0001",
                solref=".002 1",
                solimp=".95 .99 .0002",
            )
    ET.ElementTree(r).write(profile, encoding="unicode")
    info = {
        "scene": str(profile),
        "source_scene": str(src),
        "source_sha256": digest(src),
        "grid_manifest": str(FROZEN / "grids/manifest.json"),
        "grid_sha256": digest(FROZEN / "grids/manifest.json"),
        "alignment_sha256": digest(ALIGN),
        "receiver_fixed": True,
        "hand_present": False,
        "hinge_actuation": False,
        "prescribed_hinge_angle": False,
        "initial_hinge_rad": float(d.joint("target_servo_3_shaft").qpos[0]),
        "own_motor_stop": own_contact,
        "goal_motor_position": goal.geom("servo_3_case").xpos.tolist(),
        "motor_rotation": d.geom("servo_3_case").xmat.tolist(),
        "scene_sha256": digest(profile),
    }
    (out / "fixture.json").write_text(json.dumps(info, indent=2) + "\n")
    return info


def run(
    out,
    *,
    seconds=12,
    own_contact=True,
    kp=1000,
    kd=25,
    force_limit=25,
    dt=0.0005,
    preload=0.0,
):
    info = build_fixture(out, own_contact)
    m = mujoco.MjModel.from_xml_path(info["scene"])
    m.opt.timestep = dt
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    motor = m.body("target_servo_3_pivot").id
    gid = m.geom("servo_3_case").id
    h = m.joint("target_servo_3_shaft")
    qa = int(h.qposadr[0])
    root = m.joint("forearm_module_free")
    ra = int(root.qposadr[0])
    goal = np.array(info["goal_motor_position"])
    rot = d.xmat[motor].reshape(3, 3).copy()
    # Start beside receiver on table, maintaining the initial relative hinge.
    verts = []
    for g in range(m.ngeom):
        if (
            m.body(m.geom_bodyid[g]).name
            not in ("forearm_module", "target_servo_3_pivot", "target_servo_4_pivot")
            or m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH
        ):
            continue
        mid = m.geom_dataid[g]
        i = m.mesh_vertadr[mid]
        n = m.mesh_vertnum[mid]
        verts.append(
            m.mesh_vert[i : i + n] @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
        )
    shift = np.array([0, -0.23, 0.001 - min(v[:, 2].min() for v in verts)])
    d.qpos[ra : ra + 3] += shift
    mujoco.mj_forward(m, d)
    start = d.geom_xpos[gid].copy()
    high = start.copy()
    high[2] = max(goal[2] + 0.13, start[2] + 0.16)
    above = goal.copy()
    above[2] = high[2]
    active_mass = sum(
        m.body_mass[b]
        for b in range(m.nbody)
        if m.body(b).name
        in ("forearm_module", "target_servo_3_pivot", "target_servo_4_pivot")
    )
    jp = np.zeros((3, m.nv))
    jr = np.zeros_like(jp)
    records = []
    maxforce = 0.0
    selfcontacts = 0
    receivercontacts = 0
    min_contact = 0.0
    min_own = 0.0
    min_receiving = 0.0
    # Smooth motor Cartesian path, never a qpos setter during integration.
    pushgoal = goal + np.array([0, 0, -preload])
    knots = [
        (0, start),
        (0.3, start + np.array([0, 0, 0.025])),
        (3, high),
        (5, above),
        (10, goal),
        (seconds, pushgoal),
    ]
    for step in range(int(seconds / dt) + 1):
        t = float(d.time)
        for (t0, p0), (t1, p1) in itertools.pairwise(knots):
            if t <= t1 + dt:
                break
        u = np.clip((t - t0) / max(t1 - t0, dt), 0, 1)
        s = u * u * (3 - 2 * u)
        sd = 6 * u * (1 - u) / max(t1 - t0, dt)
        target = p0 + s * (p1 - p0)
        velocity = sd * (p1 - p0)
        mujoco.mj_jac(m, d, jp, jr, d.geom_xpos[gid], motor)
        force = (
            kp * (target - d.geom_xpos[gid])
            + kd * (velocity - jp @ d.qvel)
            + np.array([0, 0, active_mass * 9.81])
        )
        force = np.clip(force, -force_limit, force_limit)
        cur = d.xmat[motor].reshape(3, 3)
        err = 0.5 * sum(
            (np.cross(cur[:, i], rot[:, i]) for i in range(3)), start=np.zeros(3)
        )
        torque = 5.0 * err - 0.08 * (jr @ d.qvel)
        torque = np.clip(torque, -1, 1)
        d.qfrc_applied[:] = 0
        mujoco.mj_applyFT(m, d, force, torque, d.geom_xpos[gid], motor, d.qfrc_applied)
        own = recv = 0
        for c in d.contact:
            min_contact = min(min_contact, float(c.dist))
            names = [m.geom(int(g)).name for g in c.geom]
            if "servo_3_case__contact" in names and any(
                n.startswith("forearm_module_printed") for n in names
            ):
                own += 1
                min_own = min(min_own, float(c.dist))
            if any(n.startswith("upper_arm_module_printed") for n in names):
                recv += 1
                min_receiving = min(min_receiving, float(c.dist))
        selfcontacts += own > 0
        receivercontacts += recv > 0
        maxforce = max(maxforce, float(np.linalg.norm(force)))
        if step % 20 == 0:
            records.append(
                (
                    t,
                    d.qpos.copy(),
                    d.qvel.copy(),
                    d.geom_xpos[gid].copy(),
                    target.copy(),
                    force.copy(),
                    float(d.qpos[qa]),
                    own,
                    recv,
                )
            )
        if step % 2000 == 0:
            print(
                f"t={t:.1f} hinge={np.degrees(d.qpos[qa]):.2f}deg own={own} recv={recv} mincontact={min_contact:.6g}",
                flush=True,
            )
        mujoco.mj_step(m, d)
        if not np.isfinite(d.qpos).all() or np.max(np.abs(d.qvel)) > 1e4:
            raise RuntimeError("Numerical instability")
    out = Path(out)
    np.savez_compressed(
        out / "trajectory.npz",
        time=[r[0] for r in records],
        qpos=[r[1] for r in records],
        qvel=[r[2] for r in records],
        motor_position=[r[3] for r in records],
        motor_target=[r[4] for r in records],
        force=[r[5] for r in records],
        hinge_angle=[r[6] for r in records],
        own_contacts=[r[7] for r in records],
        receiver_contacts=[r[8] for r in records],
    )
    angles = np.array([r[6] for r in records])
    info.update(
        nq=m.nq,
        nv=m.nv,
        hinge_range_observed_rad=[float(angles.min()), float(angles.max())],
        last_motor_error_m=float(np.linalg.norm(d.geom_xpos[gid] - goal)),
        max_force_N=maxforce,
        self_contact_steps=int(selfcontacts),
        receiver_contact_steps=int(receivercontacts),
        minimum_contact_distance_m=min_contact,
        minimum_own_contact_distance_m=min_own,
        minimum_receiver_contact_distance_m=min_receiving,
        preload_target_m=preload,
        active_mass_kg=float(active_mass),
        seconds=seconds,
        controller={"kp": kp, "kd": kd, "force_limit": force_limit},
        status="diagnostic_unqualified",
    )
    (out / "result.json").write_text(json.dumps(info, indent=2) + "\n")
    return info
