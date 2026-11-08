"""Static source-collision repair; no saved dynamic/grip qualification is claimed."""

import json, xml.etree.ElementTree as ET
from pathlib import Path
import mujoco, numpy as np
from scipy.optimize import least_squares
from assembly_policy.so101.wowrobo_contact_backend import load_backend, set_surface_mode
from assembly_policy.so101.wowrobo_housing_registration import (
    apply_housing_centering,
    migrate_qpos,
)

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).parent
SOURCE = ROOT / "assets_local/wowrobo/photo-resets-repaired-cover-support-v2"
p = json.loads(
    (
        ROOT / "outputs/equilibrium-model-20261107/profile-bearing-v5/profile.json"
    ).read_text()
)
load_backend(p["grid_manifest"])
set_surface_mode(1)
old = mujoco.MjModel.from_xml_path(p["model_path"])
tree = ET.parse(p["model_path"])
apply_housing_centering(tree.getroot())
xml = OUT / "centered-coherent.xml"
tree.write(xml, encoding="unicode")
m = mujoco.MjModel.from_xml_path(str(xml))
d = mujoco.MjData(m)
before = mujoco.MjData(old)
roots = [
    "base_servo",
    "base_side_cover",
    "shoulder_connector",
    "assembly_pin",
    "upper_arm_module",
    "forearm_module",
]


def rootof(b):
    while b:
        name = m.body(b).name
        if name in roots:
            return name
        b = int(m.body_parentid[b])
    return None


geo = {g: rootof(int(m.geom_bodyid[g])) for g in range(m.ngeom)}


def update():
    mujoco.mj_kinematics(m, d)
    mujoco.mj_comPos(m, d)
    mujoco.mj_collision(m, d)


def pose(data, model, name):
    b = model.body(name).id
    return data.xpos[b].copy(), data.xmat[b].reshape(3, 3).copy()


def setcase(task, servo, position, rotation, theta):
    d.qpos[m.joint(f"target_servo_{servo}_shaft").qposadr[0]] = theta
    mujoco.mj_kinematics(m, d)
    rp, rr = pose(d, m, task)
    cp, cr = pose(d, m, f"target_servo_{servo}_pivot")
    change = rotation @ cr.T
    a = m.joint(task + "_free").qposadr[0]
    d.qpos[a : a + 3] = position + change @ (rp - cp)
    mujoco.mju_mat2Quat(d.qpos[a + 3 : a + 7], (change @ rr).ravel())
    update()


def depth(task):
    values = []
    pairs = []
    for c in d.contact:
        ga, gb = map(int, c.geom)
        an, bn = m.geom(ga).name, m.geom(gb).name
        ra, rb = geo[ga], geo[gb]
        if not any(n and n.startswith(task + "_printed") for n in [an, bn]):
            continue
        if (
            ra == rb == task
            or task in [ra, rb]
            and ((ra is not None and rb is not None) or "review_table" in [an, bn])
        ):
            values.append(float(c.dist))
            pairs.append({"geoms": [an, bn], "distance_m": float(c.dist)})
    return min(values + [0.0]), sorted(pairs, key=lambda p: p["distance_m"])[:8]


def select(task, servo, low, high, count=101):
    q = d.qpos.copy()
    cp, cr = pose(d, m, f"target_servo_{servo}_pivot")
    h0 = float(d.joint(f"target_servo_{servo}_shaft").qpos[0])
    candidates = []
    rows = []
    for theta in np.linspace(low, high, count):
        d.qpos[:] = q
        setcase(task, servo, cp, cr, float(theta))
        dep, pairs = depth(task)
        energy = sum(
            float(m.body_mass[b] * 9.81 * d.xipos[b, 2])
            for b in range(m.nbody)
            if rootof(b) == task
        )
        row = {
            "theta_rad": float(theta),
            "depth_m": dep,
            "energy_J": energy,
            "pairs": pairs,
        }
        rows.append(row)
        if dep >= -0.00005:
            candidates.append((energy, abs(theta - h0), d.qpos.copy(), row))
    if not candidates:
        d.qpos[:] = q
        update()
        return None, rows
    _, _, qnew, row = min(candidates, key=lambda r: r[:2])
    d.qpos[:] = qnew
    update()
    return row, rows


joints = {
    side: [
        m.joint(side + "_" + n).qposadr[0]
        for n in [
            "shoulder_pan",
            "shoulder_lift",
            "elbow_flex",
            "wrist_flex",
            "wrist_roll",
        ]
    ]
    for side in ["left", "right"]
}


def fit_hand(side, pgoal, rgoal):
    site = m.site(side + "_gripperframe").id
    adr = joints[side]
    jids = [
        m.joint(side + "_" + n).id
        for n in [
            "shoulder_pan",
            "shoulder_lift",
            "elbow_flex",
            "wrist_flex",
            "wrist_roll",
        ]
    ]
    bounds = m.jnt_range[jids]
    qstart = d.qpos[adr].copy()

    def fun(q):
        d.qpos[adr] = q
        mujoco.mj_kinematics(m, d)
        r = d.site_xmat[site].reshape(3, 3)
        return np.r_[
            100 * (d.site_xpos[site] - pgoal),
            2 * (r[:, 2] - rgoal[:, 2]),
            0.1 * (r[:, 0] - rgoal[:, 0]),
        ]

    sol = least_squares(
        fun,
        np.clip(qstart, bounds[:, 0] + 1e-7, bounds[:, 1] - 1e-7),
        bounds=(bounds[:, 0] + 1e-7, bounds[:, 1] - 1e-7),
        max_nfev=200,
        ftol=1e-11,
        gtol=1e-11,
        xtol=1e-11,
    )
    fun(sol.x)
    return {
        "site_position_error_m": float(np.linalg.norm(d.site_xpos[site] - pgoal)),
        "span_error": float(
            np.linalg.norm(d.site_xmat[site].reshape(3, 3)[:, 2] - rgoal[:, 2])
        ),
        "target_pos": pgoal.tolist(),
        "target_rot": rgoal.tolist(),
    }


allrows = {}
ready = {}
upper_support_theta = None
stages = [
    "21_upper_far_side",
    "22_upper_half",
    "23_upper_near",
    "25_right_support",
    "26_final_floor_grasp",
    "28_final_transport",
    "29_final_far_side",
    "30_final_half",
    "31_final_near",
]
for stage in stages:
    with np.load(SOURCE / "stages" / (stage + "_train.npz")) as z:
        mujoco.mj_setState(
            old, before, z["states"][0], mujoco.mjtState.mjSTATE_FULLPHYSICS
        )
        ctrl = z["controls"][0].copy()
    mujoco.mj_kinematics(old, before)
    q, registration = migrate_qpos(old, m, before.qpos)
    d.qpos[:] = q
    d.ctrl[:] = ctrl
    update()
    number = int(stage.split("_")[0])
    old_upper = pose(before, old, "upper_arm_module")
    cp, cr = pose(d, m, "target_servo_2_pivot")
    h0 = float(d.joint("target_servo_2_shaft").qpos[0])
    record = {"registration": registration, "source_h2": h0}
    if number < 25:
        chosen, rows = select("upper_arm_module", 2, h0 - 0.01, h0 + 0.12, 101)
    elif upper_support_theta is None:
        chosen, rows = select("upper_arm_module", 2, h0 - 0.02, h0 + 0.6, 121)
        if chosen:
            upper_support_theta = chosen["theta_rad"]
    else:
        setcase("upper_arm_module", 2, cp, cr, upper_support_theta)
        chosen = {
            "theta_rad": upper_support_theta,
            "depth_m": depth("upper_arm_module")[0],
        }
        rows = []
    record["upper_candidate"] = chosen
    record["upper_scan"] = rows
    if chosen is None:
        allrows[stage] = record
        print(stage, "NOUPPER", flush=True)
        continue
    new_upper = pose(d, m, "upper_arm_module")
    change = new_upper[1] @ old_upper[1].T
    translation = new_upper[0] - change @ old_upper[0]
    targets = {}
    if number >= 25:
        side = "right"
        sid = old.site(side + "_gripperframe").id
        targets[side] = (
            change @ before.site_xpos[sid] + translation,
            change @ before.site_xmat[sid].reshape(3, 3),
        )
    if number >= 29:
        adr = m.joint("forearm_module_free").qposadr[0]
        d.qpos[adr : adr + 3] = change @ d.qpos[adr : adr + 3] + translation
        rotation = change @ pose(d, m, "forearm_module")[1]
        mujoco.mju_mat2Quat(d.qpos[adr + 3 : adr + 7], rotation.ravel())
        update()
        h3 = float(d.joint("target_servo_3_shaft").qpos[0])
        final, scan = select("forearm_module", 3, h3 - 0.2, h3 + 0.08, 141)
        record["final_candidate"] = final
        record["final_scan"] = scan
        sid = old.site("left_gripperframe").id
        targets["left"] = (
            change @ before.site_xpos[sid] + translation,
            change @ before.site_xmat[sid].reshape(3, 3),
        )
        if final is None:
            allrows[stage] = record
            print(stage, "NOFINAL", flush=True)
            continue
    for side, (pg, rg) in targets.items():
        record.setdefault("hand_targets", {})[side] = fit_hand(side, pg, rg)
    update()
    record["upper_depth_after_ik_m"] = depth("upper_arm_module")[0]
    record["active_depth_after_ik_m"] = (
        depth("forearm_module")[0] if number >= 29 else record["upper_depth_after_ik_m"]
    )
    record["qpos"] = d.qpos.tolist()
    record["ctrl"] = ctrl.tolist()
    allrows[stage] = record
    np.savez_compressed(
        OUT / (stage + "-coherent-initializer.npz"),
        qpos=d.qpos.copy(),
        ctrl=ctrl,
        source_qpos=before.qpos.copy(),
    )
    ready[stage] = record
    print(
        stage,
        chosen,
        "final",
        record.get("final_candidate"),
        "hands",
        record.get("hand_targets"),
        flush=True,
    )
(OUT / "coherent-initializers.json").write_text(json.dumps(allrows, indent=2) + "\n")
