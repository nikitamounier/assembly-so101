"""Measured initial workcell and deterministic, grasp-tangent layout sampling.

The tangent constraint applies at an explicit local grip center, not at the
component bounding-box center. This module constructs initial poses; it does
not assert a collision-qualified insertion or a reachable robot grasp.
"""

from __future__ import annotations

import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import trimesh

from scripts.review_wowrobo_components import add, numbers, review_xml

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/wowrobo_layout.json"


def quaternion(rotation):
    q = np.empty(4)
    mujoco.mju_mat2Quat(q, np.asarray(rotation).ravel())
    return q


def rz(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def mesh_vertices(model, data, geom_id):
    mesh_id = model.geom_dataid[geom_id]
    start, count = model.mesh_vertadr[mesh_id], model.mesh_vertnum[mesh_id]
    return (
        model.mesh_vert[start : start + count] @ data.geom_xmat[geom_id].reshape(3, 3).T
        + data.geom_xpos[geom_id]
    )


class InitialLayout:
    def __init__(self, assets, config=CONFIG):
        self.assets = Path(assets).resolve()
        self.cfg = json.loads(Path(config).read_text())
        self.manifest = json.loads((self.assets / "manifest.json").read_text())
        self.vertices = {}
        self.constituent_vertices = {}
        for name, piece in self.manifest["pieces"].items():
            for key in piece["constituents"]:
                self.constituent_vertices[key] = trimesh.load_mesh(
                    self.assets / self.manifest["constituents"][key]["mesh"],
                    process=False,
                ).vertices
            self.vertices[name] = np.concatenate(
                [self.constituent_vertices[key] for key in piece["constituents"]]
            )
        root = review_xml(self.assets)
        root.set("model", self.cfg["profile"])
        world = root.find("worldbody")
        world.find("geom[@name='review_table']").set(
            "size", numbers([*self.cfg["table_half_size_m"], 0.02])
        )
        provisional = mujoco.MjModel.from_xml_string(
            ET.tostring(root, encoding="unicode")
        )
        data = mujoco.MjData(provisional)
        mujoco.mj_forward(provisional, data)
        # Explicit printed-footprint references, excluding the rear controller plate.
        gid = next(
            i
            for i in range(provisional.ngeom)
            if provisional.geom_type[i] == 7
            and provisional.mesh(provisional.geom_dataid[i]).name
            == "left_base_so101_v2"
        )
        local = (
            mesh_vertices(provisional, data, gid)
            - data.xpos[provisional.body("left_base").id]
        )
        rear = float(local[:, 0].min())
        floor = float(local[:, 2].min())
        half = self.cfg["helper_rear_edge_distance_m"] / 2
        self.helper_roots = {
            "left": np.array([-half - rear, 0, -floor]),
            "right": np.array([half + rear, 0, -floor]),
        }
        self.rear_x = {"left": -half, "right": half}
        for side in ["left", "right"]:
            world.find(f"body[@name='{side}_base']").set(
                "pos", numbers(self.helper_roots[side])
            )
        self.axes = {
            side: self.helper_roots[side]
            + (
                np.array([0.0388353, -8.97657e-9, 0.0624])
                * ([1, 1, 1] if side == "left" else [-1, -1, 1])
            )
            for side in ["left", "right"]
        }
        # CAD's forward is -Y; rotate that direction toward the right helper (+X).
        base_rotation = rz(np.pi / 2)
        v = self.vertices["base_servo"] @ base_rotation.T
        target_rear = -half + self.cfg["target_rear_from_left_rear_m"]
        self.base_pos = np.array([target_rear - v[:, 0].min(), 0, -v[:, 2].min()])
        base = world.find("body[@name='base_servo']")
        base.remove(base.find("freejoint"))
        base.set("pos", numbers(self.base_pos))
        base.set("quat", numbers(quaternion(base_rotation)))
        self.base_rotation = base_rotation
        self.grips = {}
        self.arm_for = {item["piece"]: item["arm"] for item in self.cfg["pickup_order"]}
        for step in self.cfg["pickup_order"]:
            name = step["piece"]
            points = self.vertices[name]
            bounds = np.array([points.min(0), points.max(0)])
            center = bounds.mean(axis=0)
            axis = np.array([1.0, 0, 0])
            up = np.array([0.0, 0, 1.0])
            if "servo" in step:
                servo = step["servo"]
                item = self.manifest["constituents"][f"servo_{servo}_case"]
                center = np.array(item["bounds_m"]).mean(axis=0)
                axis = np.array(self.manifest["servo_axes"][str(servo)]["axis"])
                if axis[0] < 0:
                    axis = -axis
            elif name == "base_side_cover":
                # IMG1622/1621: opposed EXTERIOR side walls, inboard of the flange.
                center[0] = self.manifest["servo_axes"]["1"]["position_m"][0]
                inset = self.cfg["cover_side_grip_inset_m"]
                center[1] = (
                    bounds[1, 1] - inset
                    if step.get("grip_axis_sign", 1) < 0
                    else bounds[0, 1] + inset
                )
            elif name == "shoulder_connector":
                # IMG1617: sides of the upper motor socket, not the lower spine.
                socket = np.array(
                    self.manifest["constituents"]["shoulder_connector_printed_1"][
                        "bounds_m"
                    ]
                )
                center = socket.mean(0)
                center[0] = self.manifest["servo_axes"]["1"]["position_m"][0]
                center[2] = (
                    socket[0, 2]
                    + self.cfg["connector_grip_height_above_socket_bottom_m"]
                )
            else:
                center = np.array([0.0, 0.0, 0.014])
                up = np.array([0.0, -1.0, 0.0])
            # A reversed span reverses the radial end while preserving table-up.
            axis *= step.get("grip_axis_sign", 1)
            axis /= np.linalg.norm(axis)
            radial = np.cross(up, axis)
            radial /= np.linalg.norm(radial)
            up = np.cross(axis, radial)
            self.grips[name] = {
                "center": center,
                "axis": axis,
                "basis": np.column_stack([axis, radial, up]),
            }
            body = world.find(f"body[@name='{name}']")
            color = ".04 .62 .80 1" if step["arm"] == "left" else ".96 .43 .12 1"
            add(
                body,
                "geom",
                name=name + "_grip_center",
                type="sphere",
                pos=numbers(center),
                size=".0025",
                rgba=color,
                group="1",
            )
            add(
                body,
                "geom",
                name=name + "_grip_tangent",
                type="capsule",
                fromto=numbers(np.r_[center - 0.032 * axis, center + 0.032 * axis]),
                size=".0012",
                rgba=color,
                group="1",
            )
            add(
                body,
                "site",
                name=name + "_grasp_frame",
                pos=numbers(center),
                size=".003",
                group="3",
            )
        # Rear-edge witnesses are fixed world geometry, not inferred joint origins.
        for name, x in [
            ("left_rear", -half),
            ("right_rear", half),
            ("assembly_rear", target_rear),
        ]:
            add(
                world,
                "geom",
                name=name + "_witness",
                type="capsule",
                fromto=numbers([x, -0.08, 0.001, x, 0.08, 0.001]),
                size=".0008",
                rgba=".38 .42 .46 1",
                group="1",
            )
        self.xml_root = root
        self.model = mujoco.MjModel.from_xml_string(
            ET.tostring(root, encoding="unicode")
        )
        self.template = mujoco.MjData(self.model)
        for side in ["left", "right"]:
            for joint, value in [
                ("shoulder_lift", -1.0),
                ("elbow_flex", 1.0),
                ("wrist_flex", -0.4),
                ("gripper", 0.5),
            ]:
                self.template.qpos[self.model.joint(side + "_" + joint).qposadr[0]] = (
                    value
                )
        for a in range(self.model.nu):
            self.template.ctrl[a] = self.template.qpos[
                self.model.jnt_qposadr[self.model.actuator_trnid[a, 0]]
            ]
        mujoco.mj_forward(self.model, self.template)
        self.base_bounds = np.array(
            [(v + self.base_pos).min(0), (v + self.base_pos).max(0)]
        )
        self.helper_bounds = []
        for side in ["left", "right"]:
            gid = next(
                i
                for i in range(self.model.ngeom)
                if self.model.geom_type[i] == 7
                and self.model.mesh(self.model.geom_dataid[i]).name
                == side + "_base_so101_v2"
            )
            vs = mesh_vertices(self.model, self.template, gid)
            self.helper_bounds.append(np.array([vs.min(0), vs.max(0)]))

    def placement(self, name, radius, relative_angle):
        side = self.arm_for[name]
        theta = relative_angle + (np.pi if side == "right" else 0)
        radial = np.array([np.cos(theta), np.sin(theta), 0.0])
        tangent = np.array([-np.sin(theta), np.cos(theta), 0.0])
        desired = np.column_stack([-tangent, radial, [0.0, 0, 1.0]])
        grip = self.grips[name]
        rotation = desired @ grip["basis"].T
        grip_xy = self.axes[side][:2] + radius * radial[:2]
        pos = np.r_[grip_xy, 0.0] - rotation @ grip["center"]
        rotated = self.vertices[name] @ rotation.T
        pos[2] = -rotated[:, 2].min()
        vs = rotated + pos
        center = rotation @ grip["center"] + pos
        step = next(item for item in self.cfg["pickup_order"] if item["piece"] == name)
        local_radial = grip["basis"][:, 1]
        if "servo" in step:
            members = [f"servo_{step['servo']}_case", f"servo_{step['servo']}_horns"]
            region = np.concatenate([self.constituent_vertices[k] for k in members])
        elif name == "assembly_pin":
            region = self.vertices[name][self.vertices[name][:, 2] <= 0.028000001]
        else:
            projections = self.vertices[name] @ local_radial
            region = self.vertices[name][projections <= projections.min() + 0.020]
        near_all = float(np.min((vs - self.axes[side]) @ radial))
        region_world = region @ rotation.T + pos
        near_grip = float(np.min((region_world - self.axes[side]) @ radial))
        nearest_gap = near_grip - near_all
        near_distance = float(
            np.min(np.linalg.norm(vs[:, :2] - self.axes[side][:2], axis=1))
        )
        grip_distance = float(
            np.min(np.linalg.norm(region_world[:, :2] - self.axes[side][:2], axis=1))
        )
        distance_gap = grip_distance - near_distance
        return {
            "nearest_grasp_end_gap_m": nearest_gap,
            "nearest_grasp_distance_gap_m": distance_gap,
            "grasp_end_faces_arm": max(nearest_gap, distance_gap)
            <= self.cfg["servo_nearest_end_tolerance_m"],
            "name": name,
            "arm": side,
            "position": pos,
            "rotation": rotation,
            "bounds": np.array([vs.min(0), vs.max(0)]),
            "grip_center": center,
            "tangent": rotation @ grip["axis"],
            "radius": radius,
            "relative_angle": relative_angle,
        }

    def clear(self, item, occupied):
        if not item["grasp_end_faces_arm"]:
            return False
        b = item["bounds"]
        half = np.array(self.cfg["table_half_size_m"])
        gap = self.cfg["minimum_planar_clearance_m"]
        if np.any(b[0, :2] < -half) or np.any(b[1, :2] > half):
            return False
        return all(
            np.any(b[1, :2] + gap < other[0, :2])
            or np.any(other[1, :2] + gap < b[0, :2])
            for other in occupied
        )

    def sample(self, seed, nominal=False):
        rng = np.random.default_rng(seed)
        for attempt in range(500):
            placed = {}
            occupied = [self.base_bounds, *self.helper_bounds]
            for step in self.cfg["pickup_order"]:
                name = step["piece"]
                success = False
                for trial in range(500):
                    if nominal and attempt == 0 and trial == 0:
                        radius, degrees = self.cfg["nominal_polar"][name]
                    else:
                        lo, hi = self.cfg["grasp_radius_m"]
                        if "servo" in step:
                            lo = self.cfg["long_part_minimum_grasp_radius_m"]
                        radius = float(rng.uniform(lo, hi))
                        degrees = float(
                            rng.uniform(
                                -self.cfg["forward_half_angle_degrees"],
                                self.cfg["forward_half_angle_degrees"],
                            )
                        )
                    item = self.placement(name, radius, np.deg2rad(degrees))
                    if self.clear(item, occupied):
                        placed[name] = item
                        occupied.append(item["bounds"])
                        success = True
                        break
                if not success:
                    break
            if len(placed) == 5:
                data = mujoco.MjData(self.model)
                data.qpos[:] = self.template.qpos
                data.ctrl[:] = self.template.ctrl
                for name, item in placed.items():
                    j = self.model.joint(name + "_free").qposadr[0]
                    data.qpos[j : j + 3] = item["position"]
                    data.qpos[j + 3 : j + 7] = quaternion(item["rotation"])
                mujoco.mj_forward(self.model, data)
                return data, placed
        raise RuntimeError("Could not pack a clear grounded tangent layout")

    def export(self, output, count=32):
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        if count < 1:
            raise ValueError("At least one layout is required")
        nominal_data, nominal = self.sample(0, nominal=True)
        for name, item in nominal.items():
            body = self.xml_root.find(f"worldbody/body[@name='{name}']")
            body.set("pos", numbers(item["position"]))
            body.set("quat", numbers(quaternion(item["rotation"])))
        keyframes = self.xml_root.find("keyframe")
        if keyframes is not None:
            self.xml_root.remove(keyframes)
        keyframes = add(self.xml_root, "keyframe")
        add(
            keyframes,
            "key",
            name="initial_scene",
            qpos=numbers(nominal_data.qpos),
            ctrl=numbers(nominal_data.ctrl),
        )
        xml = ET.tostring(self.xml_root, encoding="unicode")
        (output / "scene.xml").write_text(xml)
        signature = hashlib.sha256(xml.encode()).hexdigest()
        spec = mujoco.mjtState.mjSTATE_FULLPHYSICS
        states = []
        receipts = []
        for seed in range(count):
            data, placed = self.sample(seed, nominal=seed == 0)
            state = np.empty(mujoco.mj_stateSize(self.model, spec))
            mujoco.mj_getState(self.model, data, state, spec)
            states.append(state)
            receipts.append(
                {
                    "seed": seed,
                    "grips": {
                        name: {
                            k: (v.tolist() if isinstance(v, np.ndarray) else v)
                            for k, v in item.items()
                        }
                        for name, item in placed.items()
                    },
                }
            )
        bank = output / "layouts"
        bank.mkdir(exist_ok=True)
        np.savez_compressed(
            bank / "initial_scene_train.npz",
            states=states,
            seed=np.arange(count),
            family=np.r_[0, np.ones(count - 1, dtype=int)],
        )
        (bank / "manifest.json").write_text(
            json.dumps(
                {
                    "model_signature": signature,
                    "state_spec": int(spec),
                    "task_names": ["initial_scene"],
                    "family_names": ["Nominal layout", "Randomized tangent layout"],
                    "reset_distribution": "layout_review",
                    "display_name": "WowRobo · initial tangent layouts",
                    "default_viewer_collection": True,
                    "focus_by_task": {
                        "initial_scene": {"target": [0, 0, 0.11], "radius": 0.46}
                    },
                },
                indent=2,
            )
            + "\n"
        )
        (output / "layout-receipts.json").write_text(
            json.dumps(receipts, indent=2) + "\n"
        )
        (output / "manifest.json").write_text(
            json.dumps(
                {
                    "profile": self.cfg["profile"],
                    "model_signature": signature,
                    "bank": "layouts",
                    "count": count,
                    "config": self.cfg,
                    "helper_roots": {
                        k: v.tolist() for k, v in self.helper_roots.items()
                    },
                    "shoulder_axes": {k: v.tolist() for k, v in self.axes.items()},
                    "base_position": self.base_pos.tolist(),
                    "base_bounds": self.base_bounds.tolist(),
                    "helper_bounds": [v.tolist() for v in self.helper_bounds],
                    "base_forward_world": [1, 0, 0],
                    "grips": {
                        name: {k: v.tolist() for k, v in grip.items()}
                        for name, grip in self.grips.items()
                    },
                },
                indent=2,
            )
            + "\n"
        )
        return signature, bank
