"""Opt-in bare MuJoCo assembly sandbox, deliberately approximate physics.

Retains the original assembly task and seating reward. No native contact
plugins or penetration/velocity terminations; not a deployment controller.
Detailed source-CAD point tables are an independent observation input.
"""

import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np


def numbers(x):
    return " ".join(format(float(v), ".17g") for v in x)


def build(runtime_xml, review, destination, timestep=0.002, finger_points=None):
    """Single box per former SDF; no robot-self or part-part collision."""
    root = ET.fromstring(runtime_xml)
    for geom in root.iter("geom"):
        if geom.get("type") == "plane":
            geom.set("contype", "1")
            geom.set("conaffinity", "14")
        elif geom.get("type") == "sdf":
            name = geom.get("name")
            visual = review.geom(name.removesuffix("__contact")).id
            mid = int(review.geom_dataid[visual])
            start, count = review.mesh_vertadr[mid], review.mesh_vertnum[mid]
            vertices = review.mesh_vert[start : start + count].astype(float)
            low, high = vertices.min(0), vertices.max(0)
            rotation = np.empty(9)
            mujoco.mju_quat2Mat(rotation, review.geom_quat[visual])
            position = review.geom_pos[visual] + rotation.reshape(3, 3) @ (
                (low + high) / 2
            )
            body = review.body(int(review.geom_bodyid[visual])).name
            robot = name.startswith("physical_helper_")
            finger = robot and ("gripper" in body or "moving_jaw" in body)
            pad = finger and any(
                word in review.mesh(mid).name
                for word in ("wrist_roll_follower", "moving_jaw")
            )
            if finger_points is not None and pad:
                vertices = np.asarray(finger_points[body])
                low, high = vertices.min(0), vertices.max(0)
                position = (low + high) / 2
                quaternion = np.array([1.0, 0.0, 0.0, 0.0])
            else:
                quaternion = review.geom_quat[visual]
            finger = pad
            geom.attrib.clear()
            geom.attrib.update(
                name=name,
                type="box",
                size=numbers(np.maximum((high - low) / 2, 0.0001)),
                pos=numbers(position),
                quat=numbers(quaternion),
                mass="0",
                group="3",
                contype="4" if finger else "8" if robot else "2",
                conaffinity="3" if finger else "1" if robot else "5",
                rgba=".2 .55 .85 1" if robot else ".9 .5 .15 1",
            )
        else:
            # Original visual frames are also used by the unchanged reward.
            geom.set("contype", "0")
            geom.set("conaffinity", "0")
            continue
        for child in list(geom):
            geom.remove(child)
        geom.set("condim", "3")
        geom.set("friction", ".8 .01 .001")
        geom.set("solref", ".02 1")
        geom.set("solimp", ".9 .95 .002")
    for tag in ("extension", "sensor", "custom", "contact"):
        node = root.find(tag)
        if node is not None:
            root.remove(node)
    asset = root.find("asset")
    for mesh in list(asset.findall("mesh")):
        if mesh.find("plugin") is not None:
            asset.remove(mesh)
    option = root.find("option")
    option.attrib.update(
        timestep=str(timestep),
        iterations="10",
        ls_iterations="5",
        tolerance="1e-6",
        noslip_iterations="0",
        jacobian="auto",
        cone="pyramidal",
    )
    for key in ("noslip_tolerance", "sdf_initpoints", "sdf_iterations"):
        option.attrib.pop(key, None)
    # A zero-valued standard user sensor only satisfies the existing interface;
    # it has no callback or collision guard implementation.
    ET.SubElement(
        ET.SubElement(root, "sensor"),
        "user",
        name="training_min_contact_distance",
        dim="1",
        needstage="pos",
    )
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(ET.tostring(root, encoding="unicode"))
    model = mujoco.MjModel.from_xml_path(str(path))
    assert model.nplugin == 0 and model.nsensor == 1 and model.npair == 0
    assert not np.any(model.geom_type == mujoco.mjtGeom.mjGEOM_SDF)
    assert model.nu == 12
    receipt = {
        "scene_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "source_sha256": hashlib.sha256(runtime_xml.encode()).hexdigest(),
        "ngeom": model.ngeom,
        "nbody": model.nbody,
        "nq": model.nq,
        "nv": model.nv,
        "nu": model.nu,
        "nplugin": model.nplugin,
        "timestep": timestep,
        "solver_iterations": 10,
        "penetration_termination": False,
        "velocity_termination": False,
        "inter_part_collisions": False,
        "robot_self_collisions": False,
        "finger_pad_boxes": finger_points is not None,
        "task": "original five assembly tasks and original sparse stable-seating reward",
    }
    path.with_suffix(".json").write_text(json.dumps(receipt, indent=2) + "\n")
    return model, receipt


def environment_class(native_module, model, profile, signature):
    """Reuse original task/reset/reward/actions; omit only physical guards.

    Exact source anchors fail closed on incompatible trainer revisions. No
    installed trainer files or global native hooks are modified.
    """
    import inspect

    source = inspect.getsource(native_module.BatchedAssemblyEnv)
    first = source.index("        safety = unstable | (penetration >")
    last = source.index("        success = np.zeros", first)
    source = source[:first] + "        safety = unstable.copy()\n" + source[last:]
    first = source.index("            safety[i] |= bool(")
    last = source.index("            result = self.reward.snapshot", first)
    source = source[:first] + source[last:]
    namespace = dict(vars(native_module))
    namespace["load_runtime"] = lambda _: (model, profile, signature)
    exec(compile(source, "<bare-assembly-original-task>", "exec"), namespace)  # noqa: S102 - trusted installed class, two explicit guard edits
    cls = namespace["BatchedAssemblyEnv"]
    cls.modified_source = source
    cls.original_environment_sha256 = hashlib.sha256(
        inspect.getsource(native_module.BatchedAssemblyEnv).encode()
    ).hexdigest()
    cls.modified_environment_sha256 = hashlib.sha256(source.encode()).hexdigest()
    return cls
