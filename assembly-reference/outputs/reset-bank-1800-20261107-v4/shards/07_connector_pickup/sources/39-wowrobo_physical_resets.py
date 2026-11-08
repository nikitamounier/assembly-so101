"""Native gravity/contact reset admission with no artificial holding constraints.

Profiles preserve the approved body/joint/inertia layout and use source-triangle
SDFs for BOTH target and helper meshes. A passing geometric pose is not a reward
or loaded-grasp certificate. Admission is explicit and fails closed.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import mujoco
import numpy as np
import trimesh

from assembly_policy.so101.wowrobo_boundaries import NativeBoundaryAudit
from assembly_policy.so101.wowrobo_contact_backend import (
    load_backend,
    prepare_scene,
    set_surface_mode,
)
from assembly_policy.so101.wowrobo_friction import (
    apply_contact_friction,
    apply_friction_and_rebuild_constraints,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GRID = ROOT / "outputs/linear-insertion-updated-20261107/grids/manifest.json"
PARTS = (
    "base_servo",
    "base_side_cover",
    "shoulder_connector",
    "assembly_pin",
    "upper_arm_module",
    "forearm_module",
)


def physical_split_step(model, data, multipliers):
    """Apply per-data friction before force solving, retaining implicit damping.

    MuJoCo mj_step2 dispatches implicit/implicitfast to mj_implicit. Rebuilding
    contact constraints here follows mj_fwdPosition's ordering, then restores
    velocity references before acceleration/force solving in mj_step2.
    """
    mujoco.mj_step1(model, data)
    if model.opt.integrator == mujoco.mjtIntegrator.mjINT_EULER:
        apply_friction_and_rebuild_constraints(model, data, multipliers)
    elif model.opt.integrator in (
        mujoco.mjtIntegrator.mjINT_IMPLICIT,
        mujoco.mjtIntegrator.mjINT_IMPLICITFAST,
    ):
        apply_contact_friction(model, data, multipliers)
        mujoco.mj_makeConstraint(model, data)
        mujoco.mj_island(model, data)
        mujoco.mj_projectConstraint(model, data)
        mujoco.mj_transmission(model, data)
        mujoco.mj_fwdVelocity(model, data)
    else:
        raise ValueError(
            "Physical friction stepping requires Euler or implicit integrator"
        )
    mujoco.mj_step2(model, data)


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _grid_builder():
    spec = importlib.util.spec_from_file_location(
        "wowrobo_grid_builder", ROOT / "scripts/build_wowrobo_contact_grids.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build_one


def _build_grid(spec):
    return _grid_builder()(spec)


ASSEMBLY_ATTACHMENTS = (
    ("installed_cover_to_base", "base_servo", "base_side_cover", "06_cover_seated", 7),
    (
        "installed_connector_to_output_horns",
        "target_servo_1_pivot",
        "shoulder_connector",
        "11_connector_seated",
        12,
    ),
    (
        "installed_servo2_case_to_connector",
        "shoulder_connector",
        "target_servo_2_pivot",
        "24_upper_seated",
        25,
    ),
)

PASSIVE_BRACKET_STOPS = (
    ("servo_2_case__contact", "upper_arm_module_printed_0__contact"),
    ("servo_3_case__contact", "forearm_module_printed_0__contact"),
    ("servo_3_case__contact", "forearm_module_printed_1__contact"),
)
PASSIVE_BRACKET_STOP_LAW = {
    "condim": "4",
    "friction": ".7 .7 .005 .0001 .0001",
    "solref": ".002 1",
    "solimp": ".95 .99 .0002",
}


INTERNAL_SERVO_BODY_PAIRS = (
    ("base_servo", "target_servo_1_pivot"),
    ("upper_arm_module", "target_servo_2_pivot"),
    ("forearm_module", "target_servo_3_pivot"),
    ("forearm_module", "target_servo_4_pivot"),
    ("left_base", "left_shoulder"),
    ("right_base", "right_shoulder"),
)


def exclude_internal_servo_bearings(root):
    """Exclude manufactured servo bearings, retaining explicit real stops.

    Native parent filtering normally omits these joint-internal interfaces, but
    world-fixed target/helper bases are exceptions. Explicit bracket/motor stop pairs are
    intentionally unaffected: MuJoCo evaluates explicit pairs before exclusions.
    """
    bodies = {b.get("name") for b in root.iter("body")}
    contact = root.find("contact")
    if contact is None:
        contact = ET.SubElement(root, "contact")
    existing = {
        frozenset((p.get("body1"), p.get("body2"))) for p in contact.findall("exclude")
    }
    exclusions = []
    for parent, pivot in INTERNAL_SERVO_BODY_PAIRS:
        if not {parent, pivot} <= bodies:
            raise ValueError(f"Missing internal servo bearing body: {parent}, {pivot}")
        attrs = {"body1": parent, "body2": pivot}
        if frozenset((parent, pivot)) not in existing:
            ET.SubElement(contact, "exclude", attrs)
        exclusions.append(attrs)
    return {
        "profile": "manufactured_servo_bearings_v2",
        "exclusions": exclusions,
        "explicit_bracket_stops_preserved": True,
        "reason": "Motor horn/case bearing is a modeled hinge, not a colliding assembly interface",
    }


def _passive_stop_pairs():
    return [
        {
            "name": f"passive_motor_stop_{index}",
            "geom1": motor,
            "geom2": bracket,
            **PASSIVE_BRACKET_STOP_LAW,
        }
        for index, (motor, bracket) in enumerate(PASSIVE_BRACKET_STOPS)
    ]


def route_source_mesh_table_contacts(root):
    """Use native source-vertex support manifolds only against the flat table.

    Plane support of a mesh and its convex hull is identical. Native mesh/plane
    manifolds retain all extremal source supports, unlike the one-point SDF/plane
    route. No hull is used for target/helper/mating contacts and no shape is added.
    """
    geoms = list(root.iter("geom"))
    contacts = {
        g.get("name").removesuffix("__contact"): g
        for g in geoms
        if g.get("name", "").endswith("__contact")
    }
    source_names, table_names = [], []
    for geom in geoms:
        name = geom.get("name", "")
        if name in contacts and geom.get("type") == "mesh":
            geom.set("contype", "16")
            geom.set("conaffinity", "4")
            source_names.append(name)
        elif name.endswith("__contact"):
            geom.set("conaffinity", str(int(geom.get("conaffinity", "15")) & ~4))
        elif geom.get("type") == "plane":
            geom.set("contype", "4")
            geom.set("conaffinity", "16")
            table_names.append(name)
    if not source_names or not table_names:
        raise ValueError("Source-mesh table routing requires mesh sources and a plane")
    return {
        "profile": "source_vertices_plane_manifold_v1",
        "source_mesh_geoms": source_names,
        "table_geoms": table_names,
        "sdf_table_contacts_disabled": True,
        "floor_only_collision_bit": 16,
        "source_shape_unchanged": True,
        "contact_law_unchanged": True,
        "geometry_scope": "Native mesh/plane support only; all target/helper/mating contacts retain source SDFs",
    }


def check_table_routing_preserves_mechanics(before, after):
    """Prove routing changes only contact masks, not state layout or geometry."""
    arrays = (
        "body_parentid",
        "body_mass",
        "body_inertia",
        "body_ipos",
        "body_iquat",
        "body_pos",
        "body_quat",
        "jnt_type",
        "jnt_bodyid",
        "jnt_qposadr",
        "jnt_dofadr",
        "jnt_pos",
        "jnt_axis",
        "jnt_range",
        "jnt_stiffness",
        "dof_damping",
        "dof_frictionloss",
        "dof_armature",
        "geom_type",
        "geom_dataid",
        "geom_bodyid",
        "geom_pos",
        "geom_quat",
        "geom_size",
        "geom_friction",
        "geom_solref",
        "geom_solimp",
        "geom_condim",
        "geom_priority",
        "mesh_vert",
        "mesh_face",
        "actuator_trnid",
        "actuator_gainprm",
        "actuator_biasprm",
        "actuator_forcerange",
        "eq_data",
    )
    if any(
        getattr(before, key) != getattr(after, key)
        for key in ("nq", "nv", "nu", "neq", "ngeom", "npair")
    ):
        raise ValueError(
            "Source-mesh floor routing changed state or contact-pair layout"
        )
    for name in arrays:
        if not np.array_equal(getattr(before, name), getattr(after, name)):
            raise ValueError(f"Source-mesh floor routing changed mechanics: {name}")
    return {"state_layout_unchanged": True, "exact_native_arrays_checked": list(arrays)}


def add_passive_bracket_contacts(root):
    """Restore real bracket/motor stops omitted by native parent filtering.

    The contact law matches the force-held passive final-module diagnostic.
    No hinge angle, spring, actuation or equality is added. These are explicit
    source-SDF pairs because the motor case and printed link are parent/child.
    """
    geoms = {g.get("name") for g in root.iter("geom")}
    contact = root.find("contact")
    if contact is None:
        contact = ET.SubElement(root, "contact")
    existing = {
        frozenset((p.get("geom1"), p.get("geom2"))) for p in contact.findall("pair")
    }
    pairs = []
    for attrs in _passive_stop_pairs():
        motor, bracket = attrs["geom1"], attrs["geom2"]
        if not {motor, bracket} <= geoms:
            raise ValueError(f"Missing passive motor-stop geometry: {motor}, {bracket}")
        if frozenset((motor, bracket)) in existing:
            raise ValueError("Passive motor-stop pair already has a contact law")
        ET.SubElement(contact, "pair", attrs)
        pairs.append(attrs)
    return {
        "profile": "source_bracket_motor_stops_v1",
        "pairs": pairs,
        "hinge_angle_target": False,
        "qualification": "Source-contact candidate; equilibrium and grasp admission remain required",
    }


def check_passive_target_shafts(model):
    """Fail closed if an assembly shaft is locked or actively angle-held."""
    shafts = []
    for servo in range(1, 5):
        joint = model.joint(f"target_servo_{servo}_shaft")
        j = joint.id
        if joint.type != mujoco.mjtJoint.mjJNT_HINGE or model.jnt_stiffness[j] != 0:
            raise ValueError(
                "Assembly target shafts must remain passive spring-free hinges"
            )
        joint_transmissions = np.isin(
            model.actuator_trntype,
            (mujoco.mjtTrn.mjTRN_JOINT, mujoco.mjtTrn.mjTRN_JOINTINPARENT),
        )
        if np.any(joint_transmissions & (model.actuator_trnid[:, 0] == j)):
            raise ValueError("Assembly target shaft has an angle actuator")
        body = int(model.jnt_bodyid[j])
        parent = int(model.body_parentid[body])
        for e in range(model.neq):
            if model.eq_type[e] == mujoco.mjtEq.mjEQ_JOINT and j in (
                model.eq_obj1id[e],
                model.eq_obj2id[e],
            ):
                raise ValueError("Assembly target shaft has a joint equality")
            if model.eq_type[e] == mujoco.mjtEq.mjEQ_WELD and {
                int(model.eq_obj1id[e]),
                int(model.eq_obj2id[e]),
            } == {body, parent}:
                raise ValueError(
                    "Assembly target shaft is welded across its articulation"
                )
        shafts.append(joint.name)
    return {"shafts": shafts, "passive": True, "angle_targets": False}


def _anchor_folder(nominal_bank, completed_anchor_bank=None):
    if completed_anchor_bank is not None:
        folder = Path(completed_anchor_bank).resolve()
    elif (Path(nominal_bank) / "completed").is_dir():
        folder = Path(nominal_bank) / "completed"
    else:
        folder = Path(nominal_bank).parent / "photo-resets-v2" / "stages"
    return folder.resolve()


def _target_grid_matches_asset(record, asset_path, scale):
    """Require the field source and visual source to be the same raw geometry."""
    if not np.array_equal(scale, np.ones(3)):
        raise ValueError("Reused target grids require unscaled source mesh assets")
    expected = record.get("source_sha256")
    if expected is None:
        raise ValueError("Reused target grid lacks source geometry identity")
    if _hash(record["source_file"]) != expected or _hash(asset_path) != expected:
        raise ValueError(f"Target grid does not match visual source mesh: {asset_path}")
    if (
        record.get("grid_sha256")
        and _hash(record["grid_file"]) != record["grid_sha256"]
    ):
        raise ValueError("Target grid file identity changed")


def _apply_all_mating_pairs(root):
    """Explicit union of the accepted per-step candidate laws, no new softening."""
    from assembly_policy.so101.wowrobo_mating_profile import CONFIG

    config = json.loads(CONFIG.read_text())
    contact = root.find("contact")
    if contact is None:
        contact = ET.SubElement(root, "contact")
    geoms = {g.get("name") for g in root.iter("geom")}
    mappings, seen = [], {}
    for step, rows in config["steps"].items():
        for index, row in enumerate(rows):
            pair = tuple(sorted((row["geom1"], row["geom2"])))
            law = {
                k: v
                for k, v in config["classes"][row["contact_class"]].items()
                if k != "source_pair"
            }
            if pair in seen and seen[pair] != law:
                raise ValueError(f"Conflicting per-step laws for {pair}")
            if pair in seen:
                continue
            if not set(pair) <= geoms:
                raise ValueError(f"Missing mating contact geometry for {pair}")
            seen[pair] = law
            attrs = {
                "name": f"wowfit_{step}_{index}",
                "geom1": row["geom1"],
                "geom2": row["geom2"],
            }
            attrs.update(
                {
                    k: " ".join(format(float(x), ".17g") for x in v)
                    if isinstance(v, list)
                    else str(v)
                    for k, v in law.items()
                }
            )
            ET.SubElement(contact, "pair", attrs)
            mappings.append({"step": step, **row, "law": law})
    return {
        "profile": "legacy_mapped_all",
        "config_path": str(CONFIG),
        "config_sha256": _hash(CONFIG),
        "pairs": mappings,
        "qualification": "Uncalibrated per-step contact candidate; not automatic reset admission",
    }


def add_assembly_attachments(root, nominal_bank, completed_anchor_bank=None):
    """Completed hardware interfaces, never gripper/fixture welds."""
    review = mujoco.MjModel.from_xml_path(str(Path(nominal_bank) / "scene.xml"))
    equality = root.find("equality")
    if equality is None:
        equality = ET.SubElement(root, "equality")
    folder = _anchor_folder(nominal_bank, completed_anchor_bank)
    for name, parent, child, anchor, _ in ASSEMBLY_ATTACHMENTS:
        d = mujoco.MjData(review)
        with np.load(folder / (anchor + "_train.npz")) as f:
            mujoco.mj_setState(
                review, d, f["states"][0], mujoco.mjtState.mjSTATE_FULLPHYSICS
            )
        mujoco.mj_kinematics(review, d)
        a, b = review.body(parent).id, review.body(child).id
        r = d.xmat[a].reshape(3, 3)
        position = r.T @ (d.xpos[b] - d.xpos[a])
        rotation = r.T @ d.xmat[b].reshape(3, 3)
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, rotation.ravel())
        ET.SubElement(
            equality,
            "weld",
            name=name,
            body1=parent,
            body2=child,
            active="false",
            relpose=" ".join(map(str, np.r_[position, quat])),
            torquescale=".05",
            solref=".002 1",
            solimp=".95 .99 .0002",
        )


def build_physical_profile(
    nominal_bank,
    output_dir,
    grid_manifest=DEFAULT_GRID,
    workers=4,
    pin_mesh=None,
    pin_registration_path=None,
    completed_anchor_bank=None,
    mating_profile=None,
    pin_pose_registered=False,
    passive_motor_stops=False,
    helper_grid_profile=None,
    exclude_internal_servo_contacts=False,
    source_mesh_table_contacts=False,
    center_motor_housings=False,
):
    """Create an isolated all-source-SDF profile, reusing immutable target grids."""
    nominal_bank, output_dir = Path(nominal_bank).resolve(), Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    source = nominal_bank / "scene.xml"
    root = ET.parse(source).getroot()
    housing_registration = None
    if center_motor_housings:
        from .wowrobo_housing_registration import apply_housing_centering

        housing_registration = apply_housing_centering(root)
    original = json.loads(Path(grid_manifest).read_text())
    if mating_profile not in (None, "legacy_mapped_all"):
        raise ValueError("Unknown full-assembly mating profile")
    if pin_registration_path is None and pin_mesh is not None:
        candidate = Path(pin_mesh).resolve().parent / "registration.json"
        pin_registration_path = candidate if candidate.exists() else None
    if pin_registration_path is not None:
        pin_registration_path = Path(pin_registration_path).resolve()
    anchor_folder = _anchor_folder(nominal_bank, completed_anchor_bank)
    anchor_hashes = {
        anchor: _hash(anchor_folder / (anchor + "_train.npz"))
        for anchor in (
            "06_cover_seated",
            "11_connector_seated",
            "16_pin_seated",
            "24_upper_seated",
            "32_final_seated",
        )
    }
    if pin_mesh is not None:
        root.find("asset/mesh[@name='pin']").set("file", str(Path(pin_mesh).resolve()))
    if original.get("status") != "complete":
        raise ValueError("Target grids are incomplete")
    cache = output_dir / "grids"
    cache.mkdir(exist_ok=True)
    meshes = {}
    assets = {node.get("name"): node for node in root.find("asset").findall("mesh")}
    visual_meshes = {
        node.get("name"): node.get("mesh")
        for node in root.iter("geom")
        if node.get("mesh")
    }
    for name, record in original["meshes"].items():
        asset = assets[visual_meshes[name]]
        asset_path = Path(asset.get("file"))
        if not asset_path.is_absolute():
            asset_path = source.parent / asset_path
        scale = np.fromstring(asset.get("scale", "1 1 1"), sep=" ")
        if (
            name == "pin"
            and pin_mesh is not None
            and _hash(asset_path) != record.get("source_sha256")
        ):
            continue
        _target_grid_matches_asset(record, asset_path, scale)
        record = dict(record)
        gid = record.get("grid_id", record.get("id"))
        old = Path(record["grid_file"])
        dst = cache / f"grid_{gid}.bin"
        if not dst.exists():
            dst.symlink_to(old)
        record["grid_file"] = str(dst)
        meshes[name] = record
    assets = {node.get("name"): node for node in root.find("asset").findall("mesh")}
    asset_sources = {
        name: (
            Path(node.get("file")).resolve(),
            np.fromstring(node.get("scale", "1 1 1"), sep=" "),
        )
        for name, node in assets.items()
        if node.get("file")
    }
    specs, dedup = [], {}
    next_id = (
        max(int(r.get("grid_id", r.get("id"))) for r in original["meshes"].values()) + 1
    )
    for index, geom in enumerate(root.iter("geom")):
        mesh_name = geom.get("mesh")
        if not mesh_name or mesh_name in meshes:
            continue
        asset = assets[mesh_name]
        if asset.get("file") is None:
            continue
        name = "pin" if mesh_name == "pin" else f"physical_helper_{index}"
        geom.set("name", name)
        original_file, scale = asset_sources[mesh_name]
        fingerprint = hashlib.sha256(
            original_file.read_bytes() + scale.tobytes()
        ).hexdigest()
        if fingerprint not in dedup:
            source_mesh = output_dir / (fingerprint[:16] + ".obj")
            if not source_mesh.exists():
                tri = trimesh.load_mesh(original_file, process=False)
                tri.vertices *= scale
                tri.export(source_mesh)
            # Remove scale only after baking it into this source OBJ.
            # All uses of one asset share its scale and replaced file.
            spacing = (
                0.0001
                if mesh_name == "pin"
                else 0.0002
                if ("jaw" in mesh_name or "gripper" in mesh_name)
                else 0.0004
            )
            spec = {
                "name": name,
                "id": next_id,
                "grid_id": next_id,
                "spacing": spacing,
                "source_file": str(source_mesh),
                "grid_file": str(cache / f"grid_{next_id}.bin"),
            }
            meta = cache / f"grid_{next_id}.json"
            reuse_root = (
                Path(helper_grid_profile).resolve()
                if helper_grid_profile is not None
                else output_dir.parent / "physical-resets-v1"
            )
            reuse = reuse_root / "grids" / meta.name
            if not meta.exists() and reuse.exists():
                old_record = json.loads(reuse.read_text())
                if (
                    Path(old_record["source_file"]).name == source_mesh.name
                    and old_record["spacing"] == spacing
                    and _hash(old_record["source_file"]) == _hash(source_mesh)
                    and (
                        not old_record.get("grid_sha256")
                        or _hash(old_record["grid_file"]) == old_record["grid_sha256"]
                    )
                ):
                    old_grid = Path(old_record["grid_file"])
                    new_grid = cache / f"grid_{next_id}.bin"
                    if not new_grid.exists():
                        new_grid.symlink_to(old_grid)
                    old_record["grid_file"] = str(new_grid)
                    meta.write_text(json.dumps(old_record, indent=2) + "\n")
            if meta.exists():
                cached_spec = json.loads(meta.read_text())
                if cached_spec["spacing"] != spacing or _hash(
                    cached_spec["source_file"]
                ) != _hash(source_mesh):
                    raise ValueError(
                        "Cached source field belongs to another mesh or spacing; use a new profile directory"
                    )
                spec = cached_spec
            else:
                specs.append(spec)
            dedup[fingerprint] = spec
            next_id += 1
        record = dict(dedup[fingerprint])
        record["name"] = name
        record["source_mesh_name"] = mesh_name
        meshes[name] = record
        asset.set("file", record["source_file"])
        asset.attrib.pop("scale", None)
    if specs:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for record in pool.map(_build_grid, specs):
                (cache / f"grid_{record['grid_id']}.json").write_text(
                    json.dumps(record, indent=2) + "\n"
                )
                for name, existing in list(meshes.items()):
                    if existing["grid_id"] == record["grid_id"]:
                        meshes[name] = dict(
                            record,
                            name=name,
                            source_mesh_name=existing["source_mesh_name"],
                        )
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "grid_directory": str(cache),
        "source_scene_sha256": _hash(source),
        "meshes": meshes,
        "scope": "Source-triangle targets and helpers; no convex helper approximation",
    }
    manifest_path = cache / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    source_path = output_dir / "source.xml"
    ET.ElementTree(root).write(source_path, encoding="unicode")
    physical_path = output_dir / "physical.xml"
    prepare_scene(source_path, manifest_path, physical_path)
    reframe = None
    if any(row.get("requires_sdf_geom_reframe") for row in meshes.values()):
        from assembly_policy.so101.wowrobo_case_grid_reuse import reframe_scene

        unframed = output_dir / "physical-unframed.xml"
        unframed.write_bytes(physical_path.read_bytes())
        reframe = reframe_scene(unframed, manifest_path, physical_path)
    root = ET.parse(physical_path).getroot()
    option = root.find("option")
    option.set("gravity", "0 0 -9.81")
    option.set("integrator", "implicitfast")
    option.set("timestep", ".00025")
    # Contact categories: target1, helper2, table4. Original articulated links
    # and free bodies remain dynamic; no fixture equalities are introduced.
    for geom in root.iter("geom"):
        name = geom.get("name", "")
        if name.endswith("__contact"):
            helper = name.startswith("physical_helper_")
            source_name = meshes.get(name.removesuffix("__contact"), {}).get(
                "source_mesh_name", ""
            )
            right = source_name.startswith("right")
            geom.set("mass", "0")
            geom.set("contype", ("8" if right else "2") if helper else "1")
            geom.set("conaffinity", "15")
        elif geom.get("type") == "plane" or name in ("review_table", "table", "floor"):
            geom.set("contype", "4")
            geom.set("conaffinity", "11")
            geom.set("priority", "1")
            geom.set("condim", "4")
            geom.set("solref", ".0005 1")
            geom.set("solimp", ".95 .99 .0002")
    mating = _apply_all_mating_pairs(root) if mating_profile else None
    stops = add_passive_bracket_contacts(root) if passive_motor_stops else None
    bearing_exclusions = (
        exclude_internal_servo_bearings(root)
        if exclude_internal_servo_contacts
        else None
    )
    add_assembly_attachments(root, nominal_bank, anchor_folder)
    table_routing, before_table_model = None, None
    if source_mesh_table_contacts:
        before_table_model = mujoco.MjModel.from_xml_string(
            ET.tostring(root, encoding="unicode")
        )
        table_routing = route_source_mesh_table_contacts(root)
    ET.ElementTree(root).write(physical_path, encoding="unicode")
    compiled_model = (
        mujoco.MjModel.from_xml_path(str(physical_path))
        if passive_motor_stops or source_mesh_table_contacts
        else None
    )
    shaft_audit = (
        check_passive_target_shafts(compiled_model) if passive_motor_stops else None
    )
    if table_routing is not None:
        table_routing["native_mechanics_audit"] = (
            check_table_routing_preserves_mechanics(before_table_model, compiled_model)
        )
    result = {
        "model_path": str(physical_path),
        "model_sha256": _hash(physical_path),
        "source_model_sha256": _hash(source),
        "audit_model_path": str(source_path),
        "pin_mesh_sha256": _hash(pin_mesh) if pin_mesh is not None else None,
        "pin_mesh_path": str(Path(pin_mesh).resolve())
        if pin_mesh is not None
        else None,
        "pin_registration_path": str(pin_registration_path)
        if pin_registration_path
        else None,
        "pin_pose_registered": bool(pin_pose_registered),
        "pin_registration_sha256": _hash(pin_registration_path)
        if pin_registration_path
        else None,
        "completed_anchor_bank": str(anchor_folder),
        "completed_anchor_sha256": anchor_hashes,
        "mating_profile": mating,
        "passive_motor_stops": stops,
        "internal_servo_bearing_exclusions": bearing_exclusions,
        "source_mesh_table_contacts": table_routing,
        "housing_registration": housing_registration,
        "passive_target_shafts": shaft_audit,
        "helper_grid_profile": str(Path(helper_grid_profile).resolve())
        if helper_grid_profile is not None
        else None,
        "case_field_reframe": reframe,
        "input_grid_manifest_sha256": _hash(grid_manifest),
        "grid_manifest": str(manifest_path),
        "contact_backend": "accelerated native source-triangle SDF/surface tree",
        "gravity": [0, 0, -9.81],
        "integrator": "implicitfast",
        "timestep": 0.00025,
        "table_contact_solref": [0.0005, 1.0],
        "table_contact_priority": 1,
        "helper_geometry": "source SDF",
        "artificial_fixture_constraints_added": False,
        "assembly_attachments": [
            {"name": n, "parent": a, "child": b, "first_stage": i}
            for n, a, b, _, i in ASSEMBLY_ATTACHMENTS
        ],
        "helper_collisions": "target/table/crossarm/nonadjacent_self with native same-body/weld/parent filters",
        "training_eligible": False,
    }
    (output_dir / "profile.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def check_source_sdf_frames(model, data, profile, *, samples=96, include_targets=False):
    """Independent compiled world->native query vs raw grids and source triangles."""
    import ctypes

    from scipy.spatial.transform import Rotation

    from assembly_policy.so101.wowrobo_contact_backend import surface_library

    spec = importlib.util.spec_from_file_location(
        "wowrobo_grid_checker", ROOT / "scripts/check_wowrobo_contact_backend.py"
    )
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    query = surface_library().wowrobo_sdf_distance
    ptr = ctypes.POINTER(ctypes.c_double)
    query.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ptr]
    query.restype = ctypes.c_double
    manifest = json.loads(Path(profile["grid_manifest"]).read_text())
    report = {}
    for name, record in manifest["meshes"].items():
        if not include_targets and not name.startswith("physical_helper_"):
            continue
        mesh = trimesh.load_mesh(record["source_file"], process=False)
        points, faces = trimesh.sample.sample_surface(mesh, samples, seed=731)
        points = np.vstack(
            [points, points + mesh.face_normals[faces] * record["spacing"] * 0.5]
        )
        visual = model.geom(name).id
        contact = model.geom(name + "__contact").id
        mid = model.geom_dataid[visual]
        rotation = Rotation.from_quat(
            model.mesh_quat[mid], scalar_first=True
        ).as_matrix()
        principal = (points - model.mesh_pos[mid]) @ rotation
        world = (
            principal @ data.geom_xmat[visual].reshape(3, 3).T + data.geom_xpos[visual]
        )
        actual = np.array(
            [
                query(
                    model._address,
                    data._address,
                    contact,
                    np.ascontiguousarray(p).ctypes.data_as(ptr),
                )
                for p in world
            ]
        )
        field_points = points
        if record.get("requires_sdf_geom_reframe"):
            field_to_source = np.asarray(record["field_to_source"], dtype=float)
            source_to_field = np.linalg.inv(field_to_source)
            field_points = points @ source_to_field[:3, :3].T + source_to_field[:3, 3]
        expected, _ = helper.grid_values(record["grid_file"], field_points)
        _, distance, _ = trimesh.proximity.closest_point(mesh, points)
        unsigned_error = np.abs(np.abs(actual) - distance)
        frame_error = float(np.max(np.abs(expected - actual)))
        report[name] = {
            "samples": len(points),
            "source_mesh": record.get("source_mesh_name"),
            "spacing_m": record["spacing"],
            "max_frame_error_m": frame_error,
            "triangle_unsigned_max_error_m": float(unsigned_error.max()),
            "triangle_unsigned_p99_error_m": float(np.quantile(unsigned_error, 0.99)),
            "passed": bool(
                frame_error < 1e-9
                and unsigned_error.max() < np.sqrt(3) * record["spacing"]
            ),
        }
    return report


def _ancestor(model, body, target):
    while body:
        if body == target:
            return True
        body = int(model.body_parentid[body])
    return False


class WowRoboPhysicalValidator:
    def __init__(
        self,
        nominal_bank=None,
        output_dir=None,
        grid_manifest=None,
        pin_mesh=None,
        reward_checker=None,
        settle_seconds=0.15,
        hold_seconds=0.15,
        pin_registration_path=None,
        completed_anchor_bank=None,
        mating_profile=None,
        pin_pose_registered=False,
        passive_motor_stops=False,
        helper_grid_profile=None,
        exclude_internal_servo_contacts=False,
        source_mesh_table_contacts=False,
        center_motor_housings=False,
    ):
        self.nominal_bank = Path(
            nominal_bank or ROOT / "assets_local/wowrobo/photo-resets-v3"
        )
        output_dir = Path(
            output_dir or ROOT / "assets_local/wowrobo/physical-resets-v1"
        )
        profile_file = output_dir / "profile.json"
        if profile_file.exists():
            self.profile = json.loads(profile_file.read_text())
            if self.profile["source_model_sha256"] != _hash(
                self.nominal_bank / "scene.xml"
            ):
                raise ValueError("Physical profile belongs to another nominal model")
            if self.profile["model_sha256"] != _hash(self.profile["model_path"]):
                raise ValueError("Physical profile model identity changed")
            if self.profile.get("pin_mesh_sha256") != (
                _hash(pin_mesh) if pin_mesh is not None else None
            ):
                raise ValueError(
                    "Physical profile pin geometry changed; use a new output directory"
                )
            if self.profile.get("input_grid_manifest_sha256") != _hash(
                grid_manifest or DEFAULT_GRID
            ):
                raise ValueError("Physical profile target grid identity changed")
            if pin_registration_path is not None and self.profile.get(
                "pin_registration_sha256"
            ) != _hash(pin_registration_path):
                raise ValueError("Physical profile pin registration changed")
            if completed_anchor_bank is not None and self.profile.get(
                "completed_anchor_bank"
            ) != str(Path(completed_anchor_bank).resolve()):
                raise ValueError("Physical profile completed anchors changed")
            if bool(self.profile.get("pin_pose_registered", False)) != bool(
                pin_pose_registered
            ):
                raise ValueError(
                    "Physical profile registered pin pose selection changed"
                )
            stored_registration = self.profile.get("pin_registration_path")
            stored_registration_hash = self.profile.get("pin_registration_sha256")
            if (
                stored_registration_hash
                and _hash(stored_registration) != stored_registration_hash
            ):
                raise ValueError(
                    "Physical profile saved pin registration identity changed"
                )
            saved_law = self.profile.get("mating_profile")
            if (
                saved_law
                and _hash(saved_law["config_path"]) != saved_law["config_sha256"]
            ):
                raise ValueError("Physical profile mating law configuration changed")
            requested_mating = mating_profile
            saved_mating = (self.profile.get("mating_profile") or {}).get("profile")
            if saved_mating != requested_mating:
                raise ValueError("Physical profile mating law selection changed")
            if bool(self.profile.get("passive_motor_stops")) != bool(
                passive_motor_stops
            ):
                raise ValueError(
                    "Physical profile passive motor-stop selection changed"
                )
            if (
                passive_motor_stops
                and self.profile["passive_motor_stops"].get("pairs")
                != _passive_stop_pairs()
            ):
                raise ValueError("Physical profile passive motor-stop law changed")
            saved_exclusions = self.profile.get("internal_servo_bearing_exclusions")
            saved_table = self.profile.get("source_mesh_table_contacts")
            if bool(self.profile.get("housing_registration")) != bool(
                center_motor_housings
            ):
                raise ValueError(
                    "Physical profile housing registration changed; use a new directory"
                )
            if bool(saved_table) != bool(source_mesh_table_contacts):
                raise ValueError("Physical profile source-mesh table selection changed")
            if (
                source_mesh_table_contacts
                and saved_table.get("profile") != "source_vertices_plane_manifold_v1"
            ):
                raise ValueError("Physical profile source-mesh table routing changed")
            if bool(saved_exclusions) != bool(exclude_internal_servo_contacts):
                raise ValueError(
                    "Physical profile internal servo bearing selection changed"
                )
            if exclude_internal_servo_contacts and saved_exclusions.get(
                "exclusions"
            ) != [{"body1": a, "body2": b} for a, b in INTERNAL_SERVO_BODY_PAIRS]:
                raise ValueError(
                    "Physical profile internal servo bearing exclusions changed"
                )
            for anchor, digest in self.profile.get(
                "completed_anchor_sha256", {}
            ).items():
                if (
                    _hash(
                        Path(self.profile["completed_anchor_bank"])
                        / (anchor + "_train.npz")
                    )
                    != digest
                ):
                    raise ValueError(
                        "Physical profile completed anchor identity changed"
                    )
            load_backend(self.profile["grid_manifest"])
        else:
            self.profile = build_physical_profile(
                self.nominal_bank,
                output_dir,
                grid_manifest or DEFAULT_GRID,
                pin_mesh=pin_mesh,
                pin_registration_path=pin_registration_path,
                completed_anchor_bank=completed_anchor_bank,
                mating_profile=mating_profile,
                pin_pose_registered=pin_pose_registered,
                passive_motor_stops=passive_motor_stops,
                helper_grid_profile=helper_grid_profile,
                exclude_internal_servo_contacts=exclude_internal_servo_contacts,
                source_mesh_table_contacts=source_mesh_table_contacts,
                center_motor_housings=center_motor_housings,
            )
        set_surface_mode(1)
        self.model = mujoco.MjModel.from_xml_string(
            Path(self.profile["model_path"]).read_text()
        )
        if passive_motor_stops:
            check_passive_target_shafts(self.model)
        self.scene_path = Path(self.profile["model_path"])
        self.nominal_model = mujoco.MjModel.from_xml_string(
            Path(
                self.profile.get("audit_model_path", self.nominal_bank / "scene.xml")
            ).read_text()
        )
        self.audit = NativeBoundaryAudit(self.nominal_model)
        receipt_path = self.nominal_bank / "stage-receipts.json"
        self.support_grasp_by_stage = (
            {
                row["id"]: row.get("support_grasp_by_side", {})
                for row in json.loads(receipt_path.read_text())
            }
            if receipt_path.exists()
            else {}
        )
        self.definitions = {
            d["id"]: d
            for d in json.loads(
                (self.nominal_bank / "reset-contract.json").read_text()
            )["states"]
        }
        self.reward_checker = reward_checker
        self.pin_registration = None
        registration_path = self.profile.get("pin_registration_path")
        if registration_path:
            self.pin_registration = json.loads(Path(registration_path).read_text())
            self.profile["pin_registration_sha256"] = _hash(registration_path)
        self._load_completion_anchors()
        self.grid_records = json.loads(Path(self.profile["grid_manifest"]).read_text())[
            "meshes"
        ]
        self.settle_seconds = settle_seconds
        self.hold_seconds = hold_seconds
        if (
            self.model.nq != self.nominal_model.nq
            or self.model.nv != self.nominal_model.nv
        ):
            raise ValueError("Physical profile changed state layout")
        self.contact_geoms = {
            g
            for g in range(self.model.ngeom)
            if (self.model.geom(g).name or "").endswith("__contact")
        }

    def _native_audit(self, data):
        native = mujoco.MjData(self.nominal_model)
        native.qpos[:] = data.qpos
        mujoco.mj_forward(self.nominal_model, native)
        result = self.audit.audit(native)
        for group, geoms in (
            ("target", self.audit.targets),
            ("helper", self.audit.helper),
        ):
            minimum = (float("inf"), None)
            for g in geoms:
                mid = self.nominal_model.geom_dataid[g]
                a, n = (
                    self.nominal_model.mesh_vertadr[mid],
                    self.nominal_model.mesh_vertnum[mid],
                )
                vertices = self.nominal_model.mesh_vert[a : a + n]
                z = (
                    vertices @ native.geom_xmat[g].reshape(3, 3)[2]
                    + native.geom_xpos[g, 2]
                )
                value = float(z.min())
                if value < minimum[0]:
                    minimum = (
                        value,
                        self.nominal_model.geom(g).name
                        or self.nominal_model.mesh(mid).name,
                    )
            result[group + "_table_minimum"] = {
                "minimum_z_m": minimum[0],
                "geom_name": minimum[1],
            }
            if minimum[0] < -0.0005:
                result["candidate_rejected"] = True
                result["candidate_rejection_reasons"].append(group + "_table")
            result["max_sampled_penetration_m"] = max(
                result["max_sampled_penetration_m"], max(0.0, -minimum[0])
            )
        return result

    def _grip_pairs(self, stage):
        d = self.definitions[stage]
        task = d["task"]
        pairs = []
        for side in ("left", "right"):
            role = d[side + "_role"]
            target = (
                task
                if role == "grasp_active"
                else {
                    "hold_connector": "shoulder_connector",
                    "hold_base_assembly": "base_servo",
                    "hold_upper_arm": "upper_arm_module",
                }.get(role)
            )
            override = (
                getattr(self, "support_grasp_by_stage", {}).get(stage, {}).get(side)
            )
            if override and role != "grasp_active":
                target = override["piece"]
            if target in PARTS:
                servo = (
                    {"upper_arm_module": 2, "forearm_module": 3}.get(target)
                    if role == "grasp_active"
                    else None
                )
                contact_target = f"target_servo_{servo}_pivot" if servo else target
                pairs.append((side, contact_target))
        return pairs

    def _loaded_pads_source_extent(self, data, pairs):
        """Count the complete physical inner finger, including its curved tip."""
        return self._loaded_pads(data, pairs, source_extent=True)

    def _loaded_pads(self, data, pairs, *, source_extent=False):
        m = self.model
        result = {side: {"fixed": 0.0, "moving": 0.0} for side, _ in pairs}
        tip_limits = {}
        for side, target in pairs:
            target_id = m.body(target).id
            fixed = m.body(side + "_gripper").id
            moving = m.body(side + "_moving_jaw_so101_v1").id
            for i in range(data.ncon):
                if data.contact.dist[i] > 0:
                    continue
                a, b = map(int, data.contact.geom[i])
                ba, bb = int(m.geom_bodyid[a]), int(m.geom_bodyid[b])
                hand = None
                if _ancestor(m, bb, target_id):
                    hand = ba
                elif _ancestor(m, ba, target_id):
                    hand = bb
                if hand not in (fixed, moving):
                    continue
                hand_geom = a if hand == ba else b
                hand_name = (m.geom(hand_geom).name or "").removesuffix("__contact")
                source_mesh = self.grid_records.get(hand_name, {}).get(
                    "source_mesh_name", ""
                )
                if not any(
                    key in source_mesh for key in ("wrist_roll_follower", "moving_jaw")
                ):
                    continue
                site = m.site(side + "_gripperframe").id
                frame = data.site_xmat[site].reshape(3, 3)
                point = (data.contact.pos[i] - data.site_xpos[site]) @ frame
                normal = data.contact.frame[i, :3] * (1 if hand_geom == a else -1)
                sign = 1 if hand == fixed else -1
                tip_limit = 0.004
                if source_extent:
                    if hand_geom not in tip_limits:
                        visual = mujoco.mj_name2id(
                            m, mujoco.mjtObj.mjOBJ_GEOM, hand_name
                        )
                        if (
                            visual >= 0
                            and m.geom_type[visual] == mujoco.mjtGeom.mjGEOM_MESH
                        ):
                            mid = int(m.geom_dataid[visual])
                            start, count = (
                                int(m.mesh_vertadr[mid]),
                                int(m.mesh_vertnum[mid]),
                            )
                            vertices = m.mesh_vert[start : start + count]
                            axis = data.geom_xmat[visual].reshape(3, 3).T @ frame[:, 0]
                            origin = float(
                                (data.geom_xpos[visual] - data.site_xpos[site])
                                @ frame[:, 0]
                            )
                            tip_limits[hand_geom] = float(
                                np.max(vertices @ axis) + origin
                            )
                        else:
                            tip_limits[hand_geom] = 0.004
                    tip_limit = tip_limits[hand_geom]
                if not (
                    -0.035 < point[0] <= tip_limit + 1e-9
                    and abs(point[1]) < 0.010
                    and sign * np.dot(normal, frame[:, 2]) > 0.25
                ):
                    continue
                force = np.zeros(6)
                mujoco.mj_contactForce(m, data, i, force)
                result[side]["fixed" if hand == fixed else "moving"] += max(
                    float(force[0]), 0.0
                )
        return result

    def _load_completion_anchors(self):
        """Private completed references, not public/reset starts or fake stops."""
        folder = _anchor_folder(
            self.nominal_bank, self.profile.get("completed_anchor_bank")
        )
        manifest = json.loads(
            (ROOT / "assets_local/wowrobo/components-v2/manifest.json").read_text()
        )
        tasks = {
            "base_side_cover": "06_cover_seated",
            "shoulder_connector": "11_connector_seated",
            "assembly_pin": "16_pin_seated",
            "upper_arm_module": "24_upper_seated",
            "forearm_module": "32_final_seated",
        }
        self.goal_anchors = {}
        for task, stage in tasks.items():
            d = mujoco.MjData(self.model)
            with np.load(folder / (stage + "_train.npz")) as saved:
                mujoco.mj_setState(
                    self.model,
                    d,
                    saved["states"][0],
                    mujoco.mjtState.mjSTATE_FULLPHYSICS,
                )
            mujoco.mj_forward(self.model, d)
            servo = {"upper_arm_module": 2, "forearm_module": 3}.get(task)
            body = self.model.body(f"target_servo_{servo}_pivot" if servo else task).id
            feature = (
                np.array(
                    manifest["constituents"][f"servo_{servo}_case"]["bounds_m"]
                ).mean(0)
                if servo
                else np.zeros(3)
            )
            rotation = d.xmat[body].reshape(3, 3).copy()
            self.goal_anchors[task] = {
                "body": body,
                "feature": feature,
                "position": d.xpos[body] + rotation @ feature,
                "rotation": rotation,
                "source": str(folder / (stage + "_train.npz")),
                "goal_geometry_certified": False,
            }
        if self.pin_registration:
            goal = self.goal_anchors["assembly_pin"]
            axis = np.array(self.pin_registration["bore_axis_world"])
            goal.update(
                position=np.array(self.pin_registration["root_world_m"])
                - axis * self.pin_registration["head_gap_m"],
                rotation=np.array(self.pin_registration["rotation_world"]).reshape(
                    3, 3
                ),
                source=self.profile["pin_registration_path"],
                goal_geometry_certified=True,
            )
        path = self.scene_path.parent / "goal-registration.json"
        if path.exists():
            for task, record in json.loads(path.read_text()).items():
                self.goal_anchors[task].update(record)

    def _completion(self, stage, data, metadata):
        from scipy.spatial.transform import Rotation

        task = self.definitions[stage]["task"]
        if task not in self.goal_anchors:
            return {
                "is_terminal": False,
                "reason": "loose_scene_has_no_completed_active_task",
            }
        goal = self.goal_anchors[task]
        body = goal["body"]
        position = np.array(goal["position"])
        rotation = np.array(goal["rotation"])
        for transform in metadata.get("transforms", []):
            if transform.get("kind") == "connector_swivel" and task in transform.get(
                "pieces", []
            ):
                r = np.array(transform["rotation"])
                position = r @ position + transform["translation_m"]
                rotation = r @ rotation
        actual_rotation = data.xmat[body].reshape(3, 3)
        actual = data.xpos[body] + actual_rotation @ goal["feature"]
        error = float(np.linalg.norm(actual - position))
        if task == "assembly_pin":
            angle = float(
                np.arccos(np.clip(actual_rotation[:, 2] @ rotation[:, 2], -1, 1))
            )
        else:
            angle = float(
                Rotation.from_matrix(actual_rotation @ rotation.T).magnitude()
            )
        return {
            "is_terminal": bool(error <= 0.0005 and angle <= np.deg2rad(1)),
            "position_error_m": error,
            "orientation_error_degrees": float(np.rad2deg(angle)),
            "position_tolerance_m": 0.0005,
            "orientation_tolerance_degrees": 1.0,
            "anchor_source": goal["source"],
            "goal_geometry_certified": bool(goal["goal_geometry_certified"]),
            "quality": "registered physical feature"
            if goal["goal_geometry_certified"]
            else "provisional authored completed feature; tight residuals require registered stop",
        }

    def _register_pin(self, stage_id, data, metadata):
        if not self.pin_registration or int(stage_id.split("_", 1)[0]) < 13:
            return None
        if self.profile.get("pin_pose_registered", False):
            return {
                "registration": self.profile["pin_registration_path"],
                "registration_sha256": self.profile["pin_registration_sha256"],
                "left_gripper_ik": None,
                "noise_preserved": True,
                "stage_id": stage_id,
                "pose_overlay_applied": False,
                "scope": "Nominal bank is already registered; preserve sampled pin pose and grasp",
            }
        from assembly_policy.so101.wowrobo_pin_registration import (
            apply_pin_registration,
        )

        number = int(stage_id.split("_", 1)[0])
        m = self.model
        pin = m.body("assembly_pin").id
        mujoco.mj_forward(m, data)
        old_p = data.xpos[pin].copy()
        old_r = data.xmat[pin].reshape(3, 3).copy()
        if number <= 15:
            nominal = mujoco.MjData(m)
            with np.load(self.nominal_bank / "stages" / (stage_id + "_train.npz")) as z:
                mujoco.mj_setState(
                    m, nominal, z["states"][0], mujoco.mjtState.mjSTATE_FULLPHYSICS
                )
            mujoco.mj_forward(m, nominal)
            anchor_p = nominal.xpos[pin].copy()
            anchor_r = nominal.xmat[pin].reshape(3, 3).copy()
            for tr in metadata.get("transforms", []):
                if tr.get("kind") == "connector_swivel" and "assembly_pin" in tr.get(
                    "pieces", []
                ):
                    r = np.array(tr["rotation"])
                    anchor_p = r @ anchor_p + tr["translation_m"]
                    anchor_r = r @ anchor_r
            relative = anchor_r.T @ old_r
            receiver = m.body("shoulder_connector").id
            receiver_r = data.xmat[receiver].reshape(3, 3).copy()
            delta = receiver_r.T @ (old_p - anchor_p)
            site = m.site("left_gripperframe").id
            site_p = data.site_xpos[site].copy()
            site_r = data.site_xmat[site].reshape(3, 3).copy()
        apply_pin_registration(m, data, self.pin_registration, stage_id)
        if number <= 15:
            new_p = data.xpos[pin].copy() + receiver_r @ delta
            new_r = data.xmat[pin].reshape(3, 3).copy() @ relative
            address = m.joint("assembly_pin_free").qposadr[0]
            data.qpos[address : address + 3] = new_p
            mujoco.mju_mat2Quat(data.qpos[address + 3 : address + 7], new_r.ravel())
            mujoco.mj_forward(m, data)
            change = new_r @ old_r.T
            target_p = new_p + change @ (site_p - old_p)
            target_r = change @ site_r
            from assembly_policy.so101.wowrobo_photo_resets import PhotoResetSequence

            if not hasattr(self, "_ik_sequence"):
                self._ik_sequence = PhotoResetSequence(
                    ROOT / "assets_local/wowrobo/components-v2"
                )
            seq = self._ik_sequence
            ik_data = mujoco.MjData(seq.model)
            ik_data.qpos[:] = data.qpos
            joints = [
                m.joint("left_" + n).qposadr[0]
                for n in (
                    "shoulder_pan",
                    "shoulder_lift",
                    "elbow_flex",
                    "wrist_flex",
                    "wrist_roll",
                )
            ]
            previous = data.qpos[joints].copy()
            seq.previous["left"] = previous
            spec = dict(seq.side_grasps["assembly_pin"])
            spec["opening_radians"] = float(
                data.qpos[m.joint("left_gripper").qposadr[0]]
            )
            info = seq.solve_arm(
                ik_data,
                "left",
                target_p + target_r @ spec["center_offset_site"],
                target_r[:, 2],
                target_r[:, 0],
                spec["opening_radians"],
                np.random.default_rng(883 + number),
                grasp=spec,
                joint_reference=previous,
            )
            data.qpos[joints] = ik_data.qpos[joints]
            for a in range(m.nu):
                address = m.jnt_qposadr[m.actuator_trnid[a, 0]]
                if address in joints:
                    data.ctrl[a] = data.qpos[address]
            mujoco.mj_forward(m, data)
        else:
            info = None
        return {
            "registration": self.profile["pin_registration_path"],
            "registration_sha256": self.profile["pin_registration_sha256"],
            "left_gripper_ik": info,
            "noise_preserved": number <= 15,
            "stage_id": stage_id,
        }

    def _distribution_checks(self, stage_id, data, metadata, *, check_accessibility):
        definition = self.definitions[stage_id]
        if definition["phase"] not in ("loose", "pickup", "floor_grasp"):
            return {"required": False, "passed": True, "pieces": []}
        from assembly_policy.so101.wowrobo_photo_resets import PhotoResetSequence

        if not hasattr(self, "_ik_sequence"):
            self._ik_sequence = PhotoResetSequence(
                ROOT / "assets_local/wowrobo/components-v2"
            )
        seq = self._ik_sequence
        m = self.model
        names = seq.order if definition["phase"] == "loose" else [definition["task"]]
        records = []
        for name in names:
            side = "left" if seq.order.index(name) % 2 == 0 else "right"
            body = m.body(name).id
            r = data.xmat[body].reshape(3, 3)
            p = data.xpos[body]
            grip = seq.layout.grips[name]
            center = p + r @ grip["center"]
            radial = center[:2] - seq.layout.axes[side][:2]
            radial /= np.linalg.norm(radial)
            span = (r @ grip["axis"])[:2]
            length = np.linalg.norm(span)
            angle = float(
                np.degrees(
                    np.arcsin(np.clip(abs(radial @ (span / max(length, 1e-12))), 0, 1))
                )
            )
            if name in seq.resting:
                far = seq.resting[name]["far_side_supported"]["far_point"]
            elif name == "assembly_pin":
                registration = self.pin_registration or {}
                shaft = registration.get("shaft_length_m", 0.022)
                far = np.array([0, 0, 0.028 + shaft])
            else:
                vertices = np.vstack(
                    [
                        seq.layout.constituent_vertices[k]
                        for k in seq.layout.manifest["pieces"][name]["constituents"]
                    ]
                )
                bounds = np.array([vertices.min(0), vertices.max(0)])
                far = bounds.mean(0)
                far[1] = bounds[0, 1] if grip["center"][1] > far[1] else bounds[1, 1]
            far_world = p + r @ far
            grip_distance = float(
                np.linalg.norm(center[:2] - seq.layout.axes[side][:2])
            )
            far_distance = float(
                np.linalg.norm(far_world[:2] - seq.layout.axes[side][:2])
            )
            sampled_yaw = float(
                metadata.get("tangent_yaw_noise_degrees", {}).get(name, 0)
            )
            tangent_tolerance = 3.0 + abs(sampled_yaw)
            passed = bool(
                length > 0.2
                and angle <= tangent_tolerance
                and grip_distance <= far_distance + 0.002
            )
            row = {
                "piece": name,
                "assigned_helper": side,
                "tangent_error_degrees": angle,
                "sampled_tangent_noise_degrees": sampled_yaw,
                "tangent_tolerance_degrees": tangent_tolerance,
                "projected_span_length": float(length),
                "grip_distance_m": grip_distance,
                "far_end_distance_m": far_distance,
                "tangent_and_nearest_end_pass": passed,
                "accessibility_checked": False,
            }
            if check_accessibility and (
                definition["phase"] == "loose"
                or metadata.get("pickup_requires_hold") is False
            ):
                probe = mujoco.MjData(seq.model)
                probe.qpos[:] = data.qpos
                mujoco.mj_forward(seq.model, probe)
                target, closing = seq.grip(probe, name)
                approach = target - seq.layout.axes[side]
                approach[2] = 0
                approach -= closing * np.dot(approach, closing)
                approach /= np.linalg.norm(approach)
                if name == "base_side_cover":
                    approach = np.array([0.0, 0.0, -1.0])
                if name == "shoulder_connector":
                    approach = r @ np.array([0.0, 1.0, 0.0])
                servo = {"upper_arm_module": 2, "forearm_module": 3}.get(name)
                if servo:
                    approach = probe.xmat[
                        seq.model.body(f"target_servo_{servo}_pivot").id
                    ].reshape(3, 3) @ np.array([0.0, 0.0, -1.0])
                if (
                    abs(closing[2]) < 0.3
                    and abs(approach[2]) < 0.3
                    and np.cross(closing, approach)[2] < 0
                ):
                    closing = -closing
                seq.previous[side] = seq.joint_values(probe, side)
                ik = seq.solve_arm(
                    probe,
                    side,
                    target,
                    closing,
                    approach,
                    seq.side_grasps[name]["opening_radians"],
                    np.random.default_rng(514),
                    grasp=seq.side_grasps[name],
                    horizontal_approach=name == "shoulder_connector",
                )
                row.update(
                    accessibility_checked=True,
                    ik_position_error_m=ik["grasp_position_error_m"],
                    accessibility_pass=bool(ik["grasp_position_error_m"] < 0.002),
                )
                passed = passed and row["accessibility_pass"]
            row["passed"] = passed
            records.append(row)
        return {
            "required": True,
            "passed": all(r["passed"] for r in records),
            "pieces": records,
            "accessibility_status": "checked where no actual held grasp"
            if check_accessibility
            else "not evaluated because candidate already rejected",
        }

    def validate(self, stage_id, data, metadata):
        m = self.model
        d = mujoco.MjData(m)
        d.qpos[:] = data.qpos
        d.qvel[:] = data.qvel
        d.ctrl[:] = data.ctrl
        swivel = np.deg2rad(metadata.get("connector_swivel_degrees", 0.0))
        if int(stage_id.split("_", 1)[0]) >= 12 and not metadata.get(
            "equilibrium_input_preserved", False
        ):
            d.qpos[m.joint("target_servo_1_shaft").qposadr[0]] = swivel
        mujoco.mj_forward(m, d)
        pin_overlay = self._register_pin(stage_id, d, metadata)
        input_state = np.empty(
            mujoco.mj_stateSize(m, mujoco.mjtState.mjSTATE_FULLPHYSICS)
        )
        mujoco.mj_getState(m, d, input_state, mujoco.mjtState.mjSTATE_FULLPHYSICS)
        input_controls = d.ctrl.copy()
        before = self._native_audit(d)
        number = int(stage_id.split("_", 1)[0])
        required = []
        for name, _, _, _, first in ASSEMBLY_ATTACHMENTS:
            eq = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, name)
            if eq >= 0:
                d.eq_active[eq] = bool(
                    number >= first and not before["candidate_rejected"]
                )
                if number >= first:
                    required.append(name)
        # Installed connector is carried by the physical output horns, not an
        # independently jittered free body. Shaft1 remains a genuine free hinge.
        mujoco.mj_forward(m, d)
        input_state = np.empty(
            mujoco.mj_stateSize(m, mujoco.mjtState.mjSTATE_INTEGRATION)
        )
        mujoco.mj_getState(m, d, input_state, mujoco.mjtState.mjSTATE_INTEGRATION)
        reasons = []
        if required and before["candidate_rejected"]:
            reasons.append("installed_attachment_fit_failed")
        if before["candidate_rejected"]:
            reasons.append("initial_native_boundaries")
        if metadata.get("candidate_rejected"):
            reasons.append("candidate_sampling_constraints")
        if not metadata.get("ik_position_pass", False):
            reasons.append("candidate_ik")
        multipliers = metadata["friction_multipliers"]
        pairs = self._grip_pairs(stage_id)
        # Pickup approaches deliberately need not be held yet. Every transport,
        # insertion and support role does require a loaded two-finger grasp.
        if metadata.get("pickup_requires_hold") is False:
            active = self.definitions[stage_id]["active_arm"]
            pairs = [p for p in pairs if p[0] != active]
        reference = {}
        for side, target in pairs:
            site = m.site(side + "_gripperframe").id
            body = m.body(target).id
            frame = d.site_xmat[site].reshape(3, 3)
            reference[side] = (
                frame.T @ (d.xpos[body] - d.site_xpos[site]),
                frame.T @ d.xmat[body].reshape(3, 3),
            )
        minimum = {
            side: {"fixed": float("inf"), "moving": float("inf")} for side, _ in pairs
        }
        max_penetration = 0.0
        max_material_penetration = before["max_sampled_penetration_m"]
        worst_sdf_contact = None
        material_checks = 0
        max_drift = 0.0
        max_rotation_drift = 0.0
        hold_samples = 0
        jaw_controls = {}
        for side, _ in pairs:
            joint = m.joint(side + "_gripper").id
            actuator = np.flatnonzero(m.actuator_trnid[:, 0] == joint)
            if len(actuator) != 1:
                raise ValueError("Expected one native jaw actuator")
            a = int(actuator[0])
            jaw_controls[side] = (a, float(d.ctrl[a]))
        steps = round((self.settle_seconds + self.hold_seconds) / m.opt.timestep)
        for step in range(steps):
            if step * m.opt.timestep < self.settle_seconds:
                loads = self._loaded_pads(d, pairs)
                for side, (actuator, initial) in jaw_controls.items():
                    if min(loads[side].values()) < 0.05:
                        lower = max(
                            float(m.actuator_ctrlrange[actuator, 0]), initial - 0.12
                        )
                        d.ctrl[actuator] = max(
                            lower, d.ctrl[actuator] - 0.4 * m.opt.timestep
                        )
            physical_split_step(m, d, multipliers)
            if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all():
                reasons.append("nonfinite_dynamics")
                break
            if d.ncon:
                max_penetration = max(
                    max_penetration, max(0.0, -float(np.min(d.contact.dist[: d.ncon])))
                )
            if step * m.opt.timestep >= self.settle_seconds:
                hold_samples += 1
                for side, pads in self._loaded_pads(d, pairs).items():
                    for pad, load in pads.items():
                        minimum[side][pad] = min(minimum[side][pad], load)
                for side, target in pairs:
                    site = m.site(side + "_gripperframe").id
                    body = m.body(target).id
                    local_position, local_rotation = reference[side]
                    frame = d.site_xmat[site].reshape(3, 3)
                    relative = frame.T @ (d.xpos[body] - d.site_xpos[site])
                    max_drift = max(
                        max_drift, float(np.linalg.norm(relative - local_position))
                    )
                    relative_rotation = frame.T @ d.xmat[body].reshape(3, 3)
                    max_rotation_drift = max(
                        max_rotation_drift,
                        float(
                            np.arccos(
                                np.clip(
                                    (np.trace(relative_rotation @ local_rotation.T) - 1)
                                    / 2,
                                    -1,
                                    1,
                                )
                            )
                        ),
                    )
            if d.ncon:
                worst = int(np.argmin(d.contact.dist[: d.ncon]))
                if -float(d.contact.dist[worst]) >= max_penetration:
                    worst_sdf_contact = {
                        "geom_names": [
                            m.geom(int(g)).name for g in d.contact.geom[worst]
                        ],
                        "distance_m": float(d.contact.dist[worst]),
                    }
            if max_penetration > 0.0005 and (step % 10 == 0 or material_checks == 0):
                material_checks += 1
                probe = self._native_audit(d)
                max_material_penetration = max(
                    max_material_penetration, probe["max_sampled_penetration_m"]
                )
                if probe["candidate_rejected"]:
                    reasons.append("dynamic_material_penetration")
                    break
        if hold_samples < max(1, round(self.hold_seconds / m.opt.timestep) - 1):
            reasons.append("hold_window_not_completed")
        for loads in minimum.values():
            for pad in loads:
                if not np.isfinite(loads[pad]):
                    loads[pad] = 0.0
        if hold_samples and any(
            min(loads.values()) < 0.01 for loads in minimum.values()
        ):
            reasons.append("unloaded_finger_pad")
        if max_rotation_drift > np.deg2rad(3):
            reasons.append("grasp_rotation_slip")
        if max_drift > 0.002:
            reasons.append("grasp_slip")
        after = self._native_audit(d)
        if after["candidate_rejected"]:
            reasons.append("settled_native_boundaries")
        if np.max(np.abs(d.qvel)) > 0.05:
            reasons.append("not_settled")
        # No fabricated success geometry: calibrated seating/reward gates must
        # be supplied by the physical task factory before production admission.
        completion = self._completion(stage_id, d, metadata)
        if self.reward_checker is not None:
            if self.reward_checker(stage_id, m, d):
                reasons.append("already_terminal")
        elif completion["is_terminal"]:
            reasons.append("already_terminal")
        elif completion.get("position_error_m", 1) < 0.002 and not completion.get(
            "goal_geometry_certified", False
        ):
            reasons.append("near_completion_registration_uncertain")
        distribution = self._distribution_checks(
            stage_id, d, metadata, check_accessibility=not reasons
        )
        if not distribution["passed"]:
            reasons.append("postsettle_distribution_constraints")
        state = np.empty(mujoco.mj_stateSize(m, mujoco.mjtState.mjSTATE_INTEGRATION))
        mujoco.mj_getState(m, d, state, mujoco.mjtState.mjSTATE_INTEGRATION)
        metrics = {
            "simulated_seconds": float(d.time),
            "completion": completion,
            "postsettle_distribution": distribution,
            "contacts": int(d.ncon),
            "maximum_penetration_m": max(
                max_material_penetration, after["max_sampled_penetration_m"]
            ),
            "maximum_sdf_contact_penetration_m": max_penetration,
            "worst_sdf_contact": worst_sdf_contact,
            "material_checks_during_hold": material_checks,
            "target_table_minimum": after["target_table_minimum"],
            "minimum_pad_normal_force_N": minimum,
            "maximum_grasp_drift_m": max_drift,
            "maximum_grasp_rotation_drift_degrees": float(
                np.rad2deg(max_rotation_drift)
            ),
            "hold_samples": hold_samples,
            "loaded_grasp_check_status": "evaluated"
            if hold_samples
            else "not_evaluated_early_abort",
            "jaw_target_delta_radians": {
                side: float(d.ctrl[a] - initial)
                for side, (a, initial) in jaw_controls.items()
            },
            "maximum_abs_velocity": float(np.max(np.abs(d.qvel))),
            "largest_dof_velocities": [
                {
                    "joint": m.joint(int(m.dof_jntid[i])).name,
                    "dof": int(i),
                    "velocity": float(d.qvel[i]),
                }
                for i in np.argsort(np.abs(d.qvel))[-8:][::-1]
            ],
            "initial_boundary_rejected": before["candidate_rejected"],
            "settled_boundary_rejected": after["candidate_rejected"],
        }
        return {
            "training_eligible": not reasons,
            "physical_admission_pass": not reasons,
            "rejection_reasons": sorted(set(reasons)),
            "metrics": metrics,
            "profile": self.profile,
            "physics_profile": self.profile,
            "validation_model_signature": self.profile["model_sha256"],
            "state_spec": int(mujoco.mjtState.mjSTATE_INTEGRATION),
            "active_equalities": [
                m.equality(i).name for i in range(m.neq) if d.eq_active[i]
            ],
            "installed_attachment_model": "Only completed hardware interfaces; source fit must pass before activation; no hand welds",
            "physical_input_state": input_state,
            "physical_input_controls": input_controls,
            "pin_pose_overlay": pin_overlay,
            "settled_state": state,
            "settled_controls": d.ctrl.copy(),
        }


def main():
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--nominal", type=Path, default=ROOT / "assets_local/wowrobo/photo-resets-v3"
    )
    p.add_argument(
        "--output", type=Path, default=ROOT / "assets_local/wowrobo/physical-resets-v1"
    )
    p.add_argument("--grids", type=Path, default=DEFAULT_GRID)
    p.add_argument("--pin-mesh", type=Path)
    p.add_argument("--pin-registration", type=Path)
    p.add_argument("--completed-anchor-bank", type=Path)
    p.add_argument("--mating-profile", choices=["legacy_mapped_all"])
    p.add_argument("--assets", type=Path)
    p.add_argument("--pin-pose-registered", action="store_true")
    p.add_argument("--prepare-only", action="store_true")
    p.add_argument("--stage", default="08_connector_aligned")
    p.add_argument("--settle-seconds", type=float, default=0.15)
    p.add_argument("--hold-seconds", type=float, default=0.15)
    args = p.parse_args()
    if args.prepare_only:
        print(
            json.dumps(
                build_physical_profile(
                    args.nominal,
                    args.output,
                    args.grids,
                    pin_mesh=args.pin_mesh,
                    pin_registration_path=args.pin_registration,
                    completed_anchor_bank=args.completed_anchor_bank,
                    mating_profile=args.mating_profile,
                    pin_pose_registered=args.pin_pose_registered,
                ),
                indent=2,
            ),
            flush=True,
        )
        return
    validator = WowRoboPhysicalValidator(
        args.nominal,
        args.output,
        args.grids,
        pin_mesh=args.pin_mesh,
        settle_seconds=args.settle_seconds,
        hold_seconds=args.hold_seconds,
        pin_registration_path=args.pin_registration,
        completed_anchor_bank=args.completed_anchor_bank,
        mating_profile=args.mating_profile,
        pin_pose_registered=args.pin_pose_registered,
    )
    from assembly_policy.so101.wowrobo_randomization import WowRoboRandomizer

    generator = WowRoboRandomizer(assets=args.assets, nominal_bank=args.nominal)
    d, meta = generator.sample(args.stage, 0, difficulty=0.05)
    report = validator.validate(args.stage, d, meta)
    report.pop("settled_state", None)
    report.pop("settled_controls", None)
    output = args.output / (args.stage + "-validation.json")
    output.write_text(json.dumps(report, indent=2, default=lambda x: x.tolist()) + "\n")
    print(output, flush=True)


if __name__ == "__main__":
    main()
