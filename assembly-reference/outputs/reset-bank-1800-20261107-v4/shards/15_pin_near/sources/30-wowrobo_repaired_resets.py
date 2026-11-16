"""Versioned reset registration against the successful derived-CAD insertion paths.

This adapter never edits the approved photo-v3 bank or frozen CAD proof. It
preserves detached poses and maps engaged states onto the demonstrated guided
path, then re-solves the actual held/supporting helper grips. A path-valid pose
may violate the photographed far-end support; those gaps are measured and
explicitly rejected rather than retaining stale contact claims.
"""
from __future__ import annotations

import copy
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import trimesh
from scipy.optimize import minimize_scalar

from assembly_policy.so101.wowrobo_boundaries import world_mesh
from assembly_policy.so101.wowrobo_photo_resets import ROOT, PhotoResetSequence

PROOF = ROOT / "outputs/insertion-fix-20261107"
OLD_BANK = ROOT / "assets_local/wowrobo/photo-resets-v3"
DEFAULT_ASSETS = ROOT / "assets_local/wowrobo/components-repaired-v1"
DEFAULT_OUTPUT = ROOT / "assets_local/wowrobo/photo-resets-repaired-v1"
PIECES = ("base_side_cover", "shoulder_connector", "assembly_pin", "upper_arm_module", "forearm_module")
STEP_NAMES = ("cover", "connector", "pin", "upper", "final")
COMPLETED = ("06_cover_seated", "11_connector_seated", "16_pin_seated", "24_upper_seated", "32_final_seated")
NEAR = ("05_cover_near", "10_connector_near", "15_pin_near", "23_upper_near", "31_final_near")
ALIGNMENTS = (
    "cover/canonical-common-v1/alignment.json",
    "connector/canonical-common-v1/coherent-alignment.json",
    "pin/final-common-proof/fit/alignment.json",
    "upper/canonical-common-v1/alignment.json",
    "final/canonical/alignment-prepared.json",
)


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def body_pose(data, body):
    return data.body(body).xpos.copy(), data.body(body).xmat.reshape(3, 3).copy()


def fit_case_support_hinge(model, data, task, evidence, preferred_hinge):
    """Fit the real internal shaft within ±.12rad while keeping its case exact.

    `evidence` supplies receiver_body and far_point_piece_local_m. The solver
    uses native source triangles for that receiver. It changes only the active
    free root and that piece's shaft; helpers/receivers are untouched. This is
    kinematic support alignment, not a contact/load/material certificate.
    """
    # The solver implementation requires only model and its geometric methods,
    # so this deliberately bypasses all sequence/asset/plan initialization.
    fitter = object.__new__(RepairedResetSequence)
    fitter.model = model
    return fitter._fit_support_hinge(data, task, evidence, preferred_hinge)


class RepairedResetSequence:
    def __init__(self, assets=DEFAULT_ASSETS, old_bank=OLD_BANK):
        self.assets = Path(assets).resolve()
        self.old_bank = Path(old_bank).resolve()
        self.sequence = PhotoResetSequence(self.assets)
        self.model = self.sequence.model
        self.plan = copy.deepcopy(self.sequence.plan)
        self.plan["profile"] = "wowrobo-photo-resets-repaired-20261107-v1"
        self.definitions = {d["id"]: d for d in self.plan["states"]}
        self.old_receipts = {r["id"]: r for r in json.loads((self.old_bank / "stage-receipts.json").read_text())}
        self.ready = json.loads((PROOF / "final-common-v1/FINAL_READY.json").read_text())
        for key, hash_key in (("source_scene", "source_scene_sha256"), ("manifest", "manifest_sha256")):
            if _digest(self.ready[key]) != self.ready[hash_key]:
                raise ValueError(f"Changed frozen CAD proof {key}")
        self.alignments = {step: json.loads((PROOF / name).read_text()) for step, name in zip(STEP_NAMES, ALIGNMENTS)}
        summary = json.loads((PROOF / "summary.json").read_text())
        self.travel = {r["step"]: r["travel_mm"] * .001 for r in summary["steps"]}
        self.endpoints = {}
        for step, piece in zip(STEP_NAMES, PIECES):
            alignment = self.alignments[step]
            q = np.asarray(alignment["start_qpos"], float).copy()
            address = int(self.model.joint(piece + "_free").qposadr[0])
            q[address:address+3] += np.asarray(alignment["axis_world"]) * self.travel[step]
            self.endpoints[step] = q
        with np.load(PROOF / "connector/canonical-common-v1/drive-endpoint-full51.npz") as z:
            self.endpoints["connector"] = z["qpos"].copy()
        with np.load(PROOF / "upper/canonical-common-v1/actual-endpoint-full51.npz") as z:
            self.endpoints["upper"] = z["qpos"].copy()
        self.piece_indices = {piece: self._piece_indices(piece) for piece in PIECES}
        self.support_meshes = {}
        self.completed_mating = {}
        old_completed_receipts = self.old_bank.parent / "photo-resets-v2/stage-receipts.json"
        if old_completed_receipts.exists():
            self.completed_mating = {r["id"]: r.get("mating") for r in json.loads(old_completed_receipts.read_text())}

    def _piece_indices(self, piece):
        m = self.model
        root = m.body(piece).id
        indices = []
        widths = {int(mujoco.mjtJoint.mjJNT_FREE): 7, int(mujoco.mjtJoint.mjJNT_BALL): 4}
        for j in range(m.njnt):
            body = int(m.jnt_bodyid[j])
            while body and body != root:
                body = int(m.body_parentid[body])
            if body == root:
                address = int(m.jnt_qposadr[j])
                indices.extend(range(address, address + widths.get(int(m.jnt_type[j]), 1)))
        return np.asarray(indices, int)

    def restore(self, stage):
        d = mujoco.MjData(self.model)
        with np.load(self.old_bank / "stages" / f"{stage}_train.npz") as z:
            mujoco.mj_setState(self.model, d, z["states"][0], mujoco.mjtState.mjSTATE_FULLPHYSICS)
            d.ctrl[:] = z["controls"][0]
        mujoco.mj_forward(self.model, d)
        return d

    def _copy_piece(self, data, piece, qpos):
        indices = self.piece_indices[piece]
        data.qpos[indices] = qpos[indices]

    def _set_root(self, data, piece, position, rotation):
        address = int(self.model.joint(piece + "_free").qposadr[0])
        data.qpos[address:address+3] = position
        mujoco.mju_mat2Quat(data.qpos[address+3:address+7], np.asarray(rotation).ravel())
        mujoco.mj_forward(self.model, data)

    def _receiver(self, step):
        return {"cover": "base_servo", "connector": "base_servo", "pin": "shoulder_connector", "upper": "shoulder_connector", "final": "upper_arm_module"}[step]

    def _map_piece(self, data, piece, reference_qpos, receiver):
        """Carry the entire canonical incoming body with its actual receiver."""
        reference = mujoco.MjData(self.model)
        reference.qpos[:] = reference_qpos
        mujoco.mj_forward(self.model, reference)
        rp, rr = body_pose(reference, receiver)
        p, r = body_pose(data, receiver)
        delta = r @ rr.T
        pp, pr = body_pose(reference, piece)
        self._copy_piece(data, piece, reference_qpos)
        self._set_root(data, piece, p + delta @ (pp - rp), delta @ pr)
        return delta

    def _receiver_mesh(self, data, receiver_name):
        receiver = self.model.body(receiver_name).id
        meshes = []
        for gid in range(self.model.ngeom):
            body = int(self.model.geom_bodyid[gid])
            while body and body != receiver:
                body = int(self.model.body_parentid[body])
            if body == receiver and self.model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_MESH:
                meshes.append(world_mesh(self.model, data, gid))
        return trimesh.util.concatenate(meshes)

    def _set_case_with_hinge(self, data, task, case_position, case_rotation, hinge):
        servo = 2 if task == "upper_arm_module" else 3
        address = int(self.model.joint(f"target_servo_{servo}_shaft").qposadr[0])
        data.qpos[address] = hinge
        mujoco.mj_forward(self.model, data)
        root_p, root_r = body_pose(data, task)
        p, r = body_pose(data, f"target_servo_{servo}_pivot")
        delta = case_rotation @ r.T
        self._set_root(data, task, case_position + delta @ (root_p-p), delta @ root_r)

    def _fit_support_hinge(self, data, task, evidence, preferred_hinge):
        """Use the real shaft freedom, keeping the fitted servo case exact."""
        servo = 2 if task == "upper_arm_module" else 3
        case = f"target_servo_{servo}_pivot"
        p, r = body_pose(data, case)
        qadr = int(self.model.joint(f"target_servo_{servo}_shaft").qposadr[0])
        canonical_hinge = float(data.qpos[qadr])
        reference_offset = evidence.get("reference_signed_support_offset_m")
        mesh = self._receiver_mesh(data, evidence["receiver_body"]) if reference_offset is None else None
        local = np.asarray(evidence["far_point_piece_local_m"])
        if reference_offset is not None:
            receiver_p, receiver_r = body_pose(data, evidence["receiver_body"])
            plane_point = receiver_p + receiver_r @ np.asarray(evidence["contact_receiver_local_m"])
            plane_normal = receiver_r @ np.asarray(evidence["support_normal_receiver_local"])
            plane_normal /= np.linalg.norm(plane_normal)
        def objective(hinge):
            self._set_case_with_hinge(data, task, p, r, hinge)
            rp, rr = body_pose(data, task)
            far = rp + rr @ local
            if reference_offset is not None:
                residual = float(np.dot(far - plane_point, plane_normal) - reference_offset)
                return residual**2 + 1e-14*(hinge-preferred_hinge)**2
            _, distances, _ = trimesh.proximity.closest_point(mesh, [far])
            return float(distances[0]**2 + 1e-14*(hinge-preferred_hinge)**2)
        grid = np.linspace(preferred_hinge-.12, preferred_hinge+.12, 25)
        costs = np.array([objective(q) for q in grid])
        index = int(np.argmin(costs))
        lower, upper = grid[max(0,index-1)], grid[min(len(grid)-1,index+1)]
        fit = minimize_scalar(objective,bounds=(lower,upper),method="bounded",options={"xatol":1e-12})
        objective(float(fit.x))
        actual_p, actual_r = body_pose(data, case)
        position_error = float(np.linalg.norm(actual_p-p))
        rotation_error = float(np.linalg.norm(actual_r-r))
        if position_error>1e-10 or rotation_error>1e-10:
            raise AssertionError("Physical hinge support fit moved the registered case")
        return {"canonical_hinge_rad":canonical_hinge,"preferred_photo_hinge_rad":preferred_hinge,"fitted_hinge_rad":float(fit.x),"hinge_delta_from_photo_rad":float(fit.x-preferred_hinge),"case_position_error_m":position_error,"case_rotation_matrix_error":rotation_error,"witness_distance_m":float(np.sqrt(max(fit.fun,0))),"scope":"Bounded real internal-hinge articulation; case pose held at measured pocket path; support/material still audited separately"}

    def _mapped_targets(self, stage, *, completed=False):
        definition = self.definitions[stage]
        old = self.restore(stage)
        data = self.restore(stage)
        task, phase = definition["task"], definition["phase"]
        if task == "initial_scene":
            return data, {"installed": [], "active_path": None, "support_hinge_fits":{}}
        ordinal = PIECES.index(task)
        installed, hinge_fits = [], {}
        for index in range(ordinal):
            piece, step = PIECES[index], STEP_NAMES[index]
            self._map_piece(data,piece,self.endpoints[step],self._receiver(step))
            if piece == "upper_arm_module":
                evidence = self.completed_mating.get("24_upper_seated") or self.old_receipts["23_upper_near"]["mating"]
                preferred = float(old.qpos[self.model.joint("target_servo_2_shaft").qposadr[0]])
                hinge_fits[piece] = self._fit_support_hinge(data,piece,evidence,preferred)
            installed.append(piece)
        active_path = None
        if completed or phase in {"aligned", "entry", "half_inserted", "near_seated", "far_side_supported"}:
            step = STEP_NAMES[ordinal]
            alignment = self.alignments[step]
            travel = float(alignment["authored_axial_travel_m"])
            if completed:
                axial = self.travel[step]
                delta = self._map_piece(data,task,self.endpoints[step],self._receiver(step))
            else:
                remaining = {"aligned": travel,"entry": {"cover": .020,"connector": .025}.get(step,travel),"half_inserted": .007 if step=="pin" else travel*.5,"near_seated": .002,"far_side_supported": travel}[phase]
                axial = travel-remaining
                delta = self._map_piece(data,task,np.asarray(alignment["start_qpos"]),self._receiver(step))
                address = int(self.model.joint(task+"_free").qposadr[0])
                data.qpos[address:address+3] += delta @ np.asarray(alignment["axis_world"]) * axial
                mujoco.mj_forward(self.model,data)
            axis = delta @ np.asarray(alignment["axis_world"])
            if task in {"upper_arm_module","forearm_module"}:
                evidence = self.completed_mating.get(COMPLETED[ordinal]) if completed else self.old_receipts[stage].get("mating")
                if evidence:
                    servo = 2 if task=="upper_arm_module" else 3
                    preferred = float(old.qpos[self.model.joint(f"target_servo_{servo}_shaft").qposadr[0]])
                    hinge_fits[task] = self._fit_support_hinge(data,task,evidence,preferred)
            active_path = {"step":step,"source":str(PROOF/ALIGNMENTS[ordinal]),"axis_world":axis.tolist(),"axial_progress_m":axial,"reference_travel_m":travel,"remaining_m":travel-axial,"scope":"Canonical case-pocket path carried by actual receiver; printed link may articulate on its real shaft to retain far support"}
        mujoco.mj_forward(self.model,data)
        return data,{"installed":installed,"active_path":active_path,"support_hinge_fits":hinge_fits}

    def _grips(self, old, data, definition):
        m, seq = self.model, self.sequence
        task = definition["task"]
        result = {}
        rng = np.random.default_rng(91000 + int(definition["id"].split("_", 1)[0]))
        for side in ("left", "right"):
            role = definition[side + "_role"]
            if role == "nearby_open":
                continue
            piece = task if role == "grasp_active" else {"hold_connector": "shoulder_connector", "hold_base_assembly": "base_servo", "hold_upper_arm": "upper_arm_module"}[role]
            servo = {"upper_arm_module": 2, "forearm_module": 3}.get(piece) if role == "grasp_active" else None
            anchor = f"target_servo_{servo}_pivot" if servo else piece
            p0, r0 = body_pose(old, anchor)
            p1, r1 = body_pose(data, anchor)
            delta = r1 @ r0.T
            site = m.site(side + "_gripperframe").id
            target_r = delta @ old.site_xmat[site].reshape(3, 3)
            target_p = p1 + delta @ (old.site_xpos[site] - p0)
            old_receipt = self.old_receipts[definition["id"]].get("arms", {}).get(side, {})
            spec = copy.deepcopy(old_receipt.get("grasp_calibration") or (seq.side_grasps[piece] if role in {"grasp_active", "hold_connector"} else seq.installed_base_grip[1] if role == "hold_base_assembly" else seq.installed_upper_grip[1]))
            opening = float(old.qpos[m.joint(side + "_gripper").qposadr[0]])
            spec["opening_radians"] = opening
            offset = np.asarray(spec["center_offset_site"])
            span_axis = int(spec["span_site_axis"])
            center = target_p + target_r @ offset
            joints = [int(m.joint(side + "_" + name).qposadr[0]) for name in ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")]
            seq.previous[side] = old.qpos[joints].copy()
            info = seq.solve_arm(data, side, center, target_r[:, span_axis], target_r[:, 0], opening, rng, grasp=spec, horizontal_approach=piece == "shoulder_connector")
            frame = data.site_xmat[site].reshape(3, 3)
            info.update(anchor_body=anchor, role=role, expected_site_position_m=target_p.tolist(), expected_site_rotation=target_r.tolist(), site_orientation_error_degrees=float(np.rad2deg(np.arccos(np.clip((np.trace(target_r.T @ frame) - 1)/2, -1, 1)))))
            result[side] = info
        return result

    def _support(self, data, stage):
        evidence = self.old_receipts[stage].get("mating")
        if not evidence:
            return None
        m = self.model
        task = self.definitions[stage]["task"]
        p, r = body_pose(data, task)
        far = p + r @ np.asarray(evidence["far_point_piece_local_m"])
        receiver = m.body(evidence["receiver_body"]).id
        pieces = []
        for gid in range(m.ngeom):
            body = int(m.geom_bodyid[gid])
            while body and body != receiver:
                body = int(m.body_parentid[body])
            if body == receiver and m.geom_type[gid] == mujoco.mjtGeom.mjGEOM_MESH:
                pieces.append(world_mesh(m, data, gid))
        mesh = trimesh.util.concatenate(pieces)
        nearest, distances, old_triangles = trimesh.proximity.closest_point(mesh, [far])
        old_gap = float(distances[0])
        local = np.asarray(evidence["far_point_piece_local_m"])
        # A repaired pose can touch the same far-end region at another vertex.
        # Check the actual printed source patch, restricting receiver normals to
        # upward-facing surfaces; a nearby vertical pocket wall is not support.
        patch = []
        for gid in range(m.ngeom):
            body = int(m.geom_bodyid[gid])
            while body and body != m.body(task).id:
                body = int(m.body_parentid[body])
            if body != m.body(task).id or m.geom_type[gid] != mujoco.mjtGeom.mjGEOM_MESH or "printed" not in (m.geom(gid).name or ""):
                continue
            incoming = world_mesh(m, data, gid)
            points = np.vstack((incoming.vertices, incoming.triangles_center))
            native = (points - p) @ r
            patch.extend(points[np.linalg.norm(native - local, axis=1) <= .015])
        witness, receiver_witness, gap = far, nearest[0], old_gap
        old_normal = mesh.face_normals[old_triangles[0]]
        selected_normal = old_normal.tolist() if old_normal[2] > .35 else None
        if patch:
            patch = np.asarray(patch)
            projected, gaps, triangles = trimesh.proximity.closest_point(mesh, patch)
            upward = mesh.face_normals[triangles, 2] > .35
            if np.any(upward):
                indices = np.flatnonzero(upward)
                index = int(indices[np.argmin(gaps[indices])])
                witness, receiver_witness, gap = patch[index], projected[index], float(gaps[index])
                selected_normal = mesh.face_normals[triangles[index]].tolist()
        witness_local = r.T @ (witness - p)
        receiver_p, receiver_r = body_pose(data, evidence["receiver_body"])
        refreshed = copy.deepcopy(evidence)
        refreshed.update(far_point_piece_local_m=witness_local.tolist(), contact_receiver_local_m=(receiver_r.T @ (receiver_witness - receiver_p)).tolist(), far_contact_gap_m=gap, contact_world_m=receiver_witness.tolist())
        return {"required": True, "receiver_body": evidence["receiver_body"], "far_point_piece_local_m": witness_local.tolist(), "far_point_world_m": witness.tolist(), "nearest_receiver_surface_world_m": receiver_witness.tolist(), "gap_m": gap, "tolerance_m": .0001, "valid": gap <= .0001 and selected_normal is not None, "old_photo_witness_gap_m": old_gap, "source_patch_radius_m": .015, "source_patch_samples": len(patch), "upward_receiver_normal": selected_normal, "mating_evidence": refreshed,
                "scope": "Finite source far-end patch to upward receiver surfaces; no loaded support-force qualification"}

    def state(self, stage, *, completed=False, solve_arms=True):
        old = self.restore(stage)
        data, registration = self._mapped_targets(stage, completed=completed)
        definition = self.definitions[stage]
        arms = self._grips(old, data, definition) if solve_arms and definition["task"] != "initial_scene" else {}
        mujoco.mj_forward(self.model, data)
        data.qvel[:] = 0
        for a in range(self.model.nu):
            data.ctrl[a] = data.qpos[self.model.jnt_qposadr[self.model.actuator_trnid[a, 0]]]
        support = None if completed else self._support(data, stage)
        receipt = copy.deepcopy(self.old_receipts[stage])
        insertion_frame = None
        if registration["active_path"]:
            path = registration["active_path"]
            receiver = {"cover": "base_servo", "connector": "base_servo", "pin": "shoulder_connector", "upper": "shoulder_connector", "final": "upper_arm_module"}[path["step"]]
            receiver_r = data.body(receiver).xmat.reshape(3, 3)
            axis = np.asarray(path["axis_world"])
            axis = axis / np.linalg.norm(axis)
            insertion_frame = {"receiver_body": receiver, "axis_world": axis.tolist(), "axis_receiver_local": (receiver_r.T @ axis).tolist(), "source": path["source"], "scope": path["scope"]}
        receipt.update(id=stage, definition=copy.deepcopy(definition), arms=arms, mating=support["mating_evidence"] if support and support["valid"] else None, support_contact=support, insertion_frame=insertion_frame,
                       repaired_registration=registration, training_eligible=False, qualification="UNQUALIFIED corrected-CAD kinematic reference; dynamic retention/admission pending")
        issues = []
        if support and not support["valid"]:
            issues.append("canonical insertion path does not retain photographed far-end support")
        for side, info in arms.items():
            if info["grasp_position_error_m"] > .002 or info["span_vector_error"] > .05:
                issues.append(f"{side} grip target residual")
        receipt["registration_rejection_reasons"] = issues
        receipt["registration_pass"] = not issues
        return data, receipt

    def export(self, output=DEFAULT_OUTPUT, *, solve_arms=True):
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        root = copy.deepcopy(self.sequence.layout.xml_root)
        root.set("model", self.plan["profile"])
        xml = ET.tostring(root, encoding="unicode")
        (output / "scene.xml").write_text(xml)
        signature = hashlib.sha256(xml.encode()).hexdigest()
        spec = mujoco.mjtState.mjSTATE_FULLPHYSICS
        receipts = []
        stages = output / "stages"
        completed_dir = output / "completed"
        stages.mkdir(exist_ok=True)
        completed_dir.mkdir(exist_ok=True)
        def save(path, data, seed):
            state = np.empty(mujoco.mj_stateSize(self.model, spec))
            mujoco.mj_getState(self.model, data, state, spec)
            np.savez_compressed(path, states=[state], controls=[data.ctrl.copy()], seed=[seed], family=[0])
            return hashlib.sha256(state.tobytes()).hexdigest(), hashlib.sha256(data.ctrl.tobytes()).hexdigest()
        for index, definition in enumerate(self.plan["states"]):
            stage = definition["id"]
            data, receipt = self.state(stage, solve_arms=solve_arms)
            sha, ctrl_sha = save(stages / f"{stage}_train.npz", data, index)
            receipt.update(state_sha256=sha, control_sha256=ctrl_sha)
            receipts.append(receipt)
            print(stage, {"registration_pass": receipt["registration_pass"], "grip_error_mm": {s: round(v["grasp_position_error_m"]*1000,4) for s,v in receipt["arms"].items()}, "support_gap_mm": (receipt["support_contact"]["gap_m"]*1000 if receipt["support_contact"] else None)}, flush=True)
        for name, near in zip(COMPLETED, NEAR):
            data, receipt = self.state(near, completed=True, solve_arms=solve_arms)
            save(completed_dir / f"{name}_train.npz", data, -1)
        tasks = [d["id"] for d in self.plan["states"]]
        manifest = {"component_manifest_sha256": _digest(self.assets / "manifest.json"), "cad_revision": "insertion-fix-final-common-v1", "model_signature": signature, "state_spec": int(spec), "task_names": tasks, "family_names": ["Unqualified repaired-CAD photo reset candidate"], "reset_distribution": "photo_review", "display_name": "WowRobo · repaired CAD · nominal references", "default_viewer_collection": True, "stage_by_task": {d["id"]: d for d in self.plan["states"]}, "focus_by_task": {t:{"target":[0,0,.11],"radius":.46} for t in tasks}}
        (stages / "manifest.json").write_text(json.dumps(manifest, indent=2))
        (output / "manifest.json").write_text(json.dumps({"profile":self.plan["profile"],"component_manifest_sha256":_digest(self.assets / "manifest.json"),"cad_revision":"insertion-fix-final-common-v1","model_signature":signature,"bank":"stages","count":len(tasks),"source_frozen_cad":self.ready,"qualification":"UNQUALIFIED kinematic review; canonical guided insertion verified separately"},indent=2))
        (output / "stage-receipts.json").write_text(json.dumps(receipts,indent=2))
        (output / "reset-contract.json").write_text(json.dumps(self.plan,indent=2))
        return {"output":str(output),"model_signature":signature,"count":len(tasks),"registration_failures":{r["id"]:r["registration_rejection_reasons"] for r in receipts if not r["registration_pass"]},"completed":str(completed_dir)}
