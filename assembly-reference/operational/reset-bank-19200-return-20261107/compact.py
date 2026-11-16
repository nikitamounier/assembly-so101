"""Stream display-only metadata compaction; preserve poses and native outcomes."""

from __future__ import annotations

import hashlib
import json

PROFILE_FIELDS = (
    "model_sha256",
    "model_path",
    "source_model_sha256",
    "audit_model_path",
)
WITNESS_FIELDS = (
    "stage",
    "minimum_validity_pass",
    "minimum_validity_rejection_reasons",
    "valid_for_initial_reset_bank",
    "stationarity_required",
    "grasp_translation_drift_m",
    "grasp_rotation_drift_degrees",
    "grip_anchor_by_side",
    "bilateral_contact_fraction",
    "boundary_failure_times",
    "rejection_reasons",
    "settle_seconds",
    "hold_seconds",
    "controls_unchanged",
    "minimum_pad_normal_force_N",
    "max_sampled_penetration_m",
    "boundary_threshold_m",
    "friction_multipliers",
    "training_eligible",
    "maximum_held_part_translation_m",
    "maximum_held_part_rotation_degrees",
    "part_displacement_m",
    "part_rotation_degrees",
    "hinge_motion_degrees",
    "settled_hinge_degrees",
    "maximum_abs_hold_velocity",
    "active_feature_settling_translation_m",
    "active_feature_settling_rotation_degrees",
    "active_feature_stage_anchor_required",
    "controls_unchanged_during_hold",
    "gravity_preload_control_offset_rad",
    "velocity_cleared_after_settling",
    "prescribed_bracket_angle",
    "external_forces_used",
    "state_sha256",
    "control_sha256",
    "state_spec",
    "qualification",
)


class Reader:
    def __init__(self, stream):
        self.stream, self.buffer, self.pos, self.eof = stream, "", 0, False
        self.decoder = json.JSONDecoder()

    def fill(self):
        self.buffer = self.buffer[self.pos :]
        self.pos = 0
        chunk = self.stream.read(262144)
        self.buffer += chunk
        self.eof = not chunk

    def peek(self):
        while True:
            while self.pos < len(self.buffer) and self.buffer[self.pos].isspace():
                self.pos += 1
            if self.pos < len(self.buffer):
                return self.buffer[self.pos]
            if self.eof:
                return ""
            self.fill()

    def take(self, token):
        if self.peek() != token:
            raise ValueError(f"Malformed display JSON: expected{token!r}")
        self.pos += 1

    def value(self):
        self.peek()
        while True:
            try:
                value, end = self.decoder.raw_decode(self.buffer, self.pos)
                if end == len(self.buffer) and not self.eof:
                    self.fill()
                    continue
                if end < len(self.buffer) and self.buffer[end] not in ",]}:\t\r\n ":
                    if self.eof:
                        raise ValueError("Malformed JSON value boundary")
                    self.fill()
                    continue
                self.pos = end
                return value
            except json.JSONDecodeError:
                if self.eof:
                    raise
                self.fill()


def compact_manifest(
    source, destination, proof_reference="untouched Linux return artifacts"
):
    profiles, profile_counts, removed_counts = {}, {}, {}
    rows_seen = variations = 0

    def simplify(row):
        nonlocal rows_seen, variations
        rows_seen += 1
        if not isinstance(row, dict) or row.get("kind") != "variation":
            return row
        variations += 1
        row = dict(row)
        profile = row.pop("physics_profile", None)
        if profile is not None:
            raw = json.dumps(profile, sort_keys=True, separators=(",", ":")).encode()
            signature = hashlib.sha256(raw).hexdigest()
            profiles.setdefault(signature, profile)
            profile_counts[signature] = profile_counts.get(signature, 0) + 1
            row["shared_physics_profile_sha256"] = signature
            # Keep a small backward-compatible identity descriptor for UI details.
            row["physics_profile"] = {
                key: profile[key] for key in PROFILE_FIELDS if key in profile
            }
            removed_counts["duplicated_physics_profile"] = (
                removed_counts.get("duplicated_physics_profile", 0) + 1
            )
        witness = row.get("minimum_validity")
        metrics = row.get("physical_metrics")
        if witness is None and isinstance(metrics, dict):
            witness = metrics.get("minimum_validity")
        if isinstance(witness, dict):
            row["minimum_validity"] = {
                key: witness[key] for key in WITNESS_FIELDS if key in witness
            }
        if isinstance(witness, dict) and isinstance(witness.get("boundary"), dict):
            row["minimum_validity"]["boundary"] = {
                key: witness["boundary"][key]
                for key in (
                    "candidate_rejected",
                    "candidate_rejection_reasons",
                    "threshold_m",
                    "max_sampled_penetration_m",
                    "coverage",
                    "method",
                    "training_eligible",
                    "max_samples_per_direction_after_aabb",
                    "limitations",
                    "table_minimum",
                    "target_table_minimum",
                    "helper_table_minimum",
                )
                if key in witness["boundary"]
            }
        if "physical_metrics" in row:
            del row["physical_metrics"]
            removed_counts["duplicated_full_physical_metrics"] = (
                removed_counts.get("duplicated_full_physical_metrics", 0) + 1
            )
        row["full_qualification_details"] = proof_reference
        return row

    with source.open() as input_stream, destination.open("w") as output:
        reader = Reader(input_stream)
        dump = lambda value: output.write(
            json.dumps(value, separators=(",", ":"), ensure_ascii=False)
        )

        def mapping(handler):
            reader.take("{")
            output.write("{")
            first = True
            while reader.peek() != "}":
                if not first:
                    reader.take(",")
                    output.write(",")
                key = reader.value()
                reader.take(":")
                dump(key)
                output.write(":")
                handler(key)
                first = False
            reader.take("}")
            output.write("}")

        def entries(_key):
            reader.take("[")
            output.write("[")
            first = True
            while reader.peek() != "]":
                if not first:
                    reader.take(",")
                    output.write(",")
                dump(simplify(reader.value()))
                first = False
            reader.take("]")
            output.write("]")

        def stage(_key):
            mapping(entries)

        def top(key):
            if key == "variation_by_task":
                mapping(stage)
            else:
                dump(reader.value())

        mapping(top)
        if reader.peek():
            raise ValueError("Trailing JSON after display manifest")
    # This metadata is small after per-row compaction, so a normal load is safe.
    result = json.loads(destination.read_text())
    result["shared_physics_profiles"] = profiles
    destination.write_text(
        json.dumps(result, separators=(",", ":"), ensure_ascii=False) + "\n"
    )
    return {
        "display_only": True,
        "rows_seen": rows_seen,
        "variations": variations,
        "removed_keys": removed_counts,
        "shared_profile_counts": profile_counts,
        "shared_profile_sha256": list(profiles),
        "source_bytes": source.stat().st_size,
        "compact_bytes": destination.stat().st_size,
        "proof_artifact_reference": proof_reference,
        "outcomes_reclassified": False,
        "poses_or_controls_changed": False,
    }
