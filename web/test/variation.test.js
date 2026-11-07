import assert from "node:assert/strict";
import test from "node:test";
import {
  variationFields,
  variationStatus,
  progressFields,
} from "../src/variation.js";
import { filterEntries } from "../src/catalog.js";

test("equilibrium hold evidence does not imply task admission", () => {
  const equilibrium = {
    equilibrium_pass: true,
    prescribed_bracket_angle: false,
    hold_seconds: 0.5,
    part_displacement_m: { bracket: 0.00004 },
    part_rotation_degrees: { bracket: 0.02 },
  };
  const variation = { kind: "nominal", equilibrium };
  assert.match(variationStatus(variation), /task admission pending/);
  const fields = new Map(variationFields(variation));
  assert.equal(fields.get("BRACKET ARTICULATION"), "Passive · no angle target");
  assert.equal(fields.get("MAXIMUM PART MOTION"), "0.04 mm");
  assert.match(
    variationStatus({ equilibrium: { equilibrium_pass: false } }),
    /Hold test failed/,
  );
});

test("candidate and rejected status remain separate from physical training admission", () => {
  assert.match(variationStatus({ training_eligible: false }), /pending/);
  assert.match(variationStatus({ candidate_rejected: true }), /Rejected/);
  assert.match(variationStatus({ training_eligible: true }), /qualified/);
  const fields = new Map(
    variationFields({
      kind: "variation",
      candidate_rejected: true,
      metadata: {
        difficulty: 0.75,
        connector_swivel_degrees: -2.3,
        friction_multipliers: {
          finger_part: 0.98,
          part_table: 1.02,
          part_part: 1,
        },
      },
      rejection_reasons: ["pin intersects connector"],
    }),
  );
  assert.equal(fields.get("DIFFICULTY"), ".75".replace(/^\./, "0."));
  assert.equal(fields.get("CONNECTOR SWIVEL"), "-2.3°");
  assert.equal(fields.get("FINGER / PART FRICTION"), "×0.98 nominal");
  assert.equal(fields.get("REJECTION REASONS"), "pin intersects connector");
});

test("bank progress uses real stage counts and does not fill missing counts with zero", () => {
  const progress = {
    by_stage: {
      "08_connector_aligned": {
        accepted_train: 7,
        target_train: 640,
        accepted_eval: 2,
        target_eval: 128,
        rejected_count: 3,
        pending_count: 9,
      },
    },
  };
  assert.equal(
    progressFields(progress, "08_connector_aligned")[0][1],
    "TRAIN 7 / 640 · EVAL 2 / 128 accepted",
  );
  assert.match(progressFields({}, "missing")[0][1], /—/);
});

test("nominal versus generated filter retains task and split filtering", () => {
  const rows = [
    { id: "a", task: "08", split: "train", review_kind: "nominal", seed: 0 },
    { id: "b", task: "08", split: "train", review_kind: "variation", seed: 1 },
    { id: "c", task: "08", split: "eval", review_kind: "variation", seed: 2 },
  ];
  assert.deepEqual(
    filterEntries(rows, {
      kind: "variation",
      task: "08",
      split: "eval",
      family: "",
    }).map((e) => e.id),
    ["c"],
  );
});

test("tangent yaw review shows actual per-piece draws with friendly labels", () => {
  const draws = {
    base_side_cover: 0.125,
    shoulder_connector: -0.75,
    assembly_pin: 0,
    upper_arm_module: 0.031,
    forearm_module: -0.25,
  };
  const fields = new Map(
    variationFields({
      kind: "variation",
      metadata: {
        tangent_yaw_noise_degrees: draws,
        tangent_yaw_max_degrees: 2,
      },
    }),
  );
  assert.equal(fields.get("COVER TANGENT YAW"), "+0.125°");
  assert.equal(fields.get("CONNECTOR TANGENT YAW"), "-0.75°");
  assert.equal(fields.get("PIN TANGENT YAW"), "0°");
  assert.equal(fields.get("UPPER ARM TANGENT YAW"), "+0.031°");
  assert.equal(fields.get("FINAL MODULE TANGENT YAW"), "-0.25°");
  const exported = new Map(
    variationFields({
      kind: "variation",
      noise: { tangent_yaw_noise_degrees: draws },
    }),
  );
  assert.equal(exported.get("COVER TANGENT YAW"), "+0.125°");
  assert(
    !variationFields({ kind: "variation", tangent_yaw_max_degrees: 2 }).some(
      ([key]) => key.includes("TANGENT YAW"),
    ),
  );
  assert(
    !variationFields({
      kind: "nominal",
      metadata: { tangent_yaw_noise_degrees: draws },
    }).some(([key]) => key.includes("TANGENT YAW")),
  );
});

test("repaired references keep actual rejection status and CAD revision visible", () => {
  const reference = {
    kind: "nominal",
    cad_revision: "final-common-v1",
    candidate_rejected: true,
    rejection_reasons: ["support gap"],
  };
  assert.match(variationStatus(reference), /validation failed/);
  const fields = new Map(variationFields(reference));
  assert.equal(fields.get("CAD REVISION"), "final-common-v1");
  assert.equal(fields.get("REJECTION REASONS"), "support gap");
});
