import test from "node:test";
import assert from "node:assert/strict";
import { pointCloudWorldPositions } from "../src/point-cloud-data.js";

test("policy points align with world geometry after inverse frame and scale", () => {
  const cloud = { flat: [10, 20, 30, -10, 0, 10], scale: 10,
    frame_pose_world: { position: [4, 5, 6], rotation_row_major: [0, -1, 0, 1, 0, 0, 0, 0, 1] } };
  const original = [...cloud.flat];
  assert.deepEqual([...pointCloudWorldPositions(cloud)], [2, 6, 9, 4, 4, 7]);
  assert.deepEqual(cloud.flat, original);
});
