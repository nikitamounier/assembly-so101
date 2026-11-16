/** Convert the complete ordered policy tensor into world metres for display. */
export function pointCloudWorldPositions(cloud) {
  const {position: p, rotation_row_major: r} = cloud.frame_pose_world;
  const result = new Float32Array(cloud.flat.length);
  for (let i = 0; i < cloud.flat.length; i += 3) {
    const x = cloud.flat[i] / cloud.scale, y = cloud.flat[i + 1] / cloud.scale, z = cloud.flat[i + 2] / cloud.scale;
    for (let row = 0; row < 3; row++) result[i + row] = p[row] + r[row*3] * x + r[row*3+1] * y + r[row*3+2] * z;
  }
  return result;
}
