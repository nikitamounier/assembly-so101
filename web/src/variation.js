const formatNumber = (number) =>
  Number.isFinite(number) ? number.toLocaleString() : "—";
const quantity = (value, suffix = "") =>
  Number.isFinite(value)
    ? `${Number(value.toFixed(3))}${suffix}`
    : "Not recorded";

export function variationStatus(variation) {
  if (!variation) return null;
  if (variation.training_eligible === true)
    return "Physically qualified training reset";
  if (variation.kind === "nominal") {
    if (variation.candidate_rejected || variation.rejected)
      return "Reference pose · validation failed";
    return "Reference pose · physical admission pending";
  }
  if (variation.candidate_rejected === true || variation.rejected === true)
    return "Rejected candidate · excluded from training";
  return "Candidate only · physical admission pending";
}

export function variationFields(variation) {
  if (!variation) return [];
  const sampling = variation.metadata || variation;
  const friction =
    sampling.friction_multipliers || variation.friction_multipliers || {};
  const fields = [
    [
      "START TYPE",
      variation.kind === "nominal" ? "Reference pose" : "Generated variation",
    ],
    ["VALIDATION", variationStatus(variation)],
  ];
  if (variation.cad_revision || sampling.cad_revision)
    fields.push([
      "CAD REVISION",
      variation.cad_revision || sampling.cad_revision,
    ]);
  for (const [side, grasp] of Object.entries(
    variation.support_grasp_by_side || {},
  )) {
    if (grasp?.piece)
      fields.push([
        `${side.toUpperCase()} SUPPORT TARGET`,
        grasp.piece.replaceAll("_", " "),
      ]);
  }
  if (variation.kind !== "nominal") {
    fields.push(
      ["DIFFICULTY", quantity(sampling.difficulty)],
      ["CONNECTOR SWIVEL", quantity(sampling.connector_swivel_degrees, "°")],
    );
    for (const [key, label] of [
      ["finger_part", "FINGER / PART FRICTION"],
      ["part_table", "PART / TABLE FRICTION"],
      ["part_part", "PART / PART FRICTION"],
    ]) {
      fields.push([
        label,
        Number.isFinite(friction[key])
          ? `×${Number(friction[key].toFixed(4))} nominal`
          : "Not recorded",
      ]);
    }
    const tangentYaw =
      sampling.tangent_yaw_noise_degrees ||
      sampling.noise?.tangent_yaw_noise_degrees ||
      variation.tangent_yaw_noise_degrees ||
      variation.noise?.tangent_yaw_noise_degrees;
    const pieceNames = {
      base_side_cover: "COVER",
      shoulder_connector: "CONNECTOR",
      assembly_pin: "PIN",
      upper_arm_module: "UPPER ARM",
      forearm_module: "FINAL MODULE",
    };
    if (
      tangentYaw &&
      typeof tangentYaw === "object" &&
      !Array.isArray(tangentYaw)
    ) {
      for (const [piece, degrees] of Object.entries(tangentYaw)) {
        if (!Number.isFinite(degrees)) continue;
        fields.push([
          `${pieceNames[piece] || piece.replaceAll("_", " ").toUpperCase()} TANGENT YAW`,
          `${degrees > 0 ? "+" : ""}${quantity(degrees, "°")}`,
        ]);
      }
    }
  }
  const reasons =
    variation.rejection_reasons ||
    variation.physical?.rejection_reasons ||
    variation.validation?.rejection_reasons;
  if (Array.isArray(reasons) && reasons.length)
    fields.push(["REJECTION REASONS", reasons.join("; ")]);
  else if (typeof reasons === "string" && reasons)
    fields.push(["REJECTION REASONS", reasons]);
  return fields;
}

export function progressFields(progress, task) {
  if (!progress || typeof progress !== "object") return [];
  const stage = progress.by_stage?.[task];
  const counts = stage || progress;
  const fields = [
    [
      stage ? "STAGE BANK PROGRESS" : "BANK PROGRESS",
      `TRAIN ${formatNumber(counts.accepted_train)} / ${formatNumber(counts.target_train)} · EVAL ${formatNumber(counts.accepted_eval)} / ${formatNumber(counts.target_eval)} accepted`,
    ],
    [
      "BANK SNAPSHOT",
      `${formatNumber(counts.rejected_count)} rejected · ${formatNumber(counts.pending_count)} pending`,
    ],
    ["GENERATED VARIATIONS", formatNumber(counts.generated_count)],
  ];
  if (stage)
    fields.push([
      "TOTAL QUALIFIED BANK",
      `TRAIN ${formatNumber(progress.accepted_train)} / ${formatNumber(progress.target_train)} · EVAL ${formatNumber(progress.accepted_eval)} / ${formatNumber(progress.target_eval)} accepted`,
    ]);
  return fields;
}
