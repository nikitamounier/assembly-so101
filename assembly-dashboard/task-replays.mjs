// Select within the active run only. A newer failure never replaces a saved success.
export function latestTaskReplay(replays, task, run, modelSignature) {
  const clips = Object.entries(replays || {}).filter(([key, clip]) =>
    clip && clip.task === task && key === `${task}/${clip.stage}` &&
    clip.complete_episode === true && clip.source === 'actual_training_episode' &&
    clip.identity?.run === run && clip.identity?.model_signature === modelSignature &&
    Number.isFinite(clip.recorded_at) && Number.isFinite(clip.duration_s)
  ).map(([, clip]) => clip);
  const successes = clips.filter(clip => clip.success === true);
  return (successes.length ? successes : clips.filter(clip => clip.success === false))
    .sort((a, b) => b.recorded_at - a.recorded_at || String(b.id).localeCompare(String(a.id)))[0] || null;
}

// An explicit state stays pinned, even when another state has a newer success.
export function replayForSelection(replays, task, state, run, modelSignature) {
  if (!state) return latestTaskReplay(replays, task, run, modelSignature);
  const key = `${task}/${state}`;
  return latestTaskReplay({ [key]: replays?.[key] }, task, run, modelSignature);
}
