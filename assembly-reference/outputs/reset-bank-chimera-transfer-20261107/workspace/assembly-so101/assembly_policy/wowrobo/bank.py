"""Versioned current-assembly reset conversion with explicit admission status."""

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from ..so101.wowrobo_obs import WOW_TASKS
from ..viewer.banks import ResetCatalog
from .runtime import DEFAULT_PROFILE, DEFAULT_REVIEW, load_runtime


def directory_signature(directory):
    p = Path(directory)
    h = hashlib.sha256()
    for f in sorted([p / "manifest.json", *p.glob("*.npz")]):
        h.update(f.name.encode())
        h.update(f.read_bytes())
    return h.hexdigest()


def check_bank(directory, *, allow_debug=False):
    p = Path(directory)
    meta = json.loads((p / "manifest.json").read_text())
    _, _, signature = load_runtime(meta["profile_path"])
    if meta["model_signature"] != signature:
        raise ValueError("Current assembly reset model identity changed; rebuild bank")
    if not meta["training_qualified"] and not allow_debug:
        raise ValueError(
            "No physically qualified current-assembly bank. Use --wowrobo-debug-resets ONLY for a labeled integration test."
        )
    for task in meta["task_names"]:
        with np.load(p / f"{task}_train.npz", allow_pickle=False) as z:
            if len(z["states"]) == 0:
                raise ValueError(f"No reset for {task}")
    return meta


def build(review_bank, output, *, debug=False, profile=DEFAULT_PROFILE, review_model=DEFAULT_REVIEW):
    m, physical, signature = load_runtime(profile)
    review = mujoco.MjModel.from_xml_path(str(review_model))
    review_sig = hashlib.sha256(Path(review_model).read_bytes()).hexdigest()
    if physical["source_model_sha256"] != review_sig:
        raise ValueError("Review geometry differs from physical profile")
    spec = mujoco.mjtState.mjSTATE_FULLPHYSICS
    catalog = ResetCatalog([review_bank], review_sig, mujoco.mj_stateSize(review, spec), state_spec=spec)
    nfull = mujoco.mj_stateSize(m, spec)
    data = mujoco.MjData(m)
    rd = mujoco.MjData(review)
    rows = {t: {s: [] for s in ("train", "eval")} for t in WOW_TASKS}
    records = []
    for e in catalog.entries:
        task = (e.get("stage") or {}).get("task")
        if task not in rows:
            continue
        assessment = e.get("variation") or {}
        qualified = bool(assessment.get("training_eligible"))
        if assessment.get("candidate_rejected") or assessment.get("rejected"):
            continue
        if not qualified and not debug:
            continue
        _, state = catalog.state(e["id"])
        mujoco.mj_resetData(review, rd)
        mujoco.mj_setState(review, rd, state, spec)
        mujoco.mj_resetData(m, data)
        data.qpos[:] = rd.qpos
        data.qvel[:] = rd.qvel
        data.ctrl[:] = rd.ctrl
        number = int(e["task"].split("_")[0])
        flags = []
        for attachment in physical["assembly_attachments"]:
            flags.append(number >= attachment["first_stage"])
        out = np.zeros(nfull + m.neq)
        mujoco.mj_getState(m, data, out[:nfull], spec)
        out[nfull:] = flags
        row = dict(state=out, stage=e["task"], qualified=qualified, source=e["id"])
        rows[task][e["split"]].append(row)
        records.append({k: v for k, v in row.items() if k != "state"})
    if debug:
        # No invalid held/engaged pose is smuggled in for a missing task.
        # Use the existing, explicitly loose review starts as a diagnostic
        # full-sequence task, with every installed-part equality disabled.
        for task in WOW_TASKS:
            if rows[task]["train"]:
                continue
            for e in catalog.entries:
                if e["task"] != "00_loose" or e.get("review_kind") != "variation":
                    continue
                assessment = e.get("variation") or {}
                if assessment.get("candidate_rejected") or assessment.get("rejected"):
                    continue
                _, state = catalog.state(e["id"])
                mujoco.mj_resetData(review, rd)
                mujoco.mj_setState(review, rd, state, spec)
                mujoco.mj_resetData(m, data)
                data.qpos[:] = rd.qpos
                data.qvel[:] = rd.qvel
                out = np.zeros(nfull + m.neq)
                mujoco.mj_getState(m, data, out[:nfull], spec)
                row = dict(state=out, stage="00_loose", qualified=False, source=e["id"])
                rows[task][e["split"]].append(row)
                records.append(
                    dict(
                        task=task,
                        source=e["id"],
                        stage="00_loose",
                        qualified=False,
                        reason="Diagnostic full-sequence loose fallback; no valid staged resets",
                    )
                )
    # A task lacking an admissible staged candidate is not silently omitted.
    missing = [t for t in WOW_TASKS if not rows[t]["train"]]
    if missing:
        raise ValueError(
            f"No {'debug' if debug else 'qualified'} training resets for {missing}; fix starts before launch"
        )
    p = Path(output)
    p.mkdir(parents=True, exist_ok=False)
    for task, splits in rows.items():
        for split, items in splits.items():
            if items:
                np.savez_compressed(
                    p / f"{task}_{split}.npz",
                    states=np.stack([x["state"] for x in items]),
                    family=np.array(
                        [
                            min(
                                4,
                                0
                                if "pickup" in x["stage"] or "floor" in x["stage"]
                                else 4
                                if "near" in x["stage"]
                                else 3
                                if "half" in x["stage"]
                                else 2,
                            )
                            for x in items
                        ]
                    ),
                    stage=np.array([x["stage"] for x in items]),
                )
    meta = dict(
        schema="wowrobo-fullphysics-plus-equalities-v1",
        model_signature=signature,
        source_model_signature=review_sig,
        profile_path=str(Path(profile).resolve()),
        review_model=str(Path(review_model).resolve()),
        task_names=list(WOW_TASKS),
        training_qualified=not debug,
        debug_unqualified=debug,
        records=records,
        qualification="DIAGNOSTIC ONLY; not physically admitted"
        if debug
        else "physically admitted source rows",
        state_size=nfull + m.neq,
    )
    (p / "manifest.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--review-bank", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--debug-unqualified", action="store_true")
    a = p.parse_args()
    meta = build(a.review_bank, a.output, debug=a.debug_unqualified)
    print(json.dumps({"states": len(meta["records"]), "training_qualified": meta["training_qualified"]}))


if __name__ == "__main__":
    main()
