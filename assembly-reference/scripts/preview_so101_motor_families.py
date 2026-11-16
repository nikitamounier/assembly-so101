"""One motor pose per SGS family for review, separate from training banks.

Use the shared native model. Pickup/in-hand come from the scripted physical
pickup; near/partial are explicit prepared initializations, never insertion
demonstrations. Retain failed hold/reset checks in report.json.

Run from the repository root:
    PYTHONPATH=. .venv/bin/python scripts/preview_so101_motor_families.py
"""

from assembly_policy.so101.limits import PENETRATION_LIMIT_M
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import mujoco
import numpy as np

from assembly_policy.so101.contact_audit import sampled_motor_base_overlap
from assembly_policy.so101.env import STATE_SPEC
from assembly_policy.so101.fastener import origins
from assembly_policy.so101.robot import down_rotation, ik
from assembly_policy.so101.scene import ASSETS, metadata, model
from assembly_policy.so101.workcell import environment_identity

FAMILIES = ('loose', 'pickup', 'inhand', 'partial', 'near')


def state(m, d):
    result = np.empty(mujoco.mj_stateSize(m, STATE_SPEC))
    mujoco.mj_getState(m, d, result, STATE_SPEC)
    return result


def loaded_jaws(m, d, side):
    touched = set()
    motor = m.body('motor').id
    for i, contact in enumerate(d.contact):
        pair = m.geom_bodyid[contact.geom]
        if motor in pair:
            force = np.zeros(6)
            mujoco.mj_contactForce(m, d, i, force)
            if force[0] > .001:
                touched.update(int(b) for b in pair if b != motor)
    return {m.body(side+'_gripper').id, m.body(side+'_moving_jaw_so101_v1').id} <= touched


def hold_check(m, saved, control, side, seconds=.5):
    d = mujoco.MjData(m)
    mujoco.mj_setState(m, d, saved, STATE_SPEC)
    d.ctrl[:] = control
    mujoco.mj_forward(m, d)
    initial = d.body('motor').xpos.copy()
    initial_relative = d.site(side+'_gripperframe').xmat.reshape(3, 3).T @ (initial-d.site(side+'_gripperframe').xpos)
    peak_depth = 0.
    for _ in range(round(seconds/m.opt.timestep)):
        mujoco.mj_step(m, d)
        peak_depth = max(peak_depth, max(0., -min(d.contact.dist, default=0.)))
    mujoco.mj_forward(m, d)
    relative = d.site(side+'_gripperframe').xmat.reshape(3, 3).T @ (d.body('motor').xpos-d.site(side+'_gripperframe').xpos)
    drift = float(np.linalg.norm(d.body('motor').xpos-initial))
    slip = float(np.linalg.norm(relative-initial_relative))
    bilateral = loaded_jaws(m, d, side)
    valid = bool(np.isfinite(d.qpos).all() and not d.warning.number.any())
    report = dict(seconds=seconds, motor_drift_m=drift, grasp_slip_m=slip,
                  bilateral_loaded_contact=bilateral, peak_native_penetration_m=peak_depth,
                  numerically_valid=valid, passed=bool(valid and bilateral and slip < .002 and drift < .003 and peak_depth < PENETRATION_LIMIT_M))
    return report, d


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, default=Path('artifacts_local/motor-family-review'))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    source = args.out/'pickup.npz'
    if not source.exists():
        subprocess.run([sys.executable, '-m', 'assembly_policy.so101.demo', '--no-video',
                        '--stop-after-lift', '--out', str(args.out/'pickup.mp4')], check=True)
    receipt = json.loads(source.with_suffix('.json').read_text())
    m = model()
    identity = environment_identity()
    binary = np.empty(mujoco.mj_sizeModel(m), np.uint8)
    mujoco.mj_saveModel(m, buffer=binary)
    if hashlib.sha256(binary.tobytes()).hexdigest() != receipt['model_sha256']:
        raise ValueError('Pickup trace does not match the current shared model')
    if not receipt['completed_script'] or any(receipt['warnings']) or receipt['max_contact_penetration_m'] > PENETRATION_LIMIT_M:
        raise ValueError('Pickup trace failed qualification')
    with np.load(source) as z:
        states, controls, stages, labels = (z[n].copy() for n in ('states', 'controls', 'stage', 'labels'))
    picks = {}
    for family, label in [('loose', 'Settle loose parts'), ('pickup', 'Lower open gripper'), ('inhand', 'Lift motor')]:
        k = int(np.flatnonzero(stages == list(labels).index(label))[-1])
        picks[family] = dict(state=states[k], control=controls[k], side='right',
                             provenance='Measured actuator-driven pickup', source_index=k, prepared=False)
    # One distinct held approach and one partially engaged geometry candidate.
    # Reset preparation may assign poses; all subsequent hold checks use mj_step.
    targets = origins(metadata())
    preparation_reports = {}
    for family, gap in [('near', .04), ('partial', .008)]:
        d = mujoco.MjData(m)
        mujoco.mj_setState(m, d, picks['loose']['state'], STATE_SPEC)
        d.ctrl[:] = picks['loose']['control']
        mujoco.mj_forward(m, d)
        base = d.body('base')
        goal = base.xpos+base.xmat.reshape(3, 3) @ (targets['motor']-targets['base'])
        position = goal+np.array([0., 0., gap])
        q = int(m.joint('motor_free').qposadr[0])
        d.qpos[q:q+3] = position
        d.qpos[q+3:q+7] = base.xquat.copy()
        # Left arm reaches the elevated entry more readily than the pickup arm.
        desired_site = position+np.array([.0126, -.004, .004])
        d.qpos[:5] = ik(m, d.qpos, 'left', desired_site, down_rotation(np.pi))
        d.qpos[5] = .13
        d.qvel[:] = 0.
        d.ctrl[:6] = d.qpos[:6]
        d.ctrl[5] = -.1
        mujoco.mj_forward(m, d)
        initial = state(m, d)
        settle_check, settled = hold_check(m, initial, d.ctrl.copy(), 'left')
        check, _ = hold_check(m, state(m, settled), settled.ctrl.copy(), 'left')
        preparation_reports[family] = dict(settling=settle_check, subsequent_hold=check)
        # A settled candidate remains inspectable even when the subsequent hold
        # fails. Its failed qualification stays explicit in the receipt.
        inspectable = (settle_check['numerically_valid'] and
                       settle_check['peak_native_penetration_m'] < PENETRATION_LIMIT_M and
                       settle_check['bilateral_loaded_contact'])
        selected = settled if inspectable else d
        picks[family] = dict(state=state(m, selected), control=selected.ctrl.copy(), side='left',
                             provenance='Prepared initialization; insertion path unverified',
                             prepared=True, requested_axial_gap_m=gap,
                             preparation_hold_passed=check['passed'])
    report = dict(task='motor', scope='Five review candidates; not installed in a training bank',
                  environment=identity, model_sha256=receipt['model_sha256'],
                  state_spec=int(STATE_SPEC), family_order=list(FAMILIES), poses={},
                  preparation_holds=preparation_reports)
    poses = {}
    for family in FAMILIES:
        item = picks[family]
        d = mujoco.MjData(m)
        mujoco.mj_setState(m, d, item['state'], STATE_SPEC)
        d.ctrl[:] = item['control']
        mujoco.mj_forward(m, d)
        native_depth = max(0., -min(d.contact.dist, default=0.))
        overlap = sampled_motor_base_overlap(m, d.qpos, ASSETS)
        base = d.body('base')
        goal = base.xpos+base.xmat.reshape(3, 3) @ (targets['motor']-targets['base'])
        details = {k:v for k,v in item.items() if k not in ('state', 'control')}
        details.update(native_penetration_m=native_depth, sampled_motor_base_overlap_m=overlap,
                       motor_position_m=d.body('motor').xpos.tolist(),
                       motor_goal_m=goal.tolist(), axial_gap_m=float(d.body('motor').xpos[2]-goal[2]),
                       bilateral_loaded_contact=loaded_jaws(m, d, item['side']),
                       numerically_valid=bool(np.isfinite(d.qpos).all() and not d.warning.number.any()))
        if family in ('inhand', 'partial', 'near'):
            details['recorded_control_hold'], _ = hold_check(m, item['state'], item['control'], item['side'])
            # Exact initial actuator-target semantics used by env.set_states.
            details['current_training_reset_hold'], _ = hold_check(m, item['state'], d.qpos[:12], item['side'])
        details['review_only'] = True
        report['poses'][family] = details
        poses[family] = np.concatenate([d.xpos, d.xquat], axis=1).round(7).tolist()
    np.savez_compressed(args.out/'motor_family_candidates.npz',
                        states=np.stack([picks[f]['state'] for f in FAMILIES]),
                        controls=np.stack([picks[f]['control'] for f in FAMILIES]),
                        family=np.arange(5), names=np.array(FAMILIES))
    (args.out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    (args.out/'body_poses.json').write_text(json.dumps(dict(body_names=[m.body(i).name for i in range(m.nbody)], poses=poses), separators=(',', ':')))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
