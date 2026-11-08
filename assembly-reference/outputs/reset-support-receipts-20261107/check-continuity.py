"""Read-only zero/tiny-noise continuity regression against a prepared bank."""
import argparse
import json
from pathlib import Path

import numpy as np

from assembly_policy.so101.wowrobo_randomization import WowRoboRandomizer

p = argparse.ArgumentParser()
p.add_argument('--config', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
sampler = WowRoboRandomizer(config=a.config)
results = {}
for stage in ['21_upper_far_side','22_upper_half','23_upper_near','29_final_far_side','30_final_half','31_final_near']:
    baseline = sampler.restore(stage)
    servo = 2 if stage.startswith(('21','22','23')) else 3
    address = sampler.model.joint(f'target_servo_{servo}_shaft').qposadr[0]
    case = sampler.model.body(f'target_servo_{servo}_pivot').id
    rows = {}
    for level in [0.0, 1e-6]:
        sampled, receipt = sampler.sample(stage, 0, difficulty=level, solve_arms=False)
        support = receipt.get('resting_contact',{})
        rows[str(level)] = {
            'qpos_max_change': float(np.max(np.abs(sampled.qpos-baseline.qpos))),
            'qvel_max_change': float(np.max(np.abs(sampled.qvel-baseline.qvel))),
            'controls_max_change': float(np.max(np.abs(sampled.ctrl-baseline.ctrl))),
            'hinge_change_rad': float(sampled.qpos[address]-baseline.qpos[address]),
            'case_position_change_m': float(np.linalg.norm(sampled.xpos[case]-baseline.xpos[case])),
            'case_rotation_matrix_change': float(np.linalg.norm(sampled.xmat[case]-baseline.xmat[case])),
            'support_reference_signed_offset_m': support.get('support_reference_signed_offset_m'),
            'support_gap_m': support.get('support_gap_m'),
            'support_deviation_from_reference_m': support.get('support_deviation_from_reference_m'),
            'support_pass': support.get('support_pass'),
            'candidate_rejected': receipt.get('candidate_rejected', False),
            'rejection_reasons': receipt.get('rejection_reasons', []),
        }
    results[stage] = rows
    print(stage, json.dumps(rows), flush=True)
a.output.write_text(json.dumps(results, indent=2)+'\n')
