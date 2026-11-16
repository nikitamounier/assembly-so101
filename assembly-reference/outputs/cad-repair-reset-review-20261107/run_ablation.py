"""Matched-seed static geometry ablation. Does not change production configs/banks."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import hashlib
from itertools import combinations
import json
from pathlib import Path
import time

import mujoco
import numpy as np

from assembly_policy.so101.wowrobo_boundaries import NativeBoundaryAudit
from assembly_policy.so101.wowrobo_pin_registration import apply_pin_registration
from assembly_policy.so101.wowrobo_randomization import WowRoboRandomizer

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
KEYS = ('axial_noise_m', 'lateral_radius_m', 'angle_deg', 'rest_slide_m')
STAGES = ('04_cover_entry','05_cover_near','09_connector_entry','10_connector_near','14_pin_half','15_pin_near','21_upper_far_side','22_upper_half','23_upper_near','29_final_far_side','30_final_half','31_final_near')

def init():
    global R, BASE, AM, AUDIT, REG
    BASE = json.loads((ROOT/'configs/wowrobo_randomization_repaired.json').read_text())
    R = WowRoboRandomizer(config=deepcopy(BASE))
    AM = R.model
    AUDIT = NativeBoundaryAudit(AM, max_samples=512)

def probe(job):
    stage, seed, mode = job
    started = time.monotonic()
    cfg = deepcopy(BASE)
    level = {'nominal':0, 'all_half':.5, 'all_quarter':.25, 'all_tenth':.1}.get(mode,1.)
    if mode.startswith('active_'):
        factor = {'active_zero':0, 'active_tenth':.1, 'active_quarter':.25}[mode]
        for key in KEYS:
            if key in cfg['stages'][stage]: cfg['stages'][stage][key] *= factor
    if mode == 'no_connector': cfg['global']['connector']['yaw_deg'] = 0
    if mode == 'no_axial': cfg['stages'][stage]['axial_noise_m'] = 0
    R.config = cfg
    data, receipt = R.sample(stage, seed, difficulty=level, solve_arms=False)
    d = mujoco.MjData(AM)
    d.qpos[:] = data.qpos
    d.ctrl[:] = data.ctrl
    mujoco.mj_forward(AM,d)
    active = R.definitions[stage]['task']
    bounds = AUDIT._bounds(d, AUDIT.targets)
    pairs = [(a,b) for a,b in combinations(AUDIT.targets,2) if AUDIT.info[a]['piece'] != AUDIT.info[b]['piece']]
    active_pairs = [p for p in pairs if active in [AUDIT.info[g]['piece'] for g in p]]
    inherited_pairs = [p for p in pairs if p not in active_pairs]
    actual = AUDIT._pairs(d, active_pairs, bounds)
    inherited = AUDIT._pairs(d, inherited_pairs, bounds)
    def summary(rows):
        return {'maximum_depth_mm': max((p['penetration_m']*1000 for p in rows),default=0),
                'over_0_5mm':any(p['rejected'] for p in rows),
                'over_10um':any(p['penetration_m']>1e-5 for p in rows),
                'pairs':[{'a':p['a']['geom_name'],'b':p['b']['geom_name'],'depth_mm':p['penetration_m']*1000} for p in rows if p['penetration_m']>1e-6]}
    return {'stage':stage,'seed':seed,'mode':mode,'active':active,'active_contacts':summary(actual),'inherited_contacts':summary(inherited),
            'reference_registration_rejected': any(str(reason).startswith('reference_registration:') for reason in receipt.get('rejection_reasons',[])), 'sampler_rejection_reasons': receipt.get('rejection_reasons',[]), 'projection_rejected': any('Receiver-edge projection' in str(reason) for reason in receipt.get('rejection_reasons',[])), 'projection_rejected_fixedcap': bool(receipt.get('resting_contact',{}).get('projection_distance_m',0) > BASE['stages'][stage].get('lateral_radius_m', BASE['stages'][stage].get('rest_slide_m', .001)) + 1e-6), 'resting_contact':receipt.get('resting_contact'),
            'insertion_noise':receipt.get('insertion_noise'), 'connector_swivel_degrees':receipt['connector_swivel_degrees'],
            'seconds':time.monotonic()-started}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=4);parser.add_argument('--seeds',type=int,default=8);parser.add_argument('--stages',nargs='*',default=STAGES);parser.add_argument('--name',default='results');parser.add_argument('--modes',nargs='*',default=['current','all_half','all_quarter','all_tenth']);args=parser.parse_args()
    sources=['assembly_policy/so101/wowrobo_randomization.py','configs/wowrobo_randomization_repaired.json','assembly_policy/so101/wowrobo_boundaries.py','assets_local/wowrobo/components-repaired-v1/manifest.json','assets_local/wowrobo/photo-resets-repaired-v1/scene.xml','assets_local/wowrobo/photo-resets-repaired-v1/stage-receipts.json']
    hashes={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources}
    modes=args.modes
    jobs=[(stage,0,'nominal') for stage in args.stages]+[(stage,seed,mode) for stage in args.stages for seed in range(args.seeds) for mode in modes]
    rows=[]
    with ProcessPoolExecutor(max_workers=args.workers,initializer=init) as pool:
        for i,row in enumerate(pool.map(probe,jobs)):
            rows.append(row)
            if i%32==0: print(f'{i+1}/{len(jobs)} {row["stage"]} {row["mode"]} active={row["active_contacts"]["maximum_depth_mm"]:.4f}mm',flush=True)
    assert hashes == {p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources}, 'Source changed during experiment'
    (OUT/f'{args.name}.json').write_text(json.dumps({'method':'static reciprocal native triangle sampling;512samples/direction after AABB;no helpers/IK/dynamics;final-common-v1 repaired CAD and coherent registered reset bank;matched seeds','source_sha256':hashes,'seed_count':args.seeds,'rows':rows},indent=2))
    for stage in args.stages:
        for mode in ['nominal']+modes:
            samples=[r for r in rows if r['stage']==stage and r['mode']==mode]
            depths=[r['active_contacts']['maximum_depth_mm'] for r in samples]
            print(stage,mode,'n',len(samples),'active_fail',sum(r['active_contacts']['over_0_5mm'] for r in samples),'material10um',sum(r['active_contacts']['over_10um'] for r in samples),'max_mm',round(max(depths),6),'projection_fail',sum(r['projection_rejected'] for r in samples),flush=True)

if __name__ == '__main__':main()
