"""Audit repaired references including both helpers, cameras, targets and table."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from assembly_policy.so101.wowrobo_boundaries import NativeBoundaryAudit

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent

def init(bank):
    global BANK,M,A,MANIFEST,RECEIPTS
    BANK=Path(bank)
    MANIFEST=json.loads((BANK/'stages/manifest.json').read_text())
    RECEIPTS={r['id']:r for r in json.loads((BANK/'stage-receipts.json').read_text())}
    M=mujoco.MjModel.from_xml_path(str(BANK/'scene.xml'))
    A=NativeBoundaryAudit(M)

def one(stage):
    d=mujoco.MjData(M)
    with np.load(BANK/'stages'/f'{stage}_train.npz') as z:
        mujoco.mj_setState(M,d,z['states'][0],mujoco.mjtState(MANIFEST['state_spec']))
        d.ctrl[:]=z['controls'][0]
    mujoco.mj_forward(M,d)
    audit=A.audit(d)
    bounds=A._bounds(d,A.targets)
    target_floor=min(({'geom_name':A.info[g]['geom_name'],'minimum_z_m':float(bounds[g][0,2])} for g in A.targets),key=lambda x:x['minimum_z_m'])
    if target_floor['minimum_z_m'] < -.0005:
        audit['candidate_rejected']=True
        audit['candidate_rejection_reasons'].append('target_table')
    audit['target_table_minimum']=target_floor
    receipt=RECEIPTS[stage]
    reasons=['geometry:'+r for r in audit['candidate_rejection_reasons']]
    if receipt.get('registration_pass') is False:
        reasons+=['registration:'+str(r) for r in receipt.get('rejection_reasons',[])]
    return {'stage':stage,'candidate_rejected':bool(reasons),'rejection_reasons':reasons,'audit':audit,'support_contact':receipt.get('support_contact'),'grasp_registration':receipt.get('grasp_registration'), 'registration_pass':receipt.get('registration_pass')}

def main():
    p=argparse.ArgumentParser();p.add_argument('--bank',type=Path,default=ROOT/'assets_local/wowrobo/photo-resets-repaired-v1');p.add_argument('--workers',type=int,default=4);args=p.parse_args()
    manifest_path=args.bank/'stages/manifest.json'
    manifest=json.loads(manifest_path.read_text())
    source_hash=hashlib.sha256((args.bank/'scene.xml').read_bytes()).hexdigest()
    with ProcessPoolExecutor(max_workers=args.workers,initializer=init,initargs=(args.bank,)) as pool:rows=list(pool.map(one,manifest['task_names']))
    assert source_hash==hashlib.sha256((args.bank/'scene.xml').read_bytes()).hexdigest()
    (OUT/'nominal-audits.json').write_text(json.dumps({'model_sha256':source_hash,'rows':rows},indent=2))
    manifest['nominal_validation_by_task']={r['stage']:{'candidate_rejected':r['candidate_rejected'],'rejection_reasons':r['rejection_reasons'],'maximum_sampled_penetration_mm':r['audit']['max_sampled_penetration_m']*1000,'support_gap_mm':r.get('support_contact',{}).get('gap_m',0)*1000 if r.get('support_contact') else None,'audit_path':str((OUT/'nominal-audits.json').resolve())} for r in rows}
    manifest['randomized_display_name']='Repaired CAD · final-common-v1 · reset variations'
    for row in rows:
        stage=manifest['stage_by_task'][row['stage']]
        stage['boundary_review']='Boundary/registration check failed: '+', '.join(row['rejection_reasons']) if row['candidate_rejected'] else 'Sampled native boundaries and registration pass; dynamics and loaded grasps remain unqualified'
        stage['cad_revision']='final-common-v1'
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    for r in rows:
        print(r['stage'],r['candidate_rejected'],r['rejection_reasons'],round(r['audit']['max_sampled_penetration_m']*1000,3),flush=True)

if __name__=='__main__':main()
