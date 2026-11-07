"""Bounded equilibrium diagnostic; native arms/contact, no qvel resets or angle targets."""
import argparse,json,time
from pathlib import Path
import mujoco,numpy as np
from assembly_policy.so101.wowrobo_contact_backend import load_backend,set_surface_mode
from assembly_policy.so101.wowrobo_physical_resets import WowRoboPhysicalValidator,ASSEMBLY_ATTACHMENTS,PARTS,physical_split_step
from assembly_policy.so101.wowrobo_randomization import WowRoboRandomizer
ROOT=Path(__file__).resolve().parents[2]

def run(stage,seconds=1.0):
    profile=ROOT/'assets_local/wowrobo/physical-resets-repaired-cover-support-v2'
    p=json.loads((profile/'profile.json').read_text());load_backend(p['grid_manifest']);set_surface_mode(1)
    m=mujoco.MjModel.from_xml_path(p['model_path']);d=mujoco.MjData(m)
    sampler=WowRoboRandomizer(config=ROOT/'configs/wowrobo_randomization_repaired_cover_support.json')
    authored=sampler.restore(stage);d.qpos[:]=authored.qpos;d.qvel[:]=authored.qvel;d.ctrl[:]=authored.ctrl
    n=int(stage.split('_')[0])
    for name,_,_,_,first in ASSEMBLY_ATTACHMENTS:d.eq_active[m.equality(name).id]=n>=first
    mujoco.mj_forward(m,d)
    names=list(PARTS)+['target_servo_2_pivot','target_servo_3_pivot','left_gripper','right_gripper']
    initial={b:d.xpos[m.body(b).id].copy() for b in names}
    initial_q=d.qpos.copy();initial_ctrl=d.ctrl.copy()
    # Use validator's actual pad masks without model construction or audit abort.
    v=object.__new__(WowRoboPhysicalValidator);v.model=m;v.grid_records=json.loads(Path(p['grid_manifest']).read_text())['meshes']
    pairs=[] if stage=='00_loose' else [('left','base_side_cover'),('right','target_servo_2_pivot')]
    records=[];t0=time.monotonic(); thresholds=[0,.15,.3,.5,1,2]
    def record():
      ids=np.argsort(np.abs(d.qvel))[-6:][::-1]
      records.append({'time':float(d.time),'max_abs_qvel':float(np.max(np.abs(d.qvel))),'worst_dofs':[{'name':m.joint(int(m.dof_jntid[i])).name,'value':float(d.qvel[i])}for i in ids],
      'body_drift_m':{b:float(np.linalg.norm(d.xpos[m.body(b).id]-initial[b]))for b in names},
      'hinge_degrees':{f'shaft{i}':float(np.rad2deg(d.qpos[m.joint(f'target_servo_{i}_shaft').qposadr[0]]))for i in range(1,5)},
      'pad_loads_N':v._loaded_pads(d,pairs),'ncon':int(d.ncon),'min_contact_m':float(d.contact.dist[:d.ncon].min())if d.ncon else None})
      print(stage,records[-1],flush=True)
    record();nexti=1
    for step in range(round(seconds/m.opt.timestep)):
      physical_split_step(m,d,{'finger_part':1.,'part_table':1.,'part_part':1.})
      if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all():raise RuntimeError('Nonfinite dynamics')
      if nexti<len(thresholds) and d.time+1e-8>=thresholds[nexti]:record();nexti+=1
    if records[-1]['time']<d.time-1e-8:record()
    result={'stage':stage,'wall_seconds':time.monotonic()-t0,'profile':p,'records':records,'controls_bitwise_unchanged':np.array_equal(d.ctrl,initial_ctrl),'qvel_reset_during_integration':False,'bracket_angle_target':False,'active_equalities':[m.equality(i).name for i in range(m.neq)if d.eq_active[i]],'training_eligible':False}
    dest=Path(__file__).parent; (dest/f'{stage}.json').write_text(json.dumps(result,indent=2)+'\n');np.savez_compressed(dest/f'{stage}.npz',initial_qpos=initial_q,qpos=d.qpos,qvel=d.qvel,ctrl=d.ctrl)
    return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--stage',required=True);p.add_argument('--seconds',type=float,default=1.);a=p.parse_args();run(a.stage,a.seconds)
