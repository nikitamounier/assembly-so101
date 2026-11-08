"""Validate staged prepared sampler without modifying guarded source modules."""
from pathlib import Path
import importlib.util,sys,time,json,resource,hashlib
import numpy as np
import mujoco
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).parent
sys.path.insert(0,str(ROOT))
name='assembly_policy.so101.wowrobo_photo_resets'
spec=importlib.util.spec_from_file_location(name,HERE/'wowrobo_photo_resets.py');photo=importlib.util.module_from_spec(spec);sys.modules[name]=photo;spec.loader.exec_module(photo)
spec=importlib.util.spec_from_file_location('_prepared_rand',HERE/'wowrobo_randomization.py');rand=importlib.util.module_from_spec(spec);spec.loader.exec_module(rand)
start=time.monotonic()
config=ROOT/'outputs/reset-bank-1800-20261107-v4/config.json'
sampler=rand.WowRoboRandomizer(config=config)
constructed=time.monotonic()-start
print('prepared_constructor_seconds',constructed,flush=True)
assert sampler.sequence.prepared_for_sampling and sampler.sequence.refiner is None and sampler.sequence.lookahead=={}
for stage,piece in [('20_upper_transport','upper_arm_module'),('28_final_transport','forearm_module')]:
 np.testing.assert_array_equal(sampler.sequence.resting[piece]['far_side_supported']['far_point'],sampler.nominal_receipts[stage]['held_body_sag']['far_point_piece_local_m'])
maxima={'qpos':0.,'qvel':0.,'ctrl':0.}
for stage in sampler.definitions:
 reference=sampler.restore(stage)
 value,meta=sampler.sample(stage,0,difficulty=0,solve_arms=False)
 for key in maxima:
  actual=getattr(value,key).copy();expected=getattr(reference,key)
  if key=='ctrl':
   expected=np.clip(expected,sampler.model.actuator_ctrlrange[:,0],sampler.model.actuator_ctrlrange[:,1])
  if key=='qpos':
   for joint in range(sampler.model.njnt):
    if sampler.model.jnt_type[joint]==mujoco.mjtJoint.mjJNT_FREE:
     address=int(sampler.model.jnt_qposadr[joint])+3
     if np.dot(actual[address:address+4],expected[address:address+4])<0:actual[address:address+4]*=-1
  error=float(np.max(np.abs(actual-expected),initial=0));maxima[key]=max(maxima[key],error)
  np.testing.assert_allclose(actual,expected,atol=1e-12,rtol=0,err_msg=stage+':'+key)
 np.testing.assert_allclose(value.xpos,reference.xpos,atol=1e-12,rtol=0,err_msg=stage+':bodypos')
 np.testing.assert_allclose(value.xmat,reference.xmat,atol=1e-12,rtol=0,err_msg=stage+':bodyrotation')
 assert not meta.get('candidate_rejected',False),(stage,meta.get('rejection_reasons'))
 value,meta=sampler.sample(stage,0,difficulty=1e-6,solve_arms=False)
 assert np.isfinite(value.qpos).all() and np.isfinite(value.qvel).all()
 assert not meta.get('candidate_rejected',False),(stage,meta.get('rejection_reasons'))
print('all25 zero/tiny non-solved PASS',constructed,maxima,flush=True)
from assembly_policy.so101.wowrobo_contact_backend import load_backend,set_surface_mode
profile=json.loads((ROOT/'outputs/reset-bank-1800-20261107-v4/physical/profile.json').read_text())
load_backend(profile['grid_manifest']);set_surface_mode(1)
physical=mujoco.MjModel.from_xml_string(Path(profile['model_path']).read_text())
compared=[]
for stage in ['21_upper_far_side','29_final_far_side']:
 folder=ROOT/'outputs/reset-bank-1800-20261107-v4/shards'/stage
 paths=sorted((folder/'attempts').glob('*.json'))
 assert paths,stage
 row=json.loads(paths[0].read_text())
 value,meta=sampler.sample(stage,row['seed'],split=row['split'])
 assert abs(meta['difficulty']-row['metadata']['difficulty'])<1e-15
 with np.load(folder/row['physical_input_archive']) as f:state=f['states'][0];control=f['controls'][0]
 original=mujoco.MjData(physical);mujoco.mj_setState(physical,original,state,int(row['physical']['state_spec']))
 errors={k:float(np.max(np.abs(getattr(value,k)-getattr(original,k)),initial=0)) for k in ['qpos','qvel','ctrl']}
 for k in errors:np.testing.assert_allclose(getattr(value,k),getattr(original,k),atol=1e-12,rtol=0)
 compared.append(dict(stage=stage,seed=row['seed'],split=row['split'],errors=errors,input_state_sha256=row['physical_input_state_sha256']))
 print('savedphysicalinput PASS',compared[-1],flush=True)
result=dict(pass_=True,constructed_seconds=constructed,total_seconds=time.monotonic()-start,max_zero_errors=maxima,states_checked=25,tiny_noise_states_checked=25,compared_actual_physical_inputs=compared,max_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,far_markers_exact=True,canonical_source_modified=False)
(HERE/'validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
