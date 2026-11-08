"""Lossless integration-state conversion and append-only task-conditioned bank."""
from pathlib import Path
import hashlib,importlib.util,json,sys
import mujoco,numpy as np
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'bank-19200-admission';OUT.mkdir(exist_ok=True)
STATE=Path('/large_storage/goodarzilab/nikitamounier/assembly/reset-bank-19200-20261107-v1')
SOURCE=STATE/'exports/result/native-bank';OLD=ROOT/'native-v13/training-accepted-v13'
BANK=OUT/'training-bank-v14';assert not (BANK/'manifest.json').exists();BANK.mkdir(exist_ok=True)
oldmeta=json.loads((OLD/'manifest.json').read_text());meta=json.loads((SOURCE/'manifest.json').read_text())
assert meta['complete'] and meta['prototype_count']==19200
profile=json.loads(Path(oldmeta['profile_path']).read_text());source_profile=meta['physics_profile']
# Raw XML paths differ; compare actual SDF inputs, not their directory names.
active=json.loads(Path(profile['grid_manifest']).read_text())['meshes']
incoming=json.loads(Path(source_profile['grid_manifest']).read_text())['meshes']
assert set(active)==set(incoming)
grid_fields=('id','grid_id','spacing','native_mesh_pos','native_mesh_quat_wxyz','origin','shape','bounds','grid_sha256','source_sha256')
for name in active:
    for key in grid_fields:assert active[name].get(key)==incoming[name].get(key),(name,key)
    for row in (active[name],incoming[name]):
        assert hashlib.sha256(Path(row['grid_file']).read_bytes()).hexdigest()==row['grid_sha256'],name
probe=json.loads((OUT/'source-probe.json').read_text())
assert set(probe['changed_physical_arrays'])<= {'mesh_pathadr'}
def load(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
b=load('_bank_conversion_backend',ROOT/'native-v13/robotics-assembly-3.0/assembly_policy/so101/wowrobo_contact_backend.py')
b.load_backend(source_profile['grid_manifest'],OUT/'probe-native')
m=mujoco.MjModel.from_xml_path(source_profile['model_path'])
assert m.nq==51 and m.nv==46 and m.nu==12 and m.neq==3 and m.nuserdata==0 and m.nmocap==0
helper=load('_bank_conversion_limits',ROOT/'native-v13/wowrobo_hinge_limits.py')
fit=json.loads((ROOT/'native-v13/angles.json').read_text())
for name in helper.JOINTS:m.jnt_range[m.joint(name).id]=fit['joints'][name]['range_rad']
full=mujoco.mj_stateSize(m,mujoco.mjtState.mjSTATE_FULLPHYSICS);assert full==98
data=mujoco.MjData(m);tasks=oldmeta['task_names'];destinations={};old_arrays={};old_offsets={};offset=0
for task in tasks:
    with np.load(OLD/(task+'_train.npz')) as z:old_arrays[task]={k:z[k].copy() for k in z.files}
    fields=old_arrays[task];old_offsets[task]=offset;offset+=len(fields['states'])
    for stage,family in zip(fields['stage'],fields['family']):destinations.setdefault(str(stage),set()).add((task,int(family)))
assert len(destinations)==25 and set(destinations)==set(meta['task_names']) and len(destinations['00_loose'])==5
rows={(t,s):[] for t in tasks for s in ('train','eval')};records=[];rng=np.random.default_rng(2026110714)
counts={};invalid=[];train_keys=set();eval_keys=set();all_angles=[]
for stage in meta['task_names']:
    for split,quota in (('train',640),('eval',128)):
        path=SOURCE/(stage+'_'+split+'.npz')
        with np.load(path,allow_pickle=False) as z:
            integration=z['states'];friction=z['friction_multipliers'];seeds=z['seed']
            assert len(integration)==quota and list(z['friction_categories'])==['finger_part','part_table','part_part']
            assert integration.shape[1]==mujoco.mj_stateSize(m,meta['state_spec'])
            assert np.isfinite(integration).all() and np.isfinite(friction).all() and (friction>0).all()
            controls=z['controls'].copy()
        converted=[]
        for i,row in enumerate(integration):
            mujoco.mj_setState(m,data,row,meta['state_spec'])
            assert not np.any(data.qfrc_applied) and not np.any(data.xfrc_applied),'Nonzero external forces cannot be silently discarded'
            np.testing.assert_array_equal(data.ctrl,controls[i])
            state=np.empty(162)
            mujoco.mj_getState(m,data,state[:full],mujoco.mjtState.mjSTATE_FULLPHYSICS)
            state[0]=0  # Same episode-clock normalization used by the live environment.
            state[full:full+m.neq]=data.eq_active
            k=full+m.neq;state[k:k+m.nu]=data.ctrl;k+=m.nu
            state[k:k+m.nv]=data.qacc_warmstart;k+=m.nv;assert k==159
            state[159:162]=friction[i]
            try:augmented=helper.augment_reset_states(state[None],m,rng,amplitude=profile['hinge_limit_noise_rad'])[0]
            except ValueError as error:
                invalid.append(dict(stage=stage,split=split,row=i,reason=str(error),angles_rad=[float(data.qpos[m.joint(j).qposadr[0]]) for j in helper.JOINTS],ranges_rad=[m.joint(j).range.tolist() for j in helper.JOINTS]));continue
            assert np.array_equal(augmented[:162],state)
            key=hashlib.sha256(state.tobytes()).hexdigest()
            (train_keys if split=='train' else eval_keys).add(key)
            converted.append((augmented,int(seeds[i]),i))
        counts[stage+'_'+split]=dict(source=quota,converted=len(converted))
        for task,family in sorted(destinations[stage]):
            for state,seed,i in converted:
                rows[(task,split)].append((state,family,stage))
                records.append(dict(task=task,split=split,stage=stage,family=family,source=str(path),source_row=i,seed=seed,qualified=True))
(OUT/'rejected-source-rows.json').write_text(json.dumps(dict(invalid=invalid,counts=counts),indent=2)+'\n')
# Keep strict measured-limit support. Incompatible source rows stay quarantined.
assert len(invalid)==751 and {v['stage'] for v in invalid}=={'28_final_transport'},invalid[:10]
assert not train_keys.intersection(eval_keys),'TRAIN/EVAL physical-state leakage'
old_to_new=[];new_offset=0;totals={};old_records=oldmeta.get('records',[])
for task in tasks:
    for split in ('train','eval'):
        additions=rows[(task,split)]
        states=np.array([row[0] for row in additions]);family=np.array([row[1] for row in additions],np.int64)
        stages=np.array([row[2] for row in additions]);qualified=np.ones(len(additions),bool)
        if split=='train':
            old=old_arrays[task];n=len(old['states'])
            # Old histories refer to these exact166-value rows, including their saved noise.
            states=np.concatenate((old['states'],states));family=np.concatenate((old['family'],family));stages=np.concatenate((old['stage'],stages))
            qualified=np.concatenate((old.get('qualified',np.zeros(n,bool)),qualified))
            old_to_new.extend(range(new_offset,new_offset+n));new_offset+=len(states)
        np.savez_compressed(BANK/(task+'_'+split+'.npz'),states=states,family=family,stage=stages,qualified=qualified)
        totals[task+'_'+split]=len(states)
# Check leakage against earlier training rows too, ignoring new hinge-noise offsets.
old_physical={hashlib.sha256(np.ascontiguousarray(row[:162]).tobytes()).hexdigest() for task in tasks for row in old_arrays[task]['states']}
assert not old_physical.intersection(eval_keys),'Incoming EVAL duplicates earlier TRAIN'
assert len(old_to_new)==1143 and new_offset==19074
outmeta=dict(oldmeta)
outmeta.update(schema='wowrobo-fullphysics-eq-ctrl-warmstart-friction-hinge-v2',state_size=166,state_payload_size=166,source_bank_directory=str(OLD),new_source_bank=str(SOURCE),source_variations_train=16000,source_variations_eval=3200,source_unique_variations=19200,admitted_variations_train=15371,admitted_variations_eval=3078,quarantined_source_rows=751,task_conditioned_train_rows=new_offset,task_conditioned_eval_rows=sum(v for k,v in totals.items() if k.endswith('_eval')),legacy_rows_retained=1143,records=[*old_records,*records],qualification='Source-native19200 admission retained; exact geometry/equality/SDF grids; integration-state conversion lossless except normalized clock; new-runtime initial admission pending',hinge_noise_seed_new_variations=2026110714)
(BANK/'manifest.json').write_text(json.dumps(outmeta,indent=2)+'\n')
proof=dict(status='CONVERTED_PENDING_ADMISSION',source_native_states=19200,source_train=16000,source_eval=3200,converted_source_states=18449,admitted_source_train=15371,admitted_source_eval=3078,quarantined_source_rows=751,task_conditioned_train_rows=new_offset,task_conditioned_eval_rows=outmeta['task_conditioned_eval_rows'],old_rows=1143,old_to_new=old_to_new,physical_arrays_compared=probe['compared_arrays'],geometry_equality_exact_except_path_metadata=True,source_grids_verified=len(active),external_forces_zero=True,controls_warmstarts_friction_preserved=True,train_eval_leakage=False,invalid=invalid,counts=counts,totals=totals,physics_signature=oldmeta['model_signature'])
(OUT/'conversion.json').write_text(json.dumps(proof,indent=2)+'\n')
print(json.dumps({k:v for k,v in proof.items() if k not in ('old_to_new','counts','totals')}),flush=True)
