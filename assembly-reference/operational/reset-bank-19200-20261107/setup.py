"""Prepare a new quota-only bank against the immutable, qualified native runtime."""
import argparse,hashlib,json,os,shutil,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=a.root.resolve();assert os.environ.get('SLURM_JOB_ID');source=Path('/large_storage/goodarzilab/nikitamounier/assembly/reset-bank-1800-20261107-v4');project=source/'robotics-assembly-3.0';old=project/'outputs/reset-bank-1800-20261107-v4';plan=r/'plan';plan.mkdir(exist_ok=True)
def write(p,v):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2)+'\n');t.replace(p)
if (r/'settings.json').exists():print('Existing prepared settings preserved');raise SystemExit(0)
cfg=json.loads((old/'config.json').read_text());spec=json.loads((old/'plan.json').read_text());assert len(spec['stage_ids'])==25
cfg['profile']='wowrobo-object-relative-19200-v1';cfg['rng_namespace']='wowrobo19200-v1';s=cfg['sampling'];s.update(train_per_stage=640,eval_per_stage=128,train_total=16000,eval_total=3200)
for b,n in zip(s['train_difficulty'],[256,256,128]):b['count_per_stage']=n
shutil.copy2(old/'grasps.json',plan/'grasps.json');cfg['grasp_library']=str(plan/'grasps.json');write(plan/'config.json',cfg)
spec.update(config=str(plan/'config.json'),config_sha256=hashlib.sha256((plan/'config.json').read_bytes()).hexdigest(),train_per_stage=640,eval_per_stage=128,target_total=19200,grasp_library_sha256=hashlib.sha256((plan/'grasps.json').read_bytes()).hexdigest());write(plan/'plan.json',spec)
physical=old/'physical';profile=json.loads((physical/'profile.json').read_text());assert hashlib.sha256(Path(profile['model_path']).read_bytes()).hexdigest()==profile['model_sha256'];assert profile['model_sha256']=='1b071d9b92dc5e3e3e9ba33234a25f24bfc3b3049081a52ac673b1df423b952d'
settings=dict(root=str(r),project=str(project),source_root=str(source),plan=str(plan),physical=str(physical),train=640,eval=128,parts=64,rng_namespace='wowrobo19200-v1',target=19200,model_sha256=profile['model_sha256'],worker_memory_gib=16,created_unix=time.time(),setup_job=os.environ['SLURM_JOB_ID'],source_witness_count=25)
# Interleave stages; start the empirically slow held/insertion stages first.
hard=['31_final_near','30_final_half','29_final_far_side','14_pin_half','26_final_floor_grasp','28_final_transport','21_upper_far_side','22_upper_half','23_upper_near'];order=hard+[x for x in spec['stage_ids'] if x not in hard]
tasks=[]
for part in range(64):
 for stage in order:tasks.append(dict(index=len(tasks),stage=stage,part=part,parts=64))
write(r/'assignment.json',dict(tasks=tasks,open=True,train=640,eval=128,parts=64));write(r/'dispatch.json',dict(cursor=0,active={},failures={},attempts={}));(r/'done').mkdir(exist_ok=True);(plan/'partitions').mkdir(exist_ok=True);(plan/'shards').mkdir(exist_ok=True);write(r/'settings.json',settings)
write(r/'setup-receipt.json',dict(settings=settings,config_sha256=spec['config_sha256'],tasks=len(tasks),old_bank_modified=False,old_receipts_imported=False));print(json.dumps(settings,indent=2))
