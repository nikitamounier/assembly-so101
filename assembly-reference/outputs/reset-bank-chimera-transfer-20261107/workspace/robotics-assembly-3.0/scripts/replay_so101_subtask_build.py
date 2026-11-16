"""Replay saved nearby-arm starts without moving them or reusing old receipts.

Produces an unpublished build for subtask_curriculum --adopt-build. The physical
model must match; every surviving row receives fresh production validation and,
for partials, a fresh half-timestep replay. Failed rows remain explicit.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import hashlib
import json
import multiprocessing
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

import numpy as np
from assembly_policy.so101.curriculum import Builder, FAMILIES, candidate_digest
from assembly_policy.so101.resets import validation_identity
from assembly_policy.so101.scene import ASSEMBLY_TASKS
from assembly_policy.so101 import subtask_curriculum as build
from assembly_policy.so101.workcell import environment_identity


def source_rows(path,expected_hash):
    content=path.read_bytes() if path.exists() else None
    actual=hashlib.sha256(content).hexdigest() if content is not None else None
    if actual!=expected_hash:raise ValueError('Saved proposal source changed after the replay plan')
    return [json.loads(line) for line in content.decode().splitlines() if line] if content else []


def replay_row(builder,row):
    item=copy.deepcopy(row)
    for name in ('qualification','halfstep_qualification'):
        item.pop(name,None)
    item['state']=np.asarray(item['state'],float)
    item['controls']=np.asarray(item['controls'],float)
    if item['state'].shape!=(builder.env.nstate,) or item['controls'].shape!=(12,):
        raise ValueError('Wrong saved state/control shape')
    expected=candidate_digest(item)
    result,receipt=builder.qualify(item,warmup=False)
    if result is None:return None,receipt
    if candidate_digest(result)!=expected:
        raise RuntimeError('Replay changed the saved starting state or controls')
    result['qualification']=receipt
    if result['family']=='partial':
        timestep,substeps=builder.m.opt.timestep,builder.env.substeps
        builder.m.opt.timestep=.00025
        builder.env.substeps=round(builder.env.ecfg.control_dt/.00025)
        try:extra,fine=builder.qualify(result,warmup=False)
        finally:builder.m.opt.timestep=timestep;builder.env.substeps=substeps
        if extra is None:return None,dict(reason='halfstep_'+fine['reason'],detail=fine)
        if candidate_digest(extra)!=expected:
            raise RuntimeError('Fine replay changed the saved starting state or controls')
        result['halfstep_qualification']=fine
    return result,receipt


def replay_task(job):
    task,source,output,plan=job
    source,output=Path(source),Path(output)
    builder=Builder();counts={}
    try:
        if builder.identity!=plan['environment'] or builder.validation_identity!=plan['validator_identity']:
            raise RuntimeError('Replay source changed before initialization')
        for family in FAMILIES:
            original=source/f'{task}_{family}.jsonl'
            rows=source_rows(original,plan['source_files'].get(original.name))
            accepted=output/original.name
            failures=output/f'{task}_{family}_failures.jsonl'
            for saved in build.read_rows(accepted):build.validate_row(saved,task,family,plan)
            done={r['seed'] for p in (accepted,failures) for r in build.read_rows(p)}
            for row in rows:
                if (output/'STOP').exists():return dict(task=task,stopped=True)
                if row['seed'] in done:continue
                if shutil.disk_usage(output).free<128*1024*1024:
                    raise RuntimeError('Less than 128 MiB free; replay progress retained')
                if row['task']!=task or row['family']!=family:
                    raise ValueError('Saved row labels differ from their source group')
                try:result,receipt=replay_row(builder,row)
                except (ValueError,FloatingPointError) as error:
                    result=None;receipt=dict(reason=str(error))
                if validation_identity()!=plan['validator_identity'] or environment_identity()!=plan['environment']:
                    raise RuntimeError('Source changed during replay; receipts remain unpublished')
                if result is None:
                    record=dict(task=task,family=family,split=row['split'],seed=row['seed'],**receipt)
                    with failures.open('a') as stream:stream.write(json.dumps(record)+'\n')
                else:
                    result['replay_provenance']=dict(source=str(original.resolve()),
                        source_plan_sha256=plan['source_plan_sha256'],
                        source_row_sha256=hashlib.sha256(json.dumps(row,sort_keys=True).encode()).hexdigest())
                    build.validate_row(result,task,family,plan)
                    with accepted.open('a') as stream:stream.write(json.dumps(build.serial(result))+'\n')
                print(json.dumps(dict(task=task,family=family,split=row['split'],seed=row['seed'],
                                      result=receipt.get('reason','pass'))),flush=True)
            counts[family]=len(build.read_rows(accepted))
        return dict(task=task,accepted=counts)
    finally:builder.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=2)
    args=parser.parse_args()
    if args.workers<1:raise ValueError('Positive worker count required')
    if args.source.resolve()==args.out.resolve():raise ValueError('Choose a separate replay output')
    old_path=args.source/'plan.json';old=json.loads(old_path.read_text())
    environment=environment_identity()
    if old['environment']!=environment:
        raise ValueError('Physical model differs; this replay only updates validator qualification')
    plan=dict(environment=environment,validator_identity=validation_identity(),
        train_count=old['train_count'],eval_count=old['eval_count'],
        generation_sha256=hashlib.sha256(Path(build.__file__).read_bytes()).hexdigest(),
        source_plan_sha256=hashlib.sha256(old_path.read_bytes()).hexdigest(),
        source_files={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                      for task in ASSEMBLY_TASKS for family in FAMILIES
                      if (p:=args.source/f'{task}_{family}.jsonl').exists()},
        revalidator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='Fresh saved-state replay only; incomplete data is not a published training bank')
    args.out.mkdir(parents=True,exist_ok=True)
    with build.build_lock(args.out):
        path=args.out/'plan.json'
        if path.exists() and json.loads(path.read_text())!=plan:
            raise ValueError('Replay provenance changed; choose a new output')
        build.write_json(path,plan)
        build.write_json(args.out/'build_status.json',dict(status='running'))
        try:
            with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
                jobs=[(task,str(args.source),str(args.out),plan) for task in ASSEMBLY_TASKS]
                results=[future.result() for future in as_completed([pool.submit(replay_task,job) for job in jobs])]
            build.write_json(args.out/'build_status.json',dict(status='stopped' if (args.out/'STOP').exists() else 'replayed',results=results))
        except BaseException as error:
            build.write_json(args.out/'build_status.json',dict(status='failed',reason=str(error)))
            raise


if __name__=='__main__':main()
