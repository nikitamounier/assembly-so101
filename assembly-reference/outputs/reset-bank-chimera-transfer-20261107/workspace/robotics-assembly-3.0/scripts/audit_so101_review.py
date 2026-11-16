"""Check recorded poses at original-surface points unused by contact detection.

Run after scripts.review_so101_matings record. This checks every saved pose and
the recorded peak-contact pose. It is a sampled check, not continuous collision
detection. Analytic screw geometry is excluded from this regular-part audit.
"""

from assembly_policy.so101.limits import PENETRATION_LIMIT_M
import argparse
import hashlib
from itertools import combinations
import json
from pathlib import Path
import time

import igl
import mujoco
import numpy as np

from assembly_policy.so101 import scene, workcell
from scripts.review_so101_matings import CASES, default_output, identity

WEIGHTS = np.array([[.5,.5,0], [.5,0,.5], [0,.5,.5],
                    [.6,.2,.2], [.2,.6,.2], [.2,.2,.6]])


def surfaces():
    result = {}
    for name in scene.PARTS:
        if name == 'screw':
            continue
        with np.load(scene.ASSETS/'parts'/f'{name}.npz') as archive:
            vertices, faces = archive['vertices'].copy(), archive['faces'].copy()
        samples = np.einsum('sj,fjk->fsk', WEIGHTS, vertices[faces]).reshape(-1,3)
        tree = igl.AABB(); tree.init(vertices, faces)
        winding = igl.FastWindingNumberBVH(); winding.init(vertices, faces)
        lower, upper = vertices.min(0), vertices.max(0)
        corners = np.array(np.meshgrid(*np.stack([lower,upper]).T,
                                      indexing='ij')).reshape(3,-1).T
        result[name] = dict(vertices=vertices, faces=faces, samples=samples,
                            tree=tree, winding=winding, lower=lower, upper=upper,
                            corners=corners)
    return result


def pair_overlap(data, first, second, meshes, bounds):
    a_low, a_high = bounds[first]; b_low, b_high = bounds[second]
    # Conservative world bounds avoid transforming points from distant parts.
    if np.any(a_high+1e-12 < b_low) or np.any(b_high+1e-12 < a_low):
        return 0.
    worst = 0.
    for source, target in ((first,second), (second,first)):
        mesh = meshes[target]
        points = (meshes[source]['samples'] @ data.body(source).xmat.reshape(3,3).T
                  + data.body(source).xpos - data.body(target).xpos) @ data.body(target).xmat.reshape(3,3)
        points = points[np.all((points >= mesh['lower']) & (points <= mesh['upper']), axis=1)]
        if not len(points):
            continue
        inside = mesh['winding'].winding_number(points) > .5
        if inside.any():
            squared = mesh['tree'].squared_distance(mesh['vertices'],mesh['faces'],points[inside])[0]
            worst = max(worst, float(np.sqrt(np.maximum(squared,0)).max()))
    return worst


def audit(case, folder, model, meshes):
    path = folder/(case+'.npz')
    metadata = json.loads(path.with_suffix('.json').read_text())
    model.opt.timestep = metadata['timestep_s']
    if identity(model)['native_model_sha256'] != metadata['native_model_sha256']:
        raise ValueError('Recorded model does not match the audit geometry/model')
    with np.load(path) as archive:
        states, times = archive['qpos'].copy(), archive['times'].copy()
    if states.shape != (len(times),model.nq) or not np.isfinite(states).all():
        raise ValueError('Invalid recorded poses')
    poses = list(zip(times,states))
    if 'peak_native_pose_qpos' in metadata:
        poses.append((metadata['peak_native_contact']['time_s'],metadata['peak_native_pose_qpos']))
    if 'peak_floor_pose_qpos' in metadata:
        poses.append((metadata['peak_floor_contact']['time_s'],metadata['peak_floor_pose_qpos']))
    data = mujoco.MjData(model); start = time.monotonic()
    peak = 0.; location = None; pair_peaks = {}
    for index, (timestamp, pose) in enumerate(poses):
        data.qpos[:] = pose
        mujoco.mj_kinematics(model, data)
        bounds = {}
        plane=data.geom('table');normal=plane.xmat.reshape(3,3)[:,2]
        if int(model.geom('table').type[0])!=int(mujoco.mjtGeom.mjGEOM_PLANE):
            raise ValueError('Table audit requires the shared plane geometry')
        for name, mesh in meshes.items():
            box = mesh['corners'] @ data.body(name).xmat.reshape(3,3).T + data.body(name).xpos
            bounds[name] = (box.min(0),box.max(0))
            body=data.body(name)
            heights=mesh['vertices']@(body.xmat.reshape(3,3).T@normal)+(body.xpos-plane.xpos)@normal
            depth=max(0.,-float(heights.min()));pair=name+'/table'
            pair_peaks[pair]=max(pair_peaks.get(pair,0.),depth)
            if depth>peak:
                peak=depth;location=dict(time_s=float(timestamp),pair=pair,pose_index=index)
        for first, second in combinations(meshes,2):
            depth = pair_overlap(data,first,second,meshes,bounds)
            pair = first+'/'+second
            pair_peaks[pair] = max(pair_peaks.get(pair,0.),depth)
            if depth > peak:
                peak = depth
                location = dict(time_s=float(timestamp),pair=pair,pose_index=index)
        if index % 100 == 0:
            print(f'{case}: {index} poses, held-out peak {peak*1000:.4f} mm',flush=True)
    report = dict(case=case, passed=peak < PENETRATION_LIMIT_M, penetration_limit_m=PENETRATION_LIMIT_M, max_held_out_material_m=peak,
                  peak=location, pair_peaks_m=pair_peaks, poses_checked=len(poses),
                  samples_per_face=len(WEIGHTS), weights=WEIGHTS.tolist(),
                  sample_counts={name:len(mesh['samples']) for name,mesh in meshes.items()},
                  trace_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  recording_sha256=hashlib.sha256(path.with_suffix('.json').read_bytes()).hexdigest(),
                  recorded_identity=metadata['environment'], current_identity=workcell.environment_identity(),
                  native_model_sha256=metadata['native_model_sha256'],
                  seconds=time.monotonic()-start,
                  scope='Original-mesh distances at held-out edge/off-centre points plus exact original-vertex table-plane depth at every recorded pose. Analytic screw excluded; not continuous collision detection.')
    (folder/(case+'-dense.json')).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('case','passed','max_held_out_material_m','poses_checked')}),flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=default_output())
    parser.add_argument('--cases',default=','.join(CASES))
    args = parser.parse_args()
    model = scene.model(contact_monitor=True)
    meshes = surfaces()
    reports = [audit(case,args.out,model,meshes) for case in args.cases.split(',')]
    if not all(report['passed'] for report in reports):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
