"""Record the force-fixture path for independent geometry inspection.

The observer only copies poses. It records guidance and supported release;
the fixture's separate unassisted native-reward replay remains in the report.
These are diagnostic forces, not robot grasps or a learned policy.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from scripts.check_so101_sliding import slide
from scripts.review_so101_matings import default_output, identity


def trace(task, timestep, output):
    states, times = [], []
    peak_depth, peak_time, peak_pose = 0., 0., None
    model = None

    def observe(m, d):
        nonlocal model, peak_depth, peak_time, peak_pose
        model = m
        same_time = bool(times and abs(float(d.time)-times[-1]) < 1e-12)
        depth = max(0., -min(d.contact.dist, default=0.))
        if depth > peak_depth:
            peak_depth = depth
            # mj_step's contacts describe its pre-integration pose. The final
            # same-time callback follows mj_forward and describes current qpos.
            peak_pose = (d.qpos if same_time or not states else states[-1]).copy()
            peak_time = float(d.time) if same_time or not times else times[-1]
        if not same_time:
            states.append(d.qpos.copy()); times.append(float(d.time))

    report = slide(task, timestep, on_step=observe)
    states, times = np.asarray(states), np.asarray(times)
    expected = round((report['drive_duration_s']+.6)/timestep)+1
    if len(states) != expected or not np.allclose(np.diff(times),timestep,rtol=0,atol=1e-10):
        raise RuntimeError('Incomplete physics-step recording')
    stride = max(1,round(.01/timestep))
    indices = np.unique(np.r_[0,np.arange(1,len(states),stride),len(states)-1])
    case = f'{task}-slide-{round(timestep*1e6)}us'
    output.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output/(case+'.npz'),qpos=states[indices],times=times[indices],
                        physics_qpos=states,physics_times=times)
    report.update(case=case,**identity(model),
                  scope='Force-fixture insertion and supported release. Every physics-step pose retained; the separate unassisted native-reward result is in this report.')
    if peak_pose is not None:
        report.update(peak_native_pose_qpos=peak_pose.tolist(),
                      peak_native_contact=dict(time_s=peak_time,depth_m=float(peak_depth)))
    (output/(case+'.json')).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(case=case,passed=report['passed'],physics_poses=len(states),
                         audit_poses=len(indices))),flush=True)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task',choices=('motor','horn','retainer','yoke'),default='yoke')
    parser.add_argument('--timestep',type=float,default=.00025)
    parser.add_argument('--out',type=Path,default=default_output())
    args=parser.parse_args()
    if not 0 < args.timestep <= .0005:parser.error('Use a positive timestep no larger than 0.5 ms')
    if not trace(args.task,args.timestep,args.out)['passed']:raise SystemExit(1)


if __name__=='__main__':main()
