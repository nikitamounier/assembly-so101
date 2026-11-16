import json,time,numpy as np
from pathlib import Path
from assembly_policy.viewer.components import layout_collection
from assembly_policy.so101.wowrobo_obs import PointTables,WOW_TASKS
m,_,_=layout_collection('robotics-assembly-3.0/assets_local/wowrobo/randomization-bank-repaired-cover-support-v2')
out={}
for n in [32,48,64,96,None]:
 start=time.monotonic();t=PointTables(m,feature_budget=n);row=dict(seconds=time.monotonic()-start,budget=t.receiver_budget,coverage=t.coverage,counts=t.mating_counts,signature=t.signature);out[str(n)]=row;print(n,round(row['seconds'],2),{k:round(v['feature_probe_distance_mm']['p95'],3) for k,v in t.coverage.items()},flush=True)
path=Path('robotics-assembly-3.0/outputs/point-cloud-refinement-20261107/budget-iterations.json');path.write_text(json.dumps(out,indent=2)+'\n')
np.savez_compressed(path.with_name('selected-point-tables.npz'),points=t.points.numpy(),which=t.which.numpy(),grip=t.grip.numpy(),provenance=np.asarray(t.provenance))
