"""Compare scientific continuation state, excluding run paths and timing fields."""
import json,sys
from pathlib import Path
import numpy as np
import torch
h=Path(sys.argv[1]);a=torch.load(h/'ppo-original/ppo/checkpoint.pt',map_location='cpu',weights_only=False);b=torch.load(h/'ppo-candidate/ppo/checkpoint.pt',map_location='cpu',weights_only=False)
def equal(x,y):
 if isinstance(x,torch.Tensor):return isinstance(y,torch.Tensor) and torch.equal(x,y)
 if isinstance(x,np.ndarray):return isinstance(y,np.ndarray) and np.array_equal(x,y,equal_nan=True)
 if isinstance(x,dict):return isinstance(y,dict) and x.keys()==y.keys() and all(equal(x[k],y[k]) for k in x)
 if isinstance(x,(list,tuple)):return type(x)==type(y) and len(x)==len(y) and all(equal(u,v) for u,v in zip(x,y))
 return x==y
keys=['model','opt','sampler','it','transitions','lr'];checks={k:equal(a[k],b[k]) for k in keys}
result=dict(exact_continuation=all(checks.values()),checks=checks,it=a['it'],transitions=a['transitions']);(h/'ppo-checkpoint-parity.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result));assert all(checks.values()),checks
