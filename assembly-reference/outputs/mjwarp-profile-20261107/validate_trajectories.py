"""Apply the existing environment's fine-step velocity/depth guards to receipts."""
import json
import sys
from pathlib import Path

import numpy as np

root=Path(sys.argv[1]);out={}
free=(13,19,25,32,40)
folders=list(root.glob('candidate-w*-s80-tolNone-lsNone-cap512-j2048-selected'))
folders.append(root/'candidate-w64-s80-tolNone-lsNone-cap512-j2048-profile-contactaudit')
for folder in folders:
    with np.load(folder/'trajectory.npz') as z:
        states=z['states'];sensor=z['sensors'];v=states[:,:,52:]
        unsafe=(abs(v[:,:,:12])>5).any(axis=(1,2))
        max_linear=max_angular=0.
        for j in free:
            linear=np.linalg.norm(v[:,:,j:j+3],axis=2)
            angular=np.linalg.norm(v[:,:,j+3:j+6],axis=2)
            max_linear=max(max_linear,float(linear.max()));max_angular=max(max_angular,float(angular.max()))
            unsafe|=(linear>1).any(axis=1)|(angular>30).any(axis=1)
        unsafe|=(sensor[:,:,0]<-.0005).any(axis=1)
        out[folder.name]={'finite': bool(np.isfinite(states).all() and np.isfinite(sensor).all()),
                             'unsafe_worlds': int(unsafe.sum()),'max_arm_speed': float(abs(v[:,:,:12]).max()),
                             'max_free_linear_speed': max_linear,'max_free_angular_speed': max_angular,
                             'peak_penetration_m': float(max(0,-sensor.min()))}
        assert out[folder.name]['finite'] and not unsafe.any(),out[folder.name]
print(json.dumps(out,indent=2))
