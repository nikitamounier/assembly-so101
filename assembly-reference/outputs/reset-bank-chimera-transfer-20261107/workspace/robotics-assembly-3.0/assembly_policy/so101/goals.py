"""Shared placement errors for policy rewards and task-completion inspection."""
import numpy as np
from .scene import ASSEMBLY_TASKS
from .fastener import target_rotations

RECEIVERS={'motor':'base','horn':'motor','screw':'horn','retainer':'base','yoke':'horn'}

def placement_errors(data, task, metadata, origins, rotations=None):
    """Require every mating transform, preserving predecessor/keypoint checks."""
    rotations=target_rotations(metadata) if rotations is None else rotations
    errors={}
    for name in ASSEMBLY_TASKS[:ASSEMBLY_TASKS.index(task)+1]:
        receiver=RECEIVERS[name];body=data.body(name);parent=data.body(receiver)
        rotation=parent.xmat.reshape(3,3)@rotations[receiver].T
        offset=origins[name]-origins[receiver]
        extent=np.ptp(np.asarray(metadata[name]['bounds']),axis=0)/2
        keypoints=np.array([[0,0,0],[extent[0],0,0],[0,extent[1],0],[0,0,extent[2]]])
        if name=='screw':keypoints=np.array([[0,0,0],[0,0,.003]])
        actual=body.xpos+keypoints@body.xmat.reshape(3,3).T
        goal=parent.xpos+(keypoints@rotations[name].T+offset)@rotation.T
        errors[name]=float(np.linalg.norm(actual-goal,axis=1).max())
    return errors
