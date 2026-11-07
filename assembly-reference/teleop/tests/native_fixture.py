"""Load diagnostic fixtures only when they match the current native dimensions."""
import numpy as np


def fixture_state(model, saved):
    qpos=np.asarray(saved['qpos'],float);qvel=np.asarray(saved.get('qvel',[]),float)
    if len(qpos)!=model.nq or len(qvel) not in (0,model.nv):
        raise ValueError('Diagnostic fixture does not match the native model')
    return qpos,qvel if len(qvel) else np.zeros(model.nv)
