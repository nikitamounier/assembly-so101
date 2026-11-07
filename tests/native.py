"""Direct construction of the canonical native physics fixture."""

import mujoco
import numpy as np

from assembly_policy.so101.robot import home
from assembly_policy.so101.scene import ROOT as ROOT
from assembly_policy.so101.scene import model


def load_scene():
    m = model(contact_monitor=True)
    d = mujoco.MjData(m)
    home(m, d)
    return m, dict(home_qpos=d.qpos.copy(), home_ctrl=d.ctrl.copy())


def fixture_state(model, saved):
    q = np.asarray(saved["qpos"], float)
    v = np.asarray(saved.get("qvel", []), float)
    if len(q) != model.nq or len(v) not in (0, model.nv):
        raise ValueError("Fixture does not match native dimensions")
    return q, v if len(v) else np.zeros(model.nv)
