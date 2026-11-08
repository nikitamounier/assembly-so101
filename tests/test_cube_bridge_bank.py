from pathlib import Path
import numpy as np
import mujoco
from assembly_policy.cube_trial.core import FAMILIES, model, placement_contacts


def test_bridge_bank_keeps_all_existing_rows_and_native_contact_stages():
    root = Path(__file__).resolve().parents[1]/'data/cube_trial'
    directory = root/'banks_handoff_bridge_v8'
    if not directory.exists():
        import pytest
        pytest.skip('Qualified bridge artifacts are not installed on this host')
    m=model();d=mujoco.MjData(m)
    for split,n in [('train',128),('eval',32)]:
        new=np.load(directory/f'cube_handoff_{split}.npz')
        old_path = root/'banks_policy_transport_v7'/f'cube_handoff_{split}.npz'
        # Historical predecessor banks are optional in the portable snapshot;
        # always retain native qualification of every new contact-state row.
        if old_path.exists():
            with np.load(old_path) as old:
                for key in old.files:
                    np.testing.assert_array_equal(new[key][:len(old[key])],old[key])
        for family in ['partial_right_contact','light_right_grasp']:
            ids=np.flatnonzero(new['family']==FAMILIES.index(family))
            assert len(ids)==n
            for state in new['states'][ids]:
                mujoco.mj_setState(m,d,state,mujoco.mjtState.mjSTATE_INTEGRATION)
                mujoco.mj_forward(m,d);c=placement_contacts(m,d);loads=c['pad_loads_n']
                assert min(loads[:2])>1 and not c['table_touching']
                left_z=0.0
                cube=m.geom('cube_geom').id
                for k,contact in enumerate(d.contact):
                    if cube not in (contact.geom1,contact.geom2):
                        continue
                    other=contact.geom2 if contact.geom1==cube else contact.geom1
                    if (m.geom(other).name or '').startswith('left_'):
                        wrench=np.empty(6);mujoco.mj_contactForce(m,d,k,wrench)
                        force=contact.frame.reshape(3,3).T@wrench[:3]*(1 if contact.geom2==cube else -1)
                        left_z+=force[2]
                assert left_z>.55*m.body_mass[m.body('cube').id]*abs(m.opt.gravity[2])
                if family=='partial_right_contact':
                    assert .015<loads[2]<.4 and loads[3]<=.001
                else:
                    assert min(loads[2:])>.05 and max(loads[2:])<3.5
