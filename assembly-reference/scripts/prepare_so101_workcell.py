"""Prepare versioned reset/reference data for the current shared source and assets."""
import argparse
import json
import subprocess
import sys
import mujoco
import numpy as np
from assembly_policy.so101.scene import model,ROOT
from assembly_policy.so101.robot import home
from assembly_policy.so101.resets import check_bank,signature
from assembly_policy.so101.workcell import configuration,default_bank_directory,assert_matches_reference

ARRAYS=('body_pos','body_quat','body_mass','body_inertia','body_ipos','body_iquat',
        'jnt_type','jnt_axis','jnt_range','dof_damping','geom_type','geom_size',
        'geom_pos','geom_quat','geom_contype','geom_conaffinity','geom_friction',
        'geom_solref','geom_solimp','actuator_gainprm','actuator_biasprm',
        'actuator_forcerange','actuator_ctrlrange','mesh_vert','mesh_face',
        'cam_pos','cam_quat','cam_fovy','pair_geom1','pair_geom2','pair_dim',
        'pair_friction','pair_solref','pair_solimp','qpos0','exclude_signature','numeric_data')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--count',type=int,default=32);p.add_argument('--workers',type=int,default=2)
    a=p.parse_args();bank=default_bank_directory()
    if bank.exists():check_bank(bank)
    else:
        subprocess.run([sys.executable,'-m','assembly_policy.so101.resets','--out',str(bank),
                        '--count',str(a.count),'--workers',str(a.workers)],cwd=ROOT,check=True)
    m=model(contact_monitor=True);d=mujoco.MjData(m);home(m,d)
    prefix=ROOT/configuration()['reference_prefix'];prefix.parent.mkdir(parents=True,exist_ok=True)
    identity=signature()
    if prefix.with_suffix('.json').exists():
        saved=json.loads(prefix.with_suffix('.json').read_text())
        if saved['model_signature']!=identity:raise ValueError('Reference exists for another environment; select a new reference_prefix')
        assert_matches_reference(m)
        print(json.dumps(dict(bank=str(bank),reference=str(prefix),model_signature=identity,existing_reference_verified=True)),flush=True)
        return
    arrays={name:np.asarray(getattr(m,name)) for name in ARRAYS};arrays['home']=d.qpos
    np.savez_compressed(prefix.with_suffix('.npz'),**arrays)
    report=dict(model_signature=identity,
                scope='Frozen shared-factory baseline; independent training/Quest entrypoint parity is checked separately.',
                counts={k:int(getattr(m,k)) for k in ('nq','nv','nu','ngeom','nbody','ncam','npair','neq')},
                options={k:getattr(m.opt,k) for k in ('timestep','sdf_initpoints','sdf_iterations','iterations','tolerance','noslip_iterations','noslip_tolerance','enableflags','disableflags')})
    prefix.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(bank=str(bank),reference=str(prefix),model_signature=identity)),flush=True)


if __name__=='__main__':main()
