import json
from assembly_policy.so101.wowrobo_physical_resets import ROOT, build_physical_profile

if __name__ == '__main__':
    print(json.dumps(build_physical_profile(
        ROOT/'assets_local/wowrobo/photo-resets-repaired-cover-support-v2',
        ROOT/'outputs/equilibrium-model-20261107/profile-bearing-v3',
        grid_manifest=ROOT/'outputs/insertion-fix-20261107/final-common-v1/grids/manifest.json',
        pin_mesh=ROOT/'assets_local/wowrobo/components-repaired-v1/meshes/pin.obj',
        pin_registration_path=ROOT/'outputs/insertion-fix-20261107/pin/consistent-v3/registration.json',
        completed_anchor_bank=ROOT/'assets_local/wowrobo/photo-resets-repaired-cover-support-v2/completed',
        mating_profile='legacy_mapped_all',
        pin_pose_registered=True,
        passive_motor_stops=True,
        exclude_internal_servo_contacts=True,
        helper_grid_profile=ROOT/'assets_local/wowrobo/physical-resets-repaired-cover-support-v2',
    ),indent=2))
