"""Parent handoff: bind proven v3 mechanics to the FINAL framed shared scene.

No new run occurs on import. Keep final-common case SDF transforms exactly once.
Only the length-negative swaps pin visual/SDF, retaining every final receiver.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import xml.etree.ElementTree as ET

ROOT = Path('/Users/nikitamounier/Documents/ChatGPT/robot assembly 2/robotics-assembly-3.0')
V3 = ROOT / 'outputs/insertion-fix-20261107/pin/consistent-v3'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def negative_manifest(final_manifest, output):
    base = json.loads(final_manifest.read_text())
    replacement = json.loads((V3.parent / 'fit-grids/manifest.json').read_text())['meshes']['pin']
    folder = output / 'length-negative-grids'
    folder.mkdir(parents=True, exist_ok=True)
    for name, original in list(base['meshes'].items()):
        record = dict(replacement) if name == 'pin' else dict(original)
        record['id'] = original['id']
        record['grid_id'] = original.get('grid_id', original['id'])
        destination = folder / f"grid_{record['grid_id']}.bin"
        if not destination.exists():
            os.link(record['grid_file'], destination)
        record['grid_file'] = str(destination)
        base['meshes'][name] = record
    base.update(grid_directory=str(folder), grid_dir=str(folder),
                base_final_manifest_sha256=sha(final_manifest),
                changed_fields=['pin'],
                scope='Same FINAL receivers/solver; only v2 length14.5 pin counterfactual')
    path = folder / 'manifest.json'
    path.write_text(json.dumps(base, indent=2) + '\n')
    return path


def run(args):
    final_root = args.common_root.resolve()
    output = args.output_root.resolve()
    final_manifest = final_root / 'grids/manifest.json'
    negative_grids = output / 'length-negative-grids/manifest.json'
    if args.kind == 'length-negative':
        negative_grids = negative_manifest(final_manifest, output)
    source_script = V3 / 'run_consistent_pin.py'
    code = source_script.read_text()
    substitutions = {
        'out=OUT/kind;': 'out=FINAL_OUT/kind;',
        "grid=OUT/'grids/manifest.json'": 'grid=FINAL_GRIDS',
        "grid=OUT.parent/'fit-grids/manifest.json'": 'grid=NEGATIVE_GRIDS',
        "mesh=OUT/'assets/pin.obj'": "mesh=FINAL_ROOT/'assets/pin.obj'",
        "tree=ET.parse(source/'scene.xml')": "tree=ET.parse(FINAL_ROOT/'source-scene.xml')",
    }
    for before, after in substitutions.items():
        if code.count(before) != 1:
            raise RuntimeError(f'Frozen v3 source changed: {before}')
        code = code.replace(before, after)
    namespace = {'__name__': 'parent_pin_final_handoff', '__file__': str(source_script),
                 'FINAL_OUT': output, 'FINAL_ROOT': final_root,
                 'FINAL_GRIDS': final_manifest, 'NEGATIVE_GRIDS': negative_grids}
    exec(compile(code, str(source_script), 'exec'), namespace)

    def copy_framed_backend(full_source, grids, destination):
        # NEVER call prepare_scene here: that would lose the final case reframe.
        from assembly_policy.so101.wowrobo_contact_backend import load_backend
        load_backend(grids)
        tree = ET.parse(final_root / 'scene.xml')
        pin_pairs = [p for p in tree.getroot().findall('contact/pair')
                     if 'pin__contact' in (p.get('geom1'), p.get('geom2'))]
        if pin_pairs:
            raise RuntimeError('FINAL scene already has pin pairs; avoid duplicate profile application')
        pin_file = ET.parse(full_source).find("asset/mesh[@name='pin']").get('file')
        tree.find("asset/mesh[@name='pin']").set('file', pin_file)
        tree.write(destination, encoding='unicode')
        return {'final_scene_sha256': sha(final_root / 'scene.xml'),
                'case_frames_preserved_exactly_once': True,
                'pin_visual_override': pin_file}

    namespace['prepare_scene'] = copy_framed_backend
    namespace['main'](args.kind, resume_drive=False, full_case=True)
    folder = output / args.kind
    receipt = json.loads((folder / 'receipt.json').read_text())
    expected = json.loads((V3 / 'registration.json').read_text())
    registration = receipt['registration']
    delta = registration['head_stop_axis_projection_m'] - expected['head_stop_axis_projection_m']
    binding = {'kind': args.kind,
               'final_source_scene_sha256': sha(final_root / 'source-scene.xml'),
               'final_framed_scene_sha256': sha(final_root / 'scene.xml'),
               'final_grid_manifest_sha256': sha(final_manifest),
               'used_grid_manifest': str(final_manifest if args.kind == 'fit' else negative_grids),
               'head_stop_delta_from_v3_m': delta,
               'protected_head_stop_pass': abs(delta) < 2e-6,
               'actual_stop_to_cap_span_m': registration['stop_to_blind_end_m'],
               'case_reframe_preserved_no_rebuild': True,
               'source_script_sha256': sha(source_script),
               'handoff_script_sha256': sha(Path(__file__)),
               'scope': 'Native pin test in coherent engineered receiver targets; whole assembly qualification separate'}
    (folder / 'final-common-binding.json').write_text(json.dumps(binding, indent=2) + '\n')
    if args.kind == 'fit':
        spec = json.loads((folder / 'render-spec.json').read_text())
        spec.update(output=str(folder / 'insertion.mp4'),
                    title='PIN | final common snapshot | 2.3 x14.0mm',
                    method='1 N balanced exterior push | explicit axial guide | coherent receivers',
                    hide_helpers=True, distance=.22, lookat=[.0015, 0, .09],
                    force={'piece': 'assembly_pin', 'axis_world': receipt['guide']['axis_world'],
                           'ramp_s': .5, 'maximum_N': 1., 'off_s': 8.})
        seated = abs(receipt['head_gap_mm']) < .05
        spec['result'] = 'Native head seated; free-gravity release tested separately' if seated else 'NOT SEATED: inspect native receipt'
        (folder / 'render-spec.json').write_text(json.dumps(spec, indent=2) + '\n')
    print(json.dumps(binding, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--kind', choices=['fit', 'length-negative'], required=True)
    parser.add_argument('--common-root', type=Path, default=ROOT / 'outputs/insertion-fix-20261107/final-common-v1')
    parser.add_argument('--output-root', type=Path, default=V3.parent / 'final-common-proof')
    run(parser.parse_args())
