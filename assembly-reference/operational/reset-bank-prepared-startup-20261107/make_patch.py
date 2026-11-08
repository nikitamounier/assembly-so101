from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).parent
p=ROOT/'assembly_policy/so101/wowrobo_photo_resets.py';old=p.read_text();s=old.replace('def __init__(self, assets, plan=PLAN):','def __init__(self, assets, plan=PLAN, prepared_resting=None):',1)
a=s.index('        def piece_mesh(');b=s.index('        self.order = [',a)
block=s[a:b]
s=s[:a]+'''        self.prepared_for_sampling = prepared_resting is not None
        if self.prepared_for_sampling:
            required = {"upper_arm_module", "forearm_module"}
            if set(prepared_resting) != required:
                raise ValueError("Prepared sampling requires both transport far markers")
            self.resting = {
                name: {"far_side_supported": {"far_point": np.asarray(marker, dtype=float).copy()}}
                for name, marker in prepared_resting.items()
            }
            if any(value["far_side_supported"]["far_point"].shape != (3,)
                   or not np.isfinite(value["far_side_supported"]["far_point"]).all()
                   for value in self.resting.values()):
                raise ValueError("Invalid prepared far marker")
        else:
'''+''.join('    '+line if line.strip() else line for line in block.splitlines(True))+s[b:]
a=s.index('        self.refiner = GraspRefiner(');b=s.index('        self.previous = {',a)
block=s[a:b]
s=s[:a]+'''        if self.prepared_for_sampling:
            self.refiner = None
            self.support_cache = {}
            self.lookahead = {}
        else:
'''+''.join('    '+line if line.strip() else line for line in block.splitlines(True))+s[b:]
(HERE/p.name).write_text(s)
proof={p.name:{'source_sha256':hashlib.sha256(old.encode()).hexdigest(),'patch_sha256':hashlib.sha256(s.encode()).hexdigest(),'kept_initial_grasp_layout_block_exact':old[old.index('        self.layout ='):old.index('        def piece_mesh(')]==s[s.index('        self.layout ='):s.index('        self.prepared_for_sampling')],'runtime_math_methods_unchanged':old[old.index('    def future_obstacles'):]==s[s.index('    def future_obstacles'):]}}
p=ROOT/'assembly_policy/so101/wowrobo_randomization.py';old=p.read_text();s=old
needle='''        self.sequence = PhotoResetSequence(self.assets)'''
repl='''        self.nominal_receipts = {
            d["id"]: d
            for d in json.loads((self.nominal_bank / "stage-receipts.json").read_text())
        }
        prepared_resting = None
        if (self.config.get("grasp_library") and self.config.get("use_nominal_model")
                and self.config.get("cad_revision")):
            markers = {}
            for stage, piece in (("20_upper_transport", "upper_arm_module"),
                                 ("28_final_transport", "forearm_module")):
                marker = self.nominal_receipts.get(stage, {}).get("held_body_sag", {}).get("far_point_piece_local_m")
                if marker is not None:
                    markers[piece] = marker
            if len(markers) == 2:
                prepared_resting = markers
        self.sequence = PhotoResetSequence(self.assets, prepared_resting=prepared_resting)'''
assert needle in s;s=s.replace(needle,repl,1)
# Remove only the later duplicate read. The values and original object contents
# are unchanged; the receipt is now available before unnecessary authoring.
needle='''        self.definitions = {d["id"]: d for d in self.sequence.plan["states"]}
        self.nominal_receipts = {
            d["id"]: d
            for d in json.loads((self.nominal_bank / "stage-receipts.json").read_text())
        }'''
s=s.replace(needle,'''        self.definitions = {d["id"]: d for d in self.sequence.plan["states"]}''',1)
(HERE/p.name).write_text(s)
proof[p.name]={'source_sha256':hashlib.sha256(old.encode()).hexdigest(),'patch_sha256':hashlib.sha256(s.encode()).hexdigest(),'sample_and_ik_math_unchanged':old[old.index('    def restore('):]==s[s.index('    def restore('):]}
(HERE/'patch-proof.json').write_text(json.dumps(proof,indent=2)+'\n')
print(json.dumps(proof,indent=2))
