"""Stage an authoring-result memoizer; never edit active canonical generators."""

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parents[1] / "assembly_policy/so101/wowrobo_supported_poses.py"
original = SOURCE.read_text()
calculator = original.replace(
    "def supported_pose(", "def _calculate_supported_pose(", 1
).replace(
    'cache = Path(cache or "/private/tmp/wowrobo-rest-fields")',
    'cache = Path(cache or (Path(__file__).resolve().parents[2] / "outputs/portable-rest-fields"))',
    1,
)
# The original calculation is unchanged after the two declared I/O/name edits.
assert (
    calculator.replace(
        "def _calculate_supported_pose(", "def supported_pose(", 1
    ).replace(
        'cache = Path(cache or (Path(__file__).resolve().parents[2] / "outputs/portable-rest-fields"))',
        'cache = Path(cache or "/private/tmp/wowrobo-rest-fields")',
        1,
    )
    == original
)
wrapper = (HERE / "wrapper.txt").read_text()
wrapper = wrapper.replace(
    "__ORIGINAL_SHA__", hashlib.sha256(original.encode()).hexdigest()
)
patched = calculator + "\n\n" + wrapper
out = HERE / "wowrobo_supported_poses.py"
out.write_text(patched)
(HERE / "patch-proof.json").write_text(
    json.dumps(
        {
            "source": str(SOURCE),
            "original_sha256": hashlib.sha256(original.encode()).hexdigest(),
            "patched_sha256": hashlib.sha256(patched.encode()).hexdigest(),
            "calculator_body_unchanged_except_cache_path": True,
            "changes": [
                "portable private cache path",
                "atomic exact-input deterministic authoring-result memoization",
            ],
            "physical_admission_cache": False,
        },
        indent=2,
    )
    + "\n"
)
print(out)
