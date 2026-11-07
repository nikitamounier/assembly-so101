"""Canonicalize only visualization camera vectors across CPU architectures."""

import sys
from pathlib import Path


def patch(path):
    text = Path(path).read_text()
    helper = '''def camera_numbers(values):
    """Keep camera-only floating-point noise out of portable model identities."""
    return " ".join(format(0.0 if abs(float(v)) < 1e-14 else float(v), ".12g") for v in np.asarray(values).ravel())


'''
    if "def camera_numbers(" not in text:
        assert text.count("def axes(") == 1
        text = text.replace("def axes(", helper + "def axes(")
        for old, new in [
            ('pos=numbers(camera["position"])', 'pos=camera_numbers(camera["position"])'),
            (
                'xyaxes=numbers(axes(camera["position"], camera["look_at"]))',
                'xyaxes=camera_numbers(axes(camera["position"], camera["look_at"]))',
            ),
            ("pos=numbers(eye)", "pos=camera_numbers(eye)"),
            (
                "xyaxes=numbers(axes(eye, target, rotation[:, 2]))",
                "xyaxes=camera_numbers(axes(eye, target, rotation[:, 2]))",
            ),
        ]:
            assert text.count(old) == 1, old
            text = text.replace(old, new)
        Path(path).write_text(text)


if __name__ == "__main__":
    for path in sys.argv[1:]:
        patch(path)
