"""Visual camera roundoff must not split portable identities or round physics."""

import copy
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from assembly_policy.so101 import workcell
from assembly_policy.so101.resets import signature
from assembly_policy.so101.scene import build_xml


@pytest.mark.parametrize("direction", [-np.inf, np.inf])
def test_camera_xml_is_stable_across_one_ulp_norm_roundoff(monkeypatch, direction):
    baseline = build_xml()
    axes = workcell.axes
    monkeypatch.setattr(workcell, "axes", lambda *a, **kw: np.nextafter(axes(*a, **kw), direction))
    assert build_xml() == baseline


def test_physical_mount_precision_and_identity_are_preserved(monkeypatch):
    baseline = ET.fromstring(build_xml())
    before = signature()
    cfg = copy.deepcopy(workcell.configuration())
    cfg["mounts"]["left"]["position"][0] = float(np.nextafter(cfg["mounts"]["left"]["position"][0], np.inf))
    monkeypatch.setattr(workcell, "configuration", lambda: cfg)
    actual = ET.fromstring(build_xml())
    assert actual.find('.//body[@name="left_base"]').get("pos") != baseline.find(
        './/body[@name="left_base"]'
    ).get("pos")
    assert signature() != before
