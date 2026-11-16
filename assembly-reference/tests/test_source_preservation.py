"""The real upstream source gate accepts portability, not equation changes."""
from pathlib import Path
import subprocess
import pytest
from scripts.so101_source_checks import sdf_equations_unchanged

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def sources():
    original=subprocess.check_output(['git','show','eb5d15e004591d380c0d521da67e025ebee37703:plugin/assemblenist_sdf.cc'],cwd=ROOT,text=True)
    return original,(ROOT/'plugin/assemblenist_sdf.cc').read_text()


def test_actual_portable_plugin_preserves_upstream_equations(sources):
    assert sdf_equations_unchanged(*sources)


@pytest.mark.parametrize('old,new',[
    ('constexpr double kTan60 = 1.7320508075688772;','constexpr double kTan60 = 1.;'),
    ('const double h = G->h;','const double h = 2*G->h;'),
    ('return static_cast<float>(value);','return 0.0f;'),
])
def test_equation_and_half_conversion_changes_are_still_rejected(sources,old,new):
    original,current=sources
    assert old in current
    assert not sdf_equations_unchanged(original,current.replace(old,new,1))
