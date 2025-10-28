from __future__ import annotations

import pytest
from QPhaseSDE.visualizers.specs import PhasePortraitSpec


def test_phase_spec_validation_re_im_ok():
    s = PhasePortraitSpec(kind="re_im", modes=[0])
    assert s.kind == "re_im" and s.modes == [0]


def test_phase_spec_validation_abs_abs_bad_modes():
    with pytest.raises(Exception):
        PhasePortraitSpec(kind="abs_abs", modes=[0])
