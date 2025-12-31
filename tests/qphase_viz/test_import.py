"""Test qphase_viz package."""

import importlib.util

import pytest


def test_viz_import():
    """Test that qphase_viz can be imported."""
    spec = importlib.util.find_spec("qphase_viz")
    if spec is None:
        pytest.fail("Could not find qphase_viz module")
