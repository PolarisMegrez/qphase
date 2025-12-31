"""Test decentralized architecture imports."""

import pytest


def test_protocol_imports():
    """Test that all protocols can be imported from their new locations."""
    # Test backend protocols
    from qphase.backend.base import BackendBase

    assert BackendBase is not None

    # Test state protocols
    try:
        from qphase_sde.states.base import StateBase, TrajectorySetBase

        assert StateBase is not None
        assert TrajectorySetBase is not None

        # Test integrator protocols
        from qphase_sde.integrator.base import Integrator

        assert Integrator is not None
    except ImportError:
        pytest.skip("qphase_sde not installed")
    assert Integrator is not None

    # Test model protocols
    from qphase_sde.models.base import NoiseSpec, SDEModel

    assert SDEModel is not None
    assert NoiseSpec is not None
