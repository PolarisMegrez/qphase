"""Test decentralized architecture imports."""

import pytest

pytestmark = pytest.mark.integration


def test_protocol_imports():
    """Test that public protocols can be imported from their canonical locations."""
    # Core backend protocols
    from qphase.backend.base import BackendBase

    assert BackendBase is not None

    # qphase_sde state containers
    from qphase_sde.state import State, TrajectorySet

    assert State is not None
    assert TrajectorySet is not None

    # qphase_sde integrator protocols
    from qphase_sde.integrator.base import Integrator

    assert Integrator is not None

    # qphase_sde model protocols
    from qphase_sde.model import NoiseSpec, SDEModel

    assert SDEModel is not None
    assert NoiseSpec is not None
