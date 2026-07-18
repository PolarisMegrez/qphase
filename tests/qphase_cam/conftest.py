"""Shared CAM test models."""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
import pytest


class NoJacobianModel:
    name: ClassVar[str] = "no_jacobian"
    n_modes = 2
    steady_state_capacity: ClassVar[int] = 1
    params: dict[str, Any] = {}

    def cam_hamiltonian(self, state, params):
        del params
        return np.broadcast_to(-0.5j * np.eye(2), np.asarray(state).shape)

    def cam_diffusion(self, state, params):
        del params
        return np.broadcast_to(np.eye(2), np.asarray(state).shape)

    def cam_solution_sort_key(self, state, params):
        del params
        return float(np.real(state[0, 0]))


@pytest.fixture
def no_jacobian_model():
    return NoJacobianModel()
