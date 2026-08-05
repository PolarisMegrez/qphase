"""Bordered Lyapunov-Schmidt multiplicity condition tests."""

from __future__ import annotations

from math import factorial
from typing import Any

import numpy as np
import pytest
from qphase_cam.core.bordered import BorderedMultiplicitySystem


class ScalarNormalForm:
    """Scalar A_m normal form plus one regular stable coordinate."""

    def __init__(self, order: int) -> None:
        self.order = order

    def rhs(self, vector: Any, params: dict[str, Any]) -> np.ndarray:
        x, y = np.asarray(vector)
        value = x**self.order
        for power in range(self.order - 1):
            value += params[f"p{power}"] * x**power
        return np.asarray([value, -y])

    def jacobian(self, vector: Any, params: dict[str, Any]) -> np.ndarray:
        x, _ = np.asarray(vector)
        derivative = self.order * x ** (self.order - 1)
        for power in range(1, self.order - 1):
            derivative += power * params[f"p{power}"] * x ** (power - 1)
        return np.asarray([[derivative, 0.0], [0.0, -1.0]])

    def directional(
        self,
        order: int,
        vector: Any,
        params: dict[str, Any],
        *directions: Any,
    ) -> np.ndarray:
        x, _ = np.asarray(vector)
        derivative = 0.0
        terms = [(self.order, 1.0)]
        terms.extend((power, params[f"p{power}"]) for power in range(self.order - 1))
        for power, coefficient in terms:
            if power >= order:
                derivative += (
                    coefficient
                    * factorial(power)
                    / factorial(power - order)
                    * x ** (power - order)
                )
        contraction = np.prod([np.asarray(direction)[0] for direction in directions])
        return np.asarray([derivative * contraction, 0.0])


@pytest.mark.parametrize("order", [2, 3, 4])
def test_bordered_system_recognizes_scalar_normal_forms(order):
    controls = tuple(f"p{index}" for index in range(order - 1))
    params = {name: 0.0 for name in controls}
    system = BorderedMultiplicitySystem(
        ScalarNormalForm(order),
        n_state=2,
        order=order,
        control_names=controls,
        base_params=params,
    )
    seed = system.seed(np.zeros(2), params)
    np.testing.assert_allclose(system.residual(seed), 0.0, atol=1e-14)
    diagnostics = system.diagnostics(seed)
    np.testing.assert_allclose(diagnostics.coefficients[order], factorial(order))
    np.testing.assert_allclose(diagnostics.singular_values, [1.0, 0.0])
