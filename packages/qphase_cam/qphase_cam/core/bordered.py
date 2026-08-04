"""Bordered Lyapunov-Schmidt conditions for equilibrium multiplicity."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np


class ExactDynamics(Protocol):
    def rhs(self, vector: Any, params: dict[str, Any]) -> np.ndarray: ...

    def jacobian(self, vector: Any, params: dict[str, Any]) -> np.ndarray: ...

    def directional(
        self,
        order: int,
        vector: Any,
        params: dict[str, Any],
        *directions: Any,
    ) -> np.ndarray: ...


@dataclass(frozen=True)
class BorderedDiagnostics:
    coefficients: np.ndarray
    singular_values: np.ndarray
    right_null_vector: np.ndarray
    left_null_vector: np.ndarray


class BorderedMultiplicitySystem:
    """Square augmented system for a simple zero eigenvalue of order 2-4."""

    def __init__(
        self,
        dynamics: ExactDynamics,
        *,
        n_state: int,
        order: int,
        control_names: tuple[str, ...],
        base_params: dict[str, Any],
    ) -> None:
        if order not in {2, 3, 4}:
            raise ValueError("bordered multiplicity order must be 2, 3, or 4")
        if len(control_names) != order - 1:
            raise ValueError("multiplicity order m requires m-1 controls")
        self.dynamics = dynamics
        self.n_state = n_state
        self.order = order
        self.control_names = control_names
        self.base_params = dict(base_params)

    @property
    def size(self) -> int:
        return 3 * self.n_state + len(self.control_names)

    def residual(self, value: Any) -> np.ndarray:
        state, controls, right, left = self.unpack(value)
        params = self.params(controls)
        residual = self.dynamics.rhs(state, params)
        jacobian = self.dynamics.jacobian(state, params)
        coefficients = self.coefficients(state, params, right, left)
        return np.concatenate(
            (
                residual,
                jacobian @ right,
                jacobian.T @ left,
                np.asarray([np.dot(left, right) - 1.0]),
                coefficients[2 : self.order],
            )
        )

    def coefficients(
        self,
        state: np.ndarray,
        params: dict[str, Any],
        right: np.ndarray,
        left: np.ndarray,
    ) -> np.ndarray:
        jacobian = self.dynamics.jacobian(state, params)
        bordered = np.block(
            [
                [jacobian, right[:, None]],
                [left[None, :], np.zeros((1, 1))],
            ]
        )
        b_vv = self.dynamics.directional(
            2, state, params, right, right
        )
        coefficient_2 = float(left @ b_vv)
        h_2 = np.linalg.solve(
            bordered, np.concatenate((-b_vv, [0.0]))
        )[:-1]
        if self.order == 2:
            return np.asarray([0.0, 0.0, coefficient_2])
        t_vvv = self.dynamics.directional(
            3, state, params, right, right, right
        )
        b_vh2 = self.dynamics.directional(
            2, state, params, right, h_2
        )
        forcing_3 = t_vvv + 3.0 * b_vh2
        coefficient_3 = float(left @ forcing_3)
        if self.order < 4:
            return np.asarray([0.0, 0.0, coefficient_2, coefficient_3])
        h_3 = np.linalg.solve(
            bordered, np.concatenate((-forcing_3, [0.0]))
        )[:-1]
        q_vvvv = self.dynamics.directional(
            4, state, params, right, right, right, right
        )
        t_vvh2 = self.dynamics.directional(
            3, state, params, right, right, h_2
        )
        b_h2h2 = self.dynamics.directional(
            2, state, params, h_2, h_2
        )
        b_vh3 = self.dynamics.directional(
            2, state, params, right, h_3
        )
        forcing_4 = (
            q_vvvv
            + 6.0 * t_vvh2
            + 3.0 * b_h2h2
            + 4.0 * b_vh3
        )
        coefficient_4 = float(left @ forcing_4)
        return np.asarray(
            [0.0, 0.0, coefficient_2, coefficient_3, coefficient_4]
        )

    def seed(self, state: Any, controls: dict[str, float]) -> np.ndarray:
        vector = np.asarray(state, dtype=float).reshape(-1)
        control_values = np.asarray(
            [controls[name] for name in self.control_names]
        )
        params = self.params(control_values)
        jacobian = self.dynamics.jacobian(vector, params)
        left_vectors, _, right_vectors = np.linalg.svd(jacobian)
        right = right_vectors[-1].copy()
        left = left_vectors[:, -1].copy()
        overlap = float(left @ right)
        if abs(overlap) < 1e-12:
            raise np.linalg.LinAlgError(
                "left and right near-null vectors have negligible overlap"
            )
        right /= overlap
        return np.concatenate(
            (
                vector,
                np.asarray([controls[name] for name in self.control_names]),
                right,
                left,
            )
        )

    def diagnostics(self, value: Any) -> BorderedDiagnostics:
        state, controls, right, left = self.unpack(value)
        params = self.params(controls)
        jacobian = self.dynamics.jacobian(state, params)
        return BorderedDiagnostics(
            coefficients=self.coefficients(state, params, right, left),
            singular_values=np.linalg.svd(jacobian, compute_uv=False),
            right_null_vector=right,
            left_null_vector=left,
        )

    def unpack(
        self, value: Any
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        array = np.asarray(value, dtype=float).reshape(-1)
        if len(array) != self.size:
            raise ValueError(f"expected {self.size} bordered unknowns")
        n = self.n_state
        m = len(self.control_names)
        return (
            array[:n],
            array[n : n + m],
            array[n + m : 2 * n + m],
            array[2 * n + m :],
        )

    def params(self, controls: np.ndarray) -> dict[str, Any]:
        params = dict(self.base_params)
        params.update(zip(self.control_names, controls, strict=True))
        return params
