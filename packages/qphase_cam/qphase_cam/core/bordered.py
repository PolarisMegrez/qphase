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

    def mpmath_rhs(self, vector: Any, params: dict[str, Any]) -> Any: ...

    def mpmath_jacobian(self, vector: Any, params: dict[str, Any]) -> Any: ...

    def mpmath_directional(
        self,
        order: int,
        vector: Any,
        params: dict[str, Any],
        *directions: Any,
    ) -> Any: ...


@dataclass(frozen=True)
class BorderedDiagnostics:
    coefficients: np.ndarray
    singular_values: np.ndarray
    right_null_vector: np.ndarray
    left_null_vector: np.ndarray


@dataclass(frozen=True)
class BorderedVerificationOutcome:
    value: np.ndarray
    digits: int
    residual_norm: float
    success: bool
    decimal_values: tuple[str, ...] = ()


class BorderedMultiplicitySystem:
    """Gauge-fixed bordered system for a simple zero eigenvalue of order 2-4."""

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
                np.asarray([np.dot(right, right) - 1.0]),
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
        b_vv = self.dynamics.directional(2, state, params, right, right)
        coefficient_2 = float(left @ b_vv)
        h_2 = self._linear_solve(bordered, np.concatenate((-b_vv, [0.0])))[:-1]
        if self.order == 2:
            return np.asarray([0.0, 0.0, coefficient_2])
        t_vvv = self.dynamics.directional(3, state, params, right, right, right)
        b_vh2 = self.dynamics.directional(2, state, params, right, h_2)
        forcing_3 = t_vvv + 3.0 * b_vh2
        coefficient_3 = float(left @ forcing_3)
        if self.order < 4:
            return np.asarray([0.0, 0.0, coefficient_2, coefficient_3])
        h_3 = self._linear_solve(bordered, np.concatenate((-forcing_3, [0.0])))[:-1]
        q_vvvv = self.dynamics.directional(4, state, params, right, right, right, right)
        t_vvh2 = self.dynamics.directional(3, state, params, right, right, h_2)
        b_h2h2 = self.dynamics.directional(2, state, params, h_2, h_2)
        b_vh3 = self.dynamics.directional(2, state, params, right, h_3)
        forcing_4 = q_vvvv + 6.0 * t_vvh2 + 3.0 * b_h2h2 + 4.0 * b_vh3
        coefficient_4 = float(left @ forcing_4)
        return np.asarray([0.0, 0.0, coefficient_2, coefficient_3, coefficient_4])

    def seed(self, state: Any, controls: dict[str, float]) -> np.ndarray:
        vector = np.asarray(state, dtype=float).reshape(-1)
        control_values = np.asarray([controls[name] for name in self.control_names])
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
        left /= overlap
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

    def verify(
        self,
        value: Any,
        *,
        initial_digits: int,
        max_digits: int,
    ) -> BorderedVerificationOutcome:
        import mpmath as mp

        previous_digits = mp.mp.dps
        current = tuple(mp.mpf(str(float(item))) for item in np.asarray(value))
        digits = initial_digits
        try:
            while True:
                mp.mp.dps = digits
                functions = tuple(
                    (
                        lambda *items, index=index: self._mp_residual(
                            items, include_gauge=True
                        )[index]
                    )
                    for index in range(self.size + 1)
                )
                try:
                    solved = tuple(
                        mp.findroot(
                            functions,
                            current,
                            tol=mp.power(10, -(digits - 10)),
                            maxsteps=100,
                            verify=True,
                            solver="mdnewton",
                        )
                    )
                    residuals = self._mp_residual(solved, include_gauge=True)
                    residual = mp.sqrt(sum(item * item for item in residuals))
                    residual_float = float(residual)
                    success = residual_float <= 10.0 ** (-min(30, digits // 2))
                    if success or digits >= max_digits:
                        return BorderedVerificationOutcome(
                            value=np.asarray([float(item) for item in solved]),
                            digits=digits,
                            residual_norm=residual_float,
                            success=success,
                            decimal_values=tuple(
                                mp.nstr(item, n=digits) for item in solved
                            ),
                        )
                    current = solved
                except (
                    ArithmeticError,
                    TypeError,
                    ValueError,
                    ZeroDivisionError,
                ):
                    if digits >= max_digits:
                        return BorderedVerificationOutcome(
                            value=np.asarray(current, dtype=float),
                            digits=digits,
                            residual_norm=np.inf,
                            success=False,
                            decimal_values=tuple(
                                mp.nstr(item, n=digits) for item in current
                            ),
                        )
                digits = min(2 * digits, max_digits)
        finally:
            mp.mp.dps = previous_digits

    def _mp_residual(self, value: Any, *, include_gauge: bool = False) -> list[Any]:
        import mpmath as mp

        array = tuple(value)
        n = self.n_state
        m = len(self.control_names)
        state = array[:n]
        controls = array[n : n + m]
        right = mp.matrix(array[n + m : 2 * n + m])
        left = mp.matrix(array[2 * n + m :])
        params = self.params(controls)
        residual = self.dynamics.mpmath_rhs(state, params)
        jacobian = self.dynamics.mpmath_jacobian(state, params)
        coefficients = self._mp_coefficients(state, params, jacobian, right, left)
        output = [*residual, *(jacobian * right), *(jacobian.T * left)]
        output.append((left.T * right)[0] - 1)
        output.extend(coefficients[2 : self.order])
        if include_gauge:
            output.append((right.T * right)[0] - 1)
        return output

    def _mp_coefficients(
        self,
        state: Any,
        params: dict[str, Any],
        jacobian: Any,
        right: Any,
        left: Any,
    ) -> list[Any]:
        import mpmath as mp

        n = self.n_state
        bordered = mp.matrix(n + 1)
        for row in range(n):
            for column in range(n):
                bordered[row, column] = jacobian[row, column]
            bordered[row, n] = right[row]
            bordered[n, row] = left[row]
        b_vv = self.dynamics.mpmath_directional(2, state, params, right, right)
        coefficient_2 = (left.T * b_vv)[0]
        h_2 = mp.lu_solve(bordered, mp.matrix([*list(-b_vv), mp.mpf("0")]))[:n]
        coefficients = [mp.mpf("0"), mp.mpf("0"), coefficient_2]
        if self.order == 2:
            return coefficients
        forcing_3 = self.dynamics.mpmath_directional(
            3, state, params, right, right, right
        ) + 3 * self.dynamics.mpmath_directional(2, state, params, right, h_2)
        coefficients.append((left.T * forcing_3)[0])
        if self.order < 4:
            return coefficients
        h_3 = mp.lu_solve(bordered, mp.matrix([*list(-forcing_3), mp.mpf("0")]))[:n]
        forcing_4 = (
            self.dynamics.mpmath_directional(
                4, state, params, right, right, right, right
            )
            + 6 * self.dynamics.mpmath_directional(3, state, params, right, right, h_2)
            + 3 * self.dynamics.mpmath_directional(2, state, params, h_2, h_2)
            + 4 * self.dynamics.mpmath_directional(2, state, params, right, h_3)
        )
        coefficients.append((left.T * forcing_4)[0])
        return coefficients

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

    @staticmethod
    def _linear_solve(matrix: np.ndarray, right: np.ndarray) -> np.ndarray:
        try:
            return np.linalg.solve(matrix, right)
        except np.linalg.LinAlgError:
            solution, *_ = np.linalg.lstsq(matrix, right, rcond=None)
            return solution

    def params(self, controls: np.ndarray) -> dict[str, Any]:
        params = dict(self.base_params)
        params.update(zip(self.control_names, controls, strict=True))
        return params
