"""Numerical evaluators for exact fpgen scalar reductions."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import comb, factorial, isfinite
from typing import Any

import numpy as np
import sympy as sp
from scipy.optimize import brentq

from qphase_cam.errors import BifurcationCapabilityError


@dataclass(frozen=True)
class ReductionDiagnostics:
    regularity_determinant: float
    denominator_margin: float
    condition_number: float
    reduced_coefficients: np.ndarray


class FractionFreeScalarReduction:
    """Compiled regular-branch equations from a materialized fpgen reduction."""

    method = "reduced_fraction_free"

    def __init__(
        self,
        materialized: Any,
        *,
        order: int,
        control_names: tuple[str, ...],
        base_params: dict[str, Any],
    ) -> None:
        plan = materialized.plan
        if len(plan.retained_symbols) != 1 or len(materialized.numerators) != 1:
            raise BifurcationCapabilityError(
                "fraction-free bifurcation search requires a scalar reduction"
            )
        self.materialized = materialized
        self.plan = plan
        self.order = order
        self.q = plan.retained_symbols[0]
        self.parameter_specs = tuple(plan.dynamics.parameter_spec)
        self.parameter_symbols = tuple(item.symbol for item in self.parameter_specs)
        self.parameter_names = tuple(item.name for item in self.parameter_specs)
        self.control_names = control_names
        self.base_params = dict(base_params)
        unknown_symbols = [self.q]
        lookup = {item.name: item.symbol for item in self.parameter_specs}
        try:
            unknown_symbols.extend(lookup[name] for name in control_names)
        except KeyError as exc:
            raise BifurcationCapabilityError(
                f"unknown bifurcation control {exc.args[0]!r}"
            ) from exc
        self.unknown_symbols = tuple(unknown_symbols)
        numerator = materialized.numerators[0]
        self.condition_expressions = tuple(
            sp.diff(numerator, self.q, derivative)
            for derivative in range(order)
        )
        self.coefficient_expressions = tuple(
            sp.diff(numerator, self.q, derivative)
            for derivative in range(order + 1)
        )
        arguments = (self.q, *self.parameter_symbols)
        self._conditions = sp.lambdify(
            arguments, sp.Matrix(self.condition_expressions), modules="numpy"
        )
        search_jacobian = sp.Matrix(self.condition_expressions).jacobian(
            self.unknown_symbols
        )
        self._search_jacobian = sp.lambdify(
            arguments, search_jacobian, modules="numpy"
        )
        self._reconstruct = sp.lambdify(
            arguments, materialized.reconstruct_full_state(), modules="numpy"
        )
        self._coefficients = sp.lambdify(
            arguments, sp.Matrix(self.coefficient_expressions), modules="numpy"
        )
        self._denominators = sp.lambdify(
            arguments, materialized.denominators, modules="numpy"
        )
        self._cleared = sp.lambdify(
            arguments,
            sp.Matrix(materialized.cleared_factors),
            modules="numpy",
        )
        self._determinant = sp.lambdify(
            arguments, materialized.regularity_determinant, modules="numpy"
        )
        self._A = sp.lambdify(arguments, plan.A, modules="numpy")
        try:
            polynomial = sp.Poly(numerator, self.q)
        except sp.PolynomialError as exc:
            raise BifurcationCapabilityError(
                "the reduced numerator is not polynomial in its order parameter"
            ) from exc
        self.degree = int(polynomial.degree())
        self._polynomial_coefficients = sp.lambdify(
            self.parameter_symbols, polynomial.all_coeffs(), modules="numpy"
        )

    @property
    def retained_id(self) -> str:
        return self.plan.candidate.retained_ids[0]

    def equations(self, value: Any) -> np.ndarray:
        return self._array(self._conditions(*self._arguments(value)))

    def jacobian(self, value: Any) -> np.ndarray:
        return np.asarray(
            self._search_jacobian(*self._arguments(value)), dtype=float
        ).reshape((self.order, self.order))

    def reconstruct(self, value: Any) -> np.ndarray:
        return self._array(self._reconstruct(*self._arguments(value)))

    def diagnostics(self, value: Any) -> ReductionDiagnostics:
        arguments = self._arguments(value)
        determinant = float(np.asarray(self._determinant(*arguments)))
        denominators = self._array(self._denominators(*arguments))
        cleared = self._array(self._cleared(*arguments))
        margins = np.abs(np.concatenate((denominators, cleared)))
        matrix = np.asarray(self._A(*arguments), dtype=float)
        return ReductionDiagnostics(
            regularity_determinant=determinant,
            denominator_margin=float(np.min(margins)) if margins.size else np.inf,
            condition_number=float(np.linalg.cond(matrix)),
            reduced_coefficients=self._array(self._coefficients(*arguments)),
        )

    def initial_starts(
        self,
        control_bounds: tuple[tuple[float, float], ...],
        *,
        samples_per_control: int,
        max_starts: int,
        order_parameter_bounds: tuple[float, float] | None = None,
        order_parameter_samples: int = 41,
    ) -> list[np.ndarray]:
        del order_parameter_bounds, order_parameter_samples
        axes = [
            np.linspace(lower, upper, samples_per_control)
            for lower, upper in control_bounds
        ]
        starts: list[np.ndarray] = []
        for controls in product(*axes):
            params = self._params(np.asarray((0.0, *controls), dtype=float))
            coefficients = np.asarray(
                self._polynomial_coefficients(
                    *(params[name] for name in self.parameter_names)
                ),
                dtype=float,
            ).reshape(-1)
            first = np.flatnonzero(np.abs(coefficients) > 1e-14)
            if not first.size:
                continue
            for root in np.roots(coefficients[first[0] :]):
                if abs(root.imag) > 1e-8 * max(1.0, abs(root.real)):
                    continue
                q = float(root.real)
                if self.retained_id.startswith("r_diag_") and q < -1e-10:
                    continue
                starts.append(np.asarray((q, *controls), dtype=float))
                if len(starts) >= max_starts:
                    return starts
        return starts

    def _arguments(self, value: Any) -> tuple[float, ...]:
        array = np.asarray(value, dtype=float)
        params = self._params(array)
        parameter_values = (
            float(params[name]) for name in self.parameter_names
        )
        return (float(array[0]), *parameter_values)

    def _params(self, value: np.ndarray) -> dict[str, Any]:
        params = dict(self.base_params)
        params.update(
            zip(self.control_names, value[1:], strict=True)
        )
        return params

    @staticmethod
    def _array(value: Any) -> np.ndarray:
        output = np.asarray(value, dtype=float).reshape(-1)
        if not np.all(np.isfinite(output)):
            return np.full(output.shape, np.inf)
        return output


class CondensedScalarReduction:
    """Exact scalar reduction evaluated through implicit linear-solve jets."""

    degree: int | None = None
    method = "reduced_condensed"

    def __init__(
        self,
        plan: Any,
        *,
        order: int,
        control_names: tuple[str, ...],
        base_params: dict[str, Any],
    ) -> None:
        if len(plan.retained_symbols) != 1 or len(plan.retained_residual) != 1:
            raise BifurcationCapabilityError(
                "condensed bifurcation search requires a scalar reduction"
            )
        self.plan = plan
        self.order = order
        self.q = plan.retained_symbols[0]
        self.y = tuple(plan.eliminated_symbols)
        self.parameter_specs = tuple(plan.dynamics.parameter_spec)
        self.parameter_symbols = tuple(item.symbol for item in self.parameter_specs)
        self.parameter_names = tuple(item.name for item in self.parameter_specs)
        self.control_names = control_names
        self.base_params = dict(base_params)
        arguments = (self.q, *self.parameter_symbols)
        self._A_derivatives = tuple(
            sp.lambdify(
                arguments,
                plan.A.diff(self.q, derivative),
                modules="numpy",
            )
            for derivative in range(order + 1)
        )
        self._b_derivatives = tuple(
            sp.lambdify(
                arguments,
                plan.b.diff(self.q, derivative),
                modules="numpy",
            )
            for derivative in range(order + 1)
        )
        self._coefficient_functions = self._compile_coefficients(arguments)

    @property
    def retained_id(self) -> str:
        return self.plan.candidate.retained_ids[0]

    def equations(self, value: Any) -> np.ndarray:
        return self._evaluate(value)[1][: self.order]

    def jacobian(self, value: Any) -> np.ndarray:
        point = np.asarray(value, dtype=float)
        columns = []
        for index in range(len(point)):
            step = 2e-6 * max(1.0, abs(float(point[index])))
            offset = np.zeros_like(point)
            offset[index] = step
            columns.append(
                (self.equations(point + offset) - self.equations(point - offset))
                / (2.0 * step)
            )
        return np.stack(columns, axis=-1)

    def reconstruct(self, value: Any) -> np.ndarray:
        jets, _ = self._evaluate(value)
        values: dict[int, float] = {
            self.plan.candidate.retained_indices[0]: float(np.asarray(value)[0])
        }
        values.update(
            zip(self.plan.candidate.eliminated_indices, jets[0], strict=True)
        )
        return np.asarray([values[index] for index in range(len(values))])

    def diagnostics(self, value: Any) -> ReductionDiagnostics:
        arguments = self._arguments(value)
        matrix = np.asarray(self._A_derivatives[0](*arguments), dtype=float)
        _, coefficients = self._evaluate(value)
        return ReductionDiagnostics(
            regularity_determinant=float(np.linalg.det(matrix)),
            denominator_margin=np.inf,
            condition_number=float(np.linalg.cond(matrix)),
            reduced_coefficients=coefficients,
        )

    def initial_starts(
        self,
        control_bounds: tuple[tuple[float, float], ...],
        *,
        samples_per_control: int,
        max_starts: int,
        order_parameter_bounds: tuple[float, float] | None = None,
        order_parameter_samples: int = 41,
    ) -> list[np.ndarray]:
        q_bounds = order_parameter_bounds or self._default_q_bounds()
        q_axis = self._q_axis(q_bounds, order_parameter_samples)
        control_axes = [
            np.linspace(lower, upper, samples_per_control)
            for lower, upper in control_bounds
        ]
        starts: list[np.ndarray] = []
        for controls in product(*control_axes):
            values = []
            for q in q_axis:
                try:
                    residual = float(self.equations((q, *controls))[0])
                except (ValueError, np.linalg.LinAlgError, FloatingPointError):
                    residual = np.nan
                values.append(residual)
            for left, right, f_left, f_right in zip(
                q_axis[:-1], q_axis[1:], values[:-1], values[1:], strict=True
            ):
                if not np.isfinite(f_left) or not np.isfinite(f_right):
                    continue
                if f_left == 0.0:
                    root = float(left)
                elif np.signbit(f_left) == np.signbit(f_right):
                    continue
                else:
                    try:
                        root = float(
                            brentq(
                                lambda q, controls=controls: self.equations(
                                    (q, *controls)
                                )[0],
                                left,
                                right,
                            )
                        )
                    except (ValueError, np.linalg.LinAlgError):
                        continue
                candidate = np.asarray((root, *controls), dtype=float)
                if not any(np.linalg.norm(candidate - item) < 1e-8 for item in starts):
                    starts.append(candidate)
                if len(starts) >= max_starts:
                    return starts
        return starts

    def _evaluate(self, value: Any) -> tuple[list[np.ndarray], np.ndarray]:
        arguments = self._arguments(value)
        matrices = [
            np.asarray(function(*arguments), dtype=float)
            for function in self._A_derivatives
        ]
        vectors = [
            np.asarray(function(*arguments), dtype=float).reshape(-1)
            for function in self._b_derivatives
        ]
        jets = [np.linalg.solve(matrices[0], -vectors[0])]
        for derivative in range(1, self.order + 1):
            right = vectors[derivative].copy()
            for index in range(1, derivative + 1):
                right += (
                    comb(derivative, index)
                    * matrices[index]
                    @ jets[derivative - index]
                )
            jets.append(np.linalg.solve(matrices[0], -right))
        function_arguments: list[Any] = [*arguments]
        function_arguments.extend(item for jet in jets for item in jet)
        coefficients = np.asarray(
            [function(*function_arguments) for function in self._coefficient_functions],
            dtype=float,
        )
        return jets, coefficients

    def _compile_coefficients(self, arguments: tuple[Any, ...]) -> tuple[Any, ...]:
        time = sp.Symbol("_jet_t", real=True)
        jet_symbols = tuple(
            tuple(
                sp.Symbol(f"_jet_{derivative}_{index}", real=True)
                for index in range(len(self.y))
            )
            for derivative in range(self.order + 1)
        )
        replacements = {self.q: self.q + time}
        for index, symbol in enumerate(self.y):
            replacements[symbol] = sum(
                jet_symbols[derivative][index]
                * time**derivative
                / factorial(derivative)
                for derivative in range(self.order + 1)
            )
        expression = self.plan.retained_residual[0].subs(
            replacements, simultaneous=True
        )
        flattened_jets = tuple(item for row in jet_symbols for item in row)
        return tuple(
            sp.lambdify(
                (*arguments, *flattened_jets),
                sp.diff(expression, time, derivative).subs(time, 0),
                modules="numpy",
            )
            for derivative in range(self.order + 1)
        )

    def _arguments(self, value: Any) -> tuple[float, ...]:
        array = np.asarray(value, dtype=float)
        params = dict(self.base_params)
        params.update(zip(self.control_names, array[1:], strict=True))
        return (
            float(array[0]),
            *(float(params[name]) for name in self.parameter_names),
        )

    def _default_q_bounds(self) -> tuple[float, float]:
        nonlinear = max(
            (
                abs(float(self.base_params[name]))
                for name in ("chi", "Gamma")
                if name in self.base_params
            ),
            default=1.0,
        )
        extent = max(10.0, 10.0 / max(nonlinear, 1e-8))
        if self.retained_id.startswith("r_diag_"):
            return 0.0, extent
        return -extent, extent

    def _q_axis(
        self, bounds: tuple[float, float], samples: int
    ) -> np.ndarray:
        lower, upper = bounds
        if lower == 0.0 and upper > 0.0:
            linear = np.linspace(lower, upper, samples // 2 + 1)
            geometric = np.geomspace(
                max(upper * 1e-10, np.finfo(float).tiny),
                upper,
                samples - len(linear) + 1,
            )
            return np.unique(np.concatenate((linear, geometric)))
        return np.linspace(lower, upper, samples)


def scaled_distance(
    left: np.ndarray, right: np.ndarray, scales: np.ndarray
) -> float:
    values = np.abs((np.asarray(left) - np.asarray(right)) / scales)
    finite = values[np.isfinite(values)]
    return float(np.max(finite)) if finite.size else np.inf


def finite_vector(value: Any) -> bool:
    return all(isfinite(float(item)) for item in np.asarray(value).reshape(-1))
