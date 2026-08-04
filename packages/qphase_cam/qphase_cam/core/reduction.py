"""Numerical evaluators for exact fpgen scalar reductions."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import isfinite
from typing import Any

import numpy as np
import sympy as sp

from qphase_cam.errors import BifurcationCapabilityError


@dataclass(frozen=True)
class ReductionDiagnostics:
    regularity_determinant: float
    denominator_margin: float
    condition_number: float
    reduced_coefficients: np.ndarray


class FractionFreeScalarReduction:
    """Compiled regular-branch equations from a materialized fpgen reduction."""

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
    ) -> list[np.ndarray]:
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


def scaled_distance(
    left: np.ndarray, right: np.ndarray, scales: np.ndarray
) -> float:
    values = np.abs((np.asarray(left) - np.asarray(right)) / scales)
    finite = values[np.isfinite(values)]
    return float(np.max(finite)) if finite.size else np.inf


def finite_vector(value: Any) -> bool:
    return all(isfinite(float(item)) for item in np.asarray(value).reshape(-1))
