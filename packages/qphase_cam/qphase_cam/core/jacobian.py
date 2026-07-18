"""Analytic, symbolic, and explicitly requested numerical Jacobians."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from qphase.backend.xputil import get_xp

from qphase_cam.errors import JacobianUnavailableError

from .coordinates import matrix_to_vector, vector_to_matrix


def central_difference_jacobian(
    function: Callable[[Any], Any], vector: Any, epsilon: float = 1e-7
) -> Any:
    """Compute a central-difference Jacobian on the vector's backend."""
    xp = get_xp(vector)
    vector = xp.asarray(vector)
    n = int(vector.shape[-1])
    columns = []
    for index in range(n):
        delta = xp.zeros_like(vector)
        delta[..., index] = epsilon
        difference = function(vector + delta) - function(vector - delta)
        columns.append(difference / (2 * epsilon))
    return xp.stack(columns, axis=-1)


class SymbolicJacobian:
    """Compiled Jacobian generated from a model's symbolic H and D."""

    def __init__(self, spec: Any, n_modes: int, backend_name: str) -> None:
        import sympy as sp

        self.spec = spec
        self.n_modes = n_modes
        self.backend_name = backend_name
        operator = (
            -sp.I * spec.hamiltonian * spec.state_matrix
            + sp.I * spec.state_matrix * spec.hamiltonian.H
            + spec.diffusion
        )
        output = self._symbolic_vector(operator)
        jacobian = sp.simplify(output.jacobian(spec.state_symbols))
        if backend_name == "cupy":
            import cupy as modules
        else:
            modules = "numpy"
        self._function = sp.lambdify(
            list(spec.state_symbols) + list(spec.parameter_symbols),
            jacobian,
            modules=[{"ImmutableDenseMatrix": lambda value: value}, modules],
        )

    def _symbolic_vector(self, matrix: Any) -> Any:
        import sympy as sp

        n = self.n_modes
        diagonal = [sp.simplify(sp.re(matrix[i, i])) for i in range(n)]
        upper = [(i, j) for i in range(n) for j in range(i + 1, n)]
        real = [sp.simplify(sp.re(matrix[i, j])) for i, j in upper]
        imag = [sp.simplify(sp.im(matrix[i, j])) for i, j in upper]
        return sp.Matrix(diagonal + real + imag)

    def __call__(self, state: Any, params: dict[str, Any]) -> Any:
        xp = get_xp(state)
        vector = matrix_to_vector(state)
        arguments = [vector[..., i] for i in range(vector.shape[-1])]
        arguments.extend(params[symbol.name] for symbol in self.spec.parameter_symbols)
        output = self._function(*arguments)
        rows = []
        for row in output:
            entries = [xp.asarray(entry) for entry in row]
            shape = np.broadcast_shapes(*(entry.shape for entry in entries))
            broadcast = [xp.broadcast_to(entry, shape) for entry in entries]
            rows.append(xp.stack(broadcast, axis=-1))
        return xp.asarray(xp.stack(rows, axis=-2), dtype=state.real.dtype)


class JacobianResolver:
    """Resolve direct, symbolic, or explicitly enabled numerical Jacobians."""

    _symbolic_cache: dict[tuple[str, str, str, str], SymbolicJacobian] = {}

    def __init__(self, allow_finite_difference: bool = False, epsilon: float = 1e-7):
        self.allow_finite_difference = allow_finite_difference
        self.epsilon = epsilon
        self.last_source: str | None = None

    def resolve(
        self, model: Any, state: Any, params: dict[str, Any], backend: Any
    ) -> Any:
        direct = getattr(model, "cam_jacobian", None)
        if callable(direct):
            self.last_source = "analytic"
            return direct(state, params)

        symbolic = getattr(model, "cam_symbolic_matrices", None)
        if callable(symbolic):
            spec = symbolic()
            key = (
                str(model.name),
                str(spec.version),
                str(backend.backend_name()).lower(),
                str(state.dtype),
            )
            compiled = self._symbolic_cache.get(key)
            if compiled is None:
                compiled = SymbolicJacobian(spec, int(model.n_modes), key[2])
                self._symbolic_cache[key] = compiled
            self.last_source = "symbolic"
            return compiled(state, params)

        if self.allow_finite_difference:
            self.last_source = "finite_difference"
            vector = matrix_to_vector(state)

            def residual(value: Any) -> Any:
                from .liouvillian import residual_vector

                return residual_vector(
                    model, vector_to_matrix(value, int(model.n_modes)), params
                )

            return central_difference_jacobian(residual, vector, self.epsilon)

        raise JacobianUnavailableError(
            f"model {getattr(model, 'name', type(model).__name__)!r} does not "
            "provide an analytic or symbolic CAM Jacobian"
        )
