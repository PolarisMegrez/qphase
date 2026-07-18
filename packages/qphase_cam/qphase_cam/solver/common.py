"""Shared CPU root-solving and solution utilities."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

import numpy as np
from scipy.optimize import root

from qphase_cam.core.coordinates import matrix_to_vector, vector_to_matrix
from qphase_cam.core.jacobian import JacobianResolver
from qphase_cam.core.liouvillian import residual_vector
from qphase_cam.errors import JacobianUnavailableError
from qphase_cam.state import CAMSolution


def initial_state(value: Any, n_modes: int) -> np.ndarray:
    """Validate an initial matrix or provide the identity."""
    if value is None:
        return np.eye(n_modes, dtype=np.complex128)
    state = np.asarray(value, dtype=np.complex128)
    if state.shape != (n_modes, n_modes):
        raise ValueError(
            f"initial state must have shape ({n_modes}, {n_modes}); got {state.shape}"
        )
    return 0.5 * (state + state.conj().T)


def _lower_to_vector(lower: np.ndarray) -> np.ndarray:
    values: list[float] = []
    for i in range(lower.shape[0]):
        values.append(float(np.real(lower[i, i])))
        for j in range(i):
            values.extend((float(np.real(lower[i, j])), float(np.imag(lower[i, j]))))
    return np.asarray(values)


def _vector_to_lower(vector: np.ndarray, n_modes: int) -> np.ndarray:
    lower = np.zeros((n_modes, n_modes), dtype=np.complex128)
    cursor = 0
    for i in range(n_modes):
        lower[i, i] = vector[cursor]
        cursor += 1
        for j in range(i):
            lower[i, j] = vector[cursor] + 1j * vector[cursor + 1]
            cursor += 2
    return lower


def solve_single_state(
    model: Any,
    params: dict[str, Any],
    guess: Any = None,
    *,
    method: Literal["auto", "root", "cholesky"] = "auto",
    root_method: str = "hybr",
    tolerance: float = 1e-10,
    max_iterations: int = 1000,
    use_jacobian: bool = True,
) -> CAMSolution:
    """Solve one CAM fixed point on NumPy/SciPy."""
    state_guess = initial_state(guess, int(model.n_modes))
    attempts: list[CAMSolution] = []
    if method in {"auto", "root"}:
        attempts.append(
            _solve_root(
                model,
                params,
                state_guess,
                root_method,
                tolerance,
                max_iterations,
                use_jacobian,
            )
        )
        if attempts[-1].success or method == "root":
            return attempts[-1]
    attempts.append(
        _solve_cholesky(model, params, state_guess, tolerance, max_iterations)
    )
    successful = [attempt for attempt in attempts if attempt.success]
    return min(successful or attempts, key=lambda attempt: attempt.residual)


def _solve_root(
    model: Any,
    params: dict[str, Any],
    guess: np.ndarray,
    root_method: str,
    tolerance: float,
    max_iterations: int,
    use_jacobian: bool,
) -> CAMSolution:
    vector_guess = np.asarray(matrix_to_vector(guess), dtype=float)

    def residual(vector: np.ndarray) -> np.ndarray:
        state = vector_to_matrix(vector, int(model.n_modes))
        return np.asarray(residual_vector(model, state, params), dtype=float)

    jacobian_callback: Callable[[np.ndarray], np.ndarray] | None = None
    if use_jacobian:
        resolver = JacobianResolver()
        try:
            resolver.resolve(model, guess, params, _NumpyBackendName())

            def evaluate_jacobian(vector: np.ndarray) -> np.ndarray:
                return np.asarray(
                    resolver.resolve(
                        model,
                        vector_to_matrix(vector, int(model.n_modes)),
                        params,
                        _NumpyBackendName(),
                    )
                )

            jacobian_callback = evaluate_jacobian
        except JacobianUnavailableError:
            jacobian_callback = None
    options = {"maxfev": max_iterations} if root_method == "hybr" else {}
    solution = root(
        residual,
        vector_guess,
        jac=jacobian_callback,
        method=root_method,
        tol=tolerance,
        options=options,
    )
    state = np.asarray(vector_to_matrix(solution.x, int(model.n_modes)))
    residual_norm = float(np.linalg.norm(residual(solution.x), ord=np.inf))
    success = bool(solution.success and residual_norm <= tolerance)
    return CAMSolution(
        state=state,
        residual=residual_norm,
        success=success,
        method=f"root-{root_method}",
        message=str(solution.message),
        iterations=getattr(solution, "nit", None),
    )


def _solve_cholesky(
    model: Any,
    params: dict[str, Any],
    guess: np.ndarray,
    tolerance: float,
    max_iterations: int,
) -> CAMSolution:
    try:
        lower_guess = np.linalg.cholesky(guess)
    except np.linalg.LinAlgError:
        lower_guess = np.diag(np.sqrt(np.clip(np.real(np.diag(guess)), 0.0, None)))
    vector_guess = _lower_to_vector(lower_guess)

    def residual(vector: np.ndarray) -> np.ndarray:
        lower = _vector_to_lower(vector, int(model.n_modes))
        state = lower @ lower.conj().T
        return np.asarray(residual_vector(model, state, params), dtype=float)

    best = None
    for root_method in ("lm", "hybr"):
        options = {"maxfev": max_iterations} if root_method == "hybr" else {}
        candidate = root(
            residual,
            vector_guess,
            method=root_method,
            tol=tolerance,
            options=options,
        )
        norm = float(np.linalg.norm(residual(candidate.x), ord=np.inf))
        if best is None or norm < best[1]:
            best = (candidate, norm, root_method)
        if candidate.success and norm <= tolerance:
            break
    assert best is not None
    candidate, residual_norm, root_method = best
    lower = _vector_to_lower(candidate.x, int(model.n_modes))
    state = lower @ lower.conj().T
    return CAMSolution(
        state=state,
        residual=residual_norm,
        success=bool(candidate.success and residual_norm <= tolerance),
        method=f"cholesky-{root_method}",
        message=str(candidate.message),
        iterations=getattr(candidate, "nit", None),
    )


class _NumpyBackendName:
    def backend_name(self) -> str:
        return "numpy"


def deduplicate_solutions(
    solutions: list[CAMSolution], distance_tolerance: float
) -> list[CAMSolution]:
    """Keep the lowest-residual representative of each nearby solution."""
    accepted: list[CAMSolution] = []
    for solution in sorted(solutions, key=lambda item: item.residual):
        if not solution.success:
            continue
        vector = np.asarray(matrix_to_vector(solution.state))
        if all(
            np.linalg.norm(vector - np.asarray(matrix_to_vector(other.state)))
            >= distance_tolerance
            for other in accepted
        ):
            accepted.append(solution)
    return accepted


def random_hermitian_guesses(
    n_modes: int, count: int, scale: float, seed: int | None
) -> list[np.ndarray]:
    """Generate reproducible Hermitian multi-start guesses."""
    rng = np.random.default_rng(seed)
    guesses = [np.zeros((n_modes, n_modes), complex), np.eye(n_modes, dtype=complex)]
    while len(guesses) < count:
        raw = rng.normal(size=(n_modes, n_modes)) + 1j * rng.normal(
            size=(n_modes, n_modes)
        )
        guesses.append(scale * 0.5 * (raw + raw.conj().T))
    return guesses[:count]
