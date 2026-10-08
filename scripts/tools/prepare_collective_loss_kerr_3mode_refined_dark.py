"""Build a run-local plotting dataset for the refined-dark CLK3 candidate."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from qphase.data import Dataset, load_bundle, load_products
from qphase_cam.core.coordinates import matrix_to_vector, vector_to_matrix
from qphase_cam.core.fpgen import FPGenDynamicsAdapter
from qphase_cam.postprocessor.petermann import (
    build_bogoliubov_matrix,
    eigensystem_petermann,
)
from qphase_cam.postprocessor.stochastic_validity import (
    canonical_sample_matrix_covariance,
)
from qphase_cam.result import CAMResult
from qphase_sde.analyser.band_limited_carrier import (
    estimate_band_limited_carrier,
    track_band_limited_carrier,
)
from qphase_sde.result import SDEDataBundle
from scipy.optimize import brentq, least_squares, linear_sum_assignment, minimize_scalar

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from models.collective_loss_kerr_3mode import (  # noqa: E402
    CollectiveLossKerr3ModeModel,
)

SCHEMA = "qphase.plot-data/collective-loss-kerr-3mode-refined-dark/5"
DEFAULT_STEM = "collective_loss_kerr_3mode_refined_dark_medium_21"
PUISEUX_EXPONENT_BOUNDS = (0.30, 0.36)


@dataclass(frozen=True)
class BifurcationCandidate:
    state: np.ndarray
    params: dict[str, float]
    perturbation: str


@dataclass(frozen=True)
class PuiseuxFit:
    upper: float
    max_order: int
    exponent: float
    profile_std: float
    window_std: float
    combined_std: float
    coefficients: np.ndarray
    relative_rmse: float
    max_relative_residual: float

    @property
    def accepts_cubic_exponent(self) -> bool:
        return abs(self.exponent - 1.0 / 3.0) <= self.combined_std


def _assign_nested(payload: dict[Any, Any], dotted_key: str, value: Any) -> None:
    """Assign ``value`` into ``payload`` following a dotted key path.

    Digit-only path segments become integer keys: analyser payloads index
    per-mode tables by mode number.
    """
    parts = dotted_key.split(".")
    target = payload
    for part in parts[:-1]:
        key: Any = int(part) if part.isdigit() else part
        child = target.get(key)
        if not isinstance(child, dict):
            child = {}
            target[key] = child
        target = child
    leaf = parts[-1]
    target[int(leaf) if leaf.isdigit() else leaf] = value


def _product_payload(product: Dataset, scan_index: int | None = None) -> dict[Any, Any]:
    """Materialize a typed analyser product as a nested payload.

    Variables are materialized by dotted name into nested dicts; the scalar
    ``payload_meta`` attributes follow, with ``per_point_meta`` entries
    indexed by ``scan_index`` when one is given.
    """
    payload: dict[Any, Any] = {}
    for variable in product.variables:
        _assign_nested(
            payload, variable.name, product.handle(variable.name).materialize()
        )
    per_point = set(product.attributes.get("per_point_meta", ()))
    for meta_key, meta_value in product.attributes.get("payload_meta", {}).items():
        if meta_key in per_point and scan_index is not None:
            meta_value = list(meta_value)[scan_index]
        _assign_nested(payload, meta_key, meta_value)
    return payload


def _load_v4_payload(path: Path, product_name: str) -> dict[str, Any]:
    products = load_products(path)
    if product_name not in products:
        raise KeyError(f"Artifact {path} has no product {product_name!r}")
    return _product_payload(products[product_name])


def _load_cam_result(path: Path) -> CAMResult:
    payload = _load_v4_payload(path, "cam_result")
    return CAMResult(
        states=payload["states"],
        residuals=payload["residuals"],
        success=payload["success"],
        valid_mask=payload["valid_mask"],
        solution_count=payload["solution_count"],
        params=dict(payload["params"]),
        axes=dict(payload.get("axes") or {}),
        postprocess=dict(payload.get("postprocess") or {}),
        meta=dict(payload.get("meta") or {}),
    )


def _load_dense_cam_result(path: Path) -> CAMResult:
    """Load a dense CAM scan from either on-disk format.

    Current CAM jobs persist through the ``schema_version="2.0"`` single-npz
    path (``CAMResult.save``); older sessions hold v4 product bundles.
    """
    manifest_path = path / "artifact_manifest.json"
    if path.is_dir() and manifest_path.exists():
        version = json.loads(manifest_path.read_text(encoding="utf-8")).get(
            "schema_version"
        )
        if version == "2.0":
            singles = sorted(path.glob("*.npz"))
            if len(singles) != 1:
                raise ValueError(f"{path} holds {len(singles)} npz files")
            return CAMResult.load(singles[0])
    return _load_cam_result(path)


def _load_v4_scan(path: Path) -> SDEDataBundle:
    result = load_bundle(path)
    if not isinstance(result, SDEDataBundle):
        raise TypeError(f"Artifact {path} is not an SDEDataBundle")
    return result


def _point_analysis(result: SDEDataBundle, index: int) -> dict[str, Any]:
    point = result.point_view((index,))
    scan_index = int(point.metadata["scan_index"])
    return {
        name: _product_payload(product, scan_index)
        for name, product in point.products.items()
        if name != "trajectories"
    }


def _load_candidate(path: Path, case: int) -> BifurcationCandidate:
    payload = _load_v4_payload(path, "bifurcation_result")
    offsets = np.asarray(payload["candidate_offsets"], dtype=int)
    start = int(offsets[case])
    stop = int(offsets[case + 1])
    if start >= stop:
        raise ValueError(f"bifurcation case {case} has no candidate")
    params = {
        name: float(np.asarray(values)[case])
        for name, values in payload["case_params"].items()
    }
    params.update(
        {
            str(name): float(value)
            for name, value in zip(
                payload["candidate_control_names"],
                payload["candidate_control_values"][start],
                strict=True,
            )
        }
    )
    metadata = payload["candidate_meta"]
    return BifurcationCandidate(
        state=np.asarray(payload["candidate_state_vectors"][start], dtype=float),
        params=params,
        perturbation=str(metadata["perturbation_parameter"]),
    )


class ReducedCAMFlow:
    """Exact order-parameter flow after eliminating the other coordinates."""

    def __init__(self, candidate: BifurcationCandidate) -> None:
        self.candidate = candidate
        model = CollectiveLossKerr3ModeModel(**candidate.params)
        self.adapter = FPGenDynamicsAdapter.from_model(model)

    def evaluate(self, r_aa: float, epsilon: float) -> tuple[float, np.ndarray]:
        params = dict(self.candidate.params)
        params[self.candidate.perturbation] += epsilon
        vector = np.zeros(self.adapter.state_size, dtype=float)
        vector[0] = r_aa
        residual = self.adapter.rhs(vector, params)
        eliminated = self.adapter.jacobian(vector, params)[1:, 1:]
        vector[1:] = np.linalg.solve(eliminated, -residual[1:])
        full_residual = self.adapter.rhs(vector, params)
        if np.linalg.norm(full_residual[1:]) > 1e-8:
            raise RuntimeError("failed to eliminate non-order-parameter equations")
        return float(full_residual[0]), vector

    def root(self, epsilon: float, bounds: tuple[float, float]) -> float:
        return float(
            brentq(
                lambda value: self.evaluate(value, epsilon)[0],
                *bounds,
                xtol=1e-11,
                rtol=1e-13,
            )
        )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _load_ridge_carrier_csv(path: Path, omega_c: np.ndarray) -> dict[str, np.ndarray]:
    """Load a ridge-conditioned ``carrier_results.csv`` aligned to the scan axis."""
    rows = [row for row in _read_csv(path) if int(row["ridge_order"]) == 0]
    rows.sort(key=lambda row: float(row["omega_c"]))
    if len(rows) != omega_c.size:
        raise ValueError("ridge carrier count does not match scan")
    csv_omega_c = np.asarray([float(row["omega_c"]) for row in rows], dtype=float)
    if not np.allclose(csv_omega_c, omega_c, rtol=0.0, atol=1e-9):
        raise ValueError("ridge carrier scan axis does not align")
    return {
        "frequency": np.asarray([float(row["frequency"]) for row in rows], dtype=float),
        "sem": np.asarray(
            [float(row["diagnostic_uncertainty"]) for row in rows], dtype=float
        ),
        "status": np.asarray([row["status"] for row in rows], dtype=str),
    }


def _mode_payload(point: dict[str, Any]) -> dict[str, Any]:
    modes = point["mode_results"]
    return modes.get(0, modes.get("0"))["allan"]


def _select_cam_branch(
    result: CAMResult,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    frequencies: list[float] = []
    rates: list[float] = []
    states: list[np.ndarray] = []
    slots: list[int] = []
    for index in range(np.asarray(result.axes["omega_c"]).size):
        accepted = (
            np.asarray(result.valid_mask[index], dtype=bool)
            & np.asarray(result.postprocess["is_physical"][index], dtype=bool)
            & np.asarray(result.postprocess["is_stable"][index], dtype=bool)
        )
        valid = np.flatnonzero(accepted)
        if valid.size != 1:
            raise ValueError(
                f"CAM point {index} has {valid.size} physical stable solutions"
            )
        slot = int(valid[0])
        slots.append(slot)
        states.append(np.asarray(result.states[index, slot], dtype=complex))
        frequencies.append(float(result.postprocess["rayleigh_frequency"][index, slot]))
        eigenvalues = np.asarray(
            result.postprocess["jacobian_eigenvalues"][index, slot], dtype=complex
        )
        rates.append(float(-np.max(eigenvalues.real)))
    return (
        np.asarray(frequencies),
        np.asarray(rates),
        np.asarray(states),
        np.asarray(slots, dtype=int),
    )


def _eigensystem(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values, vectors = np.linalg.eig(matrix)
    norms = np.linalg.norm(vectors, axis=0)
    vectors = vectors / np.maximum(norms, np.finfo(float).eps)
    return values, vectors


def _match_eigensystem(
    previous_values: np.ndarray,
    previous_vectors: np.ndarray,
    values: np.ndarray,
    vectors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    spectral_scale = max(float(np.max(np.abs(previous_values))), 1e-8)
    spectral_distance = np.abs(previous_values[:, None] - values[None, :])
    vector_mismatch = 1.0 - np.abs(previous_vectors.conj().T @ vectors)
    cost = spectral_distance / spectral_scale + 0.25 * vector_mismatch
    rows, columns = linear_sum_assignment(cost)
    order = np.empty(values.size, dtype=int)
    order[rows] = columns
    return values[order], vectors[:, order]


def _track_eigenvalues(jacobians: np.ndarray, anchor: int) -> np.ndarray:
    count, size, _ = jacobians.shape
    tracked = np.empty((count, size), dtype=complex)
    vectors = np.empty((count, size, size), dtype=complex)
    values, anchor_vectors = _eigensystem(jacobians[anchor])
    order = np.asarray(
        sorted(
            range(size),
            key=lambda index: (
                -float(values[index].real),
                abs(float(values[index].imag)),
                float(values[index].imag),
            ),
        ),
        dtype=int,
    )
    tracked[anchor] = values[order]
    vectors[anchor] = anchor_vectors[:, order]
    for direction in (1, -1):
        index = anchor + direction
        while 0 <= index < count:
            previous = index - direction
            values, current_vectors = _eigensystem(jacobians[index])
            tracked[index], vectors[index] = _match_eigensystem(
                tracked[previous], vectors[previous], values, current_vectors
            )
            index += direction
    return tracked


def _distinct_real_branches(eigenvalues: np.ndarray, anchor: int) -> np.ndarray:
    center = eigenvalues[anchor].real
    tolerance = max(1e-8, 1e-7 * float(np.ptp(center)))
    groups: list[list[int]] = []
    for index in np.argsort(-center):
        for group in groups:
            if abs(center[index] - center[group[0]]) <= tolerance:
                group.append(int(index))
                break
        else:
            groups.append([int(index)])
    return np.column_stack(
        [np.mean(eigenvalues[:, group].real, axis=1) for group in groups]
    )


def _two_sided_power_fit(
    epsilon: np.ndarray, values: np.ndarray, mask: np.ndarray
) -> tuple[np.ndarray, float]:
    selected_epsilon = epsilon[mask]
    selected_values = values[mask]
    negative = selected_epsilon < 0.0
    design = np.column_stack(
        (
            negative.astype(float),
            (~negative).astype(float),
            np.log(np.abs(selected_epsilon)),
        )
    )
    coefficients, *_ = np.linalg.lstsq(design, np.log(selected_values), rcond=None)
    return np.exp(coefficients[:2]), float(coefficients[2])


def _puiseux_solution(
    epsilon: np.ndarray,
    values: np.ndarray,
    exponent: float,
    max_order: int,
    scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    powers = np.arange(2, max_order + 1, dtype=float)
    basis = (epsilon[:, None] / scale) ** (powers * exponent)
    coefficients, *_ = np.linalg.lstsq(
        basis / values[:, None], np.ones(values.size), rcond=None
    )
    prediction = basis @ coefficients
    relative_residual = (prediction - values) / values
    return coefficients, prediction, relative_residual


def _fit_puiseux_at_window(
    epsilon: np.ndarray,
    values: np.ndarray,
    *,
    lower: float,
    upper: float,
    max_order: int,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    selected = (epsilon >= lower) & (epsilon <= upper)
    x = epsilon[selected]
    y = values[selected]
    if x.size <= max_order + 2:
        raise ValueError("Puiseux fit window has too few points")

    def objective(exponent: float) -> float:
        _, _, residual = _puiseux_solution(x, y, exponent, max_order, upper)
        return float(residual @ residual)

    optimum = minimize_scalar(
        objective,
        # Restrict the fit to the cubic-normal-form basin. Without this
        # identifiability constraint, a vanishing k=2 coefficient lets the
        # k=3 term relabel the same leading exponent with p near 2/9.
        bounds=PUISEUX_EXPONENT_BOUNDS,
        method="bounded",
        options={"xatol": 1e-13},
    )
    exponent = float(optimum.x)
    coefficients, prediction, residual = _puiseux_solution(
        x, y, exponent, max_order, upper
    )
    rss = float(residual @ residual)
    degrees_of_freedom = x.size - coefficients.size - 1
    profile_target = rss + rss / degrees_of_freedom

    def profile_offset(value: float) -> float:
        return objective(value) - profile_target

    def profile_bound(bound: float) -> float:
        grid = np.linspace(exponent, bound, 1001)
        previous = float(grid[0])
        previous_value = profile_offset(previous)
        for current_raw in grid[1:]:
            current = float(current_raw)
            current_value = profile_offset(current)
            if previous_value * current_value <= 0.0:
                return float(brentq(profile_offset, previous, current))
            previous = current
            previous_value = current_value
        return bound

    lower_bound = profile_bound(PUISEUX_EXPONENT_BOUNDS[0])
    upper_bound = profile_bound(PUISEUX_EXPONENT_BOUNDS[1])
    profile_std = 0.5 * (upper_bound - lower_bound)
    return exponent, profile_std, coefficients, residual


def _fit_rate_puiseux(
    epsilon: np.ndarray,
    rate: np.ndarray,
    upper: float,
    *,
    max_order: int = 4,
) -> PuiseuxFit:
    lower = float(epsilon[0])
    exponent, profile_std, coefficients, residual = _fit_puiseux_at_window(
        epsilon,
        rate,
        lower=lower,
        upper=upper,
        max_order=max_order,
    )

    # Dense CAM points are deterministic and strongly correlated. Varying the
    # inner cutoff estimates the dominant truncation/window systematic that a
    # formal regression covariance would otherwise miss.
    sensitivity_upper = min(1e-6, upper / 100.0)
    lower_bounds = np.geomspace(lower, sensitivity_upper, 9)
    window_exponents = np.asarray(
        [
            _fit_puiseux_at_window(
                epsilon,
                rate,
                lower=float(window_lower),
                upper=upper,
                max_order=max_order,
            )[0]
            for window_lower in lower_bounds
        ]
    )
    window_std = float(np.std(window_exponents, ddof=1))
    combined_std = float(np.hypot(profile_std, window_std))
    return PuiseuxFit(
        upper=upper,
        max_order=max_order,
        exponent=exponent,
        profile_std=profile_std,
        window_std=window_std,
        combined_std=combined_std,
        coefficients=coefficients,
        relative_rmse=float(np.sqrt(np.mean(residual**2))),
        max_relative_residual=float(np.max(np.abs(residual))),
    )


def _select_rate_puiseux_window(
    epsilon: np.ndarray,
    rate: np.ndarray,
    *,
    minimum_upper: float = 1e-4,
    max_order: int = 4,
) -> tuple[PuiseuxFit, list[PuiseuxFit]]:
    first = int(np.searchsorted(epsilon, minimum_upper, side="left"))
    outward: list[PuiseuxFit] = []
    selected: PuiseuxFit | None = None
    for upper in epsilon[first:]:
        fit = _fit_rate_puiseux(epsilon, rate, float(upper), max_order=max_order)
        outward.append(fit)
        if not fit.accepts_cubic_exponent:
            break
        selected = fit
    if selected is not None:
        return selected, outward

    inward: list[PuiseuxFit] = []
    for upper in epsilon[:first][::-1]:
        fit = _fit_rate_puiseux(epsilon, rate, float(upper), max_order=max_order)
        inward.append(fit)
        if fit.accepts_cubic_exponent:
            return fit, outward + inward
    raise RuntimeError("no Jacobian Puiseux window accepts p=1/3")


def _critical_mode(jacobian: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values, right_vectors = np.linalg.eig(jacobian)
    index = int(np.argmax(np.real(values)))
    value = values[index]
    right = np.real_if_close(right_vectors[:, index], tol=1000)
    left_values, left_vectors = np.linalg.eig(jacobian.T)
    left_index = int(np.argmin(np.abs(left_values - value)))
    left = np.real_if_close(left_vectors[:, left_index], tol=1000)
    if np.iscomplexobj(right) or np.iscomplexobj(left):
        raise ValueError(f"critical CAM mode is not real: {value}")
    right = np.asarray(right, dtype=float)
    left = np.asarray(left, dtype=float)
    right /= np.linalg.norm(right)
    overlap = float(left @ right)
    if abs(overlap) <= 1e-12:
        raise ValueError("critical left-right overlap is unresolved")
    return right, left / overlap


def _rayleigh(
    model: CollectiveLossKerr3ModeModel,
    state: np.ndarray,
    params: dict[str, Any],
) -> float:
    hamiltonian = np.asarray(model.cam_hamiltonian(state, params), dtype=complex)
    return float(np.real(np.trace(hamiltonian @ state) / np.trace(state)))


def _mode_rayleigh(
    model: CollectiveLossKerr3ModeModel,
    state: np.ndarray,
    params: dict[str, Any],
    mode: int,
) -> float:
    hamiltonian = np.asarray(model.cam_hamiltonian(state, params), dtype=complex)
    return float(np.real((hamiltonian @ state)[mode, mode] / state[mode, mode]))


def _pack_eigenstate(amplitude: np.ndarray, frequency: float) -> np.ndarray:
    amplitude = np.asarray(amplitude, dtype=complex)
    amplitude *= np.exp(-1j * np.angle(amplitude[0]))
    if amplitude[0].real < 0.0:
        amplitude *= -1.0
    return np.asarray(
        [
            np.log(max(float(amplitude[0].real), 1e-12)),
            amplitude[1].real,
            amplitude[1].imag,
            amplitude[2].real,
            amplitude[2].imag,
            frequency,
        ],
        dtype=float,
    )


def _unpack_eigenstate(vector: np.ndarray) -> tuple[np.ndarray, float]:
    amplitude = np.asarray(
        [
            np.exp(vector[0]),
            vector[1] + 1j * vector[2],
            vector[3] + 1j * vector[4],
        ],
        dtype=complex,
    )
    return amplitude, float(vector[5])


def _self_consistent_eigenstate_branch(
    result: CAMResult, cam_states: np.ndarray
) -> dict[str, np.ndarray]:
    count = cam_states.shape[0]
    amplitudes = np.empty((count, 3), dtype=complex)
    frequencies = np.empty(count, dtype=float)
    residuals = np.empty(count, dtype=float)
    occupations = np.empty(count, dtype=float)
    previous: np.ndarray | None = None

    for index in range(count - 1, -1, -1):
        params = result.params_at((index,))
        model = CollectiveLossKerr3ModeModel(**params)

        if previous is None:
            values, vectors = np.linalg.eigh(cam_states[index])
            amplitude = vectors[:, -1] * np.sqrt(max(float(values[-1]), 1e-12))
            pure_state = np.outer(amplitude, amplitude.conj())
            hamiltonian = np.asarray(
                model.cam_hamiltonian(pure_state, params), dtype=complex
            )
            frequency = float(
                np.real(
                    np.vdot(amplitude, hamiltonian @ amplitude)
                    / np.vdot(amplitude, amplitude)
                )
            )
            initial = _pack_eigenstate(amplitude, frequency)
        else:
            initial = previous

        def normalized_residual(
            vector: np.ndarray,
            point_model: CollectiveLossKerr3ModeModel = model,
            point_params: dict[str, Any] = params,
        ) -> np.ndarray:
            amplitude, frequency = _unpack_eigenstate(vector)
            pure_state = np.outer(amplitude, amplitude.conj())
            hamiltonian = np.asarray(
                point_model.cam_hamiltonian(pure_state, point_params), dtype=complex
            )
            residual = hamiltonian @ amplitude - frequency * amplitude
            scale = max(float(np.linalg.norm(amplitude)), 1.0)
            return np.concatenate((residual.real, residual.imag)) / scale

        solution = least_squares(
            normalized_residual,
            initial,
            bounds=(
                np.asarray([-30.0, -1e6, -1e6, -1e6, -1e6, -100.0]),
                np.asarray([30.0, 1e6, 1e6, 1e6, 1e6, 100.0]),
            ),
            xtol=1e-13,
            ftol=1e-13,
            gtol=1e-13,
            x_scale="jac",
            max_nfev=20_000,
        )
        amplitude, frequency = _unpack_eigenstate(solution.x)
        residual = float(np.linalg.norm(normalized_residual(solution.x)))
        if not solution.success or residual > 1e-9:
            raise RuntimeError(
                f"eigenstate continuation failed at point {index}: "
                f"success={solution.success}, residual={residual:.3e}"
            )
        amplitudes[index] = amplitude
        frequencies[index] = frequency
        residuals[index] = residual
        occupations[index] = float(np.vdot(amplitude, amplitude).real)
        previous = solution.x

    return {
        "eigenstate_amplitudes": amplitudes,
        "eigenstate_frequency": frequencies,
        "eigenstate_residual": residuals,
        "eigenstate_occupation": occupations,
    }


def _frequency_gradient(
    model: CollectiveLossKerr3ModeModel,
    state: np.ndarray,
    params: dict[str, Any],
    *,
    mode: int | None = None,
) -> np.ndarray:
    vector = np.asarray(matrix_to_vector(state), dtype=float)
    gradient = np.empty_like(vector)
    for index, value in enumerate(vector):
        step = 2e-6 * max(1.0, abs(float(value)))
        plus = vector.copy()
        minus = vector.copy()
        plus[index] += step
        minus[index] -= step
        plus_state = vector_to_matrix(plus, 3)
        minus_state = vector_to_matrix(minus, 3)
        if mode is None:
            plus_frequency = _rayleigh(model, plus_state, params)
            minus_frequency = _rayleigh(model, minus_state, params)
        else:
            plus_frequency = _mode_rayleigh(model, plus_state, params, mode)
            minus_frequency = _mode_rayleigh(model, minus_state, params, mode)
        gradient[index] = (plus_frequency - minus_frequency) / (2.0 * step)
    return gradient


def _cam_noise_predictions(
    result: CAMResult, states: np.ndarray
) -> dict[str, np.ndarray]:
    count = states.shape[0]
    projected_noise = np.empty(count)
    observable_slope = np.empty(count)
    local_ou = np.empty(count)
    parameter_drive = np.empty((count, 9), dtype=float)
    frequency_gradient = np.empty((count, 9), dtype=float)
    state_response = np.empty((count, 9), dtype=float)
    jacobians = np.empty((count, 9, 9), dtype=float)
    noise_covariances = np.empty((count, 9, 9), dtype=float)
    direct_response = np.empty(count)
    state_mediated_response = np.empty(count)
    full_response = np.empty(count)
    full_ou = np.empty(count)
    projected_drive = np.empty(count)
    soft_response = np.empty(count)
    soft_ou = np.empty(count)
    soft_ratio = np.empty(count)
    jacobian_condition = np.empty(count)
    stochastic_frequency = np.empty(count)
    hamiltonian_petermann = np.empty(count)
    bogoliubov_petermann = np.empty(count)
    for index, state in enumerate(states):
        params = result.params_at((index,))
        model = CollectiveLossKerr3ModeModel(**params)
        jacobian = np.asarray(model.cam_jacobian(state, params), dtype=float)
        hamiltonian = np.asarray(model.cam_hamiltonian(state, params), dtype=complex)
        interaction = np.asarray(
            model.cam_bogoliubov_interaction(state, params), dtype=complex
        )
        _, h_factors = eigensystem_petermann(hamiltonian)
        _, b_factors = eigensystem_petermann(
            build_bogoliubov_matrix(hamiltonian, interaction)
        )
        hamiltonian_petermann[index] = float(np.sum(h_factors))
        bogoliubov_petermann[index] = float(np.sum(b_factors))
        adapter = FPGenDynamicsAdapter.from_model(model)
        vector = np.asarray(matrix_to_vector(state), dtype=float)
        parameter_index = adapter.parameter_names.index("omega_c")
        drive = np.asarray(
            adapter.parameter_jacobian(vector, params)[:, parameter_index],
            dtype=float,
        )
        gradient = _frequency_gradient(model, state, params, mode=0)
        step = 2e-6 * max(1.0, abs(float(params["omega_c"])))
        plus = dict(params)
        minus = dict(params)
        plus["omega_c"] += step
        minus["omega_c"] -= step
        explicit = (
            _mode_rayleigh(CollectiveLossKerr3ModeModel(**plus), state, plus, 0)
            - _mode_rayleigh(CollectiveLossKerr3ModeModel(**minus), state, minus, 0)
        ) / (2.0 * step)
        response = -np.linalg.solve(jacobian, drive)
        right, left = _critical_mode(jacobian)
        diffusion = np.asarray(model.cam_diffusion(state, params), dtype=complex)
        covariance = canonical_sample_matrix_covariance(state, diffusion)
        noise = float(left @ covariance @ left)
        slope = float(gradient @ right)
        eigenvalues = np.linalg.eigvals(jacobian)
        rate = float(-np.max(eigenvalues.real))
        critical_eigenvalue = float(left @ jacobian @ right)
        drive_projection = float(left @ drive)
        soft_signal = -slope * drive_projection / critical_eigenvalue
        soft_noise = noise * slope**2 / critical_eigenvalue**2
        zero_frequency_vector = np.linalg.solve(-jacobian.T, gradient)
        projected_noise[index] = noise
        observable_slope[index] = slope
        local_ou[index] = noise * slope**2 / rate**2
        parameter_drive[index] = drive
        frequency_gradient[index] = gradient
        state_response[index] = response
        jacobians[index] = jacobian
        noise_covariances[index] = covariance
        direct_response[index] = explicit
        mediated = float(gradient @ response)
        state_mediated_response[index] = mediated
        full_response[index] = explicit + mediated
        full_ou[index] = float(
            zero_frequency_vector @ covariance @ zero_frequency_vector
        )
        projected_drive[index] = drive_projection
        soft_response[index] = soft_signal
        soft_ou[index] = soft_noise
        soft_ratio[index] = noise / drive_projection**2
        jacobian_condition[index] = float(np.linalg.cond(jacobian))
        stochastic_frequency[index] = _mode_rayleigh(model, state, params, 0)
    return {
        "cam_projected_noise": projected_noise,
        "cam_observable_slope": observable_slope,
        "cam_local_ou_intensity": local_ou,
        "cam_parameter_drive": parameter_drive,
        "cam_frequency_gradient": frequency_gradient,
        "cam_state_response": state_response,
        "cam_jacobian": jacobians,
        "cam_noise_covariance": noise_covariances,
        "cam_direct_frequency_response": direct_response,
        "cam_explicit_frequency_derivative": direct_response,
        "cam_state_mediated_frequency_derivative": state_mediated_response,
        "cam_steady_frequency_derivative": full_response,
        "cam_full_frequency_response": full_response,
        "cam_full_ou_intensity": full_ou,
        "cam_projected_drive": projected_drive,
        "cam_soft_frequency_response": soft_response,
        "cam_soft_ou_intensity": soft_ou,
        "cam_soft_information_ratio": soft_ratio,
        "cam_jacobian_condition": jacobian_condition,
        "cam_stochastic_frequency": stochastic_frequency,
        "cam_stochastic_readout_mode": np.asarray(0, dtype=int),
        "cam_hamiltonian_petermann_sum": hamiltonian_petermann,
        "cam_bogoliubov_petermann_sum": bogoliubov_petermann,
    }


def _dense_cam_arrays(result: CAMResult) -> dict[str, np.ndarray]:
    """Dense-grid CAM curves for the fig-3 panels.

    SDE markers stay on the coarse scan grid; these arrays supply the smooth
    CAM theory curves from a dense (log-spaced) omega_c scan.
    """
    frequency, rate, states, _slots = _select_cam_branch(result)
    predictions = _cam_noise_predictions(result, states)
    return {
        "dense_cam_omega_c": np.asarray(result.axes["omega_c"], dtype=float),
        "dense_cam_frequency": frequency,
        "dense_cam_critical_rate": rate,
        "dense_cam_steady_frequency_derivative": predictions[
            "cam_steady_frequency_derivative"
        ],
        "dense_cam_full_ou_intensity": predictions["cam_full_ou_intensity"],
        "dense_cam_soft_information_ratio": predictions["cam_soft_information_ratio"],
        "dense_cam_hamiltonian_petermann_sum": predictions[
            "cam_hamiltonian_petermann_sum"
        ],
        "dense_cam_bogoliubov_petermann_sum": predictions[
            "cam_bogoliubov_petermann_sum"
        ],
    }


def _build_flow(
    bifurcation_result: Path, case: int
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    candidate = _load_candidate(bifurcation_result, case)
    flow = ReducedCAMFlow(candidate)
    critical_r_aa = float(candidate.state[0])
    critical_state = vector_to_matrix(candidate.state, 3)
    critical_model = CollectiveLossKerr3ModeModel(**candidate.params)
    critical_frequency = _rayleigh(critical_model, critical_state, candidate.params)
    critical_jacobian = np.asarray(
        critical_model.cam_jacobian_vector(candidate.state, candidate.params),
        dtype=float,
    )
    critical_right, critical_left = _critical_mode(critical_jacobian)
    epsilon_limit = 2e-4
    epsilon = np.linspace(-epsilon_limit, epsilon_limit, 401)
    anchor = int(np.argmin(np.abs(epsilon)))
    epsilon[anchor] = 0.0
    bounds = (0.001 * critical_r_aa, 8.0 * critical_r_aa)
    branch = np.asarray(
        [
            critical_r_aa
            if perturbation == 0.0
            else flow.root(float(perturbation), bounds)
            for perturbation in epsilon
        ]
    )
    r_aa = np.linspace(
        0.8 * float(np.min(branch)),
        1.08 * float(np.max(branch)),
        201,
    )
    drift = np.empty((r_aa.size, epsilon.size), dtype=float)
    for column, perturbation in enumerate(epsilon):
        for row, state_value in enumerate(r_aa):
            drift[row, column] = flow.evaluate(float(state_value), float(perturbation))[
                0
            ]
    branch_vectors = np.empty((epsilon.size, flow.adapter.state_size), dtype=float)
    branch_frequencies = np.empty(epsilon.size, dtype=float)
    branch_mode0_frequencies = np.empty(epsilon.size, dtype=float)
    jacobians = np.empty(
        (epsilon.size, flow.adapter.state_size, flow.adapter.state_size), dtype=float
    )
    for index, (perturbation, state_value) in enumerate(
        zip(epsilon, branch, strict=True)
    ):
        _, vector = flow.evaluate(float(state_value), float(perturbation))
        params = dict(candidate.params)
        params[candidate.perturbation] += float(perturbation)
        branch_vectors[index] = vector
        model = CollectiveLossKerr3ModeModel(**params)
        state = vector_to_matrix(vector, 3)
        branch_frequencies[index] = _rayleigh(model, state, params)
        hamiltonian = np.asarray(model.cam_hamiltonian(state, params), dtype=complex)
        branch_mode0_frequencies[index] = float(
            np.real((hamiltonian @ state)[0, 0] / state[0, 0])
        )
        jacobians[index] = flow.adapter.jacobian(vector, params)
    eigenvalues = _track_eigenvalues(jacobians, anchor)
    real_branches = _distinct_real_branches(eigenvalues, anchor)
    minimum_real_rate = np.min(np.abs(eigenvalues.real), axis=1)

    log_epsilon = np.geomspace(1e-8, epsilon_limit, 181)
    log_positive_branch = np.asarray(
        [flow.root(float(perturbation), bounds) for perturbation in log_epsilon]
    )
    log_negative_branch = np.asarray(
        [flow.root(float(-perturbation), bounds) for perturbation in log_epsilon]
    )
    log_jacobians = np.empty(
        (log_epsilon.size, flow.adapter.state_size, flow.adapter.state_size),
        dtype=float,
    )
    for index, (perturbation, state_value) in enumerate(
        zip(log_epsilon, log_positive_branch, strict=True)
    ):
        _, vector = flow.evaluate(float(state_value), float(perturbation))
        params = dict(candidate.params)
        params[candidate.perturbation] += float(perturbation)
        log_jacobians[index] = flow.adapter.jacobian(vector, params)
    log_eigenvalues = _track_eigenvalues(log_jacobians, 0)
    log_real_branches = _distinct_real_branches(log_eigenvalues, 0)
    log_critical_rate = -np.max(log_eigenvalues.real, axis=1)

    response_epsilon = np.concatenate((-log_epsilon[::-1], log_epsilon))
    response = np.abs(
        np.concatenate((log_negative_branch[::-1], log_positive_branch)) - critical_r_aa
    )
    response_fit_mask = np.ones_like(response_epsilon, dtype=bool)
    _, exponent = _two_sided_power_fit(response_epsilon, response, response_fit_mask)
    rate_fit, rate_window_scan = _select_rate_puiseux_window(
        log_epsilon, log_critical_rate
    )
    rate_fit_mask = log_epsilon <= rate_fit.upper
    rate_fit_values = _puiseux_solution(
        log_epsilon[rate_fit_mask],
        log_critical_rate[rate_fit_mask],
        rate_fit.exponent,
        rate_fit.max_order,
        rate_fit.upper,
    )[1]
    arrays = {
        "flow_epsilon": epsilon,
        "flow_r_aa": r_aa,
        "flow_drift": drift,
        "flow_branch_r_aa": branch,
        "flow_branch_state_vectors": branch_vectors,
        "flow_branch_frequency": branch_frequencies,
        "flow_branch_mode0_frequency": branch_mode0_frequencies,
        "flow_critical_r_aa": np.asarray(critical_r_aa),
        "flow_critical_frequency": np.asarray(critical_frequency),
        "flow_critical_mode0_frequency": np.asarray(branch_mode0_frequencies[anchor]),
        "flow_critical_right_vector": critical_right,
        "flow_critical_left_vector": critical_left,
        "flow_response_exponent": np.asarray(exponent),
        "flow_response_fit_mask": response_fit_mask,
        "flow_response_fit_limit": np.asarray(epsilon_limit),
        "flow_response_fit_epsilon": response_epsilon,
        "flow_jacobian_eigenvalues": eigenvalues,
        "flow_jacobian_real_branches": real_branches,
        "flow_min_abs_jacobian_real": minimum_real_rate,
        "flow_log_epsilon": log_epsilon,
        "flow_log_jacobian_eigenvalues": log_eigenvalues,
        "flow_log_jacobian_real_branches": log_real_branches,
        "flow_critical_rate": log_critical_rate,
        "flow_rate_fit_mask": rate_fit_mask,
        "flow_rate_fit_values": rate_fit_values,
        "flow_rate_fit_upper": np.asarray(rate_fit.upper),
        "flow_rate_fit_order": np.asarray(rate_fit.max_order),
        "flow_rate_fit_coefficients": rate_fit.coefficients,
        "flow_rate_exponent": np.asarray(rate_fit.exponent),
        "flow_rate_exponent_profile_std": np.asarray(rate_fit.profile_std),
        "flow_rate_exponent_window_std": np.asarray(rate_fit.window_std),
        "flow_rate_exponent_std": np.asarray(rate_fit.combined_std),
        "flow_rate_relative_rmse": np.asarray(rate_fit.relative_rmse),
        "flow_rate_max_relative_residual": np.asarray(rate_fit.max_relative_residual),
        "flow_rate_window_scan_upper": np.asarray(
            [fit.upper for fit in rate_window_scan]
        ),
        "flow_rate_window_scan_exponent": np.asarray(
            [fit.exponent for fit in rate_window_scan]
        ),
        "flow_rate_window_scan_std": np.asarray(
            [fit.combined_std for fit in rate_window_scan]
        ),
        "flow_rate_window_scan_accepts_cubic": np.asarray(
            [fit.accepts_cubic_exponent for fit in rate_window_scan], dtype=bool
        ),
    }
    metadata = {
        "case": case,
        "candidate": 0,
        "perturbation": candidate.perturbation,
        "params": candidate.params,
        "jacobian_puiseux_fit": {
            "model": "sum(k=2..K, b_k * |epsilon|^(k*p))",
            "max_order": rate_fit.max_order,
            "exponent_bounds": list(PUISEUX_EXPONENT_BOUNDS),
            "lower": float(log_epsilon[0]),
            "upper": rate_fit.upper,
            "exponent": rate_fit.exponent,
            "profile_std": rate_fit.profile_std,
            "window_std": rate_fit.window_std,
            "combined_std": rate_fit.combined_std,
            "accepts_cubic_exponent": rate_fit.accepts_cubic_exponent,
            "relative_rmse": rate_fit.relative_rmse,
            "max_relative_residual": rate_fit.max_relative_residual,
        },
    }
    return arrays, metadata


def _load_signed_sde_window(
    dataset_path: Path | Sequence[Path],
    fit_path: Path | None,
    critical_value: float,
    *,
    stem: str,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    dataset_paths = (
        [dataset_path] if isinstance(dataset_path, Path) else list(dataset_path)
    )
    # Multiple sessions may tile the signed window (e.g. a coarse grid plus an
    # interleaved supplement); merge them into one omega_c-ordered point list.
    combined: list[tuple[float, SDEDataBundle, int]] = []
    for path in dataset_paths:
        scan = _load_v4_scan(path)
        axis = np.asarray(scan.axes["omega_c"], dtype=float).reshape(-1)
        for index in range(axis.size):
            combined.append((float(axis[index]), scan, index))
    combined.sort(key=lambda item: item[0])
    omega_c = np.asarray([item[0] for item in combined], dtype=float)
    if omega_c.size > 1 and np.any(np.diff(omega_c) <= 0):
        raise ValueError("signed-window datasets hold duplicate omega_c points")
    epsilon = omega_c - critical_value
    order = np.argsort(epsilon)

    r_aa: list[float] = []
    r_aa_sem: list[float] = []
    frequency: list[float] = []
    frequency_sem: list[float] = []
    band_limited_frequency: list[float] = []
    band_limited_sem: list[float] = []
    band_limited_status: list[str] = []
    band_limited_estimates = []
    psd_peak_frequency: list[float] = []
    stationarity: list[float] = []
    first_passage_hits: list[int] = []
    soft_raw_moments: list[np.ndarray] = []
    soft_cumulants: list[np.ndarray] = []
    soft_per_trajectory_raw_moments: list[np.ndarray] = []
    soft_matrix: np.ndarray | None = None
    soft_reference_matrix: np.ndarray | None = None
    soft_center: float | None = None
    for index, (_, scan, local_index) in enumerate(combined):
        analysis = _point_analysis(scan, local_index)
        coherence = analysis["coherence_matrix"]
        carrier = analysis["coherence_carrier"]
        names = list(carrier["measurement_names"])
        trace_index = names.index("trace")
        statuses = np.asarray(carrier["status"], dtype=str)
        if statuses[trace_index] != "ok":
            raise ValueError(
                f"signed-window trace carrier failed at point {index}: "
                f"{carrier['error'][trace_index]}"
            )
        matrix = np.asarray(coherence["matrix"], dtype=complex)
        matrix_sem = np.asarray(coherence["matrix_sem_real"], dtype=float)
        r_aa.append(float(matrix[0, 0].real))
        r_aa_sem.append(float(matrix_sem[0, 0]))
        frequency.append(float(carrier["frequency"][trace_index]))
        frequency_sem.append(float(carrier["frequency_sem"][trace_index]))
        psd = analysis["psd"]
        psd_axis = np.asarray(psd["axis"], dtype=float)
        trace_psd = np.sum(np.asarray(psd["psd"], dtype=float), axis=1)
        estimate, _ = estimate_band_limited_carrier(
            psd_axis,
            trace_psd,
            freq_min=-0.3,
            freq_max=-0.1,
            maximum_lag=256.0,
        )
        band_limited_estimates.append(estimate)
        band_limited_frequency.append(float(estimate.frequency))
        band_limited_sem.append(float(estimate.diagnostic_uncertainty))
        band_limited_status.append(str(estimate.status))
        in_band = (psd_axis >= -0.3) & (psd_axis <= -0.1)
        peak_index = np.flatnonzero(in_band)[int(np.argmax(trace_psd[in_band]))]
        psd_peak_frequency.append(float(psd_axis[peak_index]))
        stationarity.append(
            float(coherence["time_blocks"]["first_last_matrix_distance"])
        )
        first_passage_hits.append(int(analysis["first_passage"]["n_hit"]))
        if "quadratic_moments" in analysis:
            quadratic = analysis["quadratic_moments"]
            names = list(quadratic["observable_names"])
            soft_index = names.index("critical_soft_coordinate")
            soft_raw_moments.append(
                np.asarray(quadratic["raw_moments"], dtype=float)[soft_index]
            )
            soft_cumulants.append(
                np.asarray(quadratic["cumulants"], dtype=float)[soft_index]
            )
            soft_per_trajectory_raw_moments.append(
                np.asarray(quadratic["per_trajectory_raw_moments"], dtype=float)[
                    :, soft_index
                ]
            )
            point_matrix = np.asarray(quadratic["matrices"], dtype=complex)[soft_index]
            point_reference = np.asarray(
                quadratic["reference_matrices"], dtype=complex
            )[soft_index]
            point_center = float(
                np.asarray(quadratic["centers"], dtype=float)[soft_index]
            )
            if soft_matrix is None:
                soft_matrix = point_matrix
                soft_reference_matrix = point_reference
                soft_center = point_center
            elif not (
                np.array_equal(soft_matrix, point_matrix)
                and np.array_equal(soft_reference_matrix, point_reference)
                and soft_center == point_center
            ):
                raise ValueError("signed-window soft-coordinate definitions differ")

    sorted_epsilon = epsilon[order]
    sorted_estimates = [band_limited_estimates[int(index)] for index in order]
    tracked = track_band_limited_carrier(sorted_epsilon, sorted_estimates)
    arrays = {
        "flow_sde_epsilon": epsilon[order],
        "flow_sde_r_aa": np.asarray(r_aa, dtype=float)[order],
        "flow_sde_r_aa_sem": np.asarray(r_aa_sem, dtype=float)[order],
        "flow_sde_trace_frequency": np.asarray(frequency, dtype=float)[order],
        "flow_sde_trace_frequency_sem": np.asarray(frequency_sem, dtype=float)[order],
        "flow_sde_band_limited_frequency": np.asarray(
            [item["tracked_frequency"] for item in tracked], dtype=float
        ),
        "flow_sde_band_limited_frequency_sem": np.asarray(
            [item["tracked_diagnostic_uncertainty"] for item in tracked],
            dtype=float,
        ),
        "flow_sde_band_limited_status": np.asarray(
            [item["tracked_status"] for item in tracked], dtype=str
        ),
        "flow_sde_band_limited_local_frequency": np.asarray(
            band_limited_frequency, dtype=float
        )[order],
        "flow_sde_band_limited_local_sem": np.asarray(band_limited_sem, dtype=float)[
            order
        ],
        "flow_sde_band_limited_local_status": np.asarray(
            band_limited_status, dtype=str
        )[order],
        "flow_sde_psd_peak_frequency": np.asarray(psd_peak_frequency, dtype=float)[
            order
        ],
        "flow_sde_stationarity": np.asarray(stationarity, dtype=float)[order],
        "flow_sde_first_passage_hits": np.asarray(first_passage_hits, dtype=int)[order],
    }
    if soft_raw_moments:
        arrays.update(
            {
                "flow_sde_soft_raw_moments": np.asarray(soft_raw_moments, dtype=float)[
                    order
                ],
                "flow_sde_soft_cumulants": np.asarray(soft_cumulants, dtype=float)[
                    order
                ],
                "flow_sde_soft_per_trajectory_raw_moments": np.asarray(
                    soft_per_trajectory_raw_moments, dtype=float
                )[order],
                "flow_sde_soft_matrix": np.asarray(soft_matrix, dtype=complex),
                "flow_sde_soft_reference_matrix": np.asarray(
                    soft_reference_matrix, dtype=complex
                ),
                "flow_sde_soft_center": np.asarray(soft_center, dtype=float),
            }
        )
    manifest_candidates = (
        dataset_paths[0] / "artifact_manifest.json",
        dataset_paths[0].parent / "artifact_manifest.json",
    )
    manifest_path = next((path for path in manifest_candidates if path.exists()), None)
    if fit_path is not None and fit_path.exists():
        fit_rows = _read_csv(fit_path)
        if len(fit_rows) != omega_c.size:
            raise ValueError("signed-window Lorentz fit count does not match scan")
        if "omega_c" in fit_rows[0]:
            fit_omega_c = np.asarray(
                [float(row["omega_c"]) for row in fit_rows], dtype=float
            )
            if not np.allclose(fit_omega_c, omega_c, rtol=0.0, atol=1e-12):
                raise ValueError("signed-window Lorentz fit grid does not match scan")
        arrays["flow_sde_lorentz_frequency"] = np.asarray(
            [float(row["center"]) for row in fit_rows], dtype=float
        )[order]
        arrays["flow_sde_lorentz_frequency_sem"] = np.asarray(
            [float(row["center_std"]) for row in fit_rows], dtype=float
        )[order]
        arrays["flow_sde_lorentz_r2"] = np.asarray(
            [float(row["R2"]) for row in fit_rows], dtype=float
        )[order]
    metadata = {
        "stem": stem,
        "point_count": int(omega_c.size),
        "dataset": [str(path.resolve()) for path in dataset_paths],
        "artifact_manifest": (
            _source(manifest_path) if manifest_path is not None else None
        ),
        "fit_results": (
            _source(fit_path) if fit_path is not None and fit_path.exists() else None
        ),
    }
    return arrays, metadata


def _normal_form_adjustment(
    arrays: dict[str, np.ndarray], *, bootstrap_samples: int = 4000
) -> dict[str, np.ndarray]:
    if "flow_sde_soft_raw_moments" not in arrays:
        return {}
    matrix = np.asarray(arrays["flow_sde_soft_matrix"], dtype=complex)
    center = float(arrays["flow_sde_soft_center"])
    branch_states = np.asarray(
        vector_to_matrix(arrays["flow_branch_state_vectors"], 3), dtype=complex
    )
    branch_x = np.einsum("ij,nji->n", matrix, branch_states).real - center
    branch_r_aa = np.asarray(arrays["flow_branch_r_aa"], dtype=float)
    critical_r_aa = float(arrays["flow_critical_r_aa"])
    critical_right = np.asarray(arrays["flow_critical_right_vector"], dtype=float)

    # Constrain the embedding to its independently normalized linear tangent,
    # then estimate only the quadratic and cubic center-manifold coefficients.
    residual = branch_r_aa - critical_r_aa - critical_right[0] * branch_x
    design = np.column_stack((branch_x**2, branch_x**3))
    nonlinear, *_ = np.linalg.lstsq(design, residual, rcond=None)
    coefficients = np.asarray(
        [critical_r_aa, critical_right[0], nonlinear[0], nonlinear[1]]
    )

    def embed(coordinate: np.ndarray) -> np.ndarray:
        return np.polynomial.polynomial.polyval(coordinate, coefficients)

    raw = np.asarray(arrays["flow_sde_soft_raw_moments"], dtype=float)
    x_adjusted = np.cbrt(raw[:, 2])
    r_aa_adjusted = embed(x_adjusted)
    per_trajectory = np.asarray(
        arrays["flow_sde_soft_per_trajectory_raw_moments"], dtype=float
    )
    rng = np.random.default_rng(20260821)
    bootstrap = np.empty((bootstrap_samples, raw.shape[0]), dtype=float)
    for point in range(raw.shape[0]):
        values = per_trajectory[point, :, 2]
        indices = rng.integers(0, values.size, size=(bootstrap_samples, values.size))
        bootstrap[:, point] = embed(np.cbrt(np.mean(values[indices], axis=1)))
    lower, upper = np.percentile(bootstrap, [2.5, 97.5], axis=0)
    return {
        "flow_center_manifold_x": branch_x,
        "flow_center_manifold_r_aa_coefficients": coefficients,
        "flow_sde_soft_x_adjusted": x_adjusted,
        "flow_sde_r_aa_adjusted": r_aa_adjusted,
        "flow_sde_r_aa_adjusted_ci_lower": lower,
        "flow_sde_r_aa_adjusted_ci_upper": upper,
        "flow_sde_r_aa_adjusted_extrapolated": (
            (x_adjusted < np.min(branch_x)) | (x_adjusted > np.max(branch_x))
        ),
        "flow_center_manifold_x_range": np.asarray(
            [np.min(branch_x), np.max(branch_x)]
        ),
    }


def _source(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size_bytes": stat.st_size,
        "modified_utc": datetime.fromtimestamp(stat.st_mtime, tz=UTC).isoformat(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sde-session", type=Path, required=True)
    parser.add_argument("--cam-result", type=Path, required=True)
    parser.add_argument(
        "--dense-cam-result",
        type=Path,
        help="dense-grid CAM scan supplying smooth fig-3 CAM curves",
    )
    parser.add_argument("--bifurcation-result", type=Path, required=True)
    parser.add_argument("--bifurcation-case", type=int, default=3)
    parser.add_argument("--critical-value", type=float, required=True)
    parser.add_argument("--stem", default=DEFAULT_STEM)
    parser.add_argument(
        "--carrier-csv",
        type=Path,
        help=(
            "ridge-conditioned carrier_results.csv replacing the adaptive "
            "band-limited carrier (same algorithm as the fig-1c positive branch)"
        ),
    )
    parser.add_argument("--signed-sde-session", type=Path)
    parser.add_argument(
        "--signed-dataset",
        type=Path,
        action="append",
        help=(
            "signed-window SDE dataset; repeat to merge multiple sessions "
            "tiling disjoint omega_c points of the same window"
        ),
    )
    parser.add_argument("--signed-fit-csv", type=Path)
    parser.add_argument(
        "--signed-stem",
        default="collective_loss_kerr_3mode_refined_dark_signed_figure1",
    )
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    sde_job = args.sde_session / f"{args.stem}_sim"
    fit_csv = args.sde_session / f"{args.stem}_fit" / "fit_results.csv"
    allan_csv = args.sde_session / f"{args.stem}_allan" / "allan_points.csv"
    output_dir = args.output_dir or (
        args.sde_session / "postprocess" / "collective_loss_kerr_3mode_refined_dark"
    )

    scan_result = _load_v4_scan(sde_job)
    omega_c = np.asarray(scan_result.axes["omega_c"], dtype=float)
    point_analyses = [
        _point_analysis(scan_result, index) for index in range(omega_c.size)
    ]
    analysis = {
        name: [point[name] for point in point_analyses]
        for name in ("psd", "allan_variance")
    }
    # Escape detection may be disabled (no first_passage analysis product).
    first_passage = [point.get("first_passage") for point in point_analyses]
    epsilon = np.abs(omega_c - args.critical_value)

    psd_points = analysis["psd"]
    carrier = []
    for point in psd_points:
        estimate, _ = estimate_band_limited_carrier(
            np.asarray(point["axis"], dtype=float),
            np.asarray(point["psd"], dtype=float)[:, 0],
            freq_min=-0.5,
            freq_max=0.2,
            bandwidth_multipliers=[0.5, 1.0, 2.0, 4.0, 8.0],
            maximum_lag=256.0,
        )
        carrier.append(estimate)
    carrier_tracked = track_band_limited_carrier(omega_c, carrier)
    if args.carrier_csv is not None:
        ridge_carrier = _load_ridge_carrier_csv(args.carrier_csv, omega_c)
        carrier_frequency = ridge_carrier["frequency"]
        carrier_sem = ridge_carrier["sem"]
        carrier_status = ridge_carrier["status"]
        carrier_local_frequency = ridge_carrier["frequency"]
        carrier_local_sem = ridge_carrier["sem"]
        carrier_local_status = ridge_carrier["status"]
    else:
        carrier_frequency = np.asarray(
            [item["tracked_frequency"] for item in carrier_tracked], dtype=float
        )
        carrier_sem = np.asarray(
            [item["tracked_diagnostic_uncertainty"] for item in carrier_tracked],
            dtype=float,
        )
        carrier_status = np.asarray(
            [item["tracked_status"] for item in carrier_tracked], dtype=str
        )
        carrier_local_frequency = np.asarray(
            [item.frequency for item in carrier], dtype=float
        )
        carrier_local_sem = np.asarray(
            [item.diagnostic_uncertainty for item in carrier], dtype=float
        )
        carrier_local_status = np.asarray(
            [item.status for item in carrier], dtype=str
        )

    # The Lorentz-fit job is optional (not part of every workflow); when absent
    # the sde_lorentz_* arrays are simply omitted from the dataset.
    fit_rows = _read_csv(fit_csv) if fit_csv.exists() else None
    allan_rows = _read_csv(allan_csv)
    if len(allan_rows) != epsilon.size:
        raise ValueError("SDE and Allan point counts do not align")
    if fit_rows is not None and len(fit_rows) != epsilon.size:
        raise ValueError("SDE and fit point counts do not align")

    allan_payloads = [_mode_payload(point) for point in analysis["allan_variance"]]
    tau = np.stack([np.asarray(item["tau"], dtype=float) for item in allan_payloads])
    variance = np.stack(
        [
            np.asarray(item["angular_frequency_variance"], dtype=float)
            for item in allan_payloads
        ]
    )
    variance_sem = np.stack(
        [
            np.asarray(item["angular_frequency_variance_sem"], dtype=float)
            for item in allan_payloads
        ]
    )
    nonoverlap_per_trajectory = np.stack(
        [
            np.asarray(item["nonoverlap_per_trajectory"], dtype=float)
            for item in allan_payloads
        ]
    )
    independent_windows = np.stack(
        [
            np.asarray(item["total_independent_window_count"], dtype=int)
            for item in allan_payloads
        ]
    )

    accepted = np.asarray([row["accepted"].lower() == "true" for row in allan_rows])
    selected = np.asarray([row["selected"].lower() == "true" for row in allan_rows])
    allan_intensity = np.asarray(
        [
            float(row["allan_intensity"]) if accepted[i] else np.nan
            for i, row in enumerate(allan_rows)
        ]
    )
    allan_intensity_sem = np.asarray(
        [
            float(row["allan_intensity_sem"]) if accepted[i] else np.nan
            for i, row in enumerate(allan_rows)
        ]
    )
    # Non-accepted points fall back to the Allan intensity at the longest common
    # averaging time (fixed-bandwidth proxy), identical to the plateau value when
    # a white-FM plateau is resolved within the record.
    tau_reference = tau[:, -1]
    allan_proxy = np.asarray(
        [
            np.nan if accepted[i] else float(tau_reference[i] * variance[i, -1])
            for i in range(epsilon.size)
        ]
    )
    allan_proxy_sem = np.asarray(
        [
            np.nan if accepted[i] else float(tau_reference[i] * variance_sem[i, -1])
            for i in range(epsilon.size)
        ]
    )

    cam_result = _load_cam_result(args.cam_result)
    cam_omega_c = np.asarray(cam_result.axes["omega_c"], dtype=float).reshape(-1)
    if not np.allclose(cam_omega_c, omega_c, rtol=0.0, atol=1e-12):
        raise ValueError("CAM and SDE scan axes do not align")
    cam_frequency, cam_rate, cam_states, cam_slots = _select_cam_branch(cam_result)
    traces = np.real(np.trace(cam_states, axis1=1, axis2=2))
    normalized = cam_states / traces[:, None, None]

    arrays: dict[str, np.ndarray] = {
        "omega_c": omega_c,
        "epsilon": epsilon,
        "sde_carrier_frequency": carrier_frequency,
        "sde_carrier_sem": carrier_sem,
        "sde_carrier_status": carrier_status,
        "sde_carrier_local_frequency": carrier_local_frequency,
        "sde_carrier_local_sem": carrier_local_sem,
        "sde_carrier_local_status": carrier_local_status,
        "allan_tau": tau,
        "allan_variance": variance,
        "allan_variance_sem": variance_sem,
        "allan_nonoverlap_per_trajectory": nonoverlap_per_trajectory,
        "allan_independent_windows": independent_windows,
        "allan_accepted": accepted,
        "allan_selected": selected,
        "allan_intensity": allan_intensity,
        "allan_intensity_sem": allan_intensity_sem,
        "allan_proxy": allan_proxy,
        "allan_proxy_sem": allan_proxy_sem,
        "allan_proxy_tau": tau_reference,
        "allan_tau_start": np.asarray(
            [
                float(row["tau_start"]) if accepted[i] else np.nan
                for i, row in enumerate(allan_rows)
            ]
        ),
        "allan_tau_end": np.asarray(
            [
                float(row["tau_end"]) if accepted[i] else np.nan
                for i, row in enumerate(allan_rows)
            ]
        ),
        "cam_solution_slot": cam_slots,
        "cam_frequency": cam_frequency,
        "cam_critical_rate": cam_rate,
        "cam_states": cam_states,
        "cam_r_aa": np.real(cam_states[:, 0, 0]),
        "cam_purity": np.real(np.trace(normalized @ normalized, axis1=1, axis2=2)),
    }
    if fit_rows is not None:
        arrays.update(
            {
                "sde_lorentz_center": np.asarray(
                    [float(row["center"]) for row in fit_rows]
                ),
                "sde_lorentz_center_sem": np.asarray(
                    [float(row["center_std"]) for row in fit_rows]
                ),
                "sde_lorentz_linewidth": np.asarray(
                    [float(row["linewidth"]) for row in fit_rows]
                ),
                "sde_lorentz_linewidth_sem": np.asarray(
                    [float(row["linewidth_std"]) for row in fit_rows]
                ),
                "sde_lorentz_r2": np.asarray([float(row["R2"]) for row in fit_rows]),
            }
        )
    if all(item is not None for item in first_passage):
        arrays["first_passage_hits"] = np.asarray(
            [int(item["n_hit"]) for item in first_passage], dtype=int
        )
    arrays.update(_cam_noise_predictions(cam_result, cam_states))
    if args.dense_cam_result is not None:
        arrays.update(_dense_cam_arrays(_load_dense_cam_result(args.dense_cam_result)))
    eigenstate = _self_consistent_eigenstate_branch(cam_result, cam_states)
    arrays.update(eigenstate)
    flow_arrays, flow_metadata = _build_flow(
        args.bifurcation_result, args.bifurcation_case
    )
    arrays.update(flow_arrays)
    signed_metadata = None
    if args.signed_dataset is not None:
        signed_arrays, signed_metadata = _load_signed_sde_window(
            args.signed_dataset,
            args.signed_fit_csv,
            args.critical_value,
            stem=args.signed_stem,
        )
        arrays.update(signed_arrays)
        arrays.update(_normal_form_adjustment(arrays))
    elif args.signed_sde_session is not None:
        signed_dataset = args.signed_sde_session / f"{args.signed_stem}_sim"
        signed_fit_csv = (
            args.signed_sde_session / f"{args.signed_stem}_fit" / "fit_results.csv"
        )
        signed_arrays, signed_metadata = _load_signed_sde_window(
            signed_dataset,
            signed_fit_csv,
            args.critical_value,
            stem=args.signed_stem,
        )
        arrays.update(signed_arrays)
        arrays.update(_normal_form_adjustment(arrays))

    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = output_dir / "plot_data.npz"
    np.savez_compressed(data_path, **arrays)
    manifest = {
        "schema": SCHEMA,
        "generated_utc": datetime.now(tz=UTC).isoformat(),
        "critical_value": args.critical_value,
        "stem": args.stem,
        "sources": {
            "sde_result": _source(sde_job / "artifact_manifest.json"),
            **({"fit_csv": _source(fit_csv)} if fit_rows is not None else {}),
            "allan_csv": _source(allan_csv),
            "cam_result": _source(args.cam_result / "artifact_manifest.json"),
            "bifurcation_result": _source(
                args.bifurcation_result / "artifact_manifest.json"
            ),
            **(
                {
                    "dense_cam_result": _source(
                        args.dense_cam_result / "artifact_manifest.json"
                    )
                }
                if args.dense_cam_result is not None
                else {}
            ),
        },
        "flow": flow_metadata,
        "signed_sde": signed_metadata,
        "carrier_source": (
            {
                "type": "ridge_conditioned_csv",
                "path": str(args.carrier_csv.resolve()),
            }
            if args.carrier_csv is not None
            else {"type": "adaptive_band_limited"}
        ),
        "carrier_estimates": [asdict(item) for item in carrier],
        "point_count": int(epsilon.size),
        "local_white_fm_points": int(np.count_nonzero(accepted)),
        "common_white_fm_points": int(np.count_nonzero(selected)),
        "eigenstate": {
            "definition": "R=alpha alpha^dagger; H(R) alpha=omega alpha; omega real",
            "maximum_normalized_residual": float(
                np.max(eigenstate["eigenstate_residual"])
            ),
        },
        "output": str(data_path.resolve()),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(data_path)


if __name__ == "__main__":
    main()
