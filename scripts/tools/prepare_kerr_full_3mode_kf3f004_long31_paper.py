"""Prepare paper plot data from aligned kf3f_004 long-31 replicates."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "tools"))
sys.path.insert(0, str(ROOT / "scripts" / "figures"))

from evaluate_kerr_full_3mode_allan_free_puiseux_orders import (  # noqa: E402
    PRIMARY_BOUNDS as ALLAN_EXPONENT_BOUNDS,
)
from evaluate_kerr_full_3mode_allan_free_puiseux_orders import (  # noqa: E402
    _fit as _modified_allan_fit,
)
from evaluate_kerr_full_3mode_cam_anchored_free_puiseux import (  # noqa: E402
    PRIMARY_BOUNDS as FREQUENCY_EXPONENT_BOUNDS,
)
from evaluate_kerr_full_3mode_cam_anchored_free_puiseux import (  # noqa: E402
    _fit as _cam_anchored_frequency_fit,
)
from plot_collective_loss_kerr_3mode_paper import (  # noqa: E402
    _local_slope,
)
from prepare_collective_loss_kerr_3mode_refined_dark import (  # noqa: E402
    _distinct_real_branches,
    _pack_eigenstate,
    _track_eigenvalues,
    _two_sided_power_fit,
    _unpack_eigenstate,
)
from prepare_kerr_full_3mode_kf3f004_q1_paper import _cam_metrics  # noqa: E402
from qphase.backend.numpy_backend import NumpyBackend  # noqa: E402
from qphase.data import Dataset, load_bundle, stack_datasets  # noqa: E402
from qphase_cam.bifurcation_result import CAMBifurcationScanResult  # noqa: E402
from qphase_cam.core.coordinates import matrix_to_vector  # noqa: E402
from qphase_cam.core.fpgen import FPGenDynamicsAdapter  # noqa: E402
from qphase_cam.postprocessor.local_response import (  # noqa: E402
    LocalResponseValidation,
    LocalResponseValidationConfig,
)
from qphase_cam.postprocessor.stochastic_validity import (  # noqa: E402
    StochasticValidity,
    canonical_sample_matrix_covariance,
)
from qphase_sde.analyser.allan_scaling import AllanScalingAnalyzer  # noqa: E402
from qphase_sde.analyser.allan_variance import (  # noqa: E402
    AllanResultAccumulator,
)
from qphase_sde.products import add_scan_parameter_coordinates  # noqa: E402
from summarize_kerr_full_3mode_frequency_q0 import (  # noqa: E402
    _product_payload,
)
from validate_kerr_full_3mode_frequency_search import (  # noqa: E402
    _snapshot_model_params,
)

from models.kerr_full_3mode import KerrFull3ModeModel  # noqa: E402

SEARCH = ROOT / "reports/models/kerr_full_3mode/frequency_search_20260904"
DEFAULT_OUTPUT = SEARCH / "kf3f_004_long31_paper"
OLD_SESSION = ROOT / "runs/2026/09/2026-09-05T03-54-15_a33ebb"
NEW_SESSION = ROOT / "runs/2026/09/2026-09-27T11-28-19_f4e04c"
OLD_SIM_JOB = "kerr_full_3mode_kf3f_004_a5000_long_31_sim"
NEW_SIM_JOB = "kerr_full_3mode_kf3f_004_publication_long31_a5000_seed_a_sim"
NEW_CARRIER_JOB = "kerr_full_3mode_kf3f_004_publication_long31_a5000_seed_a_carrier"
RESPONSE_SESSION = ROOT / "runs/2026/09/2026-09-28T00-37-37_4b815a"
RESPONSE_SIM_JOB = (
    "kerr_full_3mode_kf3f_004_publication_response_stencil_a5000_seed_a_sim"
)
DEFAULT_SIMULATION_ARTIFACTS = (
    OLD_SESSION / OLD_SIM_JOB,
    NEW_SESSION / NEW_SIM_JOB,
)
DEFAULT_FREQUENCY_CARRIERS = (
    DEFAULT_OUTPUT / "carrier_lag16_diagnostic/carrier_results.csv",
    NEW_SESSION / NEW_CARRIER_JOB / "carrier_results.csv",
)
DEFAULT_RESPONSE_ARTIFACTS = (RESPONSE_SESSION / RESPONSE_SIM_JOB,)
V0_JOB = SEARCH / "v0" / "kerr_full_3mode_d0_omega_b_gamma_b_gamma_c"
PRIMARY_MODE = 2
FLOW_EPSILON_LIMIT = 2.0e-4
FREQUENCY_PUISEUX_ORDER = 4
FREQUENCY_EXPONENT_DISPLAY_UNCERTAINTY = 1.0e-4
ALLAN_PUISEUX_ORDER = 4
ALLAN_Q_DISPLAY_DECIMALS = 4
ETA2_NONLINEAR_THRESHOLD = 0.333
STOCHASTIC = StochasticValidity()
PHASE_FREQUENCY_KEY = (
    f"mode_results.{PRIMARY_MODE}.phase_increment.mean_angular_frequency_per_trajectory"
)
RESPONSE_HALF_WINDOW = 2
RESPONSE_POLYNOMIAL_DEGREES = (2, 3, 4)
REFERENCE_RESPONSE_HALF_WINDOW = 1
REFERENCE_RESPONSE_POLYNOMIAL_DEGREES = (2, 3, 4)


class _ReducedCAMFlow:
    """One-dimensional CAM flow obtained by algebraic stable-mode elimination."""

    def __init__(
        self,
        adapter: FPGenDynamicsAdapter,
        critical_params: dict[str, float],
    ) -> None:
        self.adapter = adapter
        self.critical_params = critical_params

    def evaluate(self, r_aa: float, epsilon: float) -> tuple[float, np.ndarray]:
        params = dict(self.critical_params)
        params["omega_b"] += epsilon
        vector = np.zeros(self.adapter.state_size, dtype=float)
        vector[0] = r_aa
        residual = np.asarray(self.adapter.rhs(vector, params), dtype=float)
        stable_block = np.asarray(
            self.adapter.jacobian(vector, params)[1:, 1:], dtype=float
        )
        vector[1:] = np.linalg.solve(stable_block, -residual[1:])
        full_residual = np.asarray(self.adapter.rhs(vector, params), dtype=float)
        eliminated_residual = float(np.linalg.norm(full_residual[1:], ord=np.inf))
        residual_scale = max(1.0, float(np.linalg.norm(residual[1:], ord=np.inf)))
        if eliminated_residual > 1.0e-10 * residual_scale:
            raise RuntimeError(
                "failed to eliminate the non-order-parameter CAM equations: "
                f"relative residual {eliminated_residual / residual_scale:.3e}"
            )
        return float(full_residual[0]), vector


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted(set().union(*(row for row in rows))) if rows else ()
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _array(product: Dataset, name: str) -> np.ndarray:
    return np.asarray(product.handle(name).materialize())


def _load_aligned_replicates(paths: list[Path]) -> tuple[list[Any], np.ndarray]:
    if not paths:
        raise ValueError("at least one simulation artifact is required")
    bundles = [load_bundle(path) for path in paths]
    scan_values = _array(
        bundles[0].products["coherence_matrix"], "parameter.omega_b"
    ).astype(float)
    for path, bundle in zip(paths[1:], bundles[1:], strict=True):
        candidate = _array(
            bundle.products["coherence_matrix"], "parameter.omega_b"
        ).astype(float)
        if candidate.shape != scan_values.shape or not np.allclose(
            candidate, scan_values, rtol=0.0, atol=1.0e-12
        ):
            raise ValueError(f"simulation scan grid is not aligned: {path}")
        for product_name in (
            "allan_variance",
            "coherence_matrix",
            "moment_statistics",
            "first_passage",
        ):
            if product_name not in bundle.products:
                raise ValueError(f"{path} is missing product {product_name!r}")
    return bundles, scan_values


def _carrier_summary(
    paths: list[Path], scan_values: np.ndarray
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    if not paths:
        raise ValueError("at least one carrier CSV is required")
    frequencies = []
    diagnostic_upper = []
    for path in paths:
        rows = sorted(_read_csv(path), key=lambda row: float(row["omega_b"]))
        candidate_scan = np.asarray(
            [float(row["omega_b"]) for row in rows], dtype=float
        )
        if (
            len(rows) != scan_values.size
            or not np.allclose(candidate_scan, scan_values, rtol=0.0, atol=1.0e-12)
            or any(row["status"] != "ok" for row in rows)
        ):
            raise ValueError(
                f"carrier CSV must provide one accepted aligned row per point: {path}"
            )
        frequencies.append([float(row["frequency"]) for row in rows])
        diagnostic_upper.append(
            [float(row["ridge_conditioned_uncertainty_upper"]) for row in rows]
        )

    frequency_by_replicate = np.asarray(frequencies, dtype=float)
    diagnostic_by_replicate = np.asarray(diagnostic_upper, dtype=float)
    replicate_count = frequency_by_replicate.shape[0]
    frequency = np.mean(frequency_by_replicate, axis=0)
    if replicate_count > 1:
        seed_sem = np.std(frequency_by_replicate, axis=0, ddof=1) / math.sqrt(
            replicate_count
        )
    else:
        seed_sem = np.zeros(scan_values.size, dtype=float)
    propagated_diagnostic = (
        np.sqrt(np.sum(diagnostic_by_replicate**2, axis=0)) / replicate_count
    )
    effective_uncertainty = np.hypot(seed_sem, propagated_diagnostic)
    rows = [
        {
            "omega_b": scan_value,
            "frequency": frequency[index],
            "frequency_seed_sem": seed_sem[index],
            "frequency_propagated_diagnostic": propagated_diagnostic[index],
            "frequency_effective_uncertainty": effective_uncertainty[index],
            "replicate_count": replicate_count,
            **{
                f"replicate_{replicate}_frequency": frequency_by_replicate[
                    replicate, index
                ]
                for replicate in range(replicate_count)
            },
        }
        for index, scan_value in enumerate(scan_values)
    ]
    return (
        {
            "frequency": frequency,
            "effective_uncertainty": effective_uncertainty,
            "seed_sem": seed_sem,
            "propagated_diagnostic": propagated_diagnostic,
            "frequency_by_replicate": frequency_by_replicate,
            "diagnostic_by_replicate": diagnostic_by_replicate,
        },
        rows,
    )


def _phase_frequency_summary(
    bundles: list[Any], scan_values: np.ndarray
) -> dict[str, np.ndarray]:
    per_replicate = []
    for bundle in bundles:
        product = bundle.products["allan_variance"]
        candidate_scan = _array(product, "parameter.omega_b").astype(float)
        if candidate_scan.shape != scan_values.shape or not np.allclose(
            candidate_scan, scan_values, rtol=0.0, atol=1.0e-12
        ):
            raise ValueError("phase-frequency scan grid is not aligned")
        per_replicate.append(_array(product, PHASE_FREQUENCY_KEY).astype(float))
    by_replicate = np.asarray(
        [np.mean(values, axis=1) for values in per_replicate], dtype=float
    )
    pooled = np.concatenate(per_replicate, axis=1)
    frequency = np.mean(pooled, axis=1)
    trajectory_sem = np.std(pooled, axis=1, ddof=1) / math.sqrt(pooled.shape[1])
    seed_sem = (
        np.std(by_replicate, axis=0, ddof=1) / math.sqrt(by_replicate.shape[0])
        if by_replicate.shape[0] > 1
        else np.zeros(scan_values.size, dtype=float)
    )
    return {
        "frequency": frequency,
        "trajectory_sem": trajectory_sem,
        "seed_sem": seed_sem,
        "frequency_by_replicate": by_replicate,
        "trajectory_count": np.asarray(pooled.shape[1]),
    }


def _phase_response_summary(
    bundles: list[Any], center_epsilon: np.ndarray
) -> dict[str, np.ndarray]:
    if not bundles:
        raise ValueError("at least one response-stencil artifact is required")
    per_replicate = []
    scan_values = None
    for bundle in bundles:
        product = bundle.products["allan_variance"]
        candidate_scan = _array(product, "parameter.omega_b").astype(float)
        if scan_values is None:
            scan_values = candidate_scan
        elif candidate_scan.shape != scan_values.shape or not np.allclose(
            candidate_scan, scan_values, rtol=0.0, atol=1.0e-12
        ):
            raise ValueError("response-stencil scan grids are not aligned")
        per_replicate.append(_array(product, PHASE_FREQUENCY_KEY).astype(float))
    assert scan_values is not None
    per_trajectory = np.concatenate(per_replicate, axis=1)
    if scan_values.size != 2 * center_epsilon.size:
        raise ValueError("response-stencil phase product has the wrong point count")
    frequency = np.mean(per_trajectory, axis=1)
    frequency_sem = np.std(per_trajectory, axis=1, ddof=1) / math.sqrt(
        per_trajectory.shape[1]
    )
    minus_epsilon = scan_values[0::2] - 0.5
    plus_epsilon = scan_values[1::2] - 0.5
    centers = 0.5 * (minus_epsilon + plus_epsilon)
    if not np.allclose(centers, center_epsilon, rtol=0.0, atol=2.0e-15):
        raise ValueError("response-stencil centers do not match the long31 grid")
    spacing = plus_epsilon - minus_epsilon
    if not np.allclose(spacing, 0.2 * center_epsilon, rtol=2.0e-12, atol=1.0e-15):
        raise ValueError("response-stencil phase points are not +/-10 percent")
    response = (frequency[1::2] - frequency[0::2]) / spacing
    response_sem = np.hypot(frequency_sem[1::2], frequency_sem[0::2]) / spacing
    accepted = np.isfinite(response) & np.isfinite(response_sem) & (response_sem >= 0.0)
    return {
        "endpoint_frequency": frequency,
        "endpoint_frequency_sem": frequency_sem,
        "response": response,
        "response_sem": response_sem,
        "accepted": accepted,
        "trajectory_count": np.asarray(per_trajectory.shape[1]),
        "replicate_count": np.asarray(len(per_replicate)),
        "frequency_by_replicate": np.asarray(
            [np.mean(values, axis=1) for values in per_replicate], dtype=float
        ),
    }


def _weighted_polynomial_response(
    x: np.ndarray,
    y: np.ndarray,
    sigma: np.ndarray,
    center: float,
    degree: int,
) -> tuple[float, float, float, np.ndarray]:
    if center <= 0.0 or np.any(x <= 0.0):
        raise ValueError("log-coordinate response fit requires positive epsilon")
    coordinate = np.log(x / center)
    design = np.column_stack([coordinate**power for power in range(degree + 1)])
    weighted_design = design / sigma[:, None]
    weighted_target = y / sigma
    pseudoinverse = np.linalg.pinv(weighted_design)
    coefficients = pseudoinverse @ weighted_target
    covariance = pseudoinverse @ pseudoinverse.T
    residual = (y - design @ coefficients) / sigma
    rank = int(np.linalg.matrix_rank(weighted_design))
    if rank != degree + 1:
        raise ValueError(
            f"rank-deficient local polynomial design: rank={rank}, degree={degree}"
        )
    dof = y.size - rank
    reduced_chi2 = float(residual @ residual / dof)
    slope = float(coefficients[1] / center)
    slope_weights = pseudoinverse[1] / (sigma * center)
    slope_sem = float(
        np.sqrt(covariance[1, 1]) / center * max(1.0, math.sqrt(reduced_chi2))
    )
    return slope, slope_sem, reduced_chi2, slope_weights


def _local_polynomial_response(
    epsilon: np.ndarray,
    center_frequency: np.ndarray,
    center_sem: np.ndarray,
    response: dict[str, np.ndarray],
    *,
    half_window: int,
    degrees: tuple[int, int, int],
) -> dict[str, np.ndarray]:
    count = epsilon.size
    lower_degree, primary_degree, upper_degree = degrees
    slopes_by_degree = {
        degree: np.full(count, np.nan) for degree in degrees
    }
    formal_sem = np.full(count, np.nan)
    reduced_chi2 = np.full(count, np.nan)
    slope_weights = np.zeros((count, 3 * count), dtype=float)
    endpoint_frequency = response["endpoint_frequency"]
    endpoint_sem = response["endpoint_frequency_sem"]
    for index in range(half_window, count - half_window):
        x_values = []
        y_values = []
        sigma_values = []
        for neighbor in range(index - half_window, index + half_window + 1):
            x_values.extend(
                (0.9 * epsilon[neighbor], epsilon[neighbor], 1.1 * epsilon[neighbor])
            )
            y_values.extend(
                (
                    endpoint_frequency[2 * neighbor],
                    center_frequency[neighbor],
                    endpoint_frequency[2 * neighbor + 1],
                )
            )
            sigma_values.extend(
                (
                    endpoint_sem[2 * neighbor],
                    center_sem[neighbor],
                    endpoint_sem[2 * neighbor + 1],
                )
            )
        x = np.asarray(x_values, dtype=float)
        y = np.asarray(y_values, dtype=float)
        sigma = np.asarray(sigma_values, dtype=float)
        for degree in degrees:
            slope, slope_sem, chi2, local_weights = _weighted_polynomial_response(
                x, y, sigma, float(epsilon[index]), degree
            )
            slopes_by_degree[degree][index] = slope
            if degree == primary_degree:
                formal_sem[index] = slope_sem
                reduced_chi2[index] = chi2
                for offset, neighbor in enumerate(
                    range(index - half_window, index + half_window + 1)
                ):
                    slope_weights[index, 3 * neighbor : 3 * neighbor + 3] = (
                        local_weights[3 * offset : 3 * offset + 3]
                    )
    slope = slopes_by_degree[primary_degree]
    order_systematic = np.maximum(
        np.abs(slope - slopes_by_degree[lower_degree]),
        np.abs(slope - slopes_by_degree[upper_degree]),
    )
    effective_sem = np.hypot(formal_sem, order_systematic)
    accepted = np.zeros(count, dtype=bool)
    interior = slice(half_window, count - half_window)
    accepted[interior] = (
        np.isfinite(slope[interior])
        & np.isfinite(effective_sem[interior])
        & (effective_sem[interior] >= 0.0)
    )
    return {
        "response": slope,
        "response_sem": effective_sem,
        "formal_sem": formal_sem,
        "order_systematic": order_systematic,
        "reduced_chi2": reduced_chi2,
        "lower_order_response": slopes_by_degree[lower_degree],
        "higher_order_response": slopes_by_degree[upper_degree],
        "half_window": np.asarray(half_window),
        "primary_degree": np.asarray(primary_degree),
        "slope_weights": slope_weights,
        "accepted": accepted,
    }


def _mixed_phase_response(
    epsilon: np.ndarray,
    center_frequency: np.ndarray,
    center_sem: np.ndarray,
    response: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    primary = _local_polynomial_response(
        epsilon,
        center_frequency,
        center_sem,
        response,
        half_window=RESPONSE_HALF_WINDOW,
        degrees=RESPONSE_POLYNOMIAL_DEGREES,
    )
    reference = _local_polynomial_response(
        epsilon,
        center_frequency,
        center_sem,
        response,
        half_window=REFERENCE_RESPONSE_HALF_WINDOW,
        degrees=REFERENCE_RESPONSE_POLYNOMIAL_DEGREES,
    )
    window_systematic = np.abs(primary["response"] - reference["response"])
    raw_sem = np.empty(3 * epsilon.size, dtype=float)
    raw_sem[0::3] = response["endpoint_frequency_sem"][0::2]
    raw_sem[1::3] = center_sem
    raw_sem[2::3] = response["endpoint_frequency_sem"][1::2]
    weights = np.asarray(primary["slope_weights"], dtype=float)
    formal_covariance = (weights * raw_sem[None, :] ** 2) @ weights.T
    chi_scale = np.ones(epsilon.size, dtype=float)
    finite_chi2 = np.isfinite(primary["reduced_chi2"])
    chi_scale[finite_chi2] = np.maximum(
        1.0, np.sqrt(primary["reduced_chi2"][finite_chi2])
    )
    scaled_covariance = (
        chi_scale[:, None] * formal_covariance * chi_scale[None, :]
    )
    effective_covariance = scaled_covariance.copy()
    order_systematic_variance = np.where(
        np.isfinite(primary["order_systematic"]),
        primary["order_systematic"] ** 2,
        0.0,
    )
    effective_covariance[np.diag_indices(epsilon.size)] += (
        order_systematic_variance
    )
    primary["response_sem"] = np.sqrt(np.diag(effective_covariance))
    primary["window_systematic"] = window_systematic
    primary["reference_response"] = reference["response"]
    primary["reference_response_sem"] = reference["response_sem"]
    primary["reference_reduced_chi2"] = reference["reduced_chi2"]
    primary["reference_lower_order_response"] = reference[
        "lower_order_response"
    ]
    primary["reference_higher_order_response"] = reference[
        "higher_order_response"
    ]
    primary["formal_covariance"] = formal_covariance
    primary["effective_covariance"] = effective_covariance
    return primary


def _cam_points(scan_values: np.ndarray) -> list[dict[str, Any]]:
    result = CAMBifurcationScanResult.load(next(V0_JOB.glob("*_v0.npz")))
    source_dir = Path(str(result.meta["v0_source_npz"])).parent
    model = KerrFull3ModeModel(**_snapshot_model_params(source_dir))
    adapter = FPGenDynamicsAdapter.from_model(model)
    local = LocalResponseValidation(
        LocalResponseValidationConfig(
            epsilon_min=1.0e-9,
            epsilon_max=1.0e-1,
            epsilon_points=11,
            precision_digits=100,
            max_iterations=60,
            residual_tolerance=1.0e-35,
            fit_points=5,
        )
    )
    candidate = 1
    branch = 3
    branches = result.candidates.branches
    assert branches is not None
    critical_params = local._critical_params(result, candidate, model)
    decimals = local._critical_state_decimals(result.candidates, candidate)
    critical = local._critical_state(result.candidates, candidate, decimals)
    exponent = float(
        branches.exponent_numerator[branch] / branches.exponent_denominator[branch]
    )
    tangent = np.asarray(
        result.candidates.diagnostics["state_tangent_vector"][candidate], dtype=float
    )
    coefficient = float(branches.amplitude[branch]) * tangent
    output: list[dict[str, Any]] = []
    for index, scan_value in enumerate(scan_values):
        epsilon = float(scan_value - 0.5)
        params = dict(critical_params)
        params["omega_b"] = float(scan_value)
        state, residual, converged = local._newton(
            adapter,
            critical + coefficient * epsilon**exponent,
            params,
            critical_decimals=decimals,
            leading_coefficient=coefficient,
            epsilon_magnitude=epsilon,
            exponent=exponent,
        )
        if not converged:
            raise RuntimeError(f"CAM solve failed at omega_b={scan_value:.16g}")
        matrix = np.asarray(adapter.state_matrix(state, params), dtype=complex)
        output.append(
            {
                "index": index,
                "epsilon": epsilon,
                "scan_value": float(scan_value),
                "params": {name: float(value) for name, value in params.items()},
                "cam_matrix": [[str(value) for value in row] for row in matrix],
                "full_residual_norm": residual,
                "holdout": False,
            }
        )
    return output


def _eta2_values(points: list[dict[str, Any]]) -> tuple[np.ndarray, float]:
    result = CAMBifurcationScanResult.load(next(V0_JOB.glob("*_v0.npz")))
    source_dir = Path(str(result.meta["v0_source_npz"])).parent
    model = KerrFull3ModeModel(**_snapshot_model_params(source_dir))
    adapter = FPGenDynamicsAdapter.from_model(model)
    eta2 = np.empty(len(points), dtype=float)
    for index, point in enumerate(points):
        params = {name: float(value) for name, value in point["params"].items()}
        matrix = np.asarray(
            [[complex(value) for value in row] for row in point["cam_matrix"]]
        )
        state = np.asarray(matrix_to_vector(matrix), dtype=float)
        jacobian = np.asarray(adapter.jacobian(state, params), dtype=float)
        mode_data = STOCHASTIC._critical_mode(jacobian)
        if mode_data is None:
            raise RuntimeError(f"soft mode unresolved at CAM point {index}")
        eigenvalue, right, left, _, _, _ = mode_data
        rate = -float(np.real(eigenvalue))
        diffusion = np.asarray(model.cam_diffusion(matrix, params), dtype=complex)
        covariance = canonical_sample_matrix_covariance(matrix, diffusion)
        projected_noise = float(np.real(left @ covariance @ left))
        if rate <= 0.0 or projected_noise <= 0.0:
            raise RuntimeError(f"invalid local OU coefficients at CAM point {index}")
        sigma_x = float(np.sqrt(projected_noise / (2.0 * rate)))
        direction_step = 1.0e-5 * max(1.0, float(np.linalg.norm(state)))
        plus_jacobian = np.asarray(
            adapter.jacobian(state + direction_step * right, params), dtype=float
        )
        minus_jacobian = np.asarray(
            adapter.jacobian(state - direction_step * right, params), dtype=float
        )
        hessian_rr = (plus_jacobian - minus_jacobian) @ right / (2.0 * direction_step)
        quadratic_drift = 0.5 * float(left @ hessian_rr)
        eta2[index] = abs(quadratic_drift) * sigma_x / rate

    epsilon = np.asarray([float(point["epsilon"]) for point in points])
    above = eta2 >= ETA2_NONLINEAR_THRESHOLD
    crossings = np.flatnonzero(above[:-1] & ~above[1:])
    if crossings.size != 1:
        raise RuntimeError(
            f"expected one eta2 threshold crossing, found {crossings.size}"
        )
    lower = int(crossings[0])
    log_epsilon = np.log(epsilon[lower : lower + 2])
    log_eta2 = np.log(eta2[lower : lower + 2])
    threshold = float(
        np.exp(
            log_epsilon[0]
            + (np.log(ETA2_NONLINEAR_THRESHOLD) - log_eta2[0])
            * (log_epsilon[1] - log_epsilon[0])
            / (log_eta2[1] - log_eta2[0])
        )
    )
    return eta2, threshold


def _cam_flow_arrays() -> dict[str, np.ndarray]:
    """Build the local two-sided CAM flow used by the paper Fig. 1b panel."""
    result = CAMBifurcationScanResult.load(next(V0_JOB.glob("*_v0.npz")))
    source_dir = Path(str(result.meta["v0_source_npz"])).parent
    model = KerrFull3ModeModel(**_snapshot_model_params(source_dir))
    adapter = FPGenDynamicsAdapter.from_model(model)
    local = LocalResponseValidation(
        LocalResponseValidationConfig(
            epsilon_min=1.0e-9,
            epsilon_max=1.0e-1,
            epsilon_points=11,
            precision_digits=100,
            max_iterations=60,
            residual_tolerance=1.0e-35,
            fit_points=5,
        )
    )
    candidate = 1
    critical_params = local._critical_params(result, candidate, model)
    decimals = local._critical_state_decimals(result.candidates, candidate)
    critical = local._critical_state(result.candidates, candidate, decimals)
    tangent = np.asarray(
        result.candidates.diagnostics["state_tangent_vector"][candidate],
        dtype=float,
    )
    branches = result.candidates.branches
    if branches is None:
        raise RuntimeError("kf3f_004 bifurcation result has no Puiseux branches")
    branch_rows: dict[int, int] = {}
    for side in (-1, 1):
        matches = np.flatnonzero(
            (branches.candidate_index == candidate)
            & (branches.epsilon_side == side)
            & branches.real_branch
        )
        if matches.size != 1:
            raise RuntimeError(
                f"expected one real kf3f_004 Puiseux branch on side {side}, "
                f"found {matches.size}"
            )
        branch_rows[side] = int(matches[0])

    epsilon = np.linspace(-FLOW_EPSILON_LIMIT, FLOW_EPSILON_LIMIT, 401)
    anchor = int(np.argmin(np.abs(epsilon)))
    epsilon[anchor] = 0.0
    branch_vectors = np.empty((epsilon.size, adapter.state_size), dtype=float)
    branch_vectors[anchor] = critical
    for index, perturbation in enumerate(epsilon):
        if index == anchor:
            continue
        side = 1 if perturbation > 0.0 else -1
        row = branch_rows[side]
        exponent = float(
            branches.exponent_numerator[row] / branches.exponent_denominator[row]
        )
        coefficient = float(branches.amplitude[row]) * tangent
        params = dict(critical_params)
        params["omega_b"] += float(perturbation)
        state, residual, converged = local._newton(
            adapter,
            critical + coefficient * abs(float(perturbation)) ** exponent,
            params,
            critical_decimals=decimals,
            leading_coefficient=coefficient,
            epsilon_magnitude=abs(float(perturbation)),
            exponent=exponent,
        )
        if not converged:
            raise RuntimeError(
                "CAM flow branch solve failed at "
                f"epsilon={float(perturbation):.6g}, residual={residual:.3e}"
            )
        branch_vectors[index] = state

    branch = branch_vectors[:, 0]
    critical_r_aa = float(critical[0])
    r_aa = np.linspace(
        0.92 * float(np.min(branch)),
        1.08 * float(np.max(branch)),
        201,
    )
    reduced = _ReducedCAMFlow(adapter, critical_params)
    drift = np.empty((r_aa.size, epsilon.size), dtype=float)
    for column, perturbation in enumerate(epsilon):
        for row, state_value in enumerate(r_aa):
            drift[row, column] = reduced.evaluate(
                float(state_value), float(perturbation)
            )[0]

    jacobians = np.empty(
        (epsilon.size, adapter.state_size, adapter.state_size), dtype=float
    )
    for index, (perturbation, state) in enumerate(
        zip(epsilon, branch_vectors, strict=True)
    ):
        params = dict(critical_params)
        params["omega_b"] += float(perturbation)
        jacobians[index] = np.asarray(adapter.jacobian(state, params), dtype=float)
    eigenvalues = _track_eigenvalues(jacobians, anchor)
    real_branches = _distinct_real_branches(eigenvalues, anchor)

    nonzero = epsilon != 0.0
    _, response_exponent = _two_sided_power_fit(
        epsilon[nonzero],
        np.abs(branch[nonzero] - critical_r_aa),
        np.ones(np.count_nonzero(nonzero), dtype=bool),
    )
    return {
        "flow_epsilon": epsilon,
        "flow_r_aa": r_aa,
        "flow_drift": drift,
        "flow_branch_r_aa": branch,
        "flow_branch_state_vectors": branch_vectors,
        "flow_critical_r_aa": np.asarray(critical_r_aa),
        "flow_response_exponent": np.asarray(response_exponent),
        "flow_response_fit_limit": np.asarray(FLOW_EPSILON_LIMIT),
        "flow_jacobian_eigenvalues": eigenvalues,
        "flow_jacobian_real_branches": real_branches,
    }


def _self_consistent_eigenstate_arrays(
    points: list[dict[str, Any]], cam_matrices: np.ndarray
) -> dict[str, np.ndarray]:
    """Continue the real-frequency rank-one nonlinear eigenstate branch."""
    count = len(points)
    amplitudes = np.empty((count, 3), dtype=complex)
    frequencies = np.empty(count, dtype=float)
    residuals = np.empty(count, dtype=float)
    occupations = np.empty(count, dtype=float)
    previous: np.ndarray | None = None

    for index in range(count - 1, -1, -1):
        params = {name: float(value) for name, value in points[index]["params"].items()}
        model = KerrFull3ModeModel(**params)
        if previous is None:
            values, vectors = np.linalg.eigh(cam_matrices[index])
            amplitude = vectors[:, -1] * np.sqrt(max(float(values[-1]), 1.0e-12))
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
            point_model: KerrFull3ModeModel = model,
            point_params: dict[str, float] = params,
        ) -> np.ndarray:
            amplitude, frequency = _unpack_eigenstate(vector)
            pure_state = np.outer(amplitude, amplitude.conj())
            hamiltonian = np.asarray(
                point_model.cam_hamiltonian(pure_state, point_params), dtype=complex
            )
            residual = hamiltonian @ amplitude - frequency * amplitude
            return np.concatenate((residual.real, residual.imag)) / max(
                float(np.linalg.norm(amplitude)), 1.0
            )

        solution = least_squares(
            normalized_residual,
            initial,
            bounds=(
                np.asarray([-30.0, -1.0e6, -1.0e6, -1.0e6, -1.0e6, -100.0]),
                np.asarray([30.0, 1.0e6, 1.0e6, 1.0e6, 1.0e6, 100.0]),
            ),
            xtol=1.0e-13,
            ftol=1.0e-13,
            gtol=1.0e-13,
            x_scale="jac",
            max_nfev=20_000,
        )
        amplitude, frequency = _unpack_eigenstate(solution.x)
        residual = float(np.linalg.norm(normalized_residual(solution.x)))
        if not solution.success or residual > 1.0e-9:
            raise RuntimeError(
                "self-consistent eigenstate continuation failed at "
                f"epsilon={points[index]['epsilon']:.6g}: "
                f"success={solution.success}, residual={residual:.3e}"
            )
        amplitudes[index] = amplitude
        frequencies[index] = frequency
        residuals[index] = residual
        occupations[index] = float(np.vdot(amplitude, amplitude).real)
        previous = solution.x

    return {
        "eigenstate_epsilon": np.asarray(
            [float(point["epsilon"]) for point in points], dtype=float
        ),
        "eigenstate_amplitudes": amplitudes,
        "eigenstate_frequency": frequencies,
        "eigenstate_residual": residuals,
        "eigenstate_occupation": occupations,
    }


def _allan_analyzer(mode: int) -> AllanScalingAnalyzer:
    return AllanScalingAnalyzer(
        scan_param="omega_b",
        critical_value=0.5,
        side="positive",
        mode=mode,
        window_selection="latest",
        white_slope_min=-1.15,
        white_slope_max=-0.85,
        min_local_r2=0.98,
        min_tau_decades=0.4,
        min_averaging_samples=16,
        max_relative_sem=0.25,
        min_independent_windows=32,
        min_scaling_points=8,
        target_scaling_decades=2.0,
        max_scaling_reduced_chi2=10.0,
        bootstrap_samples=5000,
        bootstrap_seed=202609500 + mode,
        fit_frequency=False,
        normal_form={"n": 3, "k": 1, "m": 0, "observable_order": 1},
        export=[],
    )


def _pool_allan_payloads(
    bundles: list[Any], scan_values: np.ndarray
) -> dict[int, Dataset]:
    output = {}
    for index in range(scan_values.size):
        accumulator = AllanResultAccumulator()
        for bundle in bundles:
            point = bundle.point_view((index,))
            accumulator.consume_product(point.products["allan_variance"])
        finalized = accumulator.finalize_product("allan_variance")
        if not finalized:
            raise RuntimeError(f"Allan reduction produced no product at {index}")
        output[index] = finalized["allan_variance"]
    return output


def _recover_allan(
    products: dict[int, Dataset], scan_values: np.ndarray
) -> list[dict[str, Any]]:
    output = []
    source = stack_datasets(
        [products[index] for index in range(scan_values.size)], axis="scan"
    )
    source = add_scan_parameter_coordinates(
        source, {"omega_b": np.asarray(scan_values, dtype=float)}
    )
    for mode in range(3):
        typed = _allan_analyzer(mode).execute(
            {"allan_variance": source},
            backend=NumpyBackend(),
            label="allan_scaling",
        )
        result = _product_payload(typed["allan_scaling"])
        output.extend({"mode": mode, **row} for row in result["rows"])
    return output


def _allan_proxy(payload: dict[str, Any], tau_ref: float) -> tuple[float, float]:
    allan = payload["mode_results"][PRIMARY_MODE]["allan"]
    tau = np.asarray(allan["tau"], dtype=float)
    variance = np.asarray(allan["nonoverlap_angular_frequency_variance"], dtype=float)
    sem = np.asarray(allan["nonoverlap_angular_frequency_variance_sem"], dtype=float)
    selected = int(np.argmin(np.abs(tau - tau_ref)))
    return float(tau[selected] * variance[selected]), float(
        tau[selected] * sem[selected]
    )


def _state_rows(
    bundles: list[Any], points: list[dict[str, Any]], cam: dict[str, np.ndarray]
) -> list[dict[str, Any]]:
    per_trajectory_matrix = np.concatenate(
        [
            _array(bundle.products["coherence_matrix"], "per_trajectory_matrix")
            for bundle in bundles
        ],
        axis=1,
    )
    matrix = np.mean(per_trajectory_matrix, axis=1)
    trace = np.trace(matrix, axis1=1, axis2=2).real
    purity = np.empty(trace.size, dtype=float)
    for index, item in enumerate(matrix):
        eigenvalues = np.linalg.eigvalsh((item + item.conj().T) / 2.0)
        normalized = np.clip(eigenvalues / trace[index], 0.0, None)
        normalized /= np.sum(normalized)
        purity[index] = float(np.sum(normalized**2))

    per_trajectory_occupation = np.concatenate(
        [
            _array(bundle.products["moment_statistics"], "per_trajectory_occupation")
            for bundle in bundles
        ],
        axis=1,
    )
    occupation = np.mean(per_trajectory_occupation, axis=1)
    occupation_sem = np.std(per_trajectory_occupation, axis=1, ddof=1) / math.sqrt(
        per_trajectory_occupation.shape[1]
    )
    first_last_matrix_distance = np.mean(
        np.stack(
            [
                _array(
                    bundle.products["coherence_matrix"],
                    "time_blocks.first_last_matrix_distance",
                )
                for bundle in bundles
            ]
        ),
        axis=0,
    )
    first_passage_hits = np.sum(
        np.stack(
            [_array(bundle.products["first_passage"], "n_hit") for bundle in bundles]
        ),
        axis=0,
    )
    output = []
    for index, point in enumerate(points):
        cam_matrix = np.asarray(
            [[complex(value) for value in row] for row in point["cam_matrix"]]
        )
        output.append(
            {
                "index": index,
                "epsilon": point["epsilon"],
                "omega_b": point["scan_value"],
                "cam_trace": cam["trace"][index],
                "sde_trace": trace[index],
                "trace_ratio": trace[index] / cam["trace"][index],
                "cam_r_aa": cam["r_aa"][index],
                "sde_r_aa": occupation[index, 0],
                "sde_r_aa_sem": occupation_sem[index, 0],
                "r_aa_ratio": occupation[index, 0] / cam["r_aa"][index],
                "normalized_shape_distance": float(
                    np.linalg.norm(
                        matrix[index] / np.trace(matrix[index])
                        - cam_matrix / np.trace(cam_matrix)
                    )
                ),
                "purity": purity[index],
                "first_last_matrix_distance": first_last_matrix_distance[index],
                "first_passage_hits": first_passage_hits[index],
            }
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--simulation-artifact",
        type=Path,
        action="append",
        help=(
            "Aligned simulation artifact to pool; repeat for independent "
            "replicates. Defaults to the old long31 and completed new seed A."
        ),
    )
    parser.add_argument(
        "--frequency-carrier-csv",
        type=Path,
        action="append",
        help=(
            "Lag-span-16 carrier CSV corresponding to each simulation artifact; "
            "repeat in the same order."
        ),
    )
    parser.add_argument(
        "--response-artifact",
        type=Path,
        action="append",
        help=(
            "Response-stencil artifact supplying per-trajectory phase frequency; "
            "repeat for independent seeds"
        ),
    )
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    simulation_paths = [
        path.resolve()
        for path in (args.simulation_artifact or DEFAULT_SIMULATION_ARTIFACTS)
    ]
    carrier_paths = [
        path.resolve()
        for path in (args.frequency_carrier_csv or DEFAULT_FREQUENCY_CARRIERS)
    ]
    if len(simulation_paths) != len(carrier_paths):
        raise ValueError(
            "simulation artifact and frequency carrier counts must be equal"
        )
    simulations, scan_values = _load_aligned_replicates(simulation_paths)
    response_paths = [
        path.resolve()
        for path in (args.response_artifact or DEFAULT_RESPONSE_ARTIFACTS)
    ]
    response_simulations = [load_bundle(path) for path in response_paths]
    response_manifests = [
        json.loads((path / "artifact_manifest.json").read_text("utf-8"))
        for path in response_paths
    ]
    simulation_manifests = [
        json.loads((path / "artifact_manifest.json").read_text("utf-8"))
        for path in simulation_paths
    ]
    trajectory_counts = np.stack(
        [
            _array(bundle.products["allan_variance"], "n_traj").astype(int)
            for bundle in simulations
        ]
    )
    if np.any(trajectory_counts != trajectory_counts[:, :1]):
        raise ValueError("trajectory count varies across scan points")
    total_trajectories = int(np.sum(trajectory_counts[:, 0]))
    epsilon = scan_values - 0.5
    points = _cam_points(scan_values)
    cam, critical_frequency = _cam_metrics(points)
    eta2, eta2_threshold = _eta2_values(points)
    cam_matrices = np.asarray(
        [
            [[complex(value) for value in row] for row in point["cam_matrix"]]
            for point in points
        ]
    )
    eigenstate_arrays = _self_consistent_eigenstate_arrays(points, cam_matrices)
    flow_arrays = _cam_flow_arrays()

    carrier, pooled_carrier_rows = _carrier_summary(carrier_paths, scan_values)
    _write_csv(output / "pooled_carrier_results.csv", pooled_carrier_rows)
    phase_frequency = _phase_frequency_summary(simulations, scan_values)
    phase_stencil = _phase_response_summary(response_simulations, epsilon)
    phase_response = _mixed_phase_response(
        epsilon,
        phase_frequency["frequency"],
        phase_frequency["trajectory_sem"],
        phase_stencil,
    )
    sde_frequency = phase_frequency["frequency"]
    sde_frequency_sem = phase_frequency["trajectory_sem"]

    pooled_allan = _pool_allan_payloads(simulations, scan_values)
    recovered_allan = _recover_allan(pooled_allan, scan_values)
    _write_csv(output / "recovered_allan_scaling.csv", recovered_allan)
    allan_rows = sorted(
        (row for row in recovered_allan if row["mode"] == PRIMARY_MODE),
        key=lambda row: float(row["epsilon"]),
    )
    allan_accepted = np.asarray([bool(row["accepted"]) for row in allan_rows])
    allan_intensity = np.asarray([float(row["allan_intensity"]) for row in allan_rows])
    allan_sem = np.asarray([float(row["allan_intensity_sem"]) for row in allan_rows])
    allan_tau_start = np.asarray([float(row["tau_start"]) for row in allan_rows])
    allan_tau_end = np.asarray([float(row["tau_end"]) for row in allan_rows])
    tau_ref = 25_000.0
    for index in np.flatnonzero(~allan_accepted):
        allan_intensity[index], allan_sem[index] = _allan_proxy(
            _product_payload(pooled_allan[int(index)]), tau_ref
        )

    frequency_fit = _cam_anchored_frequency_fit(
        epsilon,
        cam["frequency"],
        np.ones(epsilon.size),
        critical_frequency=critical_frequency,
        order=FREQUENCY_PUISEUX_ORDER,
        weighted=False,
        bounds=FREQUENCY_EXPONENT_BOUNDS,
        calculate_loo=True,
    )
    frequency_offset = critical_frequency
    frequency_coefficients = np.asarray(
        frequency_fit["coefficients_scaled"], dtype=float
    )
    frequency_coefficient = float(frequency_coefficients[0])
    frequency_exponent = float(frequency_fit["exponent"])
    frequency_exponent_sem = float(frequency_fit["exponent_sem"])
    scale = float(frequency_fit["epsilon_scale"])
    coordinate = epsilon / scale
    frequency_prediction = frequency_offset + sum(
        coefficient * coordinate ** (power * frequency_exponent)
        for power, coefficient in enumerate(frequency_coefficients, start=1)
    )
    allan_fit = _modified_allan_fit(
        epsilon,
        cam["full_ou"],
        order=ALLAN_PUISEUX_ORDER,
        bounds=ALLAN_EXPONENT_BOUNDS,
        calculate_loo=True,
    )
    allan_exponent = float(allan_fit["exponent"])
    allan_exponent_sem = float(allan_fit["exponent_sem"])
    allan_coefficients = np.asarray(allan_fit["coefficients_scaled"], dtype=float)
    allan_scale = float(allan_fit["epsilon_scale"])
    allan_coordinate = epsilon / allan_scale
    allan_prediction = sum(
        coefficient
        * allan_coordinate ** (-2.0 * (1.0 - allan_exponent) + power * allan_exponent)
        for power, coefficient in enumerate(allan_coefficients)
    )
    allan_q = 2.0 * (1.0 - allan_exponent)
    allan_q_sem = 2.0 * allan_exponent_sem
    allan_q_display_sem = (
        math.ceil(allan_q_sem * 10**ALLAN_Q_DISPLAY_DECIMALS)
        / 10**ALLAN_Q_DISPLAY_DECIMALS
    )
    g1_slope, g1_slope_sem = _local_slope(
        epsilon,
        carrier["frequency"],
        carrier["effective_uncertainty"],
        np.ones(epsilon.size, dtype=bool),
    )
    g1_slope_wide, _ = _local_slope(
        epsilon,
        carrier["frequency"],
        carrier["effective_uncertainty"],
        np.ones(epsilon.size, dtype=bool),
        half_window=3,
    )
    g1_information_slope_accepted = np.zeros(epsilon.size, dtype=bool)
    g1_information_slope_accepted[3:-3] = True
    g1_information_slope = np.where(g1_information_slope_accepted, g1_slope, np.nan)
    g1_information_slope_window_systematic = np.where(
        g1_information_slope_accepted,
        np.abs(g1_slope - g1_slope_wide),
        np.nan,
    )
    g1_information_slope_sem = np.where(
        g1_information_slope_accepted,
        np.hypot(g1_slope_sem, g1_information_slope_window_systematic),
        np.nan,
    )
    sde_slope = phase_response["response"]
    sde_slope_sem = phase_response["response_sem"]
    sde_information_slope_accepted = phase_response["accepted"]
    sde_information_slope = np.where(sde_information_slope_accepted, sde_slope, np.nan)
    sde_information_slope_sem = np.where(
        sde_information_slope_accepted, sde_slope_sem, np.nan
    )
    sde_information_slope_window_systematic = np.where(
        sde_information_slope_accepted,
        phase_response["window_systematic"],
        np.nan,
    )
    state_rows = _state_rows(simulations, points, cam)
    _write_csv(output / "state_closure.csv", state_rows)
    flow_sde_r_aa = np.asarray([row["sde_r_aa"] for row in state_rows], dtype=float)
    flow_sde_r_aa_sem = np.asarray(
        [row["sde_r_aa_sem"] for row in state_rows], dtype=float
    )

    arrays = {
        **flow_arrays,
        **eigenstate_arrays,
        "flow_sde_epsilon": epsilon,
        "flow_sde_r_aa": flow_sde_r_aa,
        "flow_sde_r_aa_sem": flow_sde_r_aa_sem,
        "epsilon": epsilon,
        "holdout": np.zeros(epsilon.size, dtype=bool),
        "critical_frequency": np.asarray(critical_frequency),
        "cam_frequency": cam["frequency"],
        "cam_frequency_response": cam["response"],
        "cam_soft_frequency_response": cam["soft_response"],
        "cam_full_ou_intensity": cam["full_ou"],
        "cam_soft_ou_intensity": cam["soft_ou"],
        "cam_critical_rate": cam["critical_rate"],
        "cam_critical_forcing": cam["critical_forcing"],
        "cam_critical_readout": cam["critical_readout"],
        "cam_projected_noise": cam["projected_noise"],
        "cam_local_path_information": (
            cam["critical_forcing"] ** 2 / cam["projected_noise"]
        ),
        "cam_hamiltonian_petermann_sum": cam["hamiltonian_petermann"],
        "cam_bogoliubov_petermann_sum": cam["bogoliubov_petermann"],
        "cam_eta2": eta2,
        "cam_eta2_criterion": np.asarray(ETA2_NONLINEAR_THRESHOLD),
        "cam_eta2_epsilon_threshold": np.asarray(eta2_threshold),
        "cam_trace": cam["trace"],
        "cam_r_aa": cam["r_aa"],
        "sde_frequency": sde_frequency,
        "sde_frequency_sem": sde_frequency_sem,
        "sde_frequency_seed_sem": phase_frequency["seed_sem"],
        "sde_frequency_by_replicate": phase_frequency["frequency_by_replicate"],
        "sde_phase_frequency_trajectory_count": phase_frequency["trajectory_count"],
        "sde_g1_frequency": carrier["frequency"],
        "sde_g1_frequency_sem": carrier["effective_uncertainty"],
        "sde_g1_frequency_seed_sem": carrier["seed_sem"],
        "sde_g1_frequency_propagated_diagnostic": carrier["propagated_diagnostic"],
        "sde_g1_frequency_by_replicate": carrier["frequency_by_replicate"],
        "sde_allan_accepted": allan_accepted,
        "sde_allan_intensity": allan_intensity,
        "sde_allan_sem": allan_sem,
        "sde_allan_tau_start": allan_tau_start,
        "sde_allan_tau_end": allan_tau_end,
        "sde_allan_proxy_tau": np.asarray(tau_ref),
        "sde_frequency_slope": sde_slope,
        "sde_frequency_slope_sem": sde_slope_sem,
        "sde_information_slope": sde_information_slope,
        "sde_information_slope_sem": sde_information_slope_sem,
        "sde_information_slope_covariance": phase_response[
            "effective_covariance"
        ],
        "sde_information_slope_window_systematic": (
            sde_information_slope_window_systematic
        ),
        "sde_information_slope_accepted": sde_information_slope_accepted,
        "sde_phase_response_endpoint_frequency": phase_stencil["endpoint_frequency"],
        "sde_phase_response_endpoint_frequency_sem": phase_stencil[
            "endpoint_frequency_sem"
        ],
        "sde_phase_response_trajectory_count": phase_stencil["trajectory_count"],
        "sde_phase_response_replicate_count": phase_stencil["replicate_count"],
        "sde_phase_response_frequency_by_replicate": phase_stencil[
            "frequency_by_replicate"
        ],
        "sde_phase_stencil_response": phase_stencil["response"],
        "sde_phase_stencil_response_sem": phase_stencil["response_sem"],
        "sde_phase_stencil_response_accepted": phase_stencil["accepted"],
        "sde_mixed_response_formal_sem": phase_response["formal_sem"],
        "sde_mixed_response_formal_covariance": phase_response[
            "formal_covariance"
        ],
        "sde_mixed_response_order_systematic": phase_response["order_systematic"],
        "sde_mixed_response_window_systematic": phase_response[
            "window_systematic"
        ],
        "sde_mixed_response_reduced_chi2": phase_response["reduced_chi2"],
        "sde_mixed_response_lower_order": phase_response[
            "lower_order_response"
        ],
        "sde_mixed_response_higher_order": phase_response[
            "higher_order_response"
        ],
        "sde_nine_point_response": phase_response["reference_response"],
        "sde_nine_point_response_sem": phase_response["reference_response_sem"],
        "sde_nine_point_reduced_chi2": phase_response[
            "reference_reduced_chi2"
        ],
        "sde_mixed_response_quadratic": phase_response[
            "reference_lower_order_response"
        ],
        "sde_mixed_response_quartic": phase_response[
            "reference_higher_order_response"
        ],
        "sde_g1_frequency_slope": g1_slope,
        "sde_g1_frequency_slope_sem": g1_slope_sem,
        "sde_g1_information_slope": g1_information_slope,
        "sde_g1_information_slope_sem": g1_information_slope_sem,
        "sde_g1_information_slope_window_systematic": (
            g1_information_slope_window_systematic
        ),
        "sde_g1_information_slope_accepted": g1_information_slope_accepted,
        "frequency_fit_exponent": np.asarray(frequency_exponent),
        "frequency_fit_exponent_sem": np.asarray(frequency_exponent_sem),
        "frequency_fit_exponent_display_sem": np.asarray(
            FREQUENCY_EXPONENT_DISPLAY_UNCERTAINTY
        ),
        "frequency_fit_order": np.asarray(FREQUENCY_PUISEUX_ORDER),
        "frequency_fit_coefficients": frequency_coefficients,
        "frequency_fit_coefficient": np.asarray(frequency_coefficient),
        "frequency_fit_offset": np.asarray(frequency_offset),
        "frequency_fit_scale": np.asarray(scale),
        "frequency_fit_prediction": frequency_prediction,
        "allan_fit_exponent": np.asarray(allan_exponent),
        "allan_fit_exponent_sem": np.asarray(allan_exponent_sem),
        "allan_fit_order": np.asarray(ALLAN_PUISEUX_ORDER),
        "allan_fit_scale": np.asarray(allan_scale),
        "allan_fit_q": np.asarray(allan_q),
        "allan_fit_q_sem": np.asarray(allan_q_sem),
        "allan_fit_q_display_sem": np.asarray(allan_q_display_sem),
        "allan_fit_coefficients": allan_coefficients,
        "allan_fit_prediction": allan_prediction,
    }
    np.savez_compressed(output / "plot_data.npz", **arrays)
    summary = {
        "schema": "qphase.kerr-full-kf3f004-long31-plot-data/7",
        "generated_utc": datetime.now(UTC).isoformat(),
        "source_simulation_artifacts": [str(path) for path in simulation_paths],
        "source_simulation_artifact_ids": [
            manifest["artifact_id"] for manifest in simulation_manifests
        ],
        "source_response_artifacts": [str(path) for path in response_paths],
        "source_response_artifact_ids": [
            manifest["artifact_id"] for manifest in response_manifests
        ],
        "source_frequency_carrier_csvs": [str(path) for path in carrier_paths],
        "source_child_product_status": {
            "band_limited_carrier": (
                "old post-hoc lag-span-16 result plus preregistered new seed-A result"
            ),
            "spectral_ridge": (
                "new seed-A all-channel ridge recovered post hoc from retained PSD"
            ),
            "publication_seed_b": "excluded because the run was interrupted",
            "allan_scaling": "rebuilt from pooled typed raw Allan products",
        },
        "point_count": len(points),
        "successful_replicate_count": len(simulations),
        "trajectory_count_per_point": total_trajectories,
        "primary_readout": "mode_2",
        "cam_frequency_readout": "trace_rayleigh",
        "frequency_estimand": "long_time_mean_phase_increment",
        "frequency_fit": {
            "target": "CAM trace Rayleigh frequency",
            "model": "CAM-anchored O4 free-exponent Puiseux expansion",
            "critical_frequency_fixed": critical_frequency,
            "order": FREQUENCY_PUISEUX_ORDER,
            "exponent": frequency_exponent,
            "exponent_uncertainty": frequency_exponent_sem,
            "display_exponent": 0.3332,
            "display_uncertainty": FREQUENCY_EXPONENT_DISPLAY_UNCERTAINTY,
            "coefficients_scaled": frequency_coefficients.tolist(),
            "train_rmse": float(frequency_fit["train_rmse"]),
            "loo_rmse": float(frequency_fit["loo_rmse"]),
        },
        "sde_frequency": {
            "estimand": "trajectory mean mode-2 phase-increment frequency",
            "source_product": PHASE_FREQUENCY_KEY,
            "trajectory_count_per_point": int(phase_frequency["trajectory_count"]),
            "uncertainty": "trajectory SEM across pooled independent long records",
            "maximum_trajectory_sem": float(np.max(sde_frequency_sem)),
        },
        "sde_information_response": {
            "estimand": (
                "weighted fifteen-point local cubic phase-frequency fit in "
                "log-epsilon coordinates"
            ),
            "source_artifacts": [str(path) for path in response_paths],
            "response_replicate_count": int(phase_stencil["replicate_count"]),
            "trajectory_count_per_flank": int(phase_stencil["trajectory_count"]),
            "accepted_point_count": int(
                np.count_nonzero(sde_information_slope_accepted)
            ),
            "fit_points": 15,
            "neighboring_long31_centers": 5,
            "fit_coordinate": "log(epsilon/epsilon_center)",
            "primary_polynomial_degree": 3,
            "order_diagnostic_degrees": [2, 4],
            "window_reference": (
                "weighted nine-point local cubic log-coordinate fit; "
                "diagnostic only"
            ),
            "retained_center_indices": [2, int(epsilon.size - 3)],
            "uncertainty": (
                "quadrature of cubic covariance scaled by reduced chi2 and "
                "maximum quadratic/quartic slope difference"
            ),
            "cross_point_covariance": (
                "propagated from shared raw-frequency weights; order systematic "
                "is added to the diagonal"
            ),
            "linear_solver": "SVD pseudoinverse",
            "puiseux_assumption": False,
        },
        "supplementary_g1_frequency": {
            "estimand": "ridge-conditioned band-limited long-time G1 pole",
            "source_csvs": [str(path) for path in carrier_paths],
            "minimum_lag_span": 16.0,
            "uncertainty": (
                "hypot(seed SEM, independently propagated ridge-conditioned "
                "diagnostic upper bounds)"
            ),
        },
        "band_limited_carrier_accepted_count_per_replicate": int(scan_values.size),
        "eigenstate": {
            "definition": (
                "R=alpha alpha^dagger; H_eff(R) alpha=omega alpha; omega real"
            ),
            "continuation": "descending epsilon from 1e-1 to 1e-4",
            "point_count": int(eigenstate_arrays["eigenstate_frequency"].size),
            "maximum_normalized_residual": float(
                np.max(eigenstate_arrays["eigenstate_residual"])
            ),
            "occupation_range": [
                float(np.min(eigenstate_arrays["eigenstate_occupation"])),
                float(np.max(eigenstate_arrays["eigenstate_occupation"])),
            ],
        },
        "allan": {
            "recovery_reason": (
                "pooled per-trajectory typed Allan products across successful "
                "long-record replicates"
            ),
            "accepted_by_mode": {
                str(mode): sum(
                    bool(row["accepted"])
                    for row in recovered_allan
                    if row["mode"] == mode
                )
                for mode in range(3)
            },
            "proxy_count_mode_2": int(np.count_nonzero(~allan_accepted)),
            "proxy_tau": tau_ref,
            "cam_ou_fit": {
                "target": "CAM full-OU Allan intensity",
                "model": ("N_A=x^[-2(1-p)] sum(k=0..4) c_k x^(kp), x=epsilon/0.1"),
                "order": ALLAN_PUISEUX_ORDER,
                "exponent": allan_exponent,
                "exponent_uncertainty_1sigma": allan_exponent_sem,
                "allan_power_q": allan_q,
                "allan_power_q_uncertainty_1sigma": allan_q_sem,
                "allan_power_q_display_uncertainty": allan_q_display_sem,
                "display_uncertainty_rule": (
                    "1-sigma curvature uncertainty rounded upward (never down) "
                    "to four decimal places; not a 2-sigma interval"
                ),
                "relative_rmse": float(allan_fit["relative_rmse"]),
                "loo_relative_rmse": float(allan_fit["loo_relative_rmse"]),
            },
        },
        "local_ou_validity": {
            "diagnostic": "eta2=abs(f''(x_c))*sigma_x/(2*abs(lambda_c))",
            "nonlinear_region": f"eta2 >= {ETA2_NONLINEAR_THRESHOLD:g}",
            "eta2_criterion": ETA2_NONLINEAR_THRESHOLD,
            "eta2_range": [float(np.min(eta2)), float(np.max(eta2))],
            "epsilon_at_eta2_criterion": eta2_threshold,
            "threshold_interpolation": "linear in log(eta2) versus log(epsilon)",
        },
        "local_path_information": {
            "definition": "abs(F_c(epsilon))^2/Gamma_c(epsilon)",
            "critical_forcing_field": "cam_critical_forcing",
            "projected_noise_field": "cam_projected_noise",
            "information_field": "cam_local_path_information",
            "scope": (
                "local projected OU/path-information reference; not the full "
                "nonlinear path Fisher information inside the eta2 warning region"
            ),
        },
        "fig3a_reduced_enhancement": {
            "signal": "abs(dOmega/depsilon)^2 / abs(g_c F_c)^2",
            "noise": "N_A / (abs(g_c)^2 Gamma_c)",
            "critical_rate_reference": "abs(Re lambda_c)^-2",
            "normalization_epsilon": 1.0e-2,
            "scope": "local single-soft-mode OU comparison outside eta2 warning",
        },
        "state": {
            "first_passage_hits": int(
                sum(int(row["first_passage_hits"]) for row in state_rows)
            ),
            "trace_ratio_range": [
                float(min(row["trace_ratio"] for row in state_rows)),
                float(max(row["trace_ratio"] for row in state_rows)),
            ],
            "r_aa_ratio_range": [
                float(min(row["r_aa_ratio"] for row in state_rows)),
                float(max(row["r_aa_ratio"] for row in state_rows)),
            ],
            "maximum_shape_distance": float(
                max(row["normalized_shape_distance"] for row in state_rows)
            ),
            "purity_range": [
                float(min(row["purity"] for row in state_rows)),
                float(max(row["purity"] for row in state_rows)),
            ],
        },
        "cam_flow": {
            "order_parameter": "R_aa",
            "perturbation": "delta_omega_b",
            "epsilon_limit": FLOW_EPSILON_LIMIT,
            "grid_shape": [
                int(flow_arrays["flow_r_aa"].size),
                int(flow_arrays["flow_epsilon"].size),
            ],
            "response_exponent": float(flow_arrays["flow_response_exponent"]),
            "construction": "algebraic stable-mode elimination",
            "sde_preview": {
                "source": "moment_statistics.occupation[:, 0]",
                "uncertainty": "trajectory SEM",
                "available_point_count": int(epsilon.size),
                "visible_point_count": int(
                    np.count_nonzero(np.abs(epsilon) <= FLOW_EPSILON_LIMIT)
                ),
            },
        },
        "supersedes_for_active_figures": str(
            (SEARCH / "kf3f_004_q1_paper/plot_data.npz").resolve()
        ),
        "limitations": [
            "two successful long-record seeds; the planned new seed B failed "
            "during GPU spectral analysis",
            f"mixed phase-response fit pools {len(response_paths)} successful "
            "response seeds",
            "no frozen O3/O5/O7 nonlinear CAM Allan curve",
            "SDE SNR panel remains a finite-record diagnostic",
        ],
    }
    (output / "plot_data_manifest.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
