"""qphase_cam: polynomial reduction of the full CAM noise along one manifold
---------------------------------------------------------------------------
``scalar_reconstruction`` evaluates the projected noise and the readout on every grid
point of a *bounded* reconstruction grid and integrates that sampled table. A CLK3-style
figure needs the other shape of the same problem: the physical domain of the state
matrix is wide (tens of soft-mode widths), so the diffusion and the readout are fitted
as low-order polynomials of the manifold coordinate and that polynomial table — not a
sampled grid — is handed to the shared stationary quadrature.

What this module owns is only that reduction: the physical interval, the two polynomial
fits and their fit errors. Everything downstream is the shared one:
:func:`~qphase_cam.nonlinear.evaluate_generator` computes the density, the stationary
mean, the response and the intensity exactly as it does for a sampled reconstruction, so
the polynomial table and a sampled table are compared by the same arithmetic.

The defaults are the historical CLK3 numbers: a physical scan of ``±30`` soft-mode
widths on 12001 points, an 81-point fit interval and a 100001-point dense table, with
the quadrature configuration the CLK3 analysis used (``centered_cdf`` intensity, a
strict density comparison and a two-point minimum edge window). A configuration that
produces a non-positive diffusion, a non-finite fit or a density that does not normalize
is reported through :class:`~qphase_cam.outcomes.CAMOutcomeStatus` — the diffusion is
never clamped and the interval is never silently narrowed.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from .core.noise import canonical_sample_matrix_covariance
from .nonlinear import (
    NonlinearManifold,
    ScalarGeneratorEvaluation,
    ScalarGeneratorTable,
    ScalarReconstructionConfig,
    evaluate_generator,
    manifold_embedding,
)
from .observable import CAMObservable
from .outcomes import CAMOutcomeStatus
from .validity import projected_intensity

__all__ = [
    "PolynomialReconstruction",
    "PolynomialReconstructionConfig",
    "polynomial_reconstruction",
]

#: Fraction of the soft-mode width bounding the local readout fit.
LOCAL_EXTENT_FACTOR = 0.2
#: Fraction of each physical bound bounding the local readout fit as well.
LOCAL_INTERVAL_FRACTION = 0.1
#: Smallest number of points the local readout fit is sampled on.
MINIMUM_LOCAL_POINTS = 17


def _clk3_quadrature() -> ScalarReconstructionConfig:
    """Return the CLK3 stationary-quadrature configuration."""
    return ScalarReconstructionConfig(
        intensity_integration="centered_cdf",
        core_density_comparison="greater",
        edge_count_rule="ceil_min_two",
        density_floor_relative=1.0e-14,
    )


class PolynomialReconstructionConfig(BaseModel):
    """Grids, tolerances and quadrature of one polynomial reconstruction.

    ``physical_half_width`` and ``physical_scan_points`` place the scan that finds the
    physical interval, ``physical_tolerance`` is the smallest eigenvalue still accepted
    as physical, ``fit_points`` samples the two polynomial fits and ``dense_points`` is
    the table the quadrature integrates. ``quadrature`` configures the shared
    :func:`~qphase_cam.nonlinear.evaluate_generator`; its default is the CLK3 analysis'
    own (``centered_cdf`` intensity, ``greater`` density comparison, ``ceil_min_two``
    edge window, ``density_floor_relative = 1e-14``), and its ``integration_split`` has
    no effect on that intensity assembly.

    ``embedding_assembly`` states the arithmetic order of the embedding the fits sample.
    ``manifold_embedding`` adds the polynomial terms to the reference state one by one
    (its own ``progressive`` default), while the historical polynomial fits summed the
    terms first and added the state last; the polynomial is the same and the additions
    differ in order, so the two can differ in the last bit. This module defaults to
    ``terms_first`` because its tables are the historical ones, and every sampled state
    it builds — the physical scan, the fit interval and the local readout interval —
    uses the configured order.
    """

    model_config = ConfigDict(extra="forbid")

    physical_half_width: float = Field(default=30.0, gt=0.0)
    physical_scan_points: int = Field(default=12001, ge=3)
    physical_tolerance: float = Field(default=1.0e-10, ge=0.0)
    fit_points: int = Field(default=81, ge=3)
    dense_points: int = Field(default=100001, ge=11)
    embedding_assembly: Literal["progressive", "terms_first"] = Field(
        default="terms_first",
        description=(
            "Arithmetic order of the sampled manifold embedding: 'terms_first' (the "
            "default here) sums the polynomial terms from zero and adds the reference "
            "state last, the historical fits' convention; 'progressive' is "
            "manifold_embedding's own default and starts from the state. Algebraically "
            "identical, last-bit different"
        ),
    )
    quadrature: ScalarReconstructionConfig = Field(default_factory=_clk3_quadrature)


class PolynomialReconstruction(BaseModel):
    """The polynomial reduction of one manifold's noise, and its stationary summary.

    ``lower`` and ``upper`` are the physical interval the scan found, ``clipped`` says
    whether it stops at the scan's own edge rather than at a non-physical state, and
    ``physical_grid_fraction`` is the fraction of the scanned grid that is physical.
    ``diffusion_coefficients`` and ``frequency_coefficients`` are polynomials of the
    *scaled* coordinate ``x/σ``; the drift is the manifold's own polynomial in ``x``.
    ``generator_table`` is that dense table and ``evaluation`` the shared quadrature on
    it, so ``summary["scalar"]`` is the evaluation's intensity, not a second one.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: CAMOutcomeStatus
    reason: str | None = None
    reason_code: str | None = None

    order: int
    manifold: NonlinearManifold
    sigma: float

    lower: float | None = None
    upper: float | None = None
    physical_grid_fraction: float | None = None
    domain_clipped: bool | None = None

    diffusion_coefficients: tuple[float, ...] = ()
    frequency_coefficients: tuple[float, ...] = ()
    diffusion_error: float | None = None
    diffusion_minimum: float | None = None
    diffusion_maximum: float | None = None
    frequency_error: float | None = None
    frequency_minimum: float | None = None
    frequency_maximum: float | None = None

    generator_table: ScalarGeneratorTable | None = None
    evaluation: ScalarGeneratorEvaluation | None = None

    summary: dict[str, Any] = Field(default_factory=dict)
    method: dict[str, Any] = Field(default_factory=dict)
    diagnostics: dict[str, Any] = Field(default_factory=dict)


def _physical_minima(
    adapter: Any, vectors: np.ndarray, params: Mapping[str, Any]
) -> np.ndarray:
    """Return the smallest physical eigenvalue of every row of a state stack.

    The adapter's own ``physical_eigenvalues`` is used when it provides one — one call
    for the whole stack when the adapter declares it can answer a batch, one row at a
    time otherwise — and an adapter without that provider is read through its state
    matrices' Hermitian spectrum, which is the same definition. No mode count and no
    coordinate layout is assumed here.
    """
    provider = getattr(adapter, "physical_eigenvalues", None)
    batched = bool(getattr(adapter, "supports_batched_states", False))
    if callable(provider):
        if batched:
            spectrum = np.asarray(provider(vectors, params), dtype=float)
            if spectrum.ndim != 2 or spectrum.shape[0] != vectors.shape[0]:
                raise ValueError(
                    f"the adapter returned a physical spectrum of shape "
                    f"{spectrum.shape} for {vectors.shape[0]} states"
                )
            return np.min(spectrum, axis=1)
        return np.asarray(
            [
                float(np.min(np.asarray(provider(row, params), dtype=float)))
                for row in vectors
            ]
        )
    if batched:
        matrices = np.asarray(adapter.state_matrix(vectors, params), dtype=complex)
    else:
        matrices = np.stack(
            [
                np.asarray(adapter.state_matrix(row, params), dtype=complex)
                for row in vectors
            ]
        )
    hermitian = (matrices + np.conjugate(np.swapaxes(matrices, -1, -2))) / 2.0
    return np.min(np.linalg.eigvalsh(hermitian), axis=-1)


def _physical_interval(
    manifold: NonlinearManifold,
    point: np.ndarray,
    sigma: float,
    settings: PolynomialReconstructionConfig,
    adapter: Any,
    params: Mapping[str, Any],
) -> tuple[float, float, float, bool, np.ndarray, np.ndarray]:
    """Return ``(lower, upper, fraction, clipped, axis, physical)`` of one state.

    The scan covers ``±physical_half_width · σ`` and a point is physical when its state
    matrix' smallest eigenvalue is at least ``-physical_tolerance``. The interval kept
    is the contiguous physical run around ``x = 0``; a run that reaches the scan's edge
    was cut by the scan, which ``clipped`` reports.
    """
    half_width = float(settings.physical_half_width) * float(sigma)
    axis = np.linspace(-half_width, half_width, int(settings.physical_scan_points))
    vectors = manifold_embedding(
        manifold, point, axis, assembly=settings.embedding_assembly
    )
    minima = _physical_minima(adapter, vectors, params)
    physical = minima >= -float(settings.physical_tolerance)
    center = int(np.argmin(np.abs(axis)))
    if not physical[center]:
        raise ValueError(
            "the reference state is not physical (smallest eigenvalue "
            f"{float(minima[center])!r} < {-float(settings.physical_tolerance)!r})"
        )
    lower_index = center
    upper_index = center
    while lower_index > 0 and physical[lower_index - 1]:
        lower_index -= 1
    while upper_index + 1 < axis.size and physical[upper_index + 1]:
        upper_index += 1
    clipped = lower_index == 0 or upper_index == axis.size - 1
    return (
        float(axis[lower_index]),
        float(axis[upper_index]),
        float(np.mean(physical)),
        bool(clipped),
        axis,
        physical,
    )


def polynomial_reconstruction(
    adapter: Any,
    model: Any,
    observable: CAMObservable,
    *,
    state: Any,
    params: Mapping[str, Any],
    manifold: NonlinearManifold,
    sigma: float,
    config: PolynomialReconstructionConfig | None = None,
) -> PolynomialReconstruction:
    """Return the polynomial reduction of one manifold's noise.

    The manifold supplies the embedding, the drift polynomial and its order; ``sigma``
    is the soft-mode width the scan is measured in. The projected noise is
    ``leftᵀ Σ(x) left`` from the canonical sample-matrix covariance and the readout is
    the caller's observable, both sampled point by point on the fit interval and fitted
    as polynomials of ``x/σ``. The dense table those polynomials define is evaluated by
    the shared :func:`~qphase_cam.nonlinear.evaluate_generator`, so the density, the
    mean, the response, the edge diagnostics and the intensity are its own.

    A non-physical reference state, an interval too small to fit, a non-positive or
    non-finite sampled diffusion, a dense table with a non-positive diffusion and a
    density that does not normalize are reported as statuses with a reason; the
    diffusion is never clamped, the interval is never narrowed and no number is
    fabricated.
    """
    settings = config or PolynomialReconstructionConfig()
    point = np.asarray(state, dtype=float).reshape(-1)
    parameter_map = dict(params or {})
    resolved_sigma = float(sigma)
    if not np.all(np.isfinite(point)):
        raise ValueError("the state must be finite")
    if not np.isfinite(resolved_sigma) or resolved_sigma <= 0.0:
        raise ValueError(
            f"the soft-mode width must be finite and positive; got {sigma!r}"
        )
    order = int(manifold.order)
    left = np.asarray(manifold.left, dtype=float)
    if left.size != point.size:
        raise ValueError(
            f"the manifold's left vector has {left.size} entries for a "
            f"{point.size}-coordinate state"
        )
    method: dict[str, Any] = {
        "service": "qphase_cam.polynomial_reconstruction.polynomial_reconstruction",
        "manifold": dict(manifold.method),
        "physical_scan": (
            f"{int(settings.physical_scan_points)} points on "
            f"±{float(settings.physical_half_width):g} sigma, physical when the "
            f"smallest eigenvalue >= -{float(settings.physical_tolerance):g}"
        ),
        "physical_provider": (
            "adapter.physical_eigenvalues (batch when declared) or the state matrices' "
            "Hermitian spectrum"
        ),
        "fit": (
            f"diffusion degree {2 * order} on the whole interval, readout degree "
            f"{order} on the local interval, both in x/sigma"
        ),
        "embedding_assembly": settings.embedding_assembly,
        "diffusion_error": "max|pred - values| / max(|values[0]|, 1)",
        "frequency_error": "max|pred - local| / max(ptp(local), eps)",
        "dense_grid": f"{int(settings.dense_points)} points on the physical interval",
        "drift": "np.polynomial.polynomial.polyval(x, manifold.drift)",
        "quadrature": (
            "evaluate_generator on the dense polynomial table (density, mean, response "
            "and intensity are its own)"
        ),
        "config": settings.model_dump(mode="json"),
        "quadrature_config": settings.quadrature.model_dump(mode="json"),
        "local_extent_factor": LOCAL_EXTENT_FACTOR,
        "local_interval_fraction": LOCAL_INTERVAL_FRACTION,
        "minimum_local_points": MINIMUM_LOCAL_POINTS,
        "order": order,
        "sigma": resolved_sigma,
    }
    diagnostics: dict[str, Any] = {}

    def failed(
        status: CAMOutcomeStatus,
        code: str,
        reason: str,
        *,
        evaluation: ScalarGeneratorEvaluation | None = None,
        table: ScalarGeneratorTable | None = None,
    ) -> PolynomialReconstruction:
        return PolynomialReconstruction(
            status=status,
            reason=reason,
            reason_code=code,
            order=order,
            manifold=manifold,
            sigma=resolved_sigma,
            lower=diagnostics.get("lower"),
            upper=diagnostics.get("upper"),
            physical_grid_fraction=diagnostics.get("physical_grid_fraction"),
            domain_clipped=diagnostics.get("domain_clipped"),
            diffusion_coefficients=tuple(
                float(value) for value in diagnostics.get("diffusion_coefficients", ())
            ),
            frequency_coefficients=tuple(
                float(value) for value in diagnostics.get("frequency_coefficients", ())
            ),
            diffusion_error=diagnostics.get("diffusion_error"),
            diffusion_minimum=diagnostics.get("diffusion_minimum"),
            diffusion_maximum=diagnostics.get("diffusion_maximum"),
            frequency_error=diagnostics.get("frequency_error"),
            frequency_minimum=diagnostics.get("frequency_minimum"),
            frequency_maximum=diagnostics.get("frequency_maximum"),
            generator_table=table,
            evaluation=evaluation,
            summary=dict(diagnostics),
            method=dict(method),
        )

    # 1. The physical interval of the state matrix along the manifold.
    try:
        lower, upper, fraction, clipped, axis, physical = _physical_interval(
            manifold, point, resolved_sigma, settings, adapter, parameter_map
        )
    except ValueError as exc:
        diagnostics["physical_scan_points"] = int(settings.physical_scan_points)
        return failed(
            CAMOutcomeStatus.STOCHASTIC_INVALID, "center_not_physical", str(exc)
        )
    diagnostics.update(
        {
            "lower": lower,
            "upper": upper,
            "physical_grid_fraction": fraction,
            "domain_clipped": clipped,
        }
    )
    if upper <= lower:
        return failed(
            CAMOutcomeStatus.UNIDENTIFIABLE,
            "physical_interval_empty",
            f"the physical interval [{lower!r}, {upper!r}] has no width",
        )

    # 2. The two polynomial fits on that interval.
    sample = np.linspace(lower, upper, int(settings.fit_points))
    vectors = manifold_embedding(
        manifold, point, sample, assembly=settings.embedding_assembly
    )
    diffusion_values: list[float] = []
    frequency_values: list[float] = []
    matrices: list[np.ndarray] = []
    for vector in vectors:
        matrix = np.asarray(adapter.state_matrix(vector, parameter_map), dtype=complex)
        matrices.append(matrix)
        diffusion = np.asarray(
            model.cam_diffusion(matrix, parameter_map), dtype=complex
        )
        covariance = canonical_sample_matrix_covariance(matrix, diffusion)
        diffusion_values.append(projected_intensity(covariance, left, left))
        frequency_values.append(float(observable.value(adapter, vector, parameter_map)))
    diffusion_array = np.asarray(diffusion_values, dtype=float)
    frequency_array = np.asarray(frequency_values, dtype=float)
    diagnostics["diffusion_minimum"] = float(np.min(diffusion_array))
    diagnostics["diffusion_maximum"] = float(np.max(diffusion_array))
    diagnostics["frequency_minimum"] = float(np.min(frequency_array))
    diagnostics["frequency_maximum"] = float(np.max(frequency_array))
    if not np.all(np.isfinite(diffusion_array)) or not np.all(
        np.isfinite(frequency_array)
    ):
        return failed(
            CAMOutcomeStatus.STOCHASTIC_INVALID,
            "sample_not_finite",
            "the sampled projected diffusion or readout is not finite on the physical "
            "interval",
        )
    if np.any(diffusion_array <= 0.0):
        return failed(
            CAMOutcomeStatus.STOCHASTIC_INVALID,
            "diffusion_not_positive",
            "the projected diffusion is not positive on the physical interval "
            f"(minimum {float(np.min(diffusion_array))!r}); it is not clamped",
        )
    scaled_sample = sample / resolved_sigma
    diffusion_coefficients = np.polynomial.polynomial.polyfit(
        scaled_sample, diffusion_array, 2 * order
    )
    diffusion_prediction = np.polynomial.polynomial.polyval(
        scaled_sample, diffusion_coefficients
    )
    diffusion_error = float(
        np.max(np.abs(diffusion_prediction - diffusion_array))
        / max(abs(float(diffusion_array[0])), 1.0)
    )
    local_extent = min(
        LOCAL_EXTENT_FACTOR * resolved_sigma,
        LOCAL_INTERVAL_FRACTION * abs(lower),
        LOCAL_INTERVAL_FRACTION * abs(upper),
    )
    local_points = max(MINIMUM_LOCAL_POINTS, 2 * order + 5)
    local_sample = np.linspace(-local_extent, local_extent, local_points)
    local_vectors = manifold_embedding(
        manifold, point, local_sample, assembly=settings.embedding_assembly
    )
    local_frequency = np.asarray(
        [
            float(observable.value(adapter, vector, parameter_map))
            for vector in local_vectors
        ],
        dtype=float,
    )
    if not np.all(np.isfinite(local_frequency)):
        return failed(
            CAMOutcomeStatus.STOCHASTIC_INVALID,
            "sample_not_finite",
            "the sampled readout is not finite on the local interval",
        )
    frequency_coefficients = np.polynomial.polynomial.polyfit(
        local_sample / resolved_sigma, local_frequency, order
    )
    local_prediction = np.polynomial.polynomial.polyval(
        local_sample / resolved_sigma, frequency_coefficients
    )
    frequency_error = float(
        np.max(np.abs(local_prediction - local_frequency))
        / max(float(np.ptp(local_frequency)), np.finfo(float).eps)
    )
    diagnostics.update(
        {
            "diffusion_coefficients": tuple(
                float(value) for value in diffusion_coefficients
            ),
            "frequency_coefficients": tuple(
                float(value) for value in frequency_coefficients
            ),
            "diffusion_error": diffusion_error,
            "frequency_error": frequency_error,
            "local_extent": float(local_extent),
            "local_points": int(local_points),
        }
    )
    if not np.isfinite(diffusion_error) or not np.isfinite(frequency_error):
        return failed(
            CAMOutcomeStatus.STOCHASTIC_INVALID,
            "fit_error_not_finite",
            "the polynomial fit error is not finite",
        )

    # 3. The dense polynomial table and the shared stationary quadrature.
    dense_x = np.linspace(lower, upper, int(settings.dense_points))
    scaled = dense_x / resolved_sigma
    drift = np.polynomial.polynomial.polyval(
        dense_x, np.asarray(manifold.drift, dtype=float)
    )
    diffusion = np.polynomial.polynomial.polyval(scaled, diffusion_coefficients)
    frequency = np.polynomial.polynomial.polyval(scaled, frequency_coefficients)
    if not (
        np.all(np.isfinite(drift))
        and np.all(np.isfinite(diffusion))
        and np.all(np.isfinite(frequency))
    ):
        return failed(
            CAMOutcomeStatus.STOCHASTIC_INVALID,
            "dense_table_not_finite",
            "the dense polynomial table is not finite on the physical interval",
        )
    if np.any(diffusion <= 0.0):
        diagnostics["dense_diffusion_minimum"] = float(np.min(diffusion))
        return failed(
            CAMOutcomeStatus.STOCHASTIC_INVALID,
            "dense_diffusion_not_positive",
            "the dense projected diffusion is not positive (minimum "
            f"{float(np.min(diffusion))!r}); it is not clamped",
        )
    table = ScalarGeneratorTable(
        x=tuple(float(value) for value in dense_x),
        drift=tuple(float(value) for value in drift),
        diffusion=tuple(float(value) for value in diffusion),
        readout=tuple(float(value) for value in frequency),
    )
    evaluation = evaluate_generator(table, config=settings.quadrature)
    if not evaluation.normalizable or evaluation.scalar_intensity is None:
        diagnostics["density_normalization_raw"] = evaluation.density_normalization
        diagnostics["evaluation_reason"] = evaluation.reason
        return failed(
            CAMOutcomeStatus.STOCHASTIC_INVALID,
            "density_not_normalizable",
            "the dense polynomial table's stationary density does not normalize"
            + (f": {evaluation.reason}" if evaluation.reason else ""),
            evaluation=evaluation,
            table=table,
        )
    intensity = float(evaluation.scalar_intensity)
    if not np.isfinite(intensity) or intensity < 0.0:
        diagnostics["scalar"] = evaluation.scalar_intensity
        return failed(
            CAMOutcomeStatus.STOCHASTIC_INVALID,
            "intensity_not_finite",
            f"the intensity is {evaluation.scalar_intensity!r}",
            evaluation=evaluation,
            table=table,
        )
    summary: dict[str, Any] = {
        "scalar": intensity,
        "drift_highest": float(manifold.drift[order]),
        "physical_lower": lower,
        "physical_upper": upper,
        "physical_grid_fraction": fraction,
        "domain_clipped": clipped,
        "projected_diffusion_minimum": float(diagnostics["diffusion_minimum"]),
        "projected_diffusion_maximum": float(diagnostics["diffusion_maximum"]),
        "diffusion_fit_error": diffusion_error,
        "frequency_fit_error": frequency_error,
        "frequency_minimum": float(diagnostics["frequency_minimum"]),
        "frequency_maximum": float(diagnostics["frequency_maximum"]),
        "bordered_condition": float(manifold.maximum_bordered_condition),
        "edge_density_ratio": evaluation.endpoint_density_ratio,
        "edge_mass_fraction": evaluation.edge_mass_fraction,
        "minimum_projected_diffusion": evaluation.minimum_projected_diffusion,
        "stationary_mean": evaluation.stationary_mean,
        "stationary_mean_x": evaluation.stationary_mean_x,
        "density_normalization": evaluation.density_normalization,
        "core_point_count": evaluation.core_point_count,
        "edge_point_count": evaluation.edge_point_count,
        "dense_point_count": int(dense_x.size),
        "fit_point_count": int(sample.size),
        "physical_scan_point_count": int(axis.size),
        "local_point_count": int(local_points),
    }
    return PolynomialReconstruction(
        status=CAMOutcomeStatus.VALID,
        order=order,
        manifold=manifold,
        sigma=resolved_sigma,
        lower=lower,
        upper=upper,
        physical_grid_fraction=fraction,
        domain_clipped=clipped,
        diffusion_coefficients=tuple(float(v) for v in diffusion_coefficients),
        frequency_coefficients=tuple(float(v) for v in frequency_coefficients),
        diffusion_error=diffusion_error,
        diffusion_minimum=float(diagnostics["diffusion_minimum"]),
        diffusion_maximum=float(diagnostics["diffusion_maximum"]),
        frequency_error=frequency_error,
        frequency_minimum=float(diagnostics["frequency_minimum"]),
        frequency_maximum=float(diagnostics["frequency_maximum"]),
        generator_table=table,
        evaluation=evaluation,
        summary=summary,
        method=dict(method),
        diagnostics=dict(diagnostics),
    )
