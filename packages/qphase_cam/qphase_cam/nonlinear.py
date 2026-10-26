"""qphase_cam: model-independent nonlinear slow manifold and scalar reconstruction
---------------------------------------------------------
A CAM normal form is only the first term of a slow manifold. This module builds the
manifold to a requested order, embeds it into the full state, and reconstructs the
scalar Langevin problem along it — for **any** model and **any** readout:

* the Jacobian, state matrix and parameter Jacobian come from the adapter, never
  from a fixed mode count or a mode index;
* the readout is a :class:`~qphase_cam.observable.CAMObservable`, so a Rayleigh
  frequency and a matrix element are the same kind of input;
* the critical mode, its left/right normalization and the canonical noise
  covariance are the shared 10F1/10F2a objects, not a private copy.

The definitions are the ones the historical
``scripts/tools/evaluate_kerr_full_3mode_nonlinear_cam.py`` evaluation used:

* the second-order form is a symmetric forward difference of the Jacobian,
  ``B(a,b) = ½[(J(x+a) − J(x))b + (J(x+b) − J(x))a]``;
* at each degree ``d`` the embedding ``e_d`` and the drift coefficient solve the
  bordered system ``[[J − dλI, −r], [l, 0]] [e_d, c] = [transport − nonlinear, 0]``
  with ``l·e_d = 0`` and ``l·r = 1``, and the largest bordered condition number is
  reported;
* the reconstructed scalar problem uses the stationary quadrature
  ``π(x) ∝ exp(∫2f/q) / q`` on a grid between the physical boundaries, and the
  intensity comes from the integrated stationary response, exactly as compactly
  supported as a trapezoid allows.

The Fokker-Planck equation is *not* solved in time: everything here is a stationary
quadrature, and a density that does not normalize, a domain with no positive
diffusion, or a potential the truncated normal form fails to confine is reported
through :class:`~qphase_cam.outcomes.CAMOutcomeStatus` — never patched into a
plausible number.

The sampled ``f``, ``q`` and readout the quadrature integrates are also returned, as
a :class:`ScalarGeneratorTable`, so a local generator or pole solve
(:mod:`qphase_cam.carrier_pole`) starts from the coefficients this module already
computed instead of rebuilding the manifold and the noise.

``reconstructed_intensity`` (the usual ``scalar + full_ou − soft_ou`` combination)
and :func:`ou_intensities` live beside the numerical core rather than inside it, so
the kernel never guesses which correction the caller wants.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, Literal

import numpy as np
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    model_validator,
)
from scipy.integrate import cumulative_trapezoid

from .core.noise import canonical_sample_matrix_covariance, projected_noise_intensities
from .observable import CAMObservable, DerivativeRecord, observable_values
from .outcomes import CAMOutcomeStatus
from .validity import (
    CriticalMode,
    critical_mode,
    projected_intensity,
)

__all__ = [
    "DEFAULT_MANIFOLD_ORDERS",
    "ScalarGeneratorEvaluation",
    "NonlinearManifold",
    "NonlinearManifoldConfig",
    "NoiseIntensityError",
    "OuIntensities",
    "OuIntensityAlgebra",
    "ParameterResponse",
    "ResponsePrediction",
    "ScalarGeneratorTable",
    "ScalarReconstruction",
    "ScalarReconstructionConfig",
    "bilinear_form",
    "manifold_drift",
    "manifold_embedding",
    "noise_stack",
    "parameter_forcing",
    "ou_intensity_algebra",
    "parameter_response",
    "ou_intensities",
    "reconstructed_intensity",
    "response_prediction",
    "evaluate_generator",
    "scalar_reconstruction",
    "slow_manifold",
]

#: The normal-form orders the reference study reports.
DEFAULT_MANIFOLD_ORDERS: tuple[int, ...] = (3, 5, 7)

#: Exponent floor of the density quadrature (``exp(-745)`` is the smallest double).
_LOG_DENSITY_FLOOR = -745.0

#: Failure codes a reconstruction reports alongside ``reason``.
FAILURE_UNRESOLVED = "manifold_unresolved"
FAILURE_UNSTABLE = "soft_mode_not_decaying"
FAILURE_INVALID_DOMAIN = "invalid_domain"
FAILURE_NOT_NORMALIZABLE = "density_not_normalizable"
FAILURE_NOT_CONFINING = "manifold_not_confining"
FAILURE_UNBOUNDED = "physical_region_unbounded"


class NonlinearManifoldConfig(BaseModel):
    """Order and finite-difference conventions of one slow manifold.

    The three method switches exist so a caller can state *which* critical mode, which
    eigenvalue source and which second-order form the manifold is built from instead of
    relying on the historical defaults:

    ``critical_mode_rule``
        ``nearest_zero`` (the default, smallest ``|λ|``) or ``slowest`` (largest real
        part). Both are the shared :func:`~qphase_cam.validity.critical_mode` rules.
    ``critical_eigenvalue_source``
        ``eigenvalue`` (the default, ``Re λ`` of the resolved mode) or ``projected``
        (``left @ J @ right``). The two agree when the mode's left/right eigenvectors
        are a normalised eigenpair of the state Jacobian; they differ when the caller
        passes a mode whose vectors are not that pair — a spectral mode, for example —
        and then the projected value is the one the recursion and the rate use.
    ``bilinear_method``
        ``jacobian_difference`` (the default, the symmetric forward Jacobian difference
        of :func:`bilinear_form`) or ``directional``
        (``adapter.directional(2, state, params, a, b)``, the exact second directional
        derivative when the adapter provides one).
    """

    model_config = ConfigDict(extra="forbid")

    order: int = Field(default=3, ge=2, description="Normal-form truncation order")
    bilinear_offset_scale: float = Field(
        default=1.0,
        gt=0.0,
        description="Scale of the embedding offsets used by the Jacobian difference",
    )
    overlap_tolerance: float = Field(
        default=1e-10, gt=0.0, description="Minimum left-right critical overlap"
    )
    critical_mode_rule: Literal["nearest_zero", "slowest"] = Field(
        default="nearest_zero",
        description="Critical-mode selection: smallest |lambda|, or largest real part",
    )
    critical_eigenvalue_source: Literal["eigenvalue", "projected"] = Field(
        default="eigenvalue",
        description="Use Re(lambda) of the resolved mode, or left @ J @ right",
    )
    bilinear_method: Literal["jacobian_difference", "directional"] = Field(
        default="jacobian_difference",
        description=(
            "Second-order form: symmetric forward Jacobian difference, or the "
            "adapter's exact directional derivative"
        ),
    )


class NonlinearManifold(BaseModel):
    """One slow manifold to a requested order.

    ``embeddings[d - 1]`` is the state-space coefficient of ``x**d`` (so index 0
    carries ``x**1``) and ``drift[d]`` the matching drift coefficient; the drift is
    the polynomial ``Σ_d drift[d] x**d`` along the manifold coordinate.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    order: int
    soft_rate: float
    critical_eigenvalue_real: float
    critical_eigenvalue_imag: float
    noncritical_spectral_gap: float | None
    right: tuple[float, ...]
    left: tuple[float, ...]
    embeddings: tuple[tuple[float, ...], ...]
    drift: tuple[float, ...]
    maximum_bordered_condition: float
    confining: bool
    method: dict[str, Any] = Field(default_factory=dict)


class ScalarReconstructionConfig(BaseModel):
    """Grid and boundary-search controls of one scalar reconstruction.

    The defaults are the reference script's: 3001 points per side, a boundary
    search starting at 2% of the soft-mode width, growing by 1.35 up to 200 widths,
    and 60 bisection steps.
    """

    model_config = ConfigDict(extra="forbid")

    grid_points: int = Field(default=3001, ge=3, description="Points per side of x=0")
    boundary_iterations: int = Field(default=60, ge=1)
    boundary_start_relative: float = Field(default=0.02, gt=0.0)
    boundary_limit_relative: float = Field(default=200.0, gt=0.0)
    boundary_growth: float = Field(default=1.35, gt=1.0)
    physical_tolerance: float = Field(default=1e-10, ge=0.0)
    minimum_projected_diffusion: float = Field(default=1e-12, gt=0.0)
    log_density_floor: float = Field(
        default=_LOG_DENSITY_FLOOR,
        le=0.0,
        description="Smallest log-density kept before normalization",
    )
    edge_fraction: float = Field(
        default=0.02, gt=0.0, le=0.5, description="Fraction of the grid per edge"
    )
    density_floor_relative: float = Field(
        default=1e-13,
        gt=0.0,
        description="Density floor of the intensity integrand, relative to its peak",
    )
    integration_split: Literal["grid_zero", "nearest_zero"] = Field(
        default="grid_zero",
        description=(
            "Where the intensity integral switches from the left to the right "
            "antiderivative. 'grid_zero' (the default) uses the grid index at x = 0, "
            "which is exact for the reconstruction's own half-grid layout; "
            "'nearest_zero' uses the grid point nearest x = 0, the convention the "
            "historical effective-drift probe used for its dense table. The two agree "
            "on a grid that contains zero and differ on one that does not"
        ),
    )
    intensity_integration: Literal["zero_split", "centered_cdf"] = Field(
        default="zero_split",
        description=(
            "How the intensity integral is assembled. 'zero_split' (the default) "
            "switches between the left and right antiderivative as "
            "``integration_split`` says; 'centered_cdf' subtracts the density-weighted "
            "total from the cumulative integral (the historical CLK3 convention) and "
            "ignores ``integration_split``"
        ),
    )
    core_density_comparison: Literal["greater_equal", "greater"] = Field(
        default="greater_equal",
        description=(
            "How a grid point is admitted to the intensity core: "
            "'greater_equal' keeps ``density >= peak * density_floor_relative`` (the "
            "default), 'greater' keeps ``>`` (the CLK3 convention)"
        ),
    )
    edge_count_rule: Literal["floor_min_one", "ceil_min_two"] = Field(
        default="floor_min_one",
        description=(
            "How many grid points each edge window holds: 'floor_min_one' takes "
            "``max(1, int(edge_fraction * n))`` (the default), 'ceil_min_two' takes "
            "``max(2, ceil(edge_fraction * n))`` (the CLK3 convention)"
        ),
    )


class ScalarGeneratorTable(BaseModel):
    """The scalar backward-generator coefficients sampled on one grid.

    These are the coefficients of the one-dimensional Feynman-Kac (backward)
    generator ``L = f(x) ∂_x + ½ q(x) ∂_x²`` together with the readout
    ``omega(x)`` — the *coefficients*, not a discretized generator matrix: a
    consumer builds its own stencil (:func:`qphase_cam.carrier_pole.carrier_pole`
    resamples these with a monotone PCHIP onto a uniform grid and uses an upwind
    finite-volume stencil).

    The contract is what makes such a consumer safe to write: ``x`` is finite and
    strictly increasing, at least three points are present, every coefficient is
    finite and ``diffusion`` is strictly positive. A NaN, a duplicated grid point or
    a non-positive ``q`` therefore cannot be carried silently into a boundary-value
    or eigenvalue problem.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    x: tuple[float, ...]
    drift: tuple[float, ...]
    diffusion: tuple[float, ...]
    readout: tuple[float, ...]

    @model_validator(mode="after")
    def _check_table(self) -> ScalarGeneratorTable:
        size = len(self.x)
        if size < 3:
            raise ValueError(f"a generator table needs at least 3 points; got {size}")
        for name in ("drift", "diffusion", "readout"):
            length = len(getattr(self, name))
            if length != size:
                raise ValueError(
                    f"the {name} array has {length} values for {size} x points"
                )
        grid = np.asarray(self.x, dtype=float)
        if not np.all(np.isfinite(grid)):
            raise ValueError("the x grid is not finite")
        if np.any(np.diff(grid) <= 0.0):
            raise ValueError("the x grid is not strictly increasing")
        for name in ("drift", "diffusion", "readout"):
            values = np.asarray(getattr(self, name), dtype=float)
            if not np.all(np.isfinite(values)):
                raise ValueError(f"the {name} coefficients are not finite")
        if np.any(np.asarray(self.diffusion, dtype=float) <= 0.0):
            raise ValueError(
                "the diffusion coefficients must be strictly positive; a "
                "non-positive projected intensity has no generator"
            )
        return self


class ScalarReconstruction(BaseModel):
    """The stationary scalar problem along one manifold.

    ``generator_table`` is the same ``f``, ``q`` and readout the reported scalars are
    integrated from, kept so a local generator/pole solve does not have to rebuild
    the manifold; it is ``None`` for the failures that stop before the grid is
    usable, and kept for the density and boundary verdicts.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: CAMOutcomeStatus
    reason: str | None = None
    reason_code: str | None = None
    order: int
    confining: bool
    validity_domain: str | None = None

    x_min: float | None = None
    x_max: float | None = None
    negative_physical_boundary: bool | None = None
    positive_physical_boundary: bool | None = None
    sigma: float | None = None

    scalar_intensity: float | None = None
    stationary_mean: float | None = None
    stationary_response: float | None = None
    path_information_rate: float | None = None
    minimum_projected_diffusion: float | None = None
    endpoint_density_ratio: float | None = None
    edge_mass_fraction: float | None = None
    density_normalization: float | None = None

    grid_points: int | None = None
    generator_table: ScalarGeneratorTable | None = None
    manifold: NonlinearManifold
    method: dict[str, Any] = Field(default_factory=dict)
    diagnostics: dict[str, Any] = Field(default_factory=dict)


class ScalarGeneratorEvaluation(BaseModel):
    """The stationary scalar problem of one generator table, evaluated.

    ``density`` is the normalized stationary density on ``x`` (the table's own grid).
    ``normalizable`` is the evaluation's own verdict: it is false when the density
    integral is not finite and positive *and* when the requested numbers cannot be
    produced from this table at all — an empty intensity core under the configured
    density comparison, a non-finite integrand, or a non-finite intensity. In both cases
    the density is returned as far as it was computed (unnormalized when the integral
    itself failed) and every derived number is ``None``, with ``reason`` naming the
    condition; it is not a claim that only the raw density integral decides it. A
    caller that reports a status passes that reason on instead of inventing one.

    ``stationary_mean`` is the readout mean and ``stationary_mean_x`` the mean of ``x``
    itself; ``stationary_response`` and ``path_information_rate`` need a forcing vector
    and stay ``None`` without one. ``core_point_count`` counts the grid points whose
    density is admitted to the intensity core, ``edge_point_count`` the points of each
    edge window the edge mass is integrated over.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    x: tuple[float, ...]
    density: tuple[float, ...]
    density_normalization: float
    normalizable: bool
    stationary_mean: float | None = None
    stationary_mean_x: float | None = None
    stationary_response: float | None = None
    path_information_rate: float | None = None
    scalar_intensity: float | None = None
    edge_mass_fraction: float | None = None
    endpoint_density_ratio: float | None = None
    minimum_projected_diffusion: float
    core_point_count: int = 0
    edge_point_count: int = 0
    reason: str | None = None


def _not_evaluable(
    x: np.ndarray,
    density: np.ndarray,
    normalization: float,
    minimum_diffusion: float,
    reason: str,
) -> ScalarGeneratorEvaluation:
    """Return the evaluation of a table that cannot be evaluated.

    The derived numbers stay ``None`` — an unusable table has no intensity, not an
    intensity of zero — and ``normalizable`` is false, so a caller reports the failure
    through its own status instead of a valid result carrying NaN. ``reason`` says which
    condition stopped the evaluation.
    """
    return ScalarGeneratorEvaluation(
        x=tuple(float(value) for value in x),
        density=tuple(float(value) for value in density),
        density_normalization=float(normalization),
        normalizable=False,
        minimum_projected_diffusion=float(minimum_diffusion),
        reason=reason,
    )


def evaluate_generator(
    table: ScalarGeneratorTable,
    *,
    forcing: Any | None = None,
    config: ScalarReconstructionConfig | None = None,
) -> ScalarGeneratorEvaluation:
    """Evaluate the stationary scalar problem of one generator table.

    The algorithm is the reconstruction's own, line for line: the density is
    ``pi ~ exp(int 2f/q) / q`` clipped at ``log_density_floor`` and normalized; the
    intensity is ``4 (int (omega - mean) pi)^2 / (q pi)`` summed over the points
    whose density is above ``density_floor_relative`` times the peak; the edge mass
    is the density integrated over ``edge_fraction`` of the grid at both ends and the
    endpoint ratio compares the larger edge density with the peak. With ``forcing``
    the path-information rate ``int forcing^2 pi / q`` and the stationary response
    ``int (omega - mean) score pi`` with ``score = int 2 forcing / q`` are added; the
    forcing is the caller's, because it comes from the model's parameter Jacobian.

    This is a pure evaluation of a table: it never decides whether the manifold is
    confining or whether the physical region is bounded. Those verdicts belong to the
    caller, which is why the evaluation reports ``normalizable=False`` with a ``reason``
    instead of raising. That flag covers everything the requested numbers need and the
    table does not give: a density that does not normalize, a non-positive projected
    diffusion, an intensity core left empty by the configured comparison, a non-finite
    integrand and a non-finite intensity. The derived numbers are then ``None``, never a
    fabricated zero.
    """
    settings = config or ScalarReconstructionConfig()
    x = np.asarray(table.x, dtype=float)
    f = np.asarray(table.drift, dtype=float)
    q = np.asarray(table.diffusion, dtype=float)
    omega = np.asarray(table.readout, dtype=float)
    if np.any(q <= 0.0):
        # ``ScalarGeneratorTable`` already refuses a non-positive diffusion; this is the
        # guard for a table constructed around that contract.
        return _not_evaluable(
            x,
            np.full(x.size, np.nan),
            0.0,
            float(np.min(q)),
            "the projected diffusion is not positive on the grid",
        )
    potential = cumulative_trapezoid(2.0 * f / q, x, initial=0.0)
    log_density = potential - np.log(q)
    log_density -= float(np.max(log_density))
    density = np.exp(np.clip(log_density, settings.log_density_floor, 0.0))
    normalization = float(np.trapezoid(density, x))
    minimum_diffusion = float(np.min(q))
    if not np.isfinite(normalization) or normalization <= 0.0:
        return _not_evaluable(
            x,
            density,
            normalization,
            minimum_diffusion,
            f"the stationary density does not normalize (integral {normalization!r})",
        )
    density = density / normalization
    path_information: float | None = None
    stationary_score: np.ndarray | None = None
    if forcing is not None:
        force = np.asarray(forcing, dtype=float).reshape(-1)
        if force.size != x.size:
            raise ValueError(
                f"the forcing vector has {force.size} values for {x.size} x points"
            )
        path_information = float(np.trapezoid(force**2 * density / q, x))
        stationary_score = cumulative_trapezoid(2.0 * force / q, x, initial=0.0)
    stationary_mean = float(np.trapezoid(omega * density, x))
    stationary_mean_x = float(np.trapezoid(x * density, x))
    centered = omega - stationary_mean
    stationary_response: float | None = None
    if stationary_score is not None:
        response = float(np.trapezoid(centered * stationary_score * density, x))
        stationary_response = response if np.isfinite(response) else None
    weighted = centered * density
    floor = float(np.max(density)) * settings.density_floor_relative
    if settings.core_density_comparison == "greater":
        core = density > floor
    else:
        core = density >= floor
    if not np.all(np.isfinite(weighted)):
        return _not_evaluable(
            x,
            density,
            normalization,
            minimum_diffusion,
            "the weighted readout is not finite",
        )
    if settings.intensity_integration == "centered_cdf":
        # The historical CLK3 assembly: the density-weighted cumulative integral,
        # recentred so its total is zero, then integrated over the core only. A core
        # that cannot be integrated is reported, not turned into an intensity of zero.
        if np.count_nonzero(core) < 2:
            return _not_evaluable(
                x,
                density,
                normalization,
                minimum_diffusion,
                "no intensity core survives the density floor",
            )
        cumulative = cumulative_trapezoid(weighted, x, initial=0.0)
        distribution = cumulative_trapezoid(density, x, initial=0.0)
        cumulative = cumulative - distribution * cumulative[-1]
        repeated = cumulative[core] ** 2 / (q[core] * density[core])
        if not np.all(np.isfinite(repeated)):
            return _not_evaluable(
                x,
                density,
                normalization,
                minimum_diffusion,
                "the intensity integrand is not finite",
            )
        intensity = float(4.0 * np.trapezoid(repeated, x[core]))
    else:
        h_left = cumulative_trapezoid(weighted, x, initial=0.0)
        h_right = -cumulative_trapezoid(weighted[::-1], x[::-1], initial=0.0)[::-1]
        # The integral switches from the left to the right antiderivative at x = 0. The
        # default split is the grid index of zero (the reconstruction's own half-grid
        # layout); the historical probe's dense table split at the grid point nearest
        # zero, which is what ``integration_split="nearest_zero"`` selects.
        if settings.integration_split == "nearest_zero":
            zero_index = int(np.argmin(np.abs(x)))
        else:
            zero_index = int(np.searchsorted(x, 0.0, side="right")) - 1
        integrated = np.where(np.arange(x.size) <= zero_index, h_left, h_right)
        integrand = np.zeros_like(x)
        integrand[core] = 4.0 * integrated[core] ** 2 / (q[core] * density[core])
        if not np.all(np.isfinite(integrand)):
            return _not_evaluable(
                x,
                density,
                normalization,
                minimum_diffusion,
                "the intensity integrand is not finite",
            )
        intensity = float(np.trapezoid(integrand, x))
    if not np.isfinite(intensity) or intensity < 0.0:
        return _not_evaluable(
            x,
            density,
            normalization,
            minimum_diffusion,
            f"the intensity is {intensity!r}",
        )
    count = x.size
    if settings.edge_count_rule == "ceil_min_two":
        edge_count = max(2, int(math.ceil(settings.edge_fraction * count)))
    else:
        edge_count = max(1, int(settings.edge_fraction * count))
    edge_mass = float(
        np.trapezoid(density[:edge_count], x[:edge_count])
        + np.trapezoid(density[-edge_count:], x[-edge_count:])
    )
    endpoint_ratio = float(
        max(density[0], density[-1]) / max(float(np.max(density)), 1.0e-300)
    )
    return ScalarGeneratorEvaluation(
        x=tuple(float(value) for value in x),
        density=tuple(float(value) for value in density),
        density_normalization=normalization,
        normalizable=True,
        stationary_mean=stationary_mean,
        stationary_mean_x=stationary_mean_x,
        stationary_response=stationary_response,
        path_information_rate=(
            path_information
            if path_information is not None and np.isfinite(path_information)
            else None
        ),
        scalar_intensity=intensity,
        edge_mass_fraction=edge_mass,
        endpoint_density_ratio=endpoint_ratio,
        minimum_projected_diffusion=minimum_diffusion,
        core_point_count=int(np.count_nonzero(core)),
        edge_point_count=edge_count,
    )


class NoiseIntensityError(ValueError):
    """The noise does not support a Gaussian intensity.

    Raised when the projected noise is not finite and positive, or when the soft or
    full intensity comes out non-finite or negative. It is a ``ValueError`` so existing
    handlers keep raising, and it is its own type so a caller can report the failure as
    a *stochastic* one instead of an identity problem.
    """


class OuIntensities(BaseModel):
    """The Gaussian (linear-response) intensity the reconstruction corrects."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    rate: float
    projection: float
    projected_noise: float
    sigma: float
    soft_intensity: float
    full_intensity: float
    zero_frequency_gradient: tuple[float, ...]
    method: dict[str, Any] = Field(default_factory=dict)


def bilinear_form(
    adapter: Any,
    state: Any,
    params: Mapping[str, Any],
    first: Any,
    second: Any,
    *,
    offset_scale: float = 1.0,
) -> np.ndarray:
    """Return the symmetric second-order form ``B(a, b)`` at one state.

    The estimate is a forward difference of the Jacobian along each direction:
    ``½[(J(x+a) − J(x))b + (J(x+b) − J(x))a]``. Scaling the offsets is the caller's
    choice (``offset_scale``); the reference evaluation uses the embedding vectors
    themselves, which is ``offset_scale = 1``.
    """
    point = np.asarray(state, dtype=float).reshape(-1)
    left_vector = offset_scale * np.asarray(first, dtype=float).reshape(-1)
    right_vector = offset_scale * np.asarray(second, dtype=float).reshape(-1)
    parameter_map = dict(params or {})
    base = np.asarray(adapter.jacobian(point, parameter_map), dtype=float)
    first_leg = (
        np.asarray(adapter.jacobian(point + left_vector, parameter_map), dtype=float)
        - base
    ) @ right_vector
    second_leg = (
        np.asarray(adapter.jacobian(point + right_vector, parameter_map), dtype=float)
        - base
    ) @ left_vector
    return 0.5 * (first_leg + second_leg)


def _critical(
    adapter: Any,
    state: np.ndarray,
    params: Mapping[str, Any],
    overlap_tolerance: float,
    rule: str = "nearest_zero",
) -> CriticalMode:
    """Return the critical mode of one state or raise."""
    jacobian = np.asarray(adapter.jacobian(state, params), dtype=float)
    if not np.all(np.isfinite(jacobian)):
        raise ValueError("the dynamics Jacobian is not finite")
    mode = critical_mode(jacobian, overlap_tolerance=overlap_tolerance, rule=rule)
    if mode is None:
        raise ValueError(
            "the soft mode is unresolved (complex or overlapping left/right "
            "eigenvectors)"
        )
    return mode


def slow_manifold(
    adapter: Any,
    state: Any,
    params: Mapping[str, Any],
    *,
    order: int | None = None,
    config: NonlinearManifoldConfig | None = None,
    mode: CriticalMode | None = None,
) -> NonlinearManifold:
    """Build the slow manifold of one state to ``order``.

    Raises
    ------
    ValueError
        If the state or Jacobian is not finite, the critical mode is unresolved,
        the soft mode does not decay, or a bordered system is singular.

    """
    settings = config or NonlinearManifoldConfig()
    resolved_order = int(settings.order if order is None else order)
    if resolved_order < 2:
        raise ValueError(f"a slow manifold needs order >= 2; got {resolved_order}")
    point = np.asarray(state, dtype=float).reshape(-1)
    if point.size == 0 or not np.all(np.isfinite(point)):
        raise ValueError("the state must be a finite vector")
    parameter_map = dict(params or {})
    resolved_mode = mode or _critical(
        adapter,
        point,
        parameter_map,
        settings.overlap_tolerance,
        settings.critical_mode_rule,
    )
    jacobian = np.asarray(adapter.jacobian(point, parameter_map), dtype=float)
    right = np.asarray(resolved_mode.right, dtype=float)
    left = np.asarray(resolved_mode.left, dtype=float)
    eigenvalue = float(np.real(resolved_mode.value))
    lam = (
        eigenvalue
        if settings.critical_eigenvalue_source == "eigenvalue"
        else float(left @ jacobian @ right)
    )
    if lam >= 0.0:
        raise ValueError(
            f"the soft mode does not decay (real part {lam:.3e}); the manifold is "
            "not a stable slow manifold"
        )
    if settings.bilinear_method == "directional":
        if not callable(getattr(adapter, "directional", None)):
            raise ValueError(
                "bilinear_method='directional' needs an adapter that implements "
                "directional(order, state, params, *directions)"
            )

        def second_form(first: np.ndarray, second: np.ndarray) -> np.ndarray:
            return np.asarray(
                adapter.directional(2, point, parameter_map, first, second),
                dtype=float,
            ).reshape(-1)

    else:

        def second_form(first: np.ndarray, second: np.ndarray) -> np.ndarray:
            return bilinear_form(
                adapter,
                point,
                parameter_map,
                first,
                second,
                offset_scale=settings.bilinear_offset_scale,
            )

    embeddings: list[np.ndarray] = [np.zeros_like(point), right]
    drift = np.zeros(resolved_order + 1, dtype=float)
    drift[1] = lam
    maximum_condition = 0.0
    for degree in range(2, resolved_order + 1):
        nonlinear = np.zeros_like(point)
        for lower_degree in range(1, degree):
            nonlinear += 0.5 * second_form(
                embeddings[lower_degree],
                embeddings[degree - lower_degree],
            )
        transport = np.zeros_like(point)
        for lower_degree in range(2, degree):
            transport += (
                lower_degree
                * embeddings[lower_degree]
                * drift[degree - lower_degree + 1]
            )
        bordered = np.block(
            [
                [
                    jacobian - degree * lam * np.eye(point.size),
                    -right[:, None],
                ],
                [left[None, :], np.zeros((1, 1))],
            ]
        )
        condition = float(np.linalg.cond(bordered))
        if not np.isfinite(condition):
            raise ValueError(
                f"the bordered system at degree {degree} is singular; the manifold "
                "is not resolvable at this state"
            )
        maximum_condition = max(maximum_condition, condition)
        try:
            solution = np.linalg.solve(
                bordered, np.concatenate((transport - nonlinear, np.zeros(1)))
            )
        except np.linalg.LinAlgError as exc:  # pragma: no cover - cond guard above
            raise ValueError(
                f"the bordered system at degree {degree} could not be solved: {exc}"
            ) from exc
        if not np.all(np.isfinite(solution)):
            raise ValueError(f"the bordered solution at degree {degree} is not finite")
        embeddings.append(solution[:-1])
        drift[degree] = solution[-1]
    return NonlinearManifold(
        order=resolved_order,
        soft_rate=-lam,
        critical_eigenvalue_real=float(np.real(resolved_mode.value)),
        critical_eigenvalue_imag=float(np.imag(resolved_mode.value)),
        noncritical_spectral_gap=(
            None
            if not np.isfinite(resolved_mode.spectral_gap)
            else float(resolved_mode.spectral_gap)
        ),
        right=tuple(float(value) for value in right),
        left=tuple(float(value) for value in left),
        # Stored without the leading zero entry: index 0 is the x**1 coefficient.
        embeddings=tuple(
            tuple(float(value) for value in coefficient)
            for coefficient in embeddings[1:]
        ),
        drift=tuple(float(value) for value in drift),
        maximum_bordered_condition=maximum_condition,
        confining=bool(drift[resolved_order] < 0.0),
        method={
            "service": "qphase_cam.nonlinear.slow_manifold",
            "critical_mode_rule": settings.critical_mode_rule,
            "critical_eigenvalue_source": settings.critical_eigenvalue_source,
            "bilinear_method": settings.bilinear_method,
            "overlap_tolerance": float(settings.overlap_tolerance),
            "left_normalization": "left.right = 1 with a unit right vector",
            "bilinear_form": (
                "symmetric forward Jacobian difference"
                if settings.bilinear_method == "jacobian_difference"
                else "adapter.directional(2, state, params, a, b)"
            ),
            "bilinear_offset_scale": float(settings.bilinear_offset_scale),
            "critical_rate": "soft_rate = -drift[1]",
            "bordered_system": "[[J - d*lambda I, -right], [left, 0]]",
            "order": resolved_order,
            "confining_rule": "drift[order] < 0",
        },
    )


def manifold_embedding(manifold: NonlinearManifold, state: Any, x: Any) -> np.ndarray:
    """Return the full state of the manifold coordinate ``x`` (one row per value)."""
    point = np.asarray(state, dtype=float).reshape(-1)
    values = np.atleast_1d(np.asarray(x, dtype=float)).reshape(-1)
    output = np.broadcast_to(point, (values.size, point.size)).copy()
    for degree, coefficient in enumerate(manifold.embeddings, start=1):
        output += values[:, None] ** degree * np.asarray(coefficient)[None, :]
    return output


def manifold_drift(manifold: NonlinearManifold, x: Any) -> np.ndarray:
    """Return the drift polynomial ``Σ_d drift[d] x**d``."""
    values = np.atleast_1d(np.asarray(x, dtype=float)).reshape(-1)
    output = np.zeros_like(values)
    for degree, coefficient in enumerate(manifold.drift[1:], start=1):
        output += coefficient * values**degree
    return output


def _noise_covariance(
    adapter: Any, model: Any, vector: Any, params: Mapping[str, Any]
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return one state matrix, its noise covariance and the physical spectrum.

    A non-finite matrix or diffusion returns a NaN covariance instead of raising:
    the assessment turns that into an ``invalid_domain`` status, which is the
    honest answer for a model whose diffusion is not a covariance at all.
    """
    matrix = np.asarray(adapter.state_matrix(vector, params), dtype=complex)
    diffusion = np.asarray(model.cam_diffusion(matrix, params), dtype=complex)
    hermitian = (matrix + matrix.conjugate().T) / 2.0
    physical = float(np.min(np.linalg.eigvalsh(hermitian)))
    if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(diffusion)):
        return matrix, np.full((matrix.size, matrix.size), np.nan), physical
    try:
        covariance = canonical_sample_matrix_covariance(matrix, diffusion)
    except ValueError:
        return matrix, np.full((matrix.size, matrix.size), np.nan), physical
    if not np.all(np.isfinite(covariance)):
        return matrix, np.full_like(covariance, np.nan), physical
    return matrix, covariance, physical


def noise_stack(
    adapter: Any, model: Any, vectors: Any, params: Mapping[str, Any]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the state matrices, diffusions and physical spectra of a state stack.

    This is the one place that owns "ask for a batch when the adapter and the model
    declare they can answer one, otherwise call the same single-state function row by
    row": every service that samples a stack of states goes through it, so no second
    copy of that policy can drift from the reconstruction's own.

    Both the adapter and the model are asked once for the whole stack when they
    declare they can answer a batch (``supports_batched_states`` /
    ``supports_batched_matrices``); otherwise the same single-state call is made per
    row. There is no silent fallback: a batch answer whose shape is not one matrix per
    state raises, because a shape mismatch is a broken contract rather than an
    unusable state.
    """
    stack = np.asarray(vectors, dtype=float)
    if stack.ndim != 2:
        raise ValueError(
            f"a state stack must be (n, coordinates); got shape {stack.shape}"
        )
    if getattr(adapter, "supports_batched_states", False):
        matrices = np.asarray(adapter.state_matrix(stack, params), dtype=complex)
    else:
        # An adapter with only the single-state contract is sampled row by row; the
        # arithmetic is the single-state path's.
        matrices = np.stack(
            [
                np.asarray(adapter.state_matrix(row, params), dtype=complex)
                for row in stack
            ]
        )
    if matrices.shape[:-2] != (stack.shape[0],) or (
        matrices.shape[-2] != matrices.shape[-1]
    ):
        raise ValueError(
            f"the adapter returned state matrices of shape {matrices.shape} for "
            f"{stack.shape[0]} states"
        )
    if getattr(model, "supports_batched_matrices", False):
        diffusion = np.asarray(model.cam_diffusion(matrices, params), dtype=complex)
    else:
        # A model whose cam_diffusion handles exactly one matrix is sampled row by
        # row; a fresh copy keeps the evaluation free of stack strides.
        diffusion = np.stack(
            [
                np.asarray(
                    model.cam_diffusion(
                        np.array(row, dtype=complex, copy=True), params
                    ),
                    dtype=complex,
                )
                for row in matrices
            ]
        )
    if diffusion.shape != matrices.shape:
        raise ValueError(
            f"the model returned diffusion matrices of shape {diffusion.shape} for "
            f"state matrices of shape {matrices.shape}"
        )
    hermitian = (matrices + np.conjugate(np.swapaxes(matrices, -1, -2))) / 2.0
    physical = np.min(np.linalg.eigvalsh(hermitian), axis=-1)
    return matrices, diffusion, physical


def _boundary(
    adapter: Any,
    model: Any,
    point: np.ndarray,
    params: Mapping[str, Any],
    manifold: NonlinearManifold,
    sigma: float,
    sign: float,
    settings: ScalarReconstructionConfig,
) -> tuple[float, bool]:
    """Search one side for the edge of the physical, positive-diffusion region."""
    left = np.asarray(manifold.left, dtype=float)
    accepted_limit = 0.0
    trial = max(settings.boundary_start_relative * sigma, 1.0e-12)
    limit = max(settings.boundary_limit_relative * sigma, 1.0e-6)

    def usable(value: float) -> bool:
        vector = manifold_embedding(manifold, point, np.asarray([value]))[0]
        _, covariance, physical = _noise_covariance(adapter, model, vector, params)
        projected = projected_intensity(covariance, left, left)
        return bool(
            physical >= -settings.physical_tolerance
            and projected > settings.minimum_projected_diffusion
        )

    while trial <= limit:
        if not usable(sign * trial):
            low, high = accepted_limit, trial
            for _ in range(settings.boundary_iterations):
                middle = 0.5 * (low + high)
                if usable(sign * middle):
                    low = middle
                else:
                    high = middle
            return sign * low, True
        accepted_limit = trial
        trial *= settings.boundary_growth
    return sign * limit, False


def parameter_forcing(
    adapter: Any,
    vectors: Any,
    params: Mapping[str, Any],
    left: Any,
    direction: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the reduced and full parameter forcing of a state stack.

    ``full`` is ``parameter_jacobian(x) · direction`` per row and ``reduced`` is that
    vector projected on the critical left eigenvector. Both are contracted in the
    reference evaluation's order (one matrix-vector product per row, then one dot per
    row), so a published rate keeps its last bits, and the Jacobian is asked once for
    the whole stack when the adapter declares it can answer a batch — the same policy
    as :func:`noise_stack`, so a second copy of it cannot drift.

    ``direction`` lives in the adapter's **parameter** space (``parameter_direction``
    returns one entry per model parameter), while the Jacobian maps that space onto the
    canonical coordinates; the two counts are therefore independent and only the
    Jacobian's own parameter axis is checked.

    Raises
    ------
    ValueError
        If the stack is not ``(n, coordinates)``, the adapter does not return one
        Jacobian per state, or the direction does not match the Jacobian's parameter
        axis.

    """
    stack = np.asarray(vectors, dtype=float)
    if stack.ndim != 2:
        raise ValueError(
            f"a state stack must be (n, coordinates); got shape {stack.shape}"
        )
    pointer = np.asarray(left, dtype=float).reshape(-1)
    step = np.asarray(direction, dtype=float).reshape(-1)
    if pointer.size != stack.shape[1]:
        raise ValueError(
            f"the projection has {pointer.size} entries for {stack.shape[1]} "
            "coordinates"
        )
    parameter_map = dict(params or {})
    if getattr(adapter, "supports_batched_states", False):
        jacobians = np.asarray(
            adapter.parameter_jacobian(stack, parameter_map), dtype=float
        )
    else:
        jacobians = np.stack(
            [
                np.asarray(adapter.parameter_jacobian(row, parameter_map), dtype=float)
                for row in stack
            ]
        )
    if jacobians.shape[:-2] != (stack.shape[0],):
        raise ValueError(
            f"the adapter returned parameter Jacobians of shape {jacobians.shape} "
            f"for {stack.shape[0]} states"
        )
    if jacobians.shape[-1] != step.size:
        raise ValueError(
            f"the parameter direction has {step.size} entries for the Jacobian's "
            f"{jacobians.shape[-1]} parameter axis"
        )
    full = np.einsum("nij,j->ni", jacobians, step, optimize=True)
    reduced = np.einsum("i,ni->n", pointer, full, optimize=True)
    return reduced, full


def scalar_reconstruction(
    adapter: Any,
    model: Any,
    observable: CAMObservable,
    *,
    state: Any,
    params: Mapping[str, Any],
    order: int | None = None,
    parameter_direction: Any | None = None,
    sigma: float | None = None,
    manifold: NonlinearManifold | None = None,
    manifold_config: NonlinearManifoldConfig | None = None,
    config: ScalarReconstructionConfig | None = None,
) -> ScalarReconstruction:
    """Reconstruct the stationary scalar problem along one slow manifold.

    ``model`` must provide ``cam_diffusion(state_matrix, params)``. The readout is
    evaluated on the manifold through :func:`~qphase_cam.observable.observable_values`,
    which is the same readout one state at a time or one stack at a time depending on
    what the observable implements. ``parameter_direction`` (a state-space vector, e.g.
    ``adapter.parameter_direction(name, scale)``) enables the path-information rate and
    the stationary response.

    The grid is sampled in one pass: one adapter call for the state matrices, one model
    call for their diffusions, then the batch covariance projection of
    :mod:`qphase_cam.core.noise`. A model or observable without a batch form is
    sampled row by row instead, with the same single-state arithmetic — the batch path
    is a way to evaluate the same readout once, not a second definition of it.
    """
    settings = config or ScalarReconstructionConfig()
    point = np.asarray(state, dtype=float).reshape(-1)
    parameter_map = dict(params or {})
    method: dict[str, Any] = {
        "service": "qphase_cam.nonlinear.scalar_reconstruction",
        "quadrature": "stationary cumulative trapezoid (no time-domain PDE)",
        "density": "pi ~ exp(int 2f/q) / q, normalized on the physical grid",
        "intensity": "4 (int (omega - mean) pi)^2 / (q pi) summed over the core",
        "integration_split": settings.integration_split,
        "intensity_integration": settings.intensity_integration,
        "core_density_comparison": settings.core_density_comparison,
        "edge_count_rule": settings.edge_count_rule,
        "readout": observable.identity.model_dump(mode="json"),
    }
    diagnostics: dict[str, Any] = {}
    resolved_manifold = manifold
    if resolved_manifold is None:
        manifold_settings = manifold_config or NonlinearManifoldConfig()
        mode: CriticalMode | None = None
        try:
            jacobian = np.asarray(adapter.jacobian(point, parameter_map), dtype=float)
            # The initial selection uses the *same* configuration the manifold is
            # built with: fixing it to nearest_zero here would silently ignore the
            # caller's rule and overlap tolerance.
            mode = critical_mode(
                jacobian,
                overlap_tolerance=manifold_settings.overlap_tolerance,
                rule=manifold_settings.critical_mode_rule,
            )
        except (ValueError, np.linalg.LinAlgError) as exc:
            diagnostics["manifold_error"] = str(exc)
        if mode is None:
            return _failed(
                order=int(manifold_settings.order if order is None else order),
                status=CAMOutcomeStatus.UNIDENTIFIABLE,
                code=FAILURE_UNRESOLVED,
                reason=diagnostics.get("manifold_error", "the soft mode is unresolved"),
                method=method,
                diagnostics=diagnostics,
            )
        try:
            resolved_manifold = slow_manifold(
                adapter,
                point,
                parameter_map,
                order=order,
                config=manifold_config,
                mode=mode,
            )
        except ValueError as exc:
            unstable = "does not decay" in str(exc)
            return _failed(
                order=int(order or 3),
                status=(
                    CAMOutcomeStatus.UNSTABLE
                    if unstable
                    else CAMOutcomeStatus.UNIDENTIFIABLE
                ),
                code=FAILURE_UNSTABLE if unstable else FAILURE_UNRESOLVED,
                reason=str(exc),
                method=method,
                diagnostics=diagnostics,
            )

    def report(
        status: CAMOutcomeStatus,
        reason: str | None,
        *,
        code: str | None = None,
        updates: Mapping[str, Any] | None = None,
    ) -> ScalarReconstruction:
        payload: dict[str, Any] = {
            "status": status,
            "reason": reason,
            "reason_code": code,
            "order": resolved_manifold.order,
            "confining": resolved_manifold.confining,
            "manifold": resolved_manifold,
            "method": dict(method),
            "diagnostics": dict(diagnostics),
        }
        payload.update(dict(updates or {}))
        return ScalarReconstruction(**payload)

    if not callable(getattr(model, "cam_diffusion", None)):
        raise ValueError(
            "a scalar reconstruction needs a model that implements "
            "cam_diffusion(state_matrix, params)"
        )
    left = np.asarray(resolved_manifold.left, dtype=float)
    _, center_covariance, center_physical = _noise_covariance(
        adapter, model, point, parameter_map
    )
    projected_noise = projected_intensity(center_covariance, left, left)
    diagnostics["center_projected_noise"] = projected_noise
    diagnostics["center_minimum_eigenvalue"] = center_physical
    if not np.isfinite(projected_noise) or (
        projected_noise <= settings.minimum_projected_diffusion
    ):
        return report(
            CAMOutcomeStatus.STOCHASTIC_INVALID,
            "the projected noise along the critical mode is not finite and positive",
            code=FAILURE_INVALID_DOMAIN,
        )
    resolved_sigma = (
        float(sigma)
        if sigma is not None
        else float(np.sqrt(projected_noise / (2.0 * resolved_manifold.soft_rate)))
    )
    diagnostics["sigma"] = resolved_sigma
    if resolved_sigma <= 0.0 or not np.isfinite(resolved_sigma):
        return report(
            CAMOutcomeStatus.STOCHASTIC_INVALID,
            f"the soft-mode width is not finite and positive; got {resolved_sigma}",
            code=FAILURE_INVALID_DOMAIN,
        )

    negative, negative_physical = _boundary(
        adapter,
        model,
        point,
        parameter_map,
        resolved_manifold,
        resolved_sigma,
        -1.0,
        settings,
    )
    positive, positive_physical = _boundary(
        adapter,
        model,
        point,
        parameter_map,
        resolved_manifold,
        resolved_sigma,
        1.0,
        settings,
    )
    left_grid = np.linspace(negative, 0.0, settings.grid_points)
    right_grid = np.linspace(0.0, positive, settings.grid_points)[1:]
    x = np.concatenate((left_grid, right_grid))
    if np.any(np.diff(x) < 0.0) or not np.all(np.isfinite(x)):
        return report(
            CAMOutcomeStatus.STOCHASTIC_INVALID,
            "the reconstruction grid is not a finite, non-decreasing interval",
            code=FAILURE_INVALID_DOMAIN,
            updates={
                "x_min": negative,
                "x_max": positive,
                "negative_physical_boundary": negative_physical,
                "positive_physical_boundary": positive_physical,
                "sigma": resolved_sigma,
            },
        )
    vectors = manifold_embedding(resolved_manifold, point, x)
    # One adapter and one model call for the whole grid: the noise covariance and its
    # projection are then the batch primitives of ``core.noise``, which is the order
    # the reference evaluation uses for every grid point at once.
    matrices, diffusions, physical_minimum = noise_stack(
        adapter, model, vectors, parameter_map
    )
    q = projected_noise_intensities(matrices, diffusions, left)
    omega = observable_values(observable, adapter, vectors, parameter_map)
    usable = (
        np.isfinite(q) & (q > settings.minimum_projected_diffusion) & np.isfinite(omega)
    )
    diagnostics["unusable_point_count"] = int(np.count_nonzero(~usable))
    diagnostics["physical_violation_count"] = int(
        np.count_nonzero(physical_minimum < -settings.physical_tolerance)
    )
    domain: dict[str, Any] = {
        "x_min": negative,
        "x_max": positive,
        "negative_physical_boundary": negative_physical,
        "positive_physical_boundary": positive_physical,
        "sigma": resolved_sigma,
        "grid_points": int(x.size),
    }
    if not np.all(usable):
        return report(
            CAMOutcomeStatus.STOCHASTIC_INVALID,
            f"{int(np.count_nonzero(~usable))} grid points have a non-positive or "
            "non-finite projected diffusion or a non-finite readout",
            code=FAILURE_INVALID_DOMAIN,
            updates=domain,
        )
    f = manifold_drift(resolved_manifold, x)
    if not np.all(np.isfinite(f)):
        return report(
            CAMOutcomeStatus.STOCHASTIC_INVALID,
            "the manifold drift is not finite on the reconstruction grid",
            code=FAILURE_INVALID_DOMAIN,
            updates=domain,
        )
    try:
        table = ScalarGeneratorTable(
            x=tuple(float(value) for value in x),
            drift=tuple(float(value) for value in f),
            diffusion=tuple(float(value) for value in q),
            readout=tuple(float(value) for value in omega),
        )
    except ValidationError as exc:
        return report(
            CAMOutcomeStatus.STOCHASTIC_INVALID,
            f"the generator table is not usable: {exc}",
            code=FAILURE_INVALID_DOMAIN,
            updates=domain,
        )
    domain = {**domain, "generator_table": table}
    forcing: np.ndarray | None = None
    if parameter_direction is not None:
        forcing, _ = parameter_forcing(
            adapter, vectors, parameter_map, left, parameter_direction
        )
    # One evaluation owns the density and every fraction derived from it; the
    # verdicts below stay the reconstruction's own.
    evaluation = evaluate_generator(table, forcing=forcing, config=settings)
    diagnostics["density_normalization_raw"] = evaluation.density_normalization
    diagnostics["core_point_count"] = evaluation.core_point_count
    diagnostics["edge_point_count"] = evaluation.edge_point_count
    if not evaluation.normalizable:
        reason = (
            "the stationary density does not normalize (integral "
            f"{evaluation.density_normalization!r})"
        )
        if evaluation.reason is not None:
            reason = f"the generator table cannot be evaluated: {evaluation.reason}"
        return report(
            CAMOutcomeStatus.STOCHASTIC_INVALID,
            reason,
            code=FAILURE_NOT_NORMALIZABLE,
            updates=domain,
        )
    fractions = {
        "scalar_intensity": evaluation.scalar_intensity,
        "stationary_mean": evaluation.stationary_mean,
        "stationary_response": evaluation.stationary_response,
        "path_information_rate": evaluation.path_information_rate,
        "minimum_projected_diffusion": evaluation.minimum_projected_diffusion,
        "endpoint_density_ratio": evaluation.endpoint_density_ratio,
        "edge_mass_fraction": evaluation.edge_mass_fraction,
        "density_normalization": evaluation.density_normalization,
    }
    confining = resolved_manifold.confining
    bounded = bool(negative_physical and positive_physical)
    if not confining:
        validity = "not_confining"
    elif bounded:
        validity = "confining_bounded"
    else:
        validity = "confining_truncated"
    if not confining:
        return report(
            CAMOutcomeStatus.UNCONTROLLED_APPROXIMATION,
            "the truncated normal form is not confining (leading drift "
            f"coefficient {resolved_manifold.drift[resolved_manifold.order]:.3e}"
            " >= 0), so the reconstructed density is not controlled by the "
            "truncation",
            code=FAILURE_NOT_CONFINING,
            updates={"validity_domain": validity, **domain, **fractions},
        )
    if not bounded:
        return report(
            CAMOutcomeStatus.UNCONTROLLED_APPROXIMATION,
            "the physical region is unbounded within the boundary search limit "
            f"({settings.boundary_limit_relative}x the soft-mode width); the density "
            "is truncated at the grid edges",
            code=FAILURE_UNBOUNDED,
            updates={"validity_domain": validity, **domain, **fractions},
        )
    return report(
        CAMOutcomeStatus.VALID,
        None,
        updates={"validity_domain": validity, **domain, **fractions},
    )


def _failed(
    *,
    order: int,
    status: CAMOutcomeStatus,
    code: str,
    reason: str,
    method: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
) -> ScalarReconstruction:
    """Return a failure report that carries no manifold (it never resolved)."""
    placeholder = NonlinearManifold(
        order=order,
        soft_rate=float("nan"),
        critical_eigenvalue_real=float("nan"),
        critical_eigenvalue_imag=float("nan"),
        noncritical_spectral_gap=None,
        right=(),
        left=(),
        embeddings=(),
        drift=(),
        maximum_bordered_condition=float("nan"),
        confining=False,
        method={"service": "qphase_cam.nonlinear.slow_manifold", "resolved": False},
    )
    return ScalarReconstruction(
        status=status,
        reason=reason,
        reason_code=code,
        order=order,
        confining=False,
        manifold=placeholder,
        method=dict(method),
        diagnostics=dict(diagnostics),
    )


class OuIntensityAlgebra(BaseModel):
    """The Gaussian intensities of one readout, with the rate kept out of the guard.

    :func:`ou_intensities` requires a decaying soft mode and raises otherwise; a
    branch-point diagnostic has to report the same algebra on a state that does not
    decay, so the algebra lives here once. ``soft_intensity`` is ``None`` when the
    rate is not positive (it divides by it) and ``sigma`` is ``None`` when the rate or
    the projected noise is not positive; ``full_intensity`` and the projection are
    defined either way. A singular zero-frequency response still raises, because
    there is no algebraic answer to report.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    rate: float
    projection: float
    projected_noise: float
    sigma: float | None = None
    soft_intensity: float | None = None
    full_intensity: float
    zero_frequency_gradient: tuple[float, ...]
    gradient: DerivativeRecord
    method: dict[str, Any] = Field(default_factory=dict)


def ou_intensity_algebra(
    adapter: Any,
    observable: CAMObservable,
    *,
    state: Any,
    params: Mapping[str, Any],
    diffusion: Any | None = None,
    model: Any | None = None,
    mode: CriticalMode | None = None,
    jacobian: Any | None = None,
    overlap_tolerance: float = 1e-10,
) -> OuIntensityAlgebra:
    """Return the soft/full Gaussian intensities of one readout without the guard.

    ``mode``, ``diffusion`` and ``jacobian`` are optional precomputed inputs: a caller
    that already resolved the critical mode or the canonical covariance passes them so
    the eigenproblem and the transform are not repeated, and the returned numbers are
    the ones :func:`ou_intensities` gives for a decaying mode.
    """
    point = np.asarray(state, dtype=float).reshape(-1)
    parameter_map = dict(params or {})
    resolved = mode or _critical(adapter, point, parameter_map, overlap_tolerance)
    if diffusion is None:
        if model is None or not callable(getattr(model, "cam_diffusion", None)):
            raise ValueError(
                "the Gaussian intensities need either the noise covariance or a model"
            )
        _, covariance, _ = _noise_covariance(adapter, model, point, parameter_map)
    else:
        covariance = np.asarray(diffusion, dtype=float)
        if covariance.shape != (point.size, point.size):
            raise ValueError(
                f"the diffusion shape {covariance.shape} does not match the "
                f"{point.size}-coordinate state"
            )
    matrix = (
        np.asarray(jacobian, dtype=float)
        if jacobian is not None
        else np.asarray(adapter.jacobian(point, parameter_map), dtype=float)
    )
    rate = -float(np.real(resolved.value))
    record = observable.gradient(adapter, point, parameter_map)
    gradient = np.asarray(record.gradient, dtype=float)
    right = np.asarray(resolved.right, dtype=float)
    left = np.asarray(resolved.left, dtype=float)
    projection = float(gradient @ right)
    projected_noise = projected_intensity(covariance, left, left)
    try:
        response = np.linalg.solve(-matrix.T, gradient)
    except np.linalg.LinAlgError as exc:
        raise ValueError(f"the zero-frequency response does not exist: {exc}") from exc
    full = float(response @ covariance @ response)
    positive = rate > 0.0 and projected_noise > 0.0
    return OuIntensityAlgebra(
        rate=rate,
        projection=projection,
        projected_noise=projected_noise,
        sigma=(float(np.sqrt(projected_noise / (2.0 * rate))) if positive else None),
        soft_intensity=(
            projection**2 * projected_noise / rate**2 if rate > 0.0 else None
        ),
        full_intensity=full,
        zero_frequency_gradient=tuple(float(value) for value in response),
        gradient=record,
        method={
            "service": "qphase_cam.nonlinear.ou_intensity_algebra",
            "noise_covariance": "canonical diffusion covariance",
            "soft": "projection^2 * projected_noise / rate^2 (rate > 0)",
            "sigma": "sqrt(projected_noise / (2 rate)) (rate > 0 and noise > 0)",
            "full": "v^T D v with v = g^T (-J)^-1",
            "gradient_method": record.method,
        },
    )


def ou_intensities(
    adapter: Any,
    observable: CAMObservable,
    *,
    state: Any,
    params: Mapping[str, Any],
    diffusion: Any | None = None,
    model: Any | None = None,
    overlap_tolerance: float = 1e-10,
) -> OuIntensities:
    """Return the Gaussian soft- and full-mode intensities of one readout.

    ``soft`` is the soft-mode estimate ``(g·r)² (lᵀDl) / λ²`` and ``full`` is the
    zero-frequency response ``vᵀDv`` with ``v = gᵀ(-J)⁻¹``. The noise covariance
    ``D`` is the canonical one (from ``diffusion`` or the model's
    ``cam_diffusion``). Both are *noise intensities*, not state variances: the
    state covariance belongs to the 10F1 assessment.
    """
    point = np.asarray(state, dtype=float).reshape(-1)
    parameter_map = dict(params or {})
    mode = _critical(adapter, point, parameter_map, overlap_tolerance)
    lam = float(np.real(mode.value))
    if lam >= 0.0:
        raise ValueError(f"the soft mode does not decay (real part {lam:.3e})")
    algebra = ou_intensity_algebra(
        adapter,
        observable,
        state=point,
        params=parameter_map,
        diffusion=diffusion,
        model=model,
        mode=mode,
        overlap_tolerance=overlap_tolerance,
    )
    if not np.isfinite(algebra.projected_noise) or algebra.projected_noise <= 0.0:
        raise NoiseIntensityError(
            "the projected noise along the critical mode is not finite and positive "
            f"({algebra.projected_noise!r}); a stable OU intensity needs one"
        )
    if not np.isfinite(algebra.full_intensity) or algebra.full_intensity < 0.0:
        raise NoiseIntensityError(
            "the full-mode noise intensity is not finite and non-negative "
            f"({algebra.full_intensity!r})"
        )
    soft = algebra.soft_intensity
    if soft is None or not np.isfinite(soft) or soft < 0.0:
        raise NoiseIntensityError(
            f"the soft-mode noise intensity is not finite and non-negative ({soft!r})"
        )
    return OuIntensities(
        rate=algebra.rate,
        projection=algebra.projection,
        projected_noise=algebra.projected_noise,
        sigma=float(np.sqrt(algebra.projected_noise / (2.0 * algebra.rate))),
        soft_intensity=float(soft),
        full_intensity=algebra.full_intensity,
        zero_frequency_gradient=algebra.zero_frequency_gradient,
        method={
            "service": "qphase_cam.nonlinear.ou_intensities",
            "noise_covariance": "canonical diffusion covariance",
            "soft": "projection^2 * projected_noise / rate^2",
            "full": "v^T D v with v = g^T (-J)^-1",
            "gradient_method": algebra.gradient.method,
        },
    )


def reconstructed_intensity(
    scalar_intensity: float,
    *,
    full_intensity: float,
    soft_intensity: float,
) -> float:
    """Return ``scalar + full_ou − soft_ou``, the standard correction.

    The combination lives here, outside the numerical core, so no kernel decides on
    the caller's behalf which Gaussian term the reconstruction replaces.
    """
    for name, value in (
        ("scalar_intensity", scalar_intensity),
        ("full_intensity", full_intensity),
        ("soft_intensity", soft_intensity),
    ):
        if not np.isfinite(float(value)):
            raise ValueError(f"{name} must be finite; got {value!r}")
    return float(scalar_intensity) + float(full_intensity) - float(soft_intensity)


class ParameterResponse(BaseModel):
    """The explicit and static parameter response of one readout, unguarded.

    :func:`response_prediction` requires a decaying soft mode; a branch-point
    diagnostic has to report the same algebra on a state that does not decay, so the
    compute lives here once and both call it. ``derivative`` is the explicit
    central-difference record, with its step and its half-step repeat check — that
    check is never dropped. ``state_response`` is ``None`` when the static response
    does not exist.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    drive: tuple[float, ...]
    state_response: tuple[float, ...] | None
    explicit_response: float
    full_response: float | None
    gradient: DerivativeRecord
    derivative: DerivativeRecord
    method: dict[str, Any] = Field(default_factory=dict)


def parameter_response(
    adapter: Any,
    observable: CAMObservable,
    *,
    state: Any,
    params: Mapping[str, Any],
    parameter: str,
    scale: float = 1.0,
    relative_step: float = 2.0e-6,
    jacobian: Any | None = None,
    gradient: DerivativeRecord | None = None,
) -> ParameterResponse:
    """Return the explicit and static parameter response of one readout.

    The explicit term is the central difference of the readout at a fixed state,
    recorded as a :class:`~qphase_cam.observable.DerivativeRecord`; the static term is
    ``gradient . (-J)^-1 drive``; ``full_response`` is their sum and is ``None`` when
    the static response does not exist. The soft-mode term is *not* here: it divides
    by the rate and belongs to the caller that owns the mode.

    ``gradient`` is an optional precomputed record: a caller that already asked the
    observable for its gradient passes it, so one readout is differentiated once, and
    the record is checked against the state's coordinate count before it is used.
    """
    point = np.asarray(state, dtype=float).reshape(-1)
    parameter_map = dict(params or {})
    if not parameter:
        raise ValueError("a response prediction needs the parameter name")
    if not np.isfinite(relative_step) or relative_step <= 0.0:
        raise ValueError(
            f"relative_step must be finite and positive; got {relative_step}"
        )
    if parameter not in parameter_map:
        raise ValueError(f"the parameter {parameter!r} has no value in params")
    direction = np.asarray(
        adapter.parameter_direction(parameter, float(scale)), dtype=float
    ).reshape(-1)
    matrix = (
        np.asarray(jacobian, dtype=float)
        if jacobian is not None
        else np.asarray(adapter.jacobian(point, parameter_map), dtype=float)
    )
    drive = (
        np.asarray(adapter.parameter_jacobian(point, parameter_map), dtype=float)
        @ direction
    )
    record = (
        gradient
        if gradient is not None
        else observable.gradient(adapter, point, parameter_map)
    )
    if not isinstance(record, DerivativeRecord):
        raise TypeError(
            "a precomputed gradient must be a DerivativeRecord; got "
            f"{type(record).__name__}"
        )
    values = np.asarray(record.gradient, dtype=float)
    if values.size != point.size:
        raise ValueError(
            f"the gradient has {values.size} entries for a "
            f"{point.size}-coordinate state"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("the readout gradient is not finite")
    parameter_value = float(parameter_map[parameter])
    step = relative_step * max(1.0, abs(parameter_value))

    def value_at(offset: float) -> float:
        shifted = dict(parameter_map)
        shifted[parameter] = parameter_value + offset
        return float(observable.value(adapter, point, shifted))

    def explicit_at(width: float) -> float:
        return (value_at(width) - value_at(-width)) / (2.0 * width)

    explicit = explicit_at(step)
    if not np.isfinite(explicit):
        raise ValueError(
            f"the explicit parameter response is not finite ({explicit!r})"
        )
    refined = explicit_at(step / 2.0)
    denominator = abs(explicit)
    repeat_error = (
        float(abs(explicit - refined) / denominator)
        if denominator > 0.0 and np.isfinite(explicit)
        else None
    )
    derivative = DerivativeRecord(
        method="central_difference",
        order=2,
        gradient=(float(explicit),),
        step=(float(step),),
        scale=(float(max(1.0, abs(parameter_value))),),
        condition=None,
        repeat_relative_error=repeat_error,
        metadata={
            "quantity": "parameter_direction_response",
            "parameter": parameter,
            "relative_step": float(relative_step),
            "repeat_check": "recomputed at half the step",
        },
    )
    try:
        state_response = np.linalg.solve(-matrix, drive)
    except np.linalg.LinAlgError:
        state_response = None
    full = (
        None
        if state_response is None
        else float(explicit) + float(values @ state_response)
    )
    if full is not None and not np.isfinite(full):
        raise ValueError(f"the full parameter response is not finite ({full!r})")
    return ParameterResponse(
        drive=tuple(float(value) for value in drive),
        state_response=(
            None
            if state_response is None
            else tuple(float(value) for value in state_response)
        ),
        explicit_response=float(explicit),
        full_response=full,
        gradient=record,
        derivative=derivative,
        method={
            "service": "qphase_cam.nonlinear.parameter_response",
            "explicit": "central difference of the readout at a fixed state",
            "static": "gradient . (-J)^-1 drive",
            "full": "explicit + static",
            "parameter": parameter,
            "scale": float(scale),
            "relative_step": float(relative_step),
            "repeat_check": "recomputed at half the step",
        },
    )


class ResponsePrediction(BaseModel):
    """The linear and explicit frequency response of one readout to one control.

    ``soft_response`` is the critical-mode-only estimate, ``explicit_response`` the
    direct derivative of the readout at fixed state, ``full_response`` that
    derivative plus the static state displacement, and ``regular_response`` the
    full-minus-soft remainder. The explicit derivative carries its own record, so
    the prediction can say how it was taken.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    rate: float
    projection: float
    projected_drive: float
    drive: tuple[float, ...]
    soft_response: float
    explicit_response: float
    state_response: tuple[float, ...]
    full_response: float
    regular_response: float
    derivative: DerivativeRecord
    method: dict[str, Any] = Field(default_factory=dict)


def response_prediction(
    adapter: Any,
    observable: CAMObservable,
    *,
    state: Any,
    params: Mapping[str, Any],
    parameter: str,
    scale: float = 1.0,
    relative_step: float = 2.0e-6,
    overlap_tolerance: float = 1e-10,
) -> ResponsePrediction:
    """Return the soft, explicit and full parameter response of one readout.

    The explicit term is a central difference of the readout in ``parameter`` at a
    fixed state, recorded as a :class:`~qphase_cam.observable.DerivativeRecord`; the
    soft and full terms follow the reference study
    (``projection * projected_drive / rate`` and
    ``explicit + gradient . (-J)^-1 drive``).
    """
    point = np.asarray(state, dtype=float).reshape(-1)
    parameter_map = dict(params or {})
    if not parameter:
        raise ValueError("a response prediction needs the parameter name")
    if not np.isfinite(relative_step) or relative_step <= 0.0:
        raise ValueError(
            f"relative_step must be finite and positive; got {relative_step}"
        )
    mode = _critical(adapter, point, parameter_map, overlap_tolerance)
    rate = -float(np.real(mode.value))
    if rate <= 0.0:
        raise ValueError(f"the soft mode does not decay (rate {rate:.3e})")
    if parameter not in parameter_map:
        raise ValueError(f"the parameter {parameter!r} has no value in params")
    jacobian = np.asarray(adapter.jacobian(point, parameter_map), dtype=float)
    response = parameter_response(
        adapter,
        observable,
        state=point,
        params=parameter_map,
        parameter=parameter,
        scale=scale,
        relative_step=relative_step,
        jacobian=jacobian,
    )
    if response.state_response is None or response.full_response is None:
        raise ValueError("the static state response does not exist")
    gradient = np.asarray(response.gradient.gradient, dtype=float)
    drive = np.asarray(response.drive, dtype=float)
    projected_drive = float(np.asarray(mode.left, dtype=float) @ drive)
    projection = float(gradient @ np.asarray(mode.right, dtype=float))
    soft = projection * projected_drive / rate
    full = float(response.full_response)
    return ResponsePrediction(
        rate=rate,
        projection=projection,
        projected_drive=projected_drive,
        drive=response.drive,
        soft_response=float(soft),
        explicit_response=response.explicit_response,
        state_response=response.state_response,
        full_response=full,
        regular_response=float(full - soft),
        derivative=response.derivative,
        method={
            "service": "qphase_cam.nonlinear.response_prediction",
            "soft": "projection * projected_drive / rate",
            "full": "explicit parameter derivative + gradient . (-J)^-1 drive",
            "regular": "full - soft",
            "parameter": parameter,
            "scale": float(scale),
        },
    )
