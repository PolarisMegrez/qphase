"""qphase_cam: the algebraic carrier pole of the scalar backward generator
---------------------------------------------------------
The scalar reconstruction gives ``f``, ``q`` and the readout ``omega`` along the slow
manifold. The frequency at which the readout carrier decays is a pole of the
**backward** (Feynman-Kac) generator ``L = f(x) ∂_x + ½ q(x) ∂_x²`` tilted by the
readout frequency. It is not the closed-Hamiltonian pole of
:mod:`qphase_cam.coherence_pole`, not a time-domain PDE solution and not a measured
SDE spectrum.

The numbers are the ones the retired diagnostic probe's ``_generator_pole`` computed,
without that script's files, model or fixed grid (the probe is gone; its published
``poles.csv`` is reproduced through this service by
``reports/phase10_verification/carrier_pole_public_check.py``):

* every grid size resamples ``f``, ``q`` and ``omega`` from the source table with a
  monotone :class:`scipy.interpolate.PchipInterpolator` onto a uniform grid on the
  source interval, so a non-uniform source grid is fine;
* the generator is the upwind finite-volume stencil ``diffusion_rate = q / (2 dx²)``,
  ``lower = diffusion_rate + max(-f, 0) / dx``,
  ``upper = diffusion_rate + max(f, 0) / dx``, reflecting ends
  (``lower[0] = upper[-1] = 0``), ``diagonal = -(lower + upper)`` and
  ``diags((lower[1:], diagonal, upper[:-1]), (-1, 0, 1))``;
* the frequency operator ``diag(omega - median(omega))`` is switched on by a
  continuation coupling ``1/steps … 1`` and the pole follows the eigenvalue nearest to
  the previous one (shift-invert ``LM``). The shift is the previous pole once the
  tracked pole is resolvable, and ``-shift_magnitude`` otherwise, with
  ``shift_magnitude = max(shift_floor, eps · max|diagonal| · size)`` and
  ``eps = np.finfo(float).eps``. The reference's fixed ``-1e-12`` start is below the
  rounding level of ``max|diagonal|`` on a fine grid (``q / (2 dx²)`` grows like
  ``size²``), and a shift the diagonal cannot represent leaves the shifted matrix
  exactly singular — which is what a constant readout hits, because its discrete
  generator is the mass-conservation operator with an exact zero row sum. This changes
  only the shift-invert path: the discretization, the continuation coupling and the
  "eigenvalue nearest the previous pole" rule are the reference ones. ``shift_floor``
  is a floor on that numerical shift, never a decay tolerance — ``decay_tolerance``
  is the separate, physical one;
* ``carrier_frequency = omega_reference - Im(pole)`` and
  ``carrier_decay_rate = -Re(pole)``.

Each grid is solved independently and the reported pole is the **finest** grid's; the
coarser grids are the convergence witness. ``grid_converged`` compares the last two
grids on both the carrier frequency and the decay rate with an absolute plus relative
tolerance. That is a two-grid comparison, not a convergence-order proof.

Failures are statuses and never a clamped or NaN pole: an unusable source table, a
resampling that makes the diffusion non-positive or the grid spacing, diffusion rate
or generator scale lose its finite positive value, an eigensolver failure
(``ArpackNoConvergence`` and ``ArpackError`` are ``RuntimeError`` subclasses, and a
singular shift-invert factor raises a plain ``RuntimeError``) or a non-finite
eigenvalue, a non-finite pole, a negative decay rate beyond the configured tolerance
and a two-grid disagreement are all reported, with the grids that did solve kept for
diagnosis.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)
from scipy.interpolate import PchipInterpolator
from scipy.sparse import diags
from scipy.sparse.linalg import eigs

from .nonlinear import ScalarGeneratorTable
from .outcomes import CAMOutcomeStatus

__all__ = [
    "CarrierPoleAssessment",
    "CarrierPoleConfig",
    "CarrierPoleGrid",
    "carrier_pole",
]

#: Failure codes an assessment reports alongside ``reason``.
FAILURE_SOURCE_TABLE = "invalid_source_table"
FAILURE_RESAMPLED_GRID = "resampled_grid_invalid"
FAILURE_EIGENSOLVER = "eigensolver_failed"
FAILURE_POLE_NOT_FINITE = "pole_not_finite"
FAILURE_NEGATIVE_DECAY = "negative_decay_rate"
FAILURE_GRID = "grid_not_converged"


class CarrierPoleConfig(BaseModel):
    """Grids, continuation and eigensolver controls of one carrier-pole solve.

    The defaults are the reference probe's: three grid sizes, 20 continuation steps,
    six requested eigenvalues, shift-invert tolerance ``1e-10`` and ``maxiter``
    ``200000``. ``k`` is clipped to ``n - 2`` per grid, so a small test grid works.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    grid_sizes: tuple[int, ...] = Field(
        default=(401, 801, 1201),
        min_length=2,
        description="Uniform grid sizes, solved independently in increasing order",
    )
    continuation_steps: int = Field(
        default=20, ge=1, description="Steps of the frequency-operator continuation"
    )
    k: int = Field(
        default=6,
        ge=1,
        description="Requested eigenvalue count; clipped to n - 2 as ARPACK requires",
    )
    eigensolver_tolerance: float = Field(
        default=1e-10, gt=0.0, allow_inf_nan=False, description="ARPACK tolerance"
    )
    eigensolver_maxiter: int = Field(
        default=200_000, ge=1, description="ARPACK iteration limit"
    )
    grid_absolute_tolerance: float = Field(
        default=1e-3,
        ge=0.0,
        allow_inf_nan=False,
        description="Absolute tolerance of the two-finest-grid comparison",
    )
    grid_relative_tolerance: float = Field(
        default=1e-3,
        ge=0.0,
        allow_inf_nan=False,
        description="Relative tolerance of the two-finest-grid comparison",
    )
    decay_tolerance: float = Field(
        default=1e-9,
        ge=0.0,
        allow_inf_nan=False,
        description="Negative decay tolerated before the pole counts as growing",
    )
    shift_floor: float = Field(
        default=1e-12,
        gt=0.0,
        allow_inf_nan=False,
        description=(
            "Absolute floor of the shift-invert shift, which is "
            "max(shift_floor, eps * max|diagonal| * size); a numerical floor, not a "
            "decay tolerance (decay_tolerance is separate)"
        ),
    )

    @field_validator("grid_sizes")
    @classmethod
    def _check_grid_sizes(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        for size in value:
            if size < 3:
                raise ValueError(f"every grid size must be at least 3; got {size}")
        for coarse, fine in zip(value, value[1:], strict=False):
            if fine <= coarse:
                raise ValueError(
                    f"the grid sizes must be strictly increasing; got {list(value)}"
                )
        return value

    @model_validator(mode="after")
    def _check_tolerances(self) -> CarrierPoleConfig:
        if self.grid_absolute_tolerance == 0.0 and self.grid_relative_tolerance == 0.0:
            raise ValueError(
                "the grid absolute and relative tolerances cannot both be zero, "
                "which would only accept bit-identical grids"
            )
        return self


class CarrierPoleGrid(BaseModel):
    """The carrier pole of one uniform grid."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    grid_size: int
    carrier_frequency: float
    carrier_decay_rate: float
    pole_real: float
    pole_imag_shifted: float
    omega_reference: float
    continuation_steps: int


class CarrierPoleAssessment(BaseModel):
    """The carrier pole of one source table, with the per-grid witness."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: CAMOutcomeStatus
    reason: str | None = None
    reason_code: str | None = None

    grids: tuple[CarrierPoleGrid, ...] = ()
    finest_grid_size: int | None = None
    carrier_frequency: float | None = None
    carrier_decay_rate: float | None = None
    grid_frequency_differences: tuple[float, ...] = ()
    grid_decay_differences: tuple[float, ...] = ()
    grid_converged: bool | None = None
    method: dict[str, Any] = Field(default_factory=dict)


class _PoleFailureError(Exception):
    """One grid that could not produce a pole, as a status and a reason."""

    def __init__(self, status: CAMOutcomeStatus, code: str, reason: str) -> None:
        super().__init__(reason)
        self.status = status
        self.code = code
        self.reason = reason


def _grid_pole(
    table: ScalarGeneratorTable, size: int, settings: CarrierPoleConfig
) -> CarrierPoleGrid:
    """Solve one uniform grid and return its pole record."""
    source = np.asarray(table.x, dtype=float)
    x = np.linspace(float(source[0]), float(source[-1]), size)
    f = np.asarray(
        PchipInterpolator(source, np.asarray(table.drift, dtype=float))(x), dtype=float
    )
    q = np.asarray(
        PchipInterpolator(source, np.asarray(table.diffusion, dtype=float))(x),
        dtype=float,
    )
    omega = np.asarray(
        PchipInterpolator(source, np.asarray(table.readout, dtype=float))(x),
        dtype=float,
    )
    if not (np.all(np.isfinite(f)) and np.all(np.isfinite(q))):
        raise _PoleFailureError(
            CAMOutcomeStatus.STOCHASTIC_INVALID,
            FAILURE_RESAMPLED_GRID,
            f"the PCHIP resampling onto {size} points is not finite",
        )
    if not np.all(np.isfinite(omega)):
        raise _PoleFailureError(
            CAMOutcomeStatus.STOCHASTIC_INVALID,
            FAILURE_RESAMPLED_GRID,
            f"the PCHIP resampling of the readout onto {size} points is not finite",
        )
    if np.any(q <= 0.0):
        raise _PoleFailureError(
            CAMOutcomeStatus.STOCHASTIC_INVALID,
            FAILURE_RESAMPLED_GRID,
            f"the PCHIP resampling onto {size} points makes the diffusion "
            f"non-positive (minimum {float(np.min(q))!r})",
        )
    dx = float(x[1] - x[0])
    if not np.isfinite(dx) or dx <= 0.0:
        raise _PoleFailureError(
            CAMOutcomeStatus.STOCHASTIC_INVALID,
            FAILURE_RESAMPLED_GRID,
            f"the uniform {size}-point grid has a non-finite or non-positive spacing "
            f"({dx!r})",
        )
    with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
        diffusion_rate = q / (2.0 * np.float64(dx) ** 2)
    if not np.all(np.isfinite(diffusion_rate)) or np.any(diffusion_rate <= 0.0):
        # An underflowed or overflowed rate would make the generator silently
        # degenerate (an all-zero row, or an infinite diffusion), and a zero matrix
        # would still return an eigenvalue; that is not a pole.
        raise _PoleFailureError(
            CAMOutcomeStatus.STOCHASTIC_INVALID,
            FAILURE_RESAMPLED_GRID,
            f"the diffusion rate q / (2 dx^2) of the {size}-point grid is not finite "
            f"and positive ({float(np.min(diffusion_rate))!r} … "
            f"{float(np.max(diffusion_rate))!r} for dx {dx!r})",
        )
    lower = diffusion_rate + np.maximum(-f, 0.0) / dx
    upper = diffusion_rate + np.maximum(f, 0.0) / dx
    lower[0] = 0.0
    upper[-1] = 0.0
    diagonal = -(lower + upper)
    generator = diags(
        (lower[1:], diagonal, upper[:-1]), offsets=(-1, 0, 1), format="csc"
    )
    omega_reference = float(np.median(omega))
    frequency_operator = diags(omega - omega_reference, offsets=0, format="csc")
    if not (
        np.all(np.isfinite(generator.data))
        and np.all(np.isfinite(frequency_operator.data))
    ):
        raise _PoleFailureError(
            CAMOutcomeStatus.STOCHASTIC_INVALID,
            FAILURE_RESAMPLED_GRID,
            f"the generator of the {size}-point grid is not finite",
        )
    scale = float(np.max(np.abs(diagonal)))
    if not np.isfinite(scale) or scale <= 0.0:
        raise _PoleFailureError(
            CAMOutcomeStatus.STOCHASTIC_INVALID,
            FAILURE_RESAMPLED_GRID,
            f"the generator scale max|diagonal| of the {size}-point grid is not "
            f"finite and positive ({scale!r})",
        )
    # The reference starts the continuation at -1e-12. On a fine grid that absolute
    # shift is below the rounding level of the diagonal (q / (2 dx^2) grows like
    # size^2), and a shift the diagonal cannot represent leaves the matrix exactly
    # singular. The shift is therefore scale aware and floored at ``shift_floor``.
    shift_magnitude = max(
        float(settings.shift_floor), float(np.finfo(float).eps) * scale * size
    )
    count = min(int(settings.k), size - 2)
    pole = 0.0j
    for coupling in np.linspace(
        1.0 / settings.continuation_steps, 1.0, settings.continuation_steps
    ):
        tilted = generator.astype(complex) - 1j * coupling * frequency_operator
        sigma = pole if abs(pole) > shift_magnitude else -shift_magnitude
        try:
            values = eigs(
                tilted,
                k=count,
                sigma=sigma,
                which="LM",
                return_eigenvectors=False,
                tol=settings.eigensolver_tolerance,
                maxiter=settings.eigensolver_maxiter,
            )
        except (RuntimeError, np.linalg.LinAlgError) as exc:
            # ``ArpackNoConvergence`` and ``ArpackError`` are ``RuntimeError``
            # subclasses, and a failed shift-invert factor raises a plain
            # ``RuntimeError`` ("Factor is exactly singular"); all of them are
            # numeric eigensolver failures, not configuration errors.
            raise _PoleFailureError(
                CAMOutcomeStatus.NOT_CONVERGED,
                FAILURE_EIGENSOLVER,
                f"the ARPACK eigensolver failed on the {size}-point grid "
                f"(coupling {float(coupling):.3e}): {exc}",
            ) from exc
        spectrum = np.asarray(values, dtype=complex)
        if not np.all(np.isfinite(spectrum)):
            raise _PoleFailureError(
                CAMOutcomeStatus.NOT_CONVERGED,
                FAILURE_EIGENSOLVER,
                f"the ARPACK eigensolver returned a non-finite eigenvalue on the "
                f"{size}-point grid",
            )
        pole = complex(spectrum[int(np.argmin(np.abs(spectrum - pole)))])
    if not (np.isfinite(pole.real) and np.isfinite(pole.imag)):
        raise _PoleFailureError(
            CAMOutcomeStatus.NOT_CONVERGED,
            FAILURE_POLE_NOT_FINITE,
            f"the pole of the {size}-point grid is not finite ({pole!r})",
        )
    return CarrierPoleGrid(
        grid_size=size,
        carrier_frequency=float(omega_reference - pole.imag),
        carrier_decay_rate=float(-pole.real),
        pole_real=float(pole.real),
        pole_imag_shifted=float(pole.imag),
        omega_reference=omega_reference,
        continuation_steps=int(settings.continuation_steps),
    )


def _within(
    fine: CarrierPoleGrid, coarse: CarrierPoleGrid, settings: CarrierPoleConfig
) -> bool:
    """Return whether the last two grids agree on the frequency and the decay."""

    def close(first: float, second: float) -> bool:
        return abs(first - second) <= (
            settings.grid_absolute_tolerance
            + settings.grid_relative_tolerance * max(abs(first), abs(second))
        )

    return close(fine.carrier_frequency, coarse.carrier_frequency) and close(
        fine.carrier_decay_rate, coarse.carrier_decay_rate
    )


def _assessment(
    status: CAMOutcomeStatus,
    code: str | None,
    reason: str | None,
    *,
    grids: list[CarrierPoleGrid],
    method: dict[str, Any],
    converged: bool | None = None,
    finest: CarrierPoleGrid | None = None,
) -> CarrierPoleAssessment:
    records = tuple(grids)
    return CarrierPoleAssessment(
        status=status,
        reason=reason,
        reason_code=code,
        grids=records,
        finest_grid_size=None if finest is None else finest.grid_size,
        carrier_frequency=None if finest is None else finest.carrier_frequency,
        carrier_decay_rate=None if finest is None else finest.carrier_decay_rate,
        grid_frequency_differences=tuple(
            abs(fine.carrier_frequency - coarse.carrier_frequency)
            for coarse, fine in zip(records, records[1:], strict=False)
        ),
        grid_decay_differences=tuple(
            abs(fine.carrier_decay_rate - coarse.carrier_decay_rate)
            for coarse, fine in zip(records, records[1:], strict=False)
        ),
        grid_converged=converged,
        method=dict(method),
    )


def _messages(error: ValidationError) -> str:
    return "; ".join(str(item["msg"]) for item in error.errors())


def carrier_pole(
    x: Any,
    drift: Any,
    diffusion: Any,
    readout: Any,
    *,
    config: CarrierPoleConfig | None = None,
) -> CarrierPoleAssessment:
    """Return the algebraic carrier pole of one scalar generator table.

    ``x``, ``drift``, ``diffusion`` and ``readout`` are the table coefficients of
    :class:`~qphase_cam.nonlinear.ScalarGeneratorTable`; they are validated as such,
    so a shape or monotonicity error is a status instead of an ARPACK crash. No
    model, adapter or SDE sample is needed: this is a deterministic eigenvalue
    problem on the discretized backward generator.

    The discretization, the continuation coupling and the nearest-eigenvalue rule are
    the reference probe's. Only the shift-invert shift is scale aware (see the module
    docstring), and that is a numerical choice: it changes nothing about which pole is
    tracked, and it is unrelated to ``decay_tolerance``, the physical tolerance on a
    negative decay rate.
    """
    settings = config or CarrierPoleConfig()
    method: dict[str, Any] = {
        "service": "qphase_cam.carrier_pole.carrier_pole",
        "discretization": (
            "backward Feynman-Kac generator on a uniform grid: diffusion rate "
            "q/(2 dx^2), upwind drift max(-f, 0)/dx below and max(f, 0)/dx above, "
            "reflecting ends, tridiagonal"
        ),
        "interpolation": (
            "PCHIP of drift, diffusion and readout from the source table onto each "
            "uniform grid (shape preserving, no overshoot)"
        ),
        "frequency": (
            "diag(omega - median(omega)) switched on by a continuation coupling "
            "1/steps … 1"
        ),
        "pole_selection": (
            "shift-invert ARPACK eigenvalues around the previous pole, the nearest "
            "eigenvalue kept (which=LM); the shift is the previous pole, or "
            "-max(shift_floor, eps*max|diagonal|*size) when the tracked pole is not "
            "resolvable, so the shift never falls below the diagonal's rounding level"
        ),
        "carrier": (
            "carrier_frequency = omega_reference - Im(pole), "
            "carrier_decay_rate = -Re(pole)"
        ),
        "grid_convergence": (
            "the two finest grids agree on the carrier frequency and the decay rate "
            "within an absolute plus relative tolerance; a two-grid comparison, not "
            "a convergence-order proof"
        ),
        "scope": (
            "algebraic pole of the discretized backward generator; not a time-domain "
            "PDE solve, not an SDE frequency measurement and not the closed "
            "Hamiltonian pole of coherence_pole"
        ),
    }
    values: list[np.ndarray] = []
    for name, item in (
        ("x", x),
        ("drift", drift),
        ("diffusion", diffusion),
        ("readout", readout),
    ):
        try:
            values.append(np.asarray(item, dtype=float).reshape(-1))
        except (TypeError, ValueError) as exc:
            return _assessment(
                CAMOutcomeStatus.STOCHASTIC_INVALID,
                FAILURE_SOURCE_TABLE,
                f"the {name} values are not a numeric array: {exc}",
                grids=[],
                method=method,
            )
    try:
        table = ScalarGeneratorTable(
            x=tuple(float(value) for value in values[0]),
            drift=tuple(float(value) for value in values[1]),
            diffusion=tuple(float(value) for value in values[2]),
            readout=tuple(float(value) for value in values[3]),
        )
    except ValidationError as exc:
        return _assessment(
            CAMOutcomeStatus.STOCHASTIC_INVALID,
            FAILURE_SOURCE_TABLE,
            f"the source table is not a usable scalar generator: {_messages(exc)}",
            grids=[],
            method=method,
        )

    grids: list[CarrierPoleGrid] = []
    for size in settings.grid_sizes:
        try:
            grids.append(_grid_pole(table, int(size), settings))
        except _PoleFailureError as failure:
            return _assessment(
                failure.status,
                failure.code,
                failure.reason,
                grids=grids,
                method=method,
            )
    finest = grids[-1]
    converged = _within(finest, grids[-2], settings)
    if finest.carrier_decay_rate < -settings.decay_tolerance:
        return _assessment(
            CAMOutcomeStatus.UNSTABLE,
            FAILURE_NEGATIVE_DECAY,
            "the carrier decay rate is negative "
            f"({finest.carrier_decay_rate!r} < -{settings.decay_tolerance!r}), so the "
            "carrier grows instead of decaying",
            grids=grids,
            method=method,
            converged=converged,
            finest=finest,
        )
    if not converged:
        return _assessment(
            CAMOutcomeStatus.NOT_CONVERGED,
            FAILURE_GRID,
            "the two finest grids disagree: the carrier frequency differs by "
            f"{abs(finest.carrier_frequency - grids[-2].carrier_frequency)!r} and "
            f"the decay rate by "
            f"{abs(finest.carrier_decay_rate - grids[-2].carrier_decay_rate)!r}, "
            f"outside {settings.grid_absolute_tolerance!r} absolute plus "
            f"{settings.grid_relative_tolerance!r} relative tolerance",
            grids=grids,
            method=method,
            converged=False,
            finest=finest,
        )
    return _assessment(
        CAMOutcomeStatus.VALID,
        None,
        None,
        grids=grids,
        method=method,
        converged=True,
        finest=finest,
    )
