"""10G2a probe 3a: the algebraic carrier pole of the scalar backward generator.

The reference algorithm is the one of
``scripts/tools/probe_kerr_full_3mode_stochastic_carrier_pole.py::_generator_pole``
(not imported here): PCHIP resampling of ``f``, ``q`` and ``omega`` onto a uniform
grid, the upwind backward generator with reflecting ends, the frequency operator
switched on by a continuation coupling, and the eigenvalue nearest to the previous
pole.

The discretization is exactly singular when the source coefficients are constant
(every row sums to zero, which is the discrete mass-conservation mode), so the
shift-invert shift has to stay above the rounding level of ``q / (2 dx²)``; the
constant source is therefore exercised on the reference grids and not only on small
ones, since a large grid is exactly where the reference's fixed ``-1e-12`` shift
becomes unrepresentable.
"""

from __future__ import annotations

import importlib.metadata
import json
import tomllib
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError
from qphase.core.registry import discovery, registry
from qphase_cam.carrier_pole import (
    CarrierPoleConfig,
    carrier_pole,
)
from qphase_cam.core.coordinates import vector_to_matrix
from qphase_cam.data import (
    CAM_ANNOTATION_DECLARATION,
    build_candidate_product,
    build_state_product,
)
from qphase_cam.nonlinear import (
    ScalarReconstructionConfig,
    scalar_reconstruction,
    slow_manifold,
)
from qphase_cam.observable import (
    ChargeSector,
    FunctionalObservable,
    ObservableIdentity,
    ObservableKind,
)
from qphase_cam.outcomes import CAMOutcomeStatus
from qphase_cam.postprocessor import ScalarCarrierPole, ScalarCarrierPoleConfig
from qphase_cam.postprocessor.base import CAMAnnotationContext, CAMPostprocessor
from scipy.interpolate import PchipInterpolator
from scipy.sparse import diags
from scipy.sparse.linalg import ArpackNoConvergence, eigs

REAL_PCHIP = PchipInterpolator


def _constant_source(size: int = 8):
    """Uneven source grid with constant drift, diffusion and readout."""
    x = np.asarray([-1.0, -0.73, -0.41, -0.12, 0.0, 0.28, 0.64, 1.0])
    return (
        x,
        np.full(size, 0.5),
        np.ones(size),
        np.full(size, 2.0),
    )


def _curved_source():
    """Uneven source grid with a restoring drift, a curved diffusion and a tilt."""
    x = np.asarray([-2.0, -1.2, -0.6, -0.2, 0.0, 0.3, 0.9, 1.5, 2.0])
    return x, -1.5 * x, 1.0 + 0.4 * x**2, 1.0 + 0.5 * x


def _reference_pole(x_source, size, steps):
    """Compute the pole with the reference discretization, written out here.

    The shift is kept literally as the reference probe writes it (the previous pole,
    ``-1e-12`` at the start). A resolvable pole is far above either shift, so the
    parity test also shows that the service's scale-aware shift does not move it.
    """
    x, f_source, q_source, omega_source = x_source
    grid = np.linspace(float(x[0]), float(x[-1]), size)
    f = PchipInterpolator(x, f_source)(grid)
    q = PchipInterpolator(x, q_source)(grid)
    omega = PchipInterpolator(x, omega_source)(grid)
    dx = float(grid[1] - grid[0])
    rate = q / (2.0 * dx**2)
    lower = rate + np.maximum(-f, 0.0) / dx
    upper = rate + np.maximum(f, 0.0) / dx
    lower[0] = 0.0
    upper[-1] = 0.0
    generator = diags(
        (lower[1:], -(lower + upper), upper[:-1]), offsets=(-1, 0, 1), format="csc"
    )
    reference = float(np.median(omega))
    tilt = diags(omega - reference, offsets=0, format="csc")
    pole = 0.0j
    for coupling in np.linspace(1.0 / steps, 1.0, steps):
        tilted = generator.astype(complex) - 1j * coupling * tilt
        sigma = pole if pole != 0.0j else -1.0e-12
        values = eigs(
            tilted,
            k=min(6, size - 2),
            sigma=sigma,
            which="LM",
            return_eigenvectors=False,
            tol=1.0e-10,
            maxiter=200_000,
        )
        pole = values[int(np.argmin(np.abs(values - pole)))]
    return reference - pole.imag, -pole.real


# --- the reference algorithm ----------------------------------------------


def test_a_constant_source_has_the_mass_conservation_pole():
    assessment = carrier_pole(
        *_constant_source(), config=CarrierPoleConfig(grid_sizes=(9, 17, 33))
    )
    assert assessment.status is CAMOutcomeStatus.VALID
    assert assessment.reason is None
    assert assessment.grid_converged is True
    assert assessment.finest_grid_size == 33
    # A constant readout zeroes the frequency operator, so the continuation tracks
    # the mass-conservation mode at exactly zero decay.
    assert assessment.carrier_frequency == pytest.approx(2.0, abs=1e-9)
    assert assessment.carrier_decay_rate == pytest.approx(0.0, abs=1e-9)
    for grid in assessment.grids:
        assert grid.carrier_frequency == pytest.approx(2.0, abs=1e-9)
        assert grid.carrier_decay_rate == pytest.approx(0.0, abs=1e-9)
        assert grid.omega_reference == pytest.approx(2.0, rel=0.0, abs=0.0)
        assert grid.continuation_steps == 20
    assert [grid.grid_size for grid in assessment.grids] == [9, 17, 33]
    assert len(assessment.grid_frequency_differences) == 2
    assert len(assessment.grid_decay_differences) == 2
    assert max(assessment.grid_decay_differences) < 1e-9
    assert "backward Feynman-Kac generator" in assessment.method["discretization"]
    assert "PCHIP" in assessment.method["interpolation"]


@pytest.mark.parametrize(
    ("drift", "diffusion", "readout"),
    [
        (0.0, 1.0, 2.0),
        (0.5, 1.0, 2.0),
        (-0.5, 1.0, 2.0),
        (0.0, 2.5, -3.0),
    ],
)
def test_the_constant_source_converges_on_the_reference_grids(
    drift, diffusion, readout
):
    """The large-grid regression for the constant (exactly singular) source.

    The tolerance is the shift-invert accuracy of the tracked zero mode: ARPACK's
    ``tol`` is relative, so the absolute residual is bounded by
    ``tol * max|diagonal|`` (about ``1e-10 * 2 q / (2 dx^2)``, i.e. ~4e-5 on the
    finest grid), while the observed decay is ~2e-11. ``1e-8`` sits about three
    orders inside that bound and about three above the observation. The carrier
    frequency is not affected in the same way (``Im(pole)`` is ~0 when the frequency
    operator is zeroed), so it is held to ``1e-9``.
    """
    x = np.linspace(-1.0, 1.0, 9)
    assessment = carrier_pole(
        x,
        np.full(x.size, drift),
        np.full(x.size, diffusion),
        np.full(x.size, readout),
    )
    assert assessment.status is CAMOutcomeStatus.VALID
    assert assessment.reason_code is None
    assert assessment.grid_converged is True
    assert [grid.grid_size for grid in assessment.grids] == [401, 801, 1201]
    assert assessment.finest_grid_size == 1201
    assert assessment.carrier_frequency == pytest.approx(readout, abs=1e-9)
    assert abs(assessment.carrier_decay_rate) < 1e-8
    for grid in assessment.grids:
        assert grid.carrier_frequency == pytest.approx(readout, abs=1e-9)
        assert abs(grid.carrier_decay_rate) < 1e-8
        # The readout is constant, so the frequency operator is exactly zero and the
        # tracked pole is the mass-conservation mode at zero decay on every grid.
        assert grid.omega_reference == pytest.approx(readout, rel=0.0, abs=0.0)
        assert abs(grid.pole_imag_shifted) < 1e-8
    assert max(assessment.grid_decay_differences) < 1e-8
    # On these grids the scale-aware shift is what makes the solve possible: the
    # reference's fixed 1e-12 is below ulp(max|diagonal|), which is what turns the
    # shift-invert factor exactly singular.
    diagonal_scale = 2.0 * diffusion / (2.0 * (2.0 / 1200.0) ** 2)
    assert np.finfo(float).eps * diagonal_scale * 1201 > 1e-12
    assert "eps*max|diagonal|*size" in assessment.method["pole_selection"]


def test_the_pole_matches_an_independent_reference_discretization():
    source = _curved_source()
    steps = 3
    assessment = carrier_pole(
        *source,
        config=CarrierPoleConfig(
            grid_sizes=(21, 41),
            continuation_steps=steps,
            grid_absolute_tolerance=1e-2,
        ),
    )
    assert [grid.grid_size for grid in assessment.grids] == [21, 41]
    # The pole is not the trivial readout value, so the comparison is not vacuous.
    assert all(grid.carrier_decay_rate > 1e-2 for grid in assessment.grids)
    assert assessment.grids[-1].omega_reference == pytest.approx(1.0, rel=1e-12)
    for grid in assessment.grids:
        frequency, decay = _reference_pole(source, grid.grid_size, steps)
        assert grid.carrier_frequency == pytest.approx(frequency, abs=1e-12)
        assert grid.carrier_decay_rate == pytest.approx(decay, abs=1e-12)
        # ARPACK starts from an internal random vector, so the two runs agree to
        # rounding rather than bit for bit; the definitions must still line up.
        assert grid.pole_real == pytest.approx(-decay, abs=1e-12)
        assert grid.pole_imag_shifted == pytest.approx(
            grid.omega_reference - grid.carrier_frequency, abs=1e-12
        )
    # The finer source sampling is the convergence witness the status reports on.
    assert assessment.status is CAMOutcomeStatus.VALID
    assert assessment.finest_grid_size == 41
    assert assessment.grid_converged is True


def test_the_configuration_defaults_are_the_reference_probe():
    config = CarrierPoleConfig()
    assert config.grid_sizes == (401, 801, 1201)
    assert config.continuation_steps == 20
    assert config.k == 6
    assert config.eigensolver_tolerance == pytest.approx(1.0e-10)
    assert config.eigensolver_maxiter == 200_000
    assert config.grid_absolute_tolerance == pytest.approx(1.0e-3)
    assert config.grid_relative_tolerance == pytest.approx(1.0e-3)
    assert config.shift_floor == pytest.approx(1.0e-12)


# --- source-table and resampling failures ---------------------------------


def test_a_non_positive_source_diffusion_is_rejected():
    x = np.linspace(-1.0, 1.0, 5)
    assessment = carrier_pole(
        x, np.zeros(5), np.asarray([1.0, 1.0, 0.0, 1.0, 1.0]), np.zeros(5)
    )
    assert assessment.status is CAMOutcomeStatus.STOCHASTIC_INVALID
    assert assessment.reason_code == "invalid_source_table"
    assert "diffusion coefficients must be strictly positive" in assessment.reason
    assert assessment.grids == ()
    assert assessment.carrier_frequency is None
    assert assessment.grid_converged is None


def test_a_malformed_source_table_is_rejected():
    good = np.linspace(-1.0, 1.0, 5)
    cases = {
        "too few points": (good[:2], np.zeros(2), np.ones(2), np.zeros(2)),
        "duplicated grid": (
            np.asarray([-1.0, 0.0, 0.0, 0.5, 1.0]),
            np.zeros(5),
            np.ones(5),
            np.zeros(5),
        ),
        "decreasing grid": (
            np.asarray([-1.0, 0.0, 0.2, 0.1, 1.0]),
            np.zeros(5),
            np.ones(5),
            np.zeros(5),
        ),
        "length mismatch": (good, np.zeros(4), np.ones(5), np.zeros(5)),
        "non-finite grid": (
            np.asarray([-1.0, 0.0, np.inf, 0.5, 1.0]),
            np.zeros(5),
            np.ones(5),
            np.zeros(5),
        ),
        "non-finite drift": (good, np.full(5, np.nan), np.ones(5), np.zeros(5)),
        "non-finite readout": (good, np.zeros(5), np.ones(5), np.full(5, np.inf)),
        "not numeric": (good, ["a"] * 5, np.ones(5), np.zeros(5)),
    }
    for name, arguments in cases.items():
        assessment = carrier_pole(*arguments)
        assert assessment.status is CAMOutcomeStatus.STOCHASTIC_INVALID, name
        assert assessment.reason_code == "invalid_source_table", name
        assert assessment.reason, name
        assert assessment.grids == (), name


def test_a_non_positive_resampled_diffusion_is_rejected(monkeypatch):
    class NegatingPchip:
        """Resample with the reference interpolant and flip the sign."""

        def __init__(self, x, y):
            self.interpolant = REAL_PCHIP(x, y)

        def __call__(self, values):
            return -np.asarray(self.interpolant(values), dtype=float)

    monkeypatch.setattr("qphase_cam.carrier_pole.PchipInterpolator", NegatingPchip)
    assessment = carrier_pole(
        *_constant_source(), config=CarrierPoleConfig(grid_sizes=(9, 17))
    )
    assert assessment.status is CAMOutcomeStatus.STOCHASTIC_INVALID
    assert assessment.reason_code == "resampled_grid_invalid"
    assert "non-positive" in assessment.reason
    assert assessment.grids == ()


def test_a_degenerate_diffusion_rate_is_a_resampled_grid_failure():
    """Underflow and overflow of ``q / (2 dx^2)`` must not pose as a pole.

    A zero rate would zero whole rows of the generator (an all-zero matrix still
    returns an eigenvalue) and an infinite rate poisons the diagonal, so both are
    typed failures rather than a plausible-looking carrier. The PCHIP resampling stays
    finite in both cases, so this is the rate guard and not the finiteness guard.
    """
    wide = np.linspace(-1.0e50, 1.0e50, 5)
    underflow = carrier_pole(
        wide,
        np.zeros(5),
        np.full(5, 1.0e-300),
        np.zeros(5),
        config=CarrierPoleConfig(grid_sizes=(5, 7)),
    )
    assert underflow.status is CAMOutcomeStatus.STOCHASTIC_INVALID
    assert underflow.reason_code == "resampled_grid_invalid"
    assert "q / (2 dx^2)" in underflow.reason
    assert underflow.grids == ()

    narrow = np.linspace(0.0, 2.0e-9, 5)
    overflow = carrier_pole(
        narrow,
        np.zeros(5),
        np.full(5, 1.0e300),
        np.zeros(5),
        config=CarrierPoleConfig(grid_sizes=(5, 7)),
    )
    assert overflow.status is CAMOutcomeStatus.STOCHASTIC_INVALID
    assert overflow.reason_code == "resampled_grid_invalid"
    assert "q / (2 dx^2)" in overflow.reason
    assert overflow.carrier_frequency is None


# --- eigensolver failures -------------------------------------------------


def test_an_arpack_failure_is_a_not_converged_status(monkeypatch):
    def failing(*args, **kwargs):
        del args, kwargs
        raise ArpackNoConvergence(
            "ARPACK error 3: No shifts could be applied",
            np.zeros(0, dtype=complex),
            np.zeros((0, 0), dtype=complex),
        )

    monkeypatch.setattr("qphase_cam.carrier_pole.eigs", failing)
    assessment = carrier_pole(
        *_constant_source(), config=CarrierPoleConfig(grid_sizes=(9, 17))
    )
    assert assessment.status is CAMOutcomeStatus.NOT_CONVERGED
    assert assessment.reason_code == "eigensolver_failed"
    assert "ARPACK" in assessment.reason
    assert assessment.grids == ()
    assert assessment.finest_grid_size is None
    assert assessment.carrier_frequency is None
    assert assessment.carrier_decay_rate is None


def test_a_non_finite_eigenvalue_is_not_reported_as_a_pole(monkeypatch):
    def non_finite(*args, **kwargs):
        del args, kwargs
        return np.asarray([np.nan + 0.0j, -1.0 + 0.0j, -2.0 + 0.0j])

    monkeypatch.setattr("qphase_cam.carrier_pole.eigs", non_finite)
    assessment = carrier_pole(
        *_constant_source(), config=CarrierPoleConfig(grid_sizes=(9, 17))
    )
    assert assessment.status is CAMOutcomeStatus.NOT_CONVERGED
    assert assessment.reason_code == "eigensolver_failed"
    assert "non-finite eigenvalue" in assessment.reason
    assert assessment.carrier_frequency is None
    assert assessment.carrier_decay_rate is None


def test_a_growing_pole_is_reported_unstable_and_not_clamped(monkeypatch):
    def growing(*args, **kwargs):
        del args, kwargs
        return np.asarray([0.1 + 0.0j, -1.0 + 0.0j, -2.0 + 0.0j])

    monkeypatch.setattr("qphase_cam.carrier_pole.eigs", growing)
    assessment = carrier_pole(
        *_constant_source(), config=CarrierPoleConfig(grid_sizes=(9, 17))
    )
    assert assessment.status is CAMOutcomeStatus.UNSTABLE
    assert assessment.reason_code == "negative_decay_rate"
    assert assessment.carrier_decay_rate == pytest.approx(-0.1, rel=1e-12)
    assert assessment.carrier_decay_rate < 0.0
    assert assessment.grid_converged is True
    assert [grid.grid_size for grid in assessment.grids] == [9, 17]


# --- grid convergence -----------------------------------------------------


def test_a_grid_disagreement_is_not_converged():
    assessment = carrier_pole(
        *_curved_source(),
        config=CarrierPoleConfig(
            grid_sizes=(21, 41),
            continuation_steps=3,
            grid_absolute_tolerance=0.0,
            grid_relative_tolerance=1.0e-14,
        ),
    )
    assert assessment.status is CAMOutcomeStatus.NOT_CONVERGED
    assert assessment.reason_code == "grid_not_converged"
    assert assessment.grid_converged is False
    assert [grid.grid_size for grid in assessment.grids] == [21, 41]
    assert assessment.finest_grid_size == 41
    assert assessment.carrier_frequency is not None
    assert assessment.carrier_decay_rate is not None
    assert assessment.grid_decay_differences[0] > 1.0e-5


@pytest.mark.parametrize(
    "overrides",
    [
        {"grid_sizes": (41,)},
        {"grid_sizes": (41, 21)},
        {"grid_sizes": (41, 41)},
        {"grid_sizes": (2, 41)},
        {"grid_sizes": (41, float("nan"))},
        {"continuation_steps": 0},
        {"k": 0},
        {"k": float("nan")},
        {"eigensolver_tolerance": 0.0},
        {"eigensolver_tolerance": float("inf")},
        {"eigensolver_maxiter": 0},
        {"grid_absolute_tolerance": -1.0},
        {"grid_absolute_tolerance": 0.0, "grid_relative_tolerance": 0.0},
        {"grid_relative_tolerance": float("nan")},
        {"decay_tolerance": -1.0},
        {"shift_floor": 0.0},
        {"shift_floor": float("inf")},
    ],
)
def test_an_invalid_configuration_is_rejected(overrides):
    with pytest.raises(ValidationError):
        CarrierPoleConfig(**overrides)


# --- the typed postprocessor -------------------------------------------------


class _ToyAdapter:
    """Four canonical coordinates of a two-mode matrix.

    ``J(v)`` has a quadratic *and* a cubic dependence on the critical coordinate plus
    a ``v1**2`` coupling into it, so the O3 manifold of this toy is confining
    (``drift[3] = -8.74``); every one-coordinate toy has ``drift[3] = 0`` by the
    ``left·e_d = 0`` constraint.
    """

    moment_layout = "normal"

    def state_matrix(self, vector, params=None):
        del params
        return np.asarray(vector_to_matrix(np.asarray(vector, float), 2))

    def jacobian(self, vector, params=None):
        del params
        values = np.asarray(vector, float).reshape(-1)
        return np.asarray(
            [
                [
                    -1.0 - 3.0 * values[0] ** 2 - 3.0 * values[0] ** 3,
                    values[1] ** 2,
                    0.0,
                    0.0,
                ],
                [values[1] ** 2, -4.0, 0.0, 0.0],
                [0.0, 0.0, -3.0, 0.0],
                [0.0, 0.0, 0.0, -5.0],
            ]
        )

    def parameter_jacobian(self, vector, params=None):
        del vector, params
        return np.zeros((4, 1))


class _UnconfiningAdapter(_ToyAdapter):
    """The same toy without the cubic term, so its O3 truncation does not confine."""

    def jacobian(self, vector, params=None):
        del params
        values = np.asarray(vector, float).reshape(-1)
        return np.asarray(
            [
                [-1.0 - 3.0 * values[0] ** 2, values[1] ** 2, 0.0, 0.0],
                [values[1] ** 2, -4.0, 0.0, 0.0],
                [0.0, 0.0, -3.0, 0.0],
                [0.0, 0.0, 0.0, -5.0],
            ]
        )


class _ToyModel:
    """Bounded diffusion: it vanishes at ``R00 = ±2``, so the domain is finite."""

    n_modes = 2
    params = {"mu": 0.0}

    @staticmethod
    def cam_diffusion(matrix, params=None):
        r00 = float(np.real(np.asarray(matrix)[0, 0]))
        shift = float((params or {}).get("mu", 0.0))
        return np.asarray(
            [[1.0 - (r00 / 2.0) ** 2 + 0.1 * shift, 0.0], [0.0, 1.0]], dtype=complex
        )

    @staticmethod
    def cam_hamiltonian(matrix, params=None):
        del params
        return np.asarray(matrix, dtype=complex)


_TOY_STATE = np.asarray([0.5, 0.5, 0.0, 0.0])
_POLE_CONFIG = CarrierPoleConfig(
    grid_sizes=(21, 41), continuation_steps=3, grid_absolute_tolerance=1e-2
)
_RECONSTRUCTION_CONFIG = ScalarReconstructionConfig(grid_points=301)


def _readout_observable():
    return FunctionalObservable(
        identity=ObservableIdentity(
            kind=ObservableKind.STATE,
            sector=ChargeSector.CHARGE_ZERO,
            name="R00",
            measurement_mapping="R00",
        ),
        value_fn=lambda adapter, state, params: float(
            np.asarray(state, float).reshape(-1)[0]
        ),
        gradient_fn=lambda adapter, state, params: [1.0, 0.0, 0.0, 0.0],
    )


def _patch_adapter(monkeypatch, adapter):
    monkeypatch.setattr(
        "qphase_cam.postprocessor.scalar_carrier_pole.QPSGenDynamicsAdapter.from_model",
        lambda model: adapter,
    )


def _vector_state_product(vectors, *, success=None):
    matrices = np.asarray(
        [vector_to_matrix(np.asarray(vector, float), 2) for vector in vectors]
    )
    count = len(vectors)
    return build_state_product(
        states=matrices[:, None, :, :],
        residuals=np.zeros((count, 1)),
        success=(
            np.ones((count, 1), dtype=bool)
            if success is None
            else np.asarray(success, dtype=bool).reshape(count, 1)
        ),
        valid_mask=np.ones((count, 1), dtype=bool),
        solution_count=np.ones(count, dtype=np.int64),
        parameters={"mu": np.zeros(count)},
    )


def _context(product, *, role="state", metadata=None):
    return CAMAnnotationContext(
        model=_ToyModel(),
        metadata=dict(metadata or {}),
        products={role: product},
        source_role=role,
    )


def _plugin(**kwargs):
    settings = {
        "order": 3,
        "reconstruction": _RECONSTRUCTION_CONFIG,
        "pole": _POLE_CONFIG,
        "observable": _readout_observable(),
    }
    settings.update(kwargs)
    return ScalarCarrierPole(**settings)


def _direct_pole(adapter, params, config=None):
    """Run the same three services the plugin runs, for an independent reference."""
    manifold = slow_manifold(adapter, _TOY_STATE, params, order=3)
    reconstruction = scalar_reconstruction(
        adapter,
        _ToyModel(),
        _readout_observable(),
        state=_TOY_STATE,
        params=params,
        order=3,
        manifold=manifold,
        config=_RECONSTRUCTION_CONFIG,
    )
    table = reconstruction.generator_table
    assert table is not None
    assessment = carrier_pole(
        table.x,
        table.drift,
        table.diffusion,
        table.readout,
        config=config or _POLE_CONFIG,
    )
    return reconstruction, table, assessment


def _field(product, name):
    return np.asarray(product.handle(name).materialize())


def _dims(product, name):
    return next(
        variable.dims for variable in product.variables if variable.name == name
    )


def _metadata(product, row=0):
    return json.loads(str(_field(product, "method_metadata")[row]))


def test_the_postprocessor_annotates_a_real_carrier_pole(monkeypatch):
    _patch_adapter(monkeypatch, _ToyAdapter())
    product = _vector_state_product([_TOY_STATE])
    annotation = _plugin().process(product, _context(product))
    assert annotation.attributes["annotation_kind"] == "scalar_carrier_pole"
    assert annotation.attributes["target_kind"] == "state"
    reconstruction, table, reference = _direct_pole(_ToyAdapter(), {})
    assert reconstruction.validity_domain == "confining_bounded"
    assert reference.status is CAMOutcomeStatus.VALID

    identities = _field(annotation, "target_identity")
    assert list(identities) == list(_field(product, "solution_identity")[:, 0])
    assert list(_field(annotation, "case_identity")) == [""]
    assert _field(annotation, "status").tolist() == ["valid"]
    assert _field(annotation, "order").tolist() == [3]
    assert _field(annotation, "source_reconstruction_status").tolist() == ["valid"]
    assert _field(annotation, "reason").tolist() == [""]

    # One row per target, and the pole is the direct call's to rounding: ARPACK
    # starts from an internal random vector, so the two runs cannot be bit-identical.
    assert _field(annotation, "carrier_frequency").shape == (1,)
    assert _field(annotation, "carrier_frequency")[0] == pytest.approx(
        reference.carrier_frequency, abs=1e-9
    )
    assert _field(annotation, "carrier_decay_rate")[0] == pytest.approx(
        reference.carrier_decay_rate, abs=1e-10
    )
    assert _field(annotation, "finest_grid_size").tolist() == [41]
    assert _field(annotation, "grid_converged").tolist() == [1]
    assert _field(annotation, "grid_converged").dtype == np.int8
    assert _field(annotation, "finest_grid_size").dtype == np.int64
    assert _field(annotation, "carrier_frequency").dtype == np.float64
    assert _field(annotation, "order").dtype == np.int64

    # The refinement witness is a real array on the configured axis.
    assert _dims(annotation, "grid_size") == ("annotation", "refinement")
    assert _dims(annotation, "grid_frequency_differences") == (
        "annotation",
        "refinement_difference",
    )
    assert _field(annotation, "grid_size").tolist() == [[21, 41]]
    assert _field(annotation, "grid_size").dtype == np.int64
    assert _field(annotation, "pole_real").shape == (1, 2)
    assert _field(annotation, "grid_carrier_frequency")[0, 1] == pytest.approx(
        reference.carrier_frequency, abs=1e-9
    )
    assert _field(annotation, "grid_carrier_decay_rate")[0, 1] == pytest.approx(
        reference.carrier_decay_rate, abs=1e-10
    )
    assert _field(annotation, "pole_imag_shifted")[0, 1] == pytest.approx(
        reference.grids[-1].pole_imag_shifted, abs=1e-9
    )
    assert _field(annotation, "pole_real")[0, 1] == pytest.approx(
        reference.grids[-1].pole_real, abs=1e-9
    )
    assert _field(annotation, "grid_frequency_differences").shape == (1, 1)
    assert _field(annotation, "grid_decay_differences")[0, 0] == pytest.approx(
        reference.grid_decay_differences[0], abs=1e-10
    )

    metadata = _metadata(annotation)
    assert metadata["approximation"] == "nonlinear_manifold"
    assert metadata["order"] == 3
    assert metadata["generator_table_points"] == len(table.x)
    assert metadata["reconstruction_status"] == "valid"
    assert metadata["validity_domain"] == "confining_bounded"
    assert metadata["manifold"]["service"] == "qphase_cam.nonlinear.slow_manifold"
    assert metadata["configured_grid_sizes"] == [21, 41]
    assert metadata["continuation_steps"] == 3
    assert metadata["readout"]["name"] == "R00"
    scope = metadata["pole"]["scope"]
    assert "not an SDE frequency measurement" in scope
    assert "not a time-domain PDE solve" in scope


def test_the_postprocessor_keeps_each_candidates_case_parameters(monkeypatch):
    _patch_adapter(monkeypatch, _ToyAdapter())
    states = np.asarray(
        [vector_to_matrix(_TOY_STATE, 2), vector_to_matrix(_TOY_STATE, 2)]
    )
    product = build_candidate_product(
        states=states,
        state_vectors=np.asarray([_TOY_STATE, _TOY_STATE]),
        control_values=np.empty((2, 0)),
        control_names=(),
        full_residual_norm=np.zeros(2),
        search_residual_norm=np.zeros(2),
        success=np.ones(2, dtype=bool),
        status=["verified", "verified"],
        method=["test", "test"],
        verification_digits=np.full(2, 50, dtype=np.int64),
        verification_status=["verified", "verified"],
        candidate_identity=["cam:candidate:shared", "cam:candidate:shared"],
        case_metadata=[
            {"case_identity": "case:left"},
            {"case_identity": "case:right"},
        ],
        candidate_offsets=np.asarray([0, 1, 2], dtype=np.int64),
        case_params={"mu": np.asarray([0.0, 2.0])},
        case_shape=[2],
    )
    annotation = _plugin().process(product, _context(product, role="candidates"))
    assert annotation.attributes["target_kind"] == "candidate"
    assert list(_field(annotation, "target_identity")) == [
        "cam:candidate:shared",
        "cam:candidate:shared",
    ]
    assert list(_field(annotation, "case_identity")) == ["case:left", "case:right"]
    assert _field(annotation, "status").tolist() == ["valid", "valid"]
    # The two cases hold different parameter values, so their poles differ and each
    # matches the reconstruction of that case computed independently.
    frequencies = _field(annotation, "carrier_frequency")
    assert abs(frequencies[0] - frequencies[1]) > 1e-4
    for row, mu in enumerate((0.0, 2.0)):
        _, _, reference = _direct_pole(_ToyAdapter(), {"mu": mu})
        assert frequencies[row] == pytest.approx(reference.carrier_frequency, abs=1e-9)
        assert _field(annotation, "carrier_decay_rate")[row] == pytest.approx(
            reference.carrier_decay_rate, abs=1e-10
        )
    assert _metadata(annotation, 0)["order"] == 3
    assert _metadata(annotation, 1)["approximation"] == "nonlinear_manifold"


def test_an_uncontrolled_reconstruction_never_computes_a_pole(monkeypatch):
    _patch_adapter(monkeypatch, _UnconfiningAdapter())
    calls: list[Any] = []

    def spy(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("carrier_pole must not run on an uncontrolled table")

    monkeypatch.setattr(
        "qphase_cam.postprocessor.scalar_carrier_pole.carrier_pole", spy
    )
    product = _vector_state_product([_TOY_STATE])
    annotation = _plugin().process(product, _context(product))
    assert calls == []
    assert annotation.attributes["annotation_kind"] == "scalar_carrier_pole"
    assert _field(annotation, "status").tolist() == ["uncontrolled_approximation"]
    assert _field(annotation, "source_reconstruction_status").tolist() == [
        "uncontrolled_approximation"
    ]
    assert _field(annotation, "reason_code").tolist() == ["manifold_not_confining"]
    # No pole is fabricated: every slot keeps its "not computed" value.
    assert np.isnan(_field(annotation, "carrier_frequency")).all()
    assert np.isnan(_field(annotation, "carrier_decay_rate")).all()
    assert _field(annotation, "finest_grid_size").tolist() == [0]
    assert _field(annotation, "grid_converged").tolist() == [-1]
    assert _field(annotation, "grid_size").tolist() == [[0, 0]]
    assert np.isnan(_field(annotation, "pole_real")).all()
    assert np.isnan(_field(annotation, "grid_frequency_differences")).all()


def test_an_unconverged_solver_target_is_not_converged(monkeypatch):
    _patch_adapter(monkeypatch, _ToyAdapter())
    product = _vector_state_product([_TOY_STATE], success=[False])
    annotation = _plugin().process(product, _context(product))
    assert _field(annotation, "status").tolist() == ["not_converged"]
    assert _field(annotation, "reason_code").tolist() == ["not_converged"]
    assert _field(annotation, "source_reconstruction_status").tolist() == [""]
    assert np.isnan(_field(annotation, "carrier_frequency")).all()
    assert _field(annotation, "grid_converged").tolist() == [-1]


def test_a_grid_disagreement_is_propagated_with_its_witness(monkeypatch):
    _patch_adapter(monkeypatch, _ToyAdapter())
    product = _vector_state_product([_TOY_STATE])
    annotation = _plugin(
        pole=CarrierPoleConfig(
            grid_sizes=(21, 41),
            continuation_steps=3,
            grid_absolute_tolerance=0.0,
            grid_relative_tolerance=1e-14,
        )
    ).process(product, _context(product))
    assert _field(annotation, "status").tolist() == ["not_converged"]
    assert _field(annotation, "reason_code").tolist() == ["grid_not_converged"]
    assert _field(annotation, "grid_converged").tolist() == [0]
    # The solved grids are kept instead of being hidden by the failure.
    assert _field(annotation, "grid_size").tolist() == [[21, 41]]
    assert np.all(np.isfinite(_field(annotation, "grid_carrier_frequency")))
    assert _field(annotation, "grid_frequency_differences")[0, 0] > 1e-4
    assert _field(annotation, "finest_grid_size").tolist() == [41]


def test_a_valid_reconstruction_without_a_table_is_a_contract_error(monkeypatch):
    _patch_adapter(monkeypatch, _ToyAdapter())
    manifold = slow_manifold(_ToyAdapter(), _TOY_STATE, {}, order=3)
    broken = scalar_reconstruction(
        _ToyAdapter(),
        _ToyModel(),
        _readout_observable(),
        state=_TOY_STATE,
        params={},
        order=3,
        manifold=manifold,
        config=_RECONSTRUCTION_CONFIG,
    ).model_copy(update={"generator_table": None})
    assert broken.status is CAMOutcomeStatus.VALID
    assert broken.generator_table is None
    monkeypatch.setattr(
        "qphase_cam.postprocessor.scalar_carrier_pole.scalar_reconstruction",
        lambda *args, **kwargs: broken,
    )
    product = _vector_state_product([_TOY_STATE])
    with pytest.raises(RuntimeError, match="generator table"):
        _plugin().process(product, _context(product))


def test_the_plugin_declares_its_protocol_and_is_registered():
    assert ScalarCarrierPole.name == "scalar_carrier_pole"
    assert ScalarCarrierPole.annotation_kind == "scalar_carrier_pole"
    assert ScalarCarrierPole.accepted_product_roles == (
        "state",
        "state_scan",
        "candidates",
    )
    assert ScalarCarrierPole.requires_model is True
    assert ScalarCarrierPole.requires_backend is False
    assert ScalarCarrierPole.output_products == (CAM_ANNOTATION_DECLARATION,)
    assert issubclass(ScalarCarrierPole, CAMPostprocessor)
    config = ScalarCarrierPoleConfig()
    assert config.order == 7
    assert config.pole.grid_sizes == (401, 801, 1201)
    assert config.pole.continuation_steps == 20
    assert config.reconstruction.grid_points == 3001
    assert config.observable is None

    pyproject = tomllib.loads(
        (
            Path(__file__).resolve().parents[2]
            / "packages"
            / "qphase_cam"
            / "pyproject.toml"
        ).read_text(encoding="utf-8")
    )
    assert (
        pyproject["project"]["entry-points"]["qphase"][
            "cam_postprocessor.scalar_carrier_pole"
        ]
        == "qphase_cam.postprocessor.scalar_carrier_pole:ScalarCarrierPole"
    )
    # The installed metadata must agree too, or discovery would silently miss it.
    discovered = {
        entry.name: entry.value
        for entry in importlib.metadata.entry_points(group="qphase")
    }
    assert (
        discovered["cam_postprocessor.scalar_carrier_pole"]
        == "qphase_cam.postprocessor.scalar_carrier_pole:ScalarCarrierPole"
    )
    discovery.discover_plugins("qphase")
    assert (
        registry.get_plugin_class("cam_postprocessor", "scalar_carrier_pole")
        is ScalarCarrierPole
    )
    assert (
        registry.get_plugin_schema("cam_postprocessor", "scalar_carrier_pole")
        is ScalarCarrierPoleConfig
    )
