"""Phase 10F3a: model-independent slow manifold and scalar reconstruction.

The parity layout is a one-mode toy model whose canonical vector *is* its 1x1 state
matrix, ``J(v) = -1 - 3v²`` and a unit diffusion, evaluated at ``v = 0.5``. The
frozen numbers come from the reference script's own ``_slow_manifold`` and
``_scalar_intensity`` run once on that toy (the script itself is not imported).
"""

from __future__ import annotations

import numpy as np
import pytest
from qphase.data import load_bundle
from qphase_cam.core.noise import (
    canonical_hermitian_transform,
    canonical_sample_matrix_covariance,
    canonical_sample_matrix_covariances,
    projected_noise_intensities,
)
from qphase_cam.data import (
    build_candidate_product,
    build_state_product,
    cam_bundle,
)
from qphase_cam.nonlinear import (
    DEFAULT_MANIFOLD_ORDERS,
    NonlinearManifold,
    NonlinearManifoldConfig,
    ScalarGeneratorTable,
    ScalarReconstructionConfig,
    evaluate_generator,
    manifold_drift,
    manifold_embedding,
    ou_intensities,
    reconstructed_intensity,
    scalar_reconstruction,
    slow_manifold,
)
from qphase_cam.observable import (
    ChargeSector,
    FunctionalObservable,
    ObservableIdentity,
    ObservableKind,
    RayleighObservable,
    observable_values,
)
from qphase_cam.outcomes import CAMOutcomeStatus
from qphase_cam.postprocessor.base import CAMAnnotationContext
from qphase_cam.postprocessor.nonlinear_reconstruction import (
    NonlinearReconstruction,
)
from qphase_cam.validity import projected_intensity

X0 = 0.5
IDENTITY = ObservableIdentity(
    kind=ObservableKind.STATE,
    sector=ChargeSector.CHARGE_ZERO,
    name="x",
    measurement_mapping="x",
)


class _QuadraticAdapter:
    """One-mode adapter with ``J(v) = -1 - 3 v²`` and an identity state matrix."""

    moment_layout = "normal"

    def state_matrix(self, vector, params=None):
        del params
        return np.asarray(vector, dtype=float).reshape(1, 1).astype(complex)

    def jacobian(self, vector, params=None):
        del params
        value = float(np.asarray(vector, dtype=float).reshape(-1)[0])
        return np.asarray([[-1.0 - 3.0 * value**2]])

    def parameter_jacobian(self, vector, params=None):
        del params
        return np.asarray([[1.7]])


class _UnitModel:
    """One-mode model with a unit diffusion and the state matrix as Hamiltonian."""

    n_modes = 1
    params = {"mu": 0.0}

    @staticmethod
    def cam_diffusion(matrix, params=None):
        del params
        return np.ones_like(np.asarray(matrix), dtype=complex)

    @staticmethod
    def cam_hamiltonian(matrix, params=None):
        del params
        return np.asarray(matrix, dtype=complex)


class _BoundedModel(_UnitModel):
    """Diffusion vanishing at ``v = 1``, so the physical domain is bounded."""

    @staticmethod
    def cam_diffusion(matrix, params=None):
        del params
        value = float(np.real(np.asarray(matrix).reshape(-1)[0]))
        return np.asarray([[1.0 - value]], dtype=complex)


class _CubicAdapter(_QuadraticAdapter):
    """Confining toy: ``J(v) = -1 - 3 v³`` gives a negative cubic drift."""

    def jacobian(self, vector, params=None):
        del params
        value = float(np.asarray(vector, dtype=float).reshape(-1)[0])
        return np.asarray([[-1.0 - 3.0 * value**3]])


def _observable(*, gain: float = 1.0):
    return FunctionalObservable(
        identity=IDENTITY,
        value_fn=lambda adapter, state, params: gain
        * float(np.asarray(state, dtype=float).reshape(-1)[0]),
        gradient_fn=lambda adapter, state, params: [gain],
    )


# --- parity with the reference script -------------------------------------


def test_slow_manifold_recursion_matches_the_reference():
    adapter = _QuadraticAdapter()
    expected = {
        3: (
            (0.0, -1.75, -3.0, 0.0),
            14.179475529761232,
            ((1.0,), (0.0,), (0.0,)),
        ),
        5: (
            (0.0, -1.75, -3.0, 0.0, 0.0, 0.0),
            50.980384612481835,
            ((1.0,), (0.0,), (0.0,), (0.0,), (0.0,)),
        ),
        7: (
            (0.0, -1.75, -3.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            112.24109060688386,
            ((1.0,), (0.0,), (0.0,), (0.0,), (0.0,), (0.0,), (0.0,)),
        ),
    }
    for order in DEFAULT_MANIFOLD_ORDERS:
        drift, condition, embeddings = expected[order]
        manifold = slow_manifold(adapter, [X0], {}, order=order)
        assert manifold.order == order
        np.testing.assert_allclose(manifold.drift, drift, rtol=0, atol=1e-15)
        assert manifold.maximum_bordered_condition == pytest.approx(
            condition, rel=1e-12
        )
        np.testing.assert_allclose(manifold.embeddings, embeddings, atol=1e-14)
        assert manifold.right == (1.0,)
        assert manifold.left == (1.0,)
        assert manifold.soft_rate == pytest.approx(1.75, rel=1e-15)
        # A quadratic toy has no cubic term, so the O3 truncation does not confine.
        assert manifold.confining is False


def test_scalar_reconstruction_matches_the_reference():
    adapter = _QuadraticAdapter()
    model = _UnitModel()
    manifold = slow_manifold(adapter, [X0], {}, order=3)
    reconstruction = scalar_reconstruction(
        adapter,
        model,
        _observable(),
        state=[X0],
        params={},
        order=3,
        parameter_direction=np.asarray([1.0]),
        manifold=manifold,
    )
    assert reconstruction.order == 3
    assert reconstruction.confining is False
    assert reconstruction.reason_code == "manifold_not_confining"
    assert reconstruction.status is CAMOutcomeStatus.UNCONTROLLED_APPROXIMATION
    assert reconstruction.validity_domain == "not_confining"
    assert reconstruction.sigma == pytest.approx(0.5345224838248488, rel=1e-12)
    assert reconstruction.x_min == pytest.approx(-0.49999999999949996, rel=1e-9)
    assert reconstruction.x_max == pytest.approx(106.90449676496976, rel=1e-12)
    assert reconstruction.negative_physical_boundary is True
    assert reconstruction.positive_physical_boundary is False
    assert reconstruction.scalar_intensity == pytest.approx(
        0.11156067764845953, rel=1e-9
    )
    assert reconstruction.stationary_mean == pytest.approx(
        0.21290711709873192, rel=1e-9
    )
    assert reconstruction.stationary_response == pytest.approx(
        1.0141442976891994, rel=1e-9
    )
    assert reconstruction.path_information_rate == pytest.approx(
        837.9765103977287, rel=1e-9
    )
    assert reconstruction.minimum_projected_diffusion == pytest.approx(
        1.000088900582341e-12, rel=1e-6
    )
    assert reconstruction.endpoint_density_ratio < 1e-300
    assert reconstruction.edge_mass_fraction == pytest.approx(
        0.3892218547566166, rel=1e-9
    )
    assert reconstruction.diagnostics["center_projected_noise"] == pytest.approx(
        1.0, rel=1e-12
    )
    # The bounded/unbounded verdicts keep the sampled generator coefficients.
    assert reconstruction.reason_code == "manifold_not_confining"
    assert reconstruction.generator_table is not None


def test_the_gaussian_terms_and_the_reconstruction_combination():
    adapter = _QuadraticAdapter()
    model = _UnitModel()
    gaussian = ou_intensities(
        adapter, _observable(), state=[X0], params={}, model=model
    )
    assert gaussian.rate == pytest.approx(1.75, rel=1e-15)
    assert gaussian.projection == pytest.approx(1.0, rel=1e-15)
    assert gaussian.projected_noise == pytest.approx(1.0, rel=1e-12)
    assert gaussian.soft_intensity == pytest.approx(0.32653061224489793, rel=1e-12)
    # One mode means the soft and full Gaussian intensities coincide.
    assert gaussian.full_intensity == pytest.approx(gaussian.soft_intensity, rel=1e-12)
    scalar = 0.11156067764845953
    assert reconstructed_intensity(
        scalar,
        full_intensity=gaussian.full_intensity,
        soft_intensity=gaussian.soft_intensity,
    ) == pytest.approx(0.1115606776484595, rel=1e-12)
    with pytest.raises(ValueError, match="must be finite"):
        reconstructed_intensity(float("nan"), full_intensity=1.0, soft_intensity=1.0)


# --- structure and defaults ------------------------------------------------


def test_manifold_helpers_resample_the_same_law():
    manifold = slow_manifold(_QuadraticAdapter(), [X0], {}, order=3)
    values = np.asarray([-0.25, 0.0, 0.25])
    np.testing.assert_allclose(
        manifold_embedding(manifold, [X0], values).ravel(), [0.25, 0.5, 0.75]
    )
    drift = manifold_drift(manifold, values)
    expected = -1.75 * values - 3.0 * values**2
    np.testing.assert_allclose(drift, expected, atol=1e-14)
    # The left/right pair keeps the reference normalization.
    right = np.asarray(manifold.right)
    left = np.asarray(manifold.left)
    assert float(np.linalg.norm(right)) == pytest.approx(1.0, rel=1e-15)
    assert float(left @ right) == pytest.approx(1.0, rel=1e-15)


def test_the_default_configuration_matches_the_reference_semantics():
    config = ScalarReconstructionConfig()
    assert config.grid_points == 3001
    assert config.boundary_iterations == 60
    assert config.boundary_start_relative == pytest.approx(0.02)
    assert config.boundary_limit_relative == pytest.approx(200.0)
    assert config.boundary_growth == pytest.approx(1.35)
    assert config.physical_tolerance == pytest.approx(1e-10)
    assert config.minimum_projected_diffusion == pytest.approx(1e-12)
    assert config.log_density_floor == pytest.approx(-745.0)
    assert config.edge_fraction == pytest.approx(0.02)
    assert config.density_floor_relative == pytest.approx(1e-13)
    assert DEFAULT_MANIFOLD_ORDERS == (3, 5, 7)
    assert NonlinearManifoldConfig().order == 3
    assert NonlinearManifoldConfig().bilinear_offset_scale == pytest.approx(1.0)


def _confining_manifold() -> NonlinearManifold:
    """Return a confining O3 manifold with a unit ``x**1`` embedding.

    A one-coordinate state cannot produce a nonlinear embedding: the constraint
    ``left · e_d = 0`` forces ``e_d = 0`` and the reference recursion then returns
    ``drift[d] = 0`` for ``d >= 3``. The confining paths are therefore exercised
    with a manifold supplied by the caller (which the service accepts), while the
    recursion itself is covered by the exact parity test above.
    """
    return NonlinearManifold(
        order=3,
        soft_rate=1.0,
        critical_eigenvalue_real=-1.0,
        critical_eigenvalue_imag=0.0,
        noncritical_spectral_gap=None,
        right=(1.0,),
        left=(1.0,),
        embeddings=((1.0,),),
        drift=(0.0, -1.5, -0.5, -1.25),
        maximum_bordered_condition=10.0,
        confining=True,
        method={"service": "test fixture"},
    )


def test_a_confining_manifold_with_a_bounded_domain_is_valid():
    adapter = _QuadraticAdapter()
    model = _BoundedModel()
    reconstruction = scalar_reconstruction(
        adapter,
        model,
        _observable(),
        state=[X0],
        params={},
        order=3,
        parameter_direction=np.asarray([1.0]),
        manifold=_confining_manifold(),
    )
    assert reconstruction.status is CAMOutcomeStatus.VALID
    assert reconstruction.reason is None
    assert reconstruction.validity_domain == "confining_bounded"
    assert reconstruction.negative_physical_boundary is True
    assert reconstruction.positive_physical_boundary is True
    # The diffusion vanishes at v = 1, i.e. half a unit above the centre.
    assert reconstruction.x_max == pytest.approx(0.5, rel=1e-6)
    assert reconstruction.x_min == pytest.approx(-0.5, rel=1e-6)
    assert reconstruction.confining is True
    for name in (
        "scalar_intensity",
        "stationary_mean",
        "stationary_response",
        "path_information_rate",
        "edge_mass_fraction",
        "endpoint_density_ratio",
    ):
        value = getattr(reconstruction, name)
        assert value is not None and np.isfinite(value), name
    assert reconstruction.edge_mass_fraction < 1.0


def _confining_table(points: int = 201) -> ScalarGeneratorTable:
    """Return a symmetric confining table: f = -x, q = 1, omega = x."""
    x = np.linspace(-3.0, 3.0, 2 * points - 1)
    return ScalarGeneratorTable(
        x=tuple(float(value) for value in x),
        drift=tuple(float(-value) for value in x),
        diffusion=tuple(1.0 for _ in x),
        readout=tuple(float(value) for value in x),
    )


def test_the_generator_evaluation_reproduces_the_reconstruction_numbers():
    """The exported evaluation is the reconstruction's own arithmetic."""
    manifold = _confining_manifold()
    reconstruction = scalar_reconstruction(
        _QuadraticAdapter(),
        _BoundedModel(),
        _observable(),
        state=[X0],
        params={},
        order=3,
        manifold=manifold,
        config=ScalarReconstructionConfig(grid_points=51),
    )
    table = reconstruction.generator_table
    assert table is not None
    evaluation = evaluate_generator(
        table, config=ScalarReconstructionConfig(grid_points=51)
    )
    assert evaluation.normalizable
    assert evaluation.scalar_intensity == reconstruction.scalar_intensity
    assert evaluation.stationary_mean == reconstruction.stationary_mean
    assert evaluation.density_normalization == reconstruction.density_normalization
    assert evaluation.edge_mass_fraction == reconstruction.edge_mass_fraction
    assert evaluation.endpoint_density_ratio == reconstruction.endpoint_density_ratio
    assert (
        evaluation.minimum_projected_diffusion
        == reconstruction.minimum_projected_diffusion
    )
    assert evaluation.core_point_count == reconstruction.diagnostics["core_point_count"]
    assert evaluation.edge_point_count == reconstruction.diagnostics["edge_point_count"]
    # No forcing: the two forced numbers stay unavailable instead of being faked.
    assert evaluation.path_information_rate is None
    assert evaluation.stationary_response is None
    assert evaluation.stationary_mean_x is not None
    assert len(evaluation.density) == len(table.x)


def test_the_generator_evaluation_scales_with_the_forcing():
    table = _confining_table()
    forcing = np.asarray(table.x, dtype=float)
    doubled = evaluate_generator(table, forcing=2.0 * forcing)
    single = evaluate_generator(table, forcing=forcing)
    # The forcing enters quadratically, and the stationary response scales linearly
    # with the stationary score.
    assert doubled.path_information_rate == pytest.approx(
        4.0 * single.path_information_rate, rel=1e-12
    )
    assert doubled.stationary_response == pytest.approx(
        2.0 * single.stationary_response, rel=1e-12
    )
    with pytest.raises(ValueError, match="forcing vector has"):
        evaluate_generator(table, forcing=np.ones(3))


def test_grid_points_is_the_point_count_per_side():
    """``grid_points`` counts one side, so the table holds ``2 n - 1`` points."""
    manifold = _confining_manifold()
    reconstruction = scalar_reconstruction(
        _QuadraticAdapter(),
        _BoundedModel(),
        _observable(),
        state=[X0],
        params={},
        order=3,
        manifold=manifold,
        config=ScalarReconstructionConfig(grid_points=31),
    )
    table = reconstruction.generator_table
    assert table is not None
    assert len(table.x) == 2 * 31 - 1
    assert reconstruction.grid_points == 2 * 31 - 1


def test_a_density_that_does_not_normalize_is_reported_not_raised():
    """A table whose density integral is not finite reports it instead of raising."""
    points = 51
    x = np.linspace(-1.0, 1.0, 2 * points - 1)
    # A flat drift gives a flat density, which normalizes.
    flat = ScalarGeneratorTable(
        x=tuple(float(value) for value in x),
        drift=tuple(0.0 for _ in x),
        diffusion=tuple(1.0 for _ in x),
        readout=tuple(0.0 for _ in x),
    )
    assert evaluate_generator(flat).normalizable
    # A drift large enough to overflow the potential leaves no normalization.
    steep = ScalarGeneratorTable(
        x=tuple(float(value) for value in x),
        drift=tuple(1.0e308 * float(value) for value in x),
        diffusion=tuple(1.0 for _ in x),
        readout=tuple(0.0 for _ in x),
    )
    with np.errstate(over="ignore", invalid="ignore"):
        empty = evaluate_generator(steep)
    assert empty.normalizable is False
    assert empty.scalar_intensity is None
    assert empty.stationary_mean is None
    assert empty.stationary_mean_x is None
    assert empty.path_information_rate is None
    assert empty.stationary_response is None


class _BatchAdapter(_QuadraticAdapter):
    """The same adapter, declaring and implementing the stack contract."""

    supports_batched_states = True

    def state_matrix(self, vector, params=None):
        del params
        array = np.asarray(vector, dtype=float)
        if array.ndim == 1:
            return array.reshape(1, 1).astype(complex)
        return array.reshape(array.shape[0], 1, 1).astype(complex)


class _BatchObservable(FunctionalObservable):
    """A readout whose batch form is the caller's, so a test can count its calls."""

    def values(self, adapter, states, params):
        return super().values(adapter, states, params)


class _StackBoundedModel(_BoundedModel):
    """The bounded toy with the whole-stack diffusion contract."""

    supports_batched_matrices = True

    @staticmethod
    def cam_diffusion(matrix, params=None):
        del params
        array = np.asarray(matrix, dtype=complex)
        return np.asarray(1.0 - np.real(array), dtype=complex).reshape(array.shape)


class _WrongShapeModel(_StackBoundedModel):
    """A model that claims the batch contract and answers one matrix too few."""

    @staticmethod
    def cam_diffusion(matrix, params=None):
        del params
        array = np.asarray(matrix, dtype=complex)
        if array.ndim == 2:
            return np.asarray(1.0 - np.real(array), dtype=complex)
        return np.ones((array.shape[0] - 1,) + array.shape[1:], dtype=complex)


def _three_mode_stack(count: int = 5):
    """Return a Hermitian state stack, its diffusion stack and a left vector."""
    rng = np.random.default_rng(20260904)
    factor = rng.normal(size=(count, 3, 3)) + 1j * rng.normal(size=(count, 3, 3))
    states = factor @ np.conjugate(np.swapaxes(factor, -1, -2))
    diffusion = np.zeros((count, 3, 3), dtype=complex)
    diffusion[:, np.arange(3), np.arange(3)] = np.asarray([0.5, 1.5, 2.5])
    left = np.linspace(-0.4, 0.6, 9)
    return states, diffusion, left


def _reference_projection(states, diffusions, left):
    """Return the reference contraction: nested mode indices, then one projection."""
    count, modes = states.shape[0], states.shape[1]
    covariance = np.empty((count, modes * modes, modes * modes), dtype=complex)
    for i in range(modes):
        for j in range(modes):
            for k in range(modes):
                for right in range(modes):
                    covariance[:, i * modes + j, k * modes + right] = (
                        diffusions[:, i, right] * states[:, k, j]
                        + diffusions[:, k, j] * states[:, i, right]
                    )
    projected_left = canonical_hermitian_transform(modes).T @ left
    projected = np.einsum(
        "i,nij,j->n", projected_left, covariance, projected_left, optimize=True
    )
    return np.asarray(np.real(projected), dtype=float)


def test_the_batch_projection_reproduces_the_reference_order():
    """The batch projection is the reference contraction, to the last bit."""
    states, diffusion, left = _three_mode_stack()
    expected = _reference_projection(states, diffusion, left)
    actual = projected_noise_intensities(states, diffusion, left)
    np.testing.assert_array_equal(actual, expected)
    # The single-matrix projection is the same number, evaluated in its own order: the
    # two agree to rounding, not to the last bit, and that is the contract.
    pointwise = np.asarray(
        [
            projected_intensity(
                canonical_sample_matrix_covariance(state, diff), left, left
            )
            for state, diff in zip(states, diffusion, strict=True)
        ]
    )
    np.testing.assert_allclose(actual, pointwise, rtol=1.0e-12, atol=0.0)


def test_the_batch_covariance_is_the_single_matrix_covariance():
    """The batched canonical covariance is the single-matrix one, row for row."""
    states, diffusion, _ = _three_mode_stack()
    batched = canonical_sample_matrix_covariances(states, diffusion)
    assert batched.shape == (states.shape[0], 9, 9)
    expected = np.stack(
        [
            canonical_sample_matrix_covariance(state, diff)
            for state, diff in zip(states, diffusion, strict=True)
        ]
    )
    np.testing.assert_allclose(batched, expected, rtol=1.0e-12, atol=0.0)


def test_a_malformed_batch_is_an_error_not_a_nan():
    """Wrong shapes and a wrong-length left vector raise; they are not filtered."""
    states, diffusion, left = _three_mode_stack()
    with pytest.raises(ValueError, match="state-matrix stack must be"):
        projected_noise_intensities(states[0], diffusion[0], left)
    with pytest.raises(ValueError, match="state-matrix stack must be square"):
        projected_noise_intensities(states[:, :2, :3], diffusion[:, :2, :3], left)
    with pytest.raises(ValueError, match="diffusion stack has shape"):
        projected_noise_intensities(states, diffusion[:2], left)
    with pytest.raises(ValueError, match="left vector has 5 entries"):
        projected_noise_intensities(states, diffusion, left[:5])
    with pytest.raises(ValueError, match="diffusion stack has shape"):
        canonical_sample_matrix_covariances(states, diffusion[:2])


def test_a_non_real_projection_is_an_error_and_a_non_finite_row_is_nan():
    """A mixed-coordinate projection raises; an unusable state is reported as NaN."""
    states, diffusion, left = _three_mode_stack()
    skewed = states + 0.3j * np.swapaxes(states, -1, -2).conjugate()
    with pytest.raises(ValueError, match="projection is not real"):
        projected_noise_intensities(skewed, diffusion, left)
    broken = np.array(states, copy=True)
    broken[2] = np.nan
    intensities = projected_noise_intensities(broken, diffusion, left)
    assert np.isnan(intensities[2])
    assert np.all(np.isfinite(np.delete(intensities, 2)))


def test_the_batched_readout_matches_the_point_readout():
    """A batch readout answers the point readout's value on every row.

    The two differ only in summation order, so the parity is a tolerance rather than
    bitwise equality: a batch readout exists to avoid one adapter and model call per
    point, not to define a second readout.
    """
    model = _UnitModel()
    adapter = _QuadraticAdapter()
    stack = np.linspace(0.4, 0.6, 7)[:, None]
    observable = RayleighObservable(model)
    batched = observable.values(adapter, stack, {})
    point = np.asarray([observable.value(adapter, row, {}) for row in stack])
    np.testing.assert_allclose(batched, point, rtol=1.0e-13, atol=0.0)
    # A readout without a batch form is sampled by the helper, with the same values.
    functional = _observable()
    np.testing.assert_allclose(
        observable_values(functional, adapter, stack, {}),
        np.asarray([functional.value(adapter, row, {}) for row in stack]),
        rtol=1.0e-13,
        atol=0.0,
    )
    with pytest.raises(ValueError, match="state stack must be"):
        observable_values(functional, adapter, stack.reshape(-1), {})


def test_a_batch_readout_is_used_only_when_it_exists():
    """The helper prefers the batch call and validates its length."""
    adapter = _QuadraticAdapter()
    stack = np.linspace(0.4, 0.6, 5)[:, None]
    observable = _observable()
    recorder: list[int] = []
    batch = _BatchObservable(
        identity=observable.identity,
        value_fn=observable.value_fn,
        values_fn=lambda adapter_, states_, params_: (
            recorder.append(int(np.asarray(states_).shape[0]))
            or np.asarray(
                [observable.value_fn(adapter_, row, params_) for row in states_]
            )
        ),
    )
    values = observable_values(batch, adapter, stack, {})
    assert recorder == [5]
    np.testing.assert_array_equal(
        values, np.asarray([observable.value(adapter, row, {}) for row in stack])
    )

    class _WrongLength(_BatchObservable):
        def values(self, adapter, states, params):
            return np.zeros(2)

    with pytest.raises(ValueError, match="2 values for 5"):
        observable_values(
            _WrongLength(identity=observable.identity, value_fn=observable.value_fn),
            adapter,
            stack,
            {},
        )


def test_the_batch_adapter_and_the_point_adapter_reconstruct_the_same_table():
    """Batching the adapter call must not change one number of the reconstruction."""
    manifold = _confining_manifold()
    config = ScalarReconstructionConfig(grid_points=31)
    point = scalar_reconstruction(
        _QuadraticAdapter(),
        _BoundedModel(),
        _observable(),
        state=[X0],
        params={},
        order=3,
        manifold=manifold,
        config=config,
    )
    batched = scalar_reconstruction(
        _BatchAdapter(),
        _BoundedModel(),
        _observable(),
        state=[X0],
        params={},
        order=3,
        manifold=manifold,
        config=config,
    )
    assert point.status is batched.status
    assert point.scalar_intensity == batched.scalar_intensity
    assert point.stationary_mean == batched.stationary_mean
    assert point.density_normalization == batched.density_normalization
    assert point.edge_mass_fraction == batched.edge_mass_fraction
    assert point.endpoint_density_ratio == batched.endpoint_density_ratio
    assert point.generator_table == batched.generator_table


def test_a_point_only_model_reconstructs_the_same_table_as_a_batch_model():
    """A model without the batch contract is sampled row by row, not guessed at."""
    manifold = _confining_manifold()
    config = ScalarReconstructionConfig(grid_points=31)
    point = scalar_reconstruction(
        _QuadraticAdapter(),
        _BoundedModel(),
        _observable(),
        state=[X0],
        params={},
        order=3,
        manifold=manifold,
        config=config,
    )
    batched = scalar_reconstruction(
        _QuadraticAdapter(),
        _StackBoundedModel(),
        _observable(),
        state=[X0],
        params={},
        order=3,
        manifold=manifold,
        config=config,
    )
    assert point.status is batched.status
    assert point.generator_table is not None and batched.generator_table is not None
    np.testing.assert_allclose(
        np.asarray(batched.generator_table.diffusion),
        np.asarray(point.generator_table.diffusion),
        rtol=1.0e-13,
        atol=0.0,
    )
    np.testing.assert_allclose(
        np.asarray(batched.generator_table.readout),
        np.asarray(point.generator_table.readout),
        rtol=1.0e-13,
        atol=0.0,
    )


def test_a_model_that_breaks_the_batch_contract_raises():
    """A diffusion stack of the wrong shape is a broken contract, not a NaN domain."""
    config = ScalarReconstructionConfig(grid_points=11)
    with pytest.raises(ValueError, match="diffusion matrices of shape"):
        scalar_reconstruction(
            _QuadraticAdapter(),
            _WrongShapeModel(),
            _observable(),
            state=[X0],
            params={},
            order=3,
            manifold=_confining_manifold(),
            config=config,
        )


def test_the_rayleigh_batch_readout_refuses_an_undefined_row():
    """One undefined row makes the whole batch readout undefined."""
    adapter = _QuadraticAdapter()
    stack = np.linspace(0.4, 0.6, 5)[:, None]

    class _WrongShapeHamiltonian(_UnitModel):
        supports_batched_matrices = True

        @staticmethod
        def cam_hamiltonian(matrix, params=None):
            del params
            array = np.asarray(matrix, dtype=complex)
            return np.ones((array.shape[0] - 1, 1, 1), dtype=complex)

    with pytest.raises(ValueError, match="does not match the state matrix stack"):
        RayleighObservable(_WrongShapeHamiltonian()).values(adapter, stack, {})
    with pytest.raises(ValueError, match="vanishing trace"):
        RayleighObservable(_UnitModel()).values(adapter, np.zeros((3, 1)), {})
    with pytest.raises(ValueError, match="state stack must be"):
        RayleighObservable(_UnitModel()).values(adapter, stack.reshape(-1), {})


def test_a_batch_readout_in_the_reconstruction_matches_a_point_readout():
    """A batched readout and a point readout give the same reconstruction."""
    manifold = _confining_manifold()
    config = ScalarReconstructionConfig(grid_points=31)
    point = scalar_reconstruction(
        _QuadraticAdapter(),
        _BoundedModel(),
        _observable(),
        state=[X0],
        params={},
        order=3,
        manifold=manifold,
        config=config,
    )
    recorder: list[int] = []
    base = _observable()
    batch = FunctionalObservable(
        identity=base.identity,
        value_fn=base.value_fn,
        values_fn=lambda adapter_, states_, params_: (
            recorder.append(int(np.asarray(states_).shape[0]))
            or np.asarray([base.value_fn(adapter_, row, params_) for row in states_])
        ),
    )
    batched = scalar_reconstruction(
        _QuadraticAdapter(),
        _BoundedModel(),
        batch,
        state=[X0],
        params={},
        order=3,
        manifold=manifold,
        config=config,
    )
    assert recorder == [2 * 31 - 1]
    assert point.scalar_intensity == batched.scalar_intensity
    assert point.path_information_rate is None and batched.path_information_rate is None
    assert point.generator_table == batched.generator_table


def test_the_generator_table_carries_the_reconstruction_coefficients():
    manifold = _confining_manifold()
    reconstruction = scalar_reconstruction(
        _QuadraticAdapter(),
        _BoundedModel(),
        _observable(),
        state=[X0],
        params={},
        order=3,
        manifold=manifold,
        config=ScalarReconstructionConfig(grid_points=51),
    )
    assert reconstruction.status is CAMOutcomeStatus.VALID
    table = reconstruction.generator_table
    assert table is not None
    x = np.asarray(table.x)
    assert x.size == reconstruction.grid_points
    assert len(table.drift) == x.size
    assert len(table.diffusion) == x.size
    assert len(table.readout) == x.size
    assert np.all(np.diff(x) > 0.0)
    # The table is exactly the coefficients the reconstruction integrated.
    np.testing.assert_array_equal(np.asarray(table.drift), manifold_drift(manifold, x))
    # The bounded model's diffusion is 1 - v at v = X0 + x, and the canonical sample
    # covariance of a one-mode matrix is 2 v (1 - v), which one mode projects onto
    # the critical direction unchanged.
    state = X0 + x
    np.testing.assert_allclose(
        np.asarray(table.diffusion), 2.0 * state * (1.0 - state), atol=1e-12
    )
    assert np.all(np.asarray(table.diffusion) > 0.0)
    np.testing.assert_allclose(np.asarray(table.readout), X0 + x, atol=1e-12)


def test_a_confining_but_unbounded_domain_is_reported_as_truncated():
    reconstruction = scalar_reconstruction(
        _QuadraticAdapter(),
        _UnitModel(),
        _observable(),
        state=[X0],
        params={},
        order=3,
        manifold=_confining_manifold(),
    )
    assert reconstruction.confining is True
    assert reconstruction.status is CAMOutcomeStatus.UNCONTROLLED_APPROXIMATION
    assert reconstruction.reason_code == "physical_region_unbounded"
    assert reconstruction.validity_domain == "confining_truncated"
    assert reconstruction.positive_physical_boundary is False
    # The path-information fields need a drive direction and stay unavailable.
    assert reconstruction.path_information_rate is None
    assert reconstruction.stationary_response is None


# --- failure statuses ------------------------------------------------------


def test_a_growing_soft_mode_is_unstable():
    class Growing(_QuadraticAdapter):
        def jacobian(self, vector, params=None):
            del vector, params
            return np.asarray([[0.5]])

    with pytest.raises(ValueError, match="does not decay"):
        slow_manifold(Growing(), [X0], {}, order=3)
    assessment = scalar_reconstruction(
        Growing(), _UnitModel(), _observable(), state=[X0], params={}, order=3
    )
    assert assessment.status is CAMOutcomeStatus.UNSTABLE
    assert assessment.reason_code == "soft_mode_not_decaying"


def test_a_rotating_jacobian_has_no_manifold():
    class Rotating(_QuadraticAdapter):
        def jacobian(self, vector, params=None):
            del vector, params
            return np.asarray([[-0.5, -1.0], [1.0, -0.5]])

        def state_matrix(self, vector, params=None):
            del params
            return np.asarray(vector, dtype=float).reshape(2, 2).astype(complex)

    with pytest.raises(ValueError, match="unresolved"):
        slow_manifold(Rotating(), [1.0, 0.0], {}, order=3)
    assessment = scalar_reconstruction(
        Rotating(), _UnitModel(), _observable(), state=[1.0, 0.0], params={}, order=3
    )
    assert assessment.status is CAMOutcomeStatus.UNIDENTIFIABLE
    assert assessment.reason_code == "manifold_unresolved"


def test_a_negative_diffusion_is_reported_not_used():
    class Negative(_UnitModel):
        @staticmethod
        def cam_diffusion(matrix, params=None):
            del params
            return -np.ones_like(np.asarray(matrix), dtype=complex)

    assessment = scalar_reconstruction(
        _QuadraticAdapter(),
        Negative(),
        _observable(),
        state=[X0],
        params={},
        order=3,
    )
    assert assessment.status is CAMOutcomeStatus.STOCHASTIC_INVALID
    assert assessment.reason_code == "invalid_domain"
    assert "projected noise" in (assessment.reason or "")
    # The grid never became usable, so no generator table is fabricated.
    assert assessment.generator_table is None


def test_a_non_finite_diffusion_is_an_invalid_domain():
    class NotFinite(_UnitModel):
        @staticmethod
        def cam_diffusion(matrix, params=None):
            del params
            return np.full_like(np.asarray(matrix), np.nan, dtype=complex)

    assessment = scalar_reconstruction(
        _QuadraticAdapter(),
        NotFinite(),
        _observable(),
        state=[X0],
        params={},
        order=3,
        manifold=_confining_manifold(),
    )
    assert assessment.status is CAMOutcomeStatus.STOCHASTIC_INVALID
    assert assessment.reason_code == "invalid_domain"
    assert assessment.scalar_intensity is None


def test_a_model_without_diffusion_is_a_configuration_error():
    class NoDiffusion:
        params = {"mu": 0.0}

    with pytest.raises(ValueError, match="cam_diffusion"):
        scalar_reconstruction(
            _QuadraticAdapter(),
            NoDiffusion(),
            _observable(),
            state=[X0],
            params={},
            order=3,
        )


# --- postprocessor ---------------------------------------------------------


class _Patch:
    """Patch the adapter factory of the postprocessor module."""

    def __init__(self, monkeypatch, adapter):
        monkeypatch.setattr(
            "qphase_cam.postprocessor.nonlinear_reconstruction."
            "QPSGenDynamicsAdapter.from_model",
            lambda model: adapter,
        )


def _state_product(values, *, success=None):
    states = np.zeros((len(values), 1, 1, 1), dtype=complex)
    states[:, 0, 0, 0] = np.asarray(values, dtype=complex)
    count = len(values)
    return build_state_product(
        states=states,
        residuals=np.zeros((count, 1)),
        success=(
            np.ones((count, 1), dtype=bool)
            if success is None
            else np.asarray(success, dtype=bool).reshape(count, 1)
        ),
        valid_mask=np.ones((count, 1), dtype=bool),
        solution_count=np.ones(count, dtype=np.int64),
        parameters={"mu": np.asarray(values, dtype=float)},
    )


def _context(product, role="state"):
    return CAMAnnotationContext(
        model=_BoundedModel(), products={role: product}, source_role=role
    )


def test_the_postprocessor_emits_one_row_per_order(monkeypatch):
    _Patch(monkeypatch, _CubicAdapter())
    product = _state_product([X0])
    annotation = NonlinearReconstruction(
        orders=(3, 5),
        reconstruction=ScalarReconstructionConfig(grid_points=401),
    ).process(product, _context(product))
    assert annotation.attributes["annotation_kind"] == "nonlinear_reconstruction"
    assert annotation.attributes["target_kind"] == "state"
    assert np.asarray(annotation.handle("order").materialize()).tolist() == [[3, 5]]
    # One row per target with the orders on an axis: the target identity stays the
    # solution identity, and every order reports its own status.
    identities = np.asarray(annotation.handle("target_identity").materialize())
    assert len(identities) == 1
    assert list(np.asarray(annotation.handle("case_identity").materialize())) == [""]
    # A one-coordinate toy cannot confine its truncation, which is reported per
    # order instead of being hidden.
    assert np.asarray(annotation.handle("order_status").materialize()).tolist() == [
        ["uncontrolled_approximation", "uncontrolled_approximation"]
    ]
    assert list(np.asarray(annotation.handle("status").materialize())) == [
        "uncontrolled_approximation"
    ]
    assert np.asarray(annotation.handle("confining").materialize()).tolist() == [[0, 0]]
    assert np.all(
        np.isfinite(np.asarray(annotation.handle("scalar_intensity").materialize()))
    )
    # The Gaussian correction is reported next to the scalar intensity.
    reconstructed = np.asarray(
        annotation.handle("reconstructed_intensity").materialize()
    )
    assert np.all(np.isfinite(reconstructed))
    metadata = str(np.asarray(annotation.handle("method_metadata").materialize())[0])
    assert "scalar_reconstruction" in metadata
    assert "rayleigh_frequency" in metadata


def test_the_postprocessor_keeps_a_candidate_case_scope(monkeypatch):
    _Patch(monkeypatch, _CubicAdapter())
    product = build_candidate_product(
        states=np.asarray([[[X0]]], dtype=float),
        state_vectors=np.asarray([[X0]], dtype=float),
        control_values=np.empty((1, 0)),
        control_names=(),
        full_residual_norm=np.zeros(1),
        search_residual_norm=np.zeros(1),
        success=np.ones(1, dtype=bool),
        status=["verified"],
        method=["test"],
        verification_digits=np.full(1, 50, dtype=np.int64),
        verification_status=["verified"],
        candidate_identity=["cam:candidate:test"],
        case_metadata=[{"case_identity": "case:7"}],
    )
    annotation = NonlinearReconstruction(
        orders=(3,),
        reconstruction=ScalarReconstructionConfig(grid_points=401),
        observable=_observable(gain=2.0),
    ).process(product, _context(product, role="candidates"))
    assert annotation.attributes["target_kind"] == "candidate"
    assert list(np.asarray(annotation.handle("case_identity").materialize())) == [
        "case:7"
    ]
    assert str(np.asarray(annotation.handle("status").materialize())[0]) == (
        "uncontrolled_approximation"
    )
    assert np.asarray(annotation.handle("order_status").materialize()).tolist() == [
        ["uncontrolled_approximation"]
    ]


def test_an_unconverged_target_is_not_converged(monkeypatch):
    _Patch(monkeypatch, _CubicAdapter())
    product = _state_product([X0], success=[False])
    annotation = NonlinearReconstruction(
        orders=(3,), reconstruction=ScalarReconstructionConfig(grid_points=201)
    ).process(product, _context(product))
    assert list(np.asarray(annotation.handle("status").materialize())) == [
        "not_converged"
    ]
    assert np.isnan(np.asarray(annotation.handle("scalar_intensity").materialize())[0])


def test_the_annotation_round_trips_through_an_artifact(monkeypatch, tmp_path):
    _Patch(monkeypatch, _CubicAdapter())
    product = _state_product([X0])
    annotation = NonlinearReconstruction(
        orders=(3,), reconstruction=ScalarReconstructionConfig(grid_points=201)
    ).process(product, _context(product))
    cam_bundle({"nonlinear_reconstruction": annotation}).save_dataset(
        tmp_path, layout="single", shard_target_bytes=4096
    )
    restored = load_bundle(tmp_path).require("nonlinear_reconstruction")
    assert list(np.asarray(restored.handle("status").materialize())) == [
        "uncontrolled_approximation"
    ]
    assert np.isfinite(np.asarray(restored.handle("scalar_intensity").materialize())[0])
    assert np.asarray(restored.handle("order").materialize()).tolist() == [[3]]
    assert np.asarray(restored.handle("edge_mass_fraction").materialize()).shape == (
        1,
        1,
    )
