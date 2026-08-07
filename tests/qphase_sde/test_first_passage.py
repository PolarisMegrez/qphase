"""Tests for the online first-passage observer plugin."""

from types import SimpleNamespace

import numpy as np
import pytest
from qphase.backend.numpy_backend import NumpyBackend
from qphase.backend.xputil import get_xp
from qphase.core.scan import ScanSpec
from qphase_sde.analyser.psd import PsdAnalyzer
from qphase_sde.engine import Engine, EngineConfig
from qphase_sde.integrator.base import ChunkStepResult
from qphase_sde.integrator.euler_maruyama import EulerMaruyama
from qphase_sde.observer import (
    FirstPassageObserver,
    FirstPassageObserverConfig,
    FirstPassageTriggeredError,
)
from qphase_sde.planning import build_execution_plan

pytestmark = pytest.mark.integration


def _cupy_available() -> bool:
    try:
        import cupy as cp

        cp.cuda.runtime.getDevice()
        return True
    except Exception:
        return False


class LinearGrowthModel:
    """Deterministic per-mode linear growth: alpha_i(k) = alpha_i(0) + rate_i*k*dt."""

    name = "linear_growth"
    noise_basis = "real"

    def __init__(self, rates=(1.0,)):
        self.rates = tuple(float(rate) for rate in rates)
        self.n_modes = len(self.rates)
        self.noise_dim = len(self.rates)
        self.params: dict[str, object] = {}

    def drift(self, y, t, p):
        del t, p
        xp = get_xp(y)
        return xp.broadcast_to(xp.asarray(self.rates, dtype=y.dtype), y.shape)

    def diffusion(self, y, t, p):
        del t, p
        xp = get_xp(y)
        return xp.zeros(y.shape[:-1] + (self.n_modes, self.noise_dim), dtype=y.dtype)


class PulseDriftModel:
    """One-step blip at t in [0.5, 0.6) and sustained unit drift from t=1.0."""

    name = "pulse_drift"
    n_modes = 1
    noise_basis = "real"
    noise_dim = 1
    params: dict[str, object] = {}

    def drift(self, y, t, p):
        del p
        xp = get_xp(y)
        rate = 0.0
        if 0.5 - 1e-12 <= t < 0.6 - 1e-12:
            rate = 10.0
        elif 0.6 - 1e-12 <= t < 0.7 - 1e-12:
            rate = -10.0
        elif t >= 1.0 - 1e-12:
            rate = 1.0
        return xp.full(y.shape, rate, dtype=y.dtype)

    def diffusion(self, y, t, p):
        del t, p
        xp = get_xp(y)
        return xp.zeros(y.shape[:-1] + (1, 1), dtype=y.dtype)


class DecayScanModel:
    """Scanned exponential decay model: drift = -rate * y."""

    name = "fp_scan"
    n_modes = 1
    noise_basis = "real"
    noise_dim = 1

    def __init__(self):
        self._params = {"rate": 1.0}

    @property
    def params(self):
        return self._params

    def drift(self, y, t, params):
        del t
        return -np.asarray(params["rate"])[..., None] * y

    def diffusion(self, y, t, params):
        del t, params
        return np.zeros(y.shape + (1,))


class StochasticModel:
    """Ornstein-Uhlenbeck model for RNG-neutrality checks."""

    name = "stochastic"
    n_modes = 1
    noise_basis = "real"
    noise_dim = 1
    params: dict[str, object] = {}

    def drift(self, y, t, p):
        del t, p
        return -y

    def diffusion(self, y, t, p):
        del t, p
        xp = get_xp(y)
        return xp.ones(y.shape[:-1] + (1, 1), dtype=y.dtype)


class LinearChunkIntegrator:
    """Chunk-capable test integrator with exact linear dynamics y += k*dt."""

    def __init__(self, chunk_steps):
        self.config = SimpleNamespace(chunk_steps=chunk_steps)

    def supports_chunk_step(self, model, backend):
        del model, backend
        return True

    def step_chunk(
        self,
        y,
        t,
        dt,
        model,
        noise,
        backend,
        *,
        n_steps,
        save_offsets,
        record_modes,
    ):
        del t, model, noise, backend, record_modes
        if save_offsets:
            saved = np.stack([y + offset * dt for offset in save_offsets], axis=1)
        else:
            saved = np.empty((y.shape[0], 0, y.shape[1]), dtype=y.dtype)
        return ChunkStepResult(final_state=y + n_steps * dt, saved_states=saved)


def _observer(**overrides):
    config = {
        "rule": "state_norm",
        "threshold": 0.45,
        "direction": "above",
        "check_interval_steps": 3,
    }
    config.update(overrides)
    return FirstPassageObserver(**config)


def _engine(model, observer, *, integrator=None, n_traj=2, ic=None, **config):
    config.setdefault("dt", 0.1)
    config.setdefault("t0", 0.0)
    config.setdefault("t1", 2.0)
    config.setdefault("seed", 7)
    plugins = {
        "backend": NumpyBackend(),
        "integrator": integrator or EulerMaruyama(),
        "model": model,
    }
    if observer is not None:
        plugins["observer"] = observer if isinstance(observer, dict) else {
            "first_passage": observer
        }
    return Engine(
        config=EngineConfig(n_traj=n_traj, ic=ic, **config),
        plugins=plugins,
    )


def _growth_payload(observer=None, **kwargs):
    observer = observer if observer is not None else _observer()
    result = _engine(LinearGrowthModel(), observer, ic=[[0.0]], **kwargs).run()
    return result.analysis["first_passage"], result


def test_state_norm_hit_records_first_passage():
    payload, result = _growth_payload()

    np.testing.assert_array_equal(payload["hit"], [True, True])
    np.testing.assert_array_equal(payload["censored"], [False, False])
    np.testing.assert_array_equal(payload["first_hit_step"], [6, 6])
    np.testing.assert_allclose(payload["first_hit_time"], [0.6, 0.6])
    # The true crossing (t=0.45) is bracketed by the checks around the hit.
    np.testing.assert_allclose(payload["value_before_hit"], [0.3, 0.3])
    np.testing.assert_allclose(payload["value_at_hit"], [0.6, 0.6])
    assert payload["value_before_hit"][0] < 0.45 <= payload["value_at_hit"][0]
    np.testing.assert_allclose(payload["effective_end_time"], [2.0, 2.0])
    assert payload["n_hit"] == 2
    assert payload["n_censored"] == 0
    assert payload["observable"] == "state_norm"
    assert payload["time_unit"] == "seconds"
    # record-only observers keep the full trajectory.
    assert result.trajectory.data.shape == (2, 21, 1)


def test_state_norm_censored_when_threshold_never_crossed():
    payload, _ = _growth_payload(_observer(threshold=5.0))

    np.testing.assert_array_equal(payload["hit"], [False, False])
    np.testing.assert_array_equal(payload["censored"], [True, True])
    np.testing.assert_array_equal(payload["first_hit_step"], [-1, -1])
    assert np.all(np.isnan(payload["first_hit_time"]))
    assert np.all(np.isnan(payload["value_before_hit"]))
    assert np.all(np.isnan(payload["value_at_hit"]))
    # Right-censoring keeps the finite effective end time, never inf or zero.
    np.testing.assert_allclose(payload["effective_end_time"], [2.0, 2.0])
    assert payload["n_hit"] == 0
    assert payload["n_censored"] == 2


def test_direction_below_with_negative_rate():
    observer = FirstPassageObserver(
        rule="state_norm",
        threshold=0.75,
        direction="below",
        check_interval_steps=3,
    )
    payload, _ = _growth_payload_with_model(
        LinearGrowthModel(rates=(-1.0,)), observer
    )

    # alpha(k) = 1 - 0.1*k; checks at k=0 (1.0) and k=3 (0.7 < 0.75).
    np.testing.assert_array_equal(payload["first_hit_step"], [3, 3])
    np.testing.assert_allclose(payload["value_before_hit"], [1.0, 1.0])
    np.testing.assert_allclose(payload["value_at_hit"], [0.7, 0.7])


def _growth_payload_with_model(model, observer, **kwargs):
    kwargs.setdefault("ic", [[1.0]])
    result = _engine(model, observer, **kwargs).run()
    return result.analysis["first_passage"], result


def test_mode_magnitude_outside_and_inside():
    model = LinearGrowthModel(rates=(1.0, 0.0))
    outside = FirstPassageObserver(
        rule="mode_magnitude",
        mode=0,
        upper=0.45,
        direction="outside",
        check_interval_steps=3,
    )
    payload, _ = _growth_payload_with_model(model, outside, ic=[[0.0, 5.0]])
    # |alpha_0|(k) = 0.1*k leaves the [.., 0.45] interval at k=6 (0.6 > 0.45).
    np.testing.assert_array_equal(payload["first_hit_step"], [6, 6])
    np.testing.assert_allclose(payload["value_at_hit"], [0.6, 0.6])
    assert payload["observable"] == "|alpha_0|"

    inside = FirstPassageObserver(
        rule="mode_magnitude",
        mode=1,
        lower=4.0,
        upper=6.0,
        direction="inside",
        check_interval_steps=3,
    )
    payload, _ = _growth_payload_with_model(model, inside, ic=[[0.0, 5.0]])
    # |alpha_1| = 5.0 is inside [4, 6] from the initial condition check.
    np.testing.assert_array_equal(payload["first_hit_step"], [0, 0])
    np.testing.assert_allclose(payload["first_hit_time"], [0.0, 0.0])
    np.testing.assert_allclose(payload["value_at_hit"], [5.0, 5.0])


def test_mode_magnitude_requires_a_bound():
    with pytest.raises(ValueError, match="lower"):
        FirstPassageObserver(
            rule="mode_magnitude",
            mode=0,
            direction="outside",
            check_interval_steps=1,
        )


def test_linear_projection_weights():
    model = LinearGrowthModel(rates=(1.0, 0.0))
    observer = FirstPassageObserver(
        rule="linear_projection",
        weights=[2.0, 1.0],
        component="real",
        threshold=0.9,
        direction="above",
        check_interval_steps=2,
    )
    payload, _ = _growth_payload_with_model(model, observer, ic=[[0.0, 0.5]])

    # Re(w . alpha)(k) = 2*(0.1*k) + 0.5; checks at even steps:
    # k=0: 0.5, k=2: 0.9 (not > 0.9), k=4: 1.3 confirms.
    np.testing.assert_array_equal(payload["first_hit_step"], [4, 4])
    np.testing.assert_allclose(payload["value_before_hit"], [0.9, 0.9])
    np.testing.assert_allclose(payload["value_at_hit"], [1.3, 1.3])
    assert payload["observable"] == "real(w . alpha)"


def test_linear_projection_validates_weights_against_n_modes():
    observer = FirstPassageObserver(
        rule="linear_projection",
        weights=[1.0, 1.0],
        threshold=1.0,
        direction="above",
        check_interval_steps=1,
    )
    with pytest.raises(ValueError, match="n_modes=1"):
        _growth_payload(observer)


def test_matrix_projection_population_with_reference():
    model = LinearGrowthModel(rates=(1.0, 0.0))
    observer = FirstPassageObserver(
        rule="matrix_projection",
        left_vector=[1.0, 0.0, 0.0, 0.0],
        reference=[0.25, 0.0, 0.0, 0.0],
        threshold=0.0,
        direction="above",
        check_interval_steps=3,
    )
    payload, _ = _growth_payload_with_model(model, observer, ic=[[0.0, 0.0]])

    # c(k) = |alpha_0|^2 - 0.25 = 0.01*k^2 - 0.25; checks at k=0 (-0.25),
    # k=3 (-0.16), k=6 (0.11) confirms.
    np.testing.assert_array_equal(payload["first_hit_step"], [6, 6])
    np.testing.assert_allclose(payload["value_before_hit"], [-0.16, -0.16])
    np.testing.assert_allclose(payload["value_at_hit"], [0.11, 0.11])
    assert (
        payload["observable"] == "left_vector . (vec(R) - vec(reference))"
    )
    np.testing.assert_allclose(
        payload["left_vector"], [1.0, 0.0, 0.0, 0.0]
    )
    np.testing.assert_allclose(
        payload["reference_vector"], [0.25, 0.0, 0.0, 0.0]
    )


def test_matrix_projection_requires_perfect_square_vectors():
    with pytest.raises(ValueError, match="perfect square"):
        FirstPassageObserver(
            rule="matrix_projection",
            left_vector=[1.0, 0.0, 0.0],
            threshold=0.0,
            direction="above",
            check_interval_steps=1,
        )


def test_debounce_rejects_single_check_blip():
    observer = FirstPassageObserver(
        rule="state_norm",
        threshold=0.15,
        direction="above",
        check_interval_steps=1,
        debounce_checks=2,
    )
    payload, _ = _growth_payload_with_model(
        PulseDriftModel(), observer, ic=[[0.0]]
    )

    # The one-check blip at k=6 never confirms; the sustained drift confirms
    # at k=13 with the first-hit time of the run start at k=12.
    np.testing.assert_array_equal(payload["first_hit_step"], [12, 12])
    np.testing.assert_allclose(
        payload["first_hit_time"], [1.2, 1.2], atol=1e-12
    )
    np.testing.assert_allclose(payload["value_before_hit"], [0.1, 0.1])
    np.testing.assert_allclose(payload["value_at_hit"], [0.3, 0.3])


def test_debounce_one_confirms_the_blip():
    observer = FirstPassageObserver(
        rule="state_norm",
        threshold=0.15,
        direction="above",
        check_interval_steps=1,
        debounce_checks=1,
    )
    payload, _ = _growth_payload_with_model(
        PulseDriftModel(), observer, ic=[[0.0]]
    )

    np.testing.assert_array_equal(payload["first_hit_step"], [6, 6])
    np.testing.assert_allclose(payload["value_at_hit"], [1.0, 1.0])


def test_multiple_observers_compose_mixed_cadences():
    fast = FirstPassageObserver(
        rule="state_norm",
        threshold=0.25,
        direction="above",
        check_interval_steps=2,
    )
    slow = FirstPassageObserver(
        rule="state_norm",
        threshold=0.45,
        direction="above",
        check_interval_steps=3,
    )
    result = _engine(
        LinearGrowthModel(),
        {"fast": fast, "slow": slow},
        ic=[[0.0]],
    ).run()

    # Each observer fires exactly on its own cadence multiples.
    np.testing.assert_array_equal(
        result.analysis["fast"]["first_hit_step"], [4, 4]
    )
    np.testing.assert_array_equal(
        result.analysis["slow"]["first_hit_step"], [6, 6]
    )


def test_record_observer_preserves_trajectories_bitwise():
    config = dict(dt=0.01, t1=0.5, n_traj=4, seed=123, ic=[[0.5]])
    reference = _engine(StochasticModel(), None, **config).run()
    no_hit = _engine(
        StochasticModel(),
        _observer(threshold=100.0, check_interval_steps=1),
        **config,
    ).run()
    immediate_hit = _engine(
        StochasticModel(),
        _observer(threshold=0.0, check_interval_steps=1),
        **config,
    ).run()

    np.testing.assert_array_equal(
        reference.trajectory.data, no_hit.trajectory.data
    )
    np.testing.assert_array_equal(
        reference.trajectory.data, immediate_hit.trajectory.data
    )
    payload = immediate_hit.analysis["first_passage"]
    np.testing.assert_array_equal(payload["first_hit_step"], [0, 0, 0, 0])


def test_chunked_and_stepwise_hits_match():
    stepwise_payload, _ = _growth_payload()
    chunked_payload, _ = _growth_payload(
        integrator=LinearChunkIntegrator(chunk_steps=8)
    )

    # Cadence 3 is not a divisor of chunk_steps 8; chunk clamping must keep
    # the checks on exact cadence multiples (6 rather than the boundary 8).
    np.testing.assert_array_equal(
        chunked_payload["hit"], stepwise_payload["hit"]
    )
    np.testing.assert_array_equal(
        chunked_payload["censored"], stepwise_payload["censored"]
    )
    np.testing.assert_array_equal(
        chunked_payload["first_hit_step"], stepwise_payload["first_hit_step"]
    )
    np.testing.assert_allclose(
        chunked_payload["first_hit_time"],
        stepwise_payload["first_hit_time"],
        atol=1e-12,
    )


def test_stop_batch_truncates_output():
    payload, result = _growth_payload(_observer(action="stop_batch"))

    assert result.trajectory.data.shape == (2, 7, 1)
    meta = result.trajectory.meta
    assert meta["stopped_early"] is True
    assert meta["stop_reason"] == "observer:first_passage"
    assert meta["effective_steps"] == 6
    np.testing.assert_array_equal(payload["hit"], [True, True])
    np.testing.assert_allclose(payload["effective_end_time"], [0.6, 0.6])


def test_fail_job_raises_with_trajectory_ids_and_times():
    engine = _engine(
        LinearGrowthModel(), _observer(action="fail_job"), ic=[[0.0]]
    )

    with pytest.raises(FirstPassageTriggeredError) as excinfo:
        engine.run()

    message = str(excinfo.value)
    assert "first_passage" in message
    assert "state_norm" in message
    assert "[0, 1]" in message
    payload = excinfo.value.payload
    assert payload["step"] == 6
    assert payload["time"] == pytest.approx(0.6)
    details = payload["observers"]["first_passage"]
    assert details["rule"] == "state_norm"
    assert details["hit_trajectories"] == [0, 1]
    assert details["first_hit_times"] == pytest.approx([0.6, 0.6])


@pytest.mark.skipif(not _cupy_available(), reason="CuPy not available")
def test_cupy_matches_numpy_hits():
    from qphase.backend.cupy_backend import CuPyBackend

    def run(backend):
        observer = FirstPassageObserver(
            rule="matrix_projection",
            left_vector=[1.0, 0.0, 0.0, 0.0],
            reference=[0.25, 0.0, 0.0, 0.0],
            threshold=0.0,
            direction="above",
            check_interval_steps=3,
        )
        engine = Engine(
            config=EngineConfig(
                dt=0.1, t0=0.0, t1=2.0, n_traj=2, seed=7, ic=[[0.0, 0.0]]
            ),
            plugins={
                "backend": backend,
                "integrator": EulerMaruyama(),
                "model": LinearGrowthModel(rates=(1.0, 0.0)),
                "observer": {"first_passage": observer},
            },
        )
        return engine.run().analysis["first_passage"]

    numpy_payload = run(NumpyBackend())
    cupy_payload = run(CuPyBackend())

    np.testing.assert_array_equal(cupy_payload["hit"], numpy_payload["hit"])
    np.testing.assert_array_equal(
        cupy_payload["censored"], numpy_payload["censored"]
    )
    np.testing.assert_array_equal(
        cupy_payload["first_hit_step"], numpy_payload["first_hit_step"]
    )
    np.testing.assert_allclose(
        cupy_payload["first_hit_time"],
        numpy_payload["first_hit_time"],
        atol=1e-12,
    )


def test_trajectory_batching_merges_observer_payloads():
    # Dyadic increments keep every check value exact in binary floating point.
    ic = [[0.0625 * index] for index in range(8)]

    def run(**config):
        config.setdefault("dt", 0.125)
        engine = _engine(
            LinearGrowthModel(),
            _observer(check_interval_steps=2),
            n_traj=8,
            ic=ic,
            **config,
        )
        engine.plugins["analyser"] = {
            "psd": PsdAnalyzer(kind="complex", modes=[0])
        }
        return engine.run()

    reference = run(keep_traj=False).analysis["first_passage"]
    merged = run(keep_traj=False, trajectory_batch_size=3).analysis[
        "first_passage"
    ]

    # Global trajectory ids are the row index: 8 rows, no renumbering.
    assert merged["hit"].shape == (8,)
    assert merged["n_traj"] == 8
    np.testing.assert_array_equal(merged["hit"], reference["hit"])
    np.testing.assert_array_equal(
        merged["first_hit_step"], reference["first_hit_step"]
    )
    np.testing.assert_allclose(
        merged["first_hit_time"], reference["first_hit_time"]
    )
    # threshold 0.45 on 0.0625*i + 0.125*k, checked at even steps.
    np.testing.assert_array_equal(
        merged["first_hit_step"], [4, 4, 4, 4, 2, 2, 2, 2]
    )
    assert merged["n_hit"] == int(np.count_nonzero(reference["hit"]))


def test_scan_fused_payload_keeps_full_trajectory_axis():
    grid = ScanSpec.model_validate(
        {
            "axes": {
                "rate": {
                    "target": "model.fp_scan.rate",
                    "values": [1.0, 2.0],
                }
            }
        }
    ).compile()
    observer = FirstPassageObserver(
        rule="state_norm",
        threshold=0.9,
        direction="below",
        check_interval_steps=1,
    )
    engine = Engine(
        config=EngineConfig(
            t0=0.0, t1=0.2, dt=0.01, n_traj=2, seed=7, ic=[[1.0]]
        ),
        plugins={
            "backend": NumpyBackend(),
            "integrator": EulerMaruyama(),
            "model": DecayScanModel(),
            "observer": {"first_passage": observer},
        },
    )

    result = engine.run(context=SimpleNamespace(parameter_grid=grid, progress=None))

    payload = result.combined.analysis["first_passage"]
    assert payload["n_traj"] == 4
    assert payload["hit"].shape == (4,)
    # rate=1 crosses 0.9 at k=11 (0.99^k), rate=2 at k=6 (0.98^k).
    np.testing.assert_array_equal(
        payload["first_hit_step"], [11, 11, 6, 6]
    )


def test_planner_reports_observer_cadence_cost():
    class FakeBackend:
        config = SimpleNamespace(float_dtype="float64")

        def backend_name(self):
            return "numpy"

    class FakeModel:
        name = "fake"
        n_modes = 2
        noise_dim = 4

    config = EngineConfig(t0=0.0, t1=1.0, dt=0.1, n_traj=2)
    integrator = SimpleNamespace(config=SimpleNamespace(chunk_steps=8))

    plan = build_execution_plan(
        config=config,
        grid=None,
        model=FakeModel(),
        backend=FakeBackend(),
        integrator=integrator,
        analysers={},
        observers={"first_passage": SimpleNamespace(check_interval_steps=3)},
        resources=None,
    )
    assert plan.observer_check_interval_steps == 3
    assert plan.effective_chunk_steps == 3
    assert plan.to_dict()["observer_check_interval_steps"] == 3
    assert plan.to_dict()["effective_chunk_steps"] == 3

    no_observers = build_execution_plan(
        config=config,
        grid=None,
        model=FakeModel(),
        backend=FakeBackend(),
        integrator=integrator,
        analysers={},
        resources=None,
    )
    assert no_observers.observer_check_interval_steps is None
    assert no_observers.effective_chunk_steps is None
    assert "observer_check_interval_steps" not in no_observers.to_dict()
    assert "effective_chunk_steps" not in no_observers.to_dict()

    unchunked = build_execution_plan(
        config=config,
        grid=None,
        model=FakeModel(),
        backend=FakeBackend(),
        integrator=SimpleNamespace(config=SimpleNamespace(chunk_steps=1)),
        analysers={},
        observers={"first_passage": SimpleNamespace(check_interval_steps=3)},
        resources=None,
    )
    assert unchunked.observer_check_interval_steps == 3
    assert unchunked.effective_chunk_steps is None


def test_observer_plugin_is_discoverable():
    from qphase.core.registry import discovery, registry

    discovery.discover_plugins()
    discovered = registry.get_plugin_class("observer", "first_passage")
    assert discovered is FirstPassageObserver
    assert "observer" in Engine.manifest.optional_plugins


def test_config_validation_rejects_incomplete_rules():
    with pytest.raises(ValueError, match="threshold"):
        FirstPassageObserverConfig(
            rule="state_norm", direction="above", check_interval_steps=1
        )
    with pytest.raises(ValueError, match="direction"):
        FirstPassageObserverConfig(
            rule="mode_magnitude",
            mode=0,
            lower=1.0,
            direction="above",
            check_interval_steps=1,
        )
    with pytest.raises(ValueError, match="weights"):
        FirstPassageObserverConfig(
            rule="linear_projection",
            threshold=1.0,
            direction="above",
            check_interval_steps=1,
        )
    with pytest.raises(ValueError, match="left_vector"):
        FirstPassageObserverConfig(
            rule="matrix_projection",
            threshold=1.0,
            direction="above",
            check_interval_steps=1,
        )
    with pytest.raises(ValueError, match="debounce|greater than or equal"):
        FirstPassageObserverConfig(
            rule="state_norm",
            threshold=1.0,
            direction="above",
            check_interval_steps=1,
            debounce_checks=0,
        )
