"""Phase 10G2d1c1: the shared closure state reused by several readouts.

The fixture is the one-mode toy of the closure tests: ``J = -1 - 3x²``, a unit noise
covariance and a quadratic readout, so the numbers are closed forms and the point of
the test is *how many times* the state-level work runs, not what it computes.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from qphase_cam.closure import (
    ClosureState,
    assess_closure,
    closure_state,
)
from qphase_cam.outcomes import CAMOutcomeStatus
from qphase_cam.validity import assess_stochastic_state

from tests.qphase_cam.test_closure import (
    X0,
    _scalar_observable,
    _ScalarAdapter,
)

DIFFUSION = np.asarray([[1.0]])


def _paired_readouts():
    """Two readouts of the same state with different gradients and curvatures."""
    return _scalar_observable(curvature=2.0), _scalar_observable(
        curvature=-1.0, gradient_scale=0.5
    )


@pytest.fixture
def counters(monkeypatch):
    """Count the state-level work in both the closure pass and the readout pass."""
    import qphase_cam.closure as closure
    import qphase_cam.validity as validity

    counts: dict[str, Any] = {
        "closure_mode": 0,
        "closure_lyapunov": 0,
        "closure_contraction": 0,
        "readout_mode": 0,
        "readout_calls": 0,
    }
    precomputed: list[dict[str, bool]] = []
    original_mode = closure.critical_mode
    original_lyapunov = closure.lyapunov_covariance
    original_contraction = closure.closure_contraction
    original_validity_mode = validity.critical_mode

    def counted(name, function):
        def wrapper(*args, **kwargs):
            counts[name] += 1
            return function(*args, **kwargs)

        return wrapper

    monkeypatch.setattr(
        closure, "critical_mode", counted("closure_mode", original_mode)
    )
    monkeypatch.setattr(
        closure, "lyapunov_covariance", counted("closure_lyapunov", original_lyapunov)
    )
    monkeypatch.setattr(
        closure,
        "closure_contraction",
        counted("closure_contraction", original_contraction),
    )
    monkeypatch.setattr(
        validity, "critical_mode", counted("readout_mode", original_validity_mode)
    )

    def counted_readout(*args, **kwargs):
        counts["readout_calls"] += 1
        precomputed.append(
            {
                "covariance": kwargs.get("stationary_covariance") is not None,
                "mode": kwargs.get("mode") is not None,
            }
        )
        return original_readout(*args, **kwargs)

    original_readout = validity.assess_stochastic_state
    monkeypatch.setattr(validity, "assess_stochastic_state", counted_readout)
    monkeypatch.setattr(
        "qphase_cam.closure.assess_stochastic_state", counted_readout
    )
    return counts, precomputed


def test_one_state_pass_serves_every_readout(counters):
    """Three readouts cost one mode, one Lyapunov solve and one contraction."""
    counts, precomputed = counters
    adapter = _ScalarAdapter()
    first, second = _paired_readouts()
    context = closure_state(adapter, state=[X0], params={}, diffusion=DIFFUSION)
    assert isinstance(context, ClosureState)
    assert context.status is CAMOutcomeStatus.VALID
    assert counts["closure_mode"] == 1
    assert counts["closure_lyapunov"] == 1
    assert counts["closure_contraction"] == 1
    for observable in (first, second, _scalar_observable(curvature=0.5)):
        assessment = assess_closure(
            adapter,
            observable,
            state=[X0],
            params={},
            diffusion=DIFFUSION,
            state_context=context,
        )
        assert assessment.status is CAMOutcomeStatus.VALID
    # Nothing state-level ran again, in either layer.
    assert counts["closure_mode"] == 1
    assert counts["closure_lyapunov"] == 1
    assert counts["closure_contraction"] == 1
    # The readout layer resolved no mode of its own ...
    assert counts["readout_mode"] == 0
    # ... and every readout assessment was handed both precomputed inputs.
    assert counts["readout_calls"] == 3
    assert all(item == {"covariance": True, "mode": True} for item in precomputed)


def test_the_shared_path_is_the_fresh_path():
    """The reused context and a fresh assessment agree field by field."""
    adapter = _ScalarAdapter()
    context = closure_state(adapter, state=[X0], params={}, diffusion=DIFFUSION)
    for observable in _paired_readouts():
        fresh = assess_closure(
            adapter, observable, state=[X0], params={}, diffusion=DIFFUSION
        )
        shared = assess_closure(
            adapter,
            observable,
            state=[X0],
            params={},
            diffusion=DIFFUSION,
            state_context=context,
        )
        assert shared.status is fresh.status
        for name in (
            "closure_contraction_norm",
            "closure_displacement_norm",
            "drift_bias",
            "jensen_bias",
            "closure_bias_total",
            "closure_shape_fraction",
            "soft_mode_rate",
            "soft_mode_sigma",
            "soft_sigma",
            "soft_kubo",
            "full_kubo",
            "jacobian_condition_number",
            "covariance_condition_number",
            "physical_minimum_eigenvalue",
            "validity_domain",
            "physical_is_psd",
        ):
            assert getattr(shared, name) == getattr(fresh, name), name
        assert shared.closure_displacement == fresh.closure_displacement
        assert shared.stochastic.projected_variance == pytest.approx(
            fresh.stochastic.projected_variance, rel=1e-15
        )


def test_each_readout_keeps_its_own_numbers():
    """Sharing the state does not share the readout-specific quantities."""
    adapter = _ScalarAdapter()
    context = closure_state(adapter, state=[X0], params={}, diffusion=DIFFUSION)
    first, second = _paired_readouts()
    left = assess_closure(
        adapter,
        first,
        state=[X0],
        params={},
        diffusion=DIFFUSION,
        state_context=context,
    )
    right = assess_closure(
        adapter,
        second,
        state=[X0],
        params={},
        diffusion=DIFFUSION,
        state_context=context,
    )
    # The state-level quantities are identical ...
    assert left.soft_mode_rate == right.soft_mode_rate
    assert left.closure_displacement == right.closure_displacement
    assert left.stochastic.lyapunov_covariance_trace == (
        right.stochastic.lyapunov_covariance_trace
    )
    # ... and every readout-specific one is not.
    assert left.drift_bias != right.drift_bias
    assert left.jensen_bias != right.jensen_bias
    assert left.soft_kubo != right.soft_kubo
    assert left.stochastic.observable_gradient != right.stochastic.observable_gradient
    # The Kubo regime is readout-specific too, so the shared path cannot decide it once.
    for name in ("soft_kubo", "full_kubo"):
        value = getattr(left, name)
        assert value is not None
    assert left.validity_domain in {"weak_noise", "noise_dominated", "unresolved"}


def test_a_state_level_failure_is_the_same_for_every_readout():
    """A failed state pass fails every readout, with no numbers invented on top."""
    adapter = _ScalarAdapter()
    context = closure_state(
        adapter, state=[X0], params={}, diffusion=np.asarray([[-1.0]])
    )
    assert context.status is CAMOutcomeStatus.STOCHASTIC_INVALID
    for observable in _paired_readouts():
        assessment = assess_closure(
            adapter,
            observable,
            state=[X0],
            params={},
            diffusion=np.asarray([[-1.0]]),
            state_context=context,
        )
        assert assessment.status is CAMOutcomeStatus.STOCHASTIC_INVALID
        assert "positive semidefinite" in (assessment.reason or "")
        assert assessment.closure_contraction_norm is None
        assert assessment.drift_bias is None
        assert assessment.jensen_bias is None


def test_a_mismatched_state_context_is_refused():
    adapter = _ScalarAdapter()
    context = closure_state(adapter, state=[X0], params={}, diffusion=DIFFUSION)
    with pytest.raises(ValueError, match="coordinates|shape"):
        assess_closure(
            adapter,
            _scalar_observable(),
            state=[X0, 0.1],
            params={},
            diffusion=np.eye(2),
            state_context=context,
        )


def test_a_precomputed_readout_input_is_verified_not_trusted():
    """The precomputed covariance and mode are checked before they are used."""
    adapter = _ScalarAdapter()
    observable = _scalar_observable()
    mode = __import__(
        "qphase_cam.validity", fromlist=["critical_mode"]
    ).critical_mode(adapter.jacobian(np.asarray([X0])), rule="slowest")
    assert mode is not None
    with pytest.raises(ValueError, match="shape"):
        assess_stochastic_state(
            adapter,
            observable,
            state=[X0],
            params={},
            diffusion=DIFFUSION,
            stationary_covariance=np.eye(2),
            mode=mode,
        )
    # A covariance that does not solve the Lyapunov equation is rejected.
    wrong = assess_stochastic_state(
        adapter,
        observable,
        state=[X0],
        params={},
        diffusion=DIFFUSION,
        stationary_covariance=np.zeros((1, 1)),
        mode=mode,
    )
    assert wrong.status is CAMOutcomeStatus.STOCHASTIC_INVALID
    assert "Lyapunov" in (wrong.reason or "")
    assert wrong.method["lyapunov_source"] == "precomputed"
    # A finite but inconsistent covariance is caught by the residual, and the honest
    # one passes and records that it was precomputed.
    context = closure_state(adapter, state=[X0], params={}, diffusion=DIFFUSION)
    good = assess_stochastic_state(
        adapter,
        observable,
        state=[X0],
        params={},
        diffusion=DIFFUSION,
        stationary_covariance=context.stationary_covariance,
        mode=context.mode,
    )
    assert good.status is CAMOutcomeStatus.VALID
    assert good.method["lyapunov_source"] == "precomputed"
    assert good.method["critical_mode_source"] == "precomputed"
    assert good.lyapunov_relative_residual < 1.0e-12


def test_the_critical_mode_rule_is_explicit():
    """The historical rule is selected by config, not by a global default change."""
    adapter = _ScalarAdapter()
    observable = _scalar_observable()
    default = assess_stochastic_state(
        adapter, observable, state=[X0], params={}, diffusion=DIFFUSION
    )
    assert default.method["critical_mode_rule"] == "slowest"
    historical = assess_stochastic_state(
        adapter,
        observable,
        state=[X0],
        params={},
        diffusion=DIFFUSION,
        config=__import__(
            "qphase_cam.validity", fromlist=["StochasticAssessmentConfig"]
        ).StochasticAssessmentConfig(critical_mode_rule="nearest_zero"),
    )
    assert historical.method["critical_mode_rule"] == "nearest_zero"
