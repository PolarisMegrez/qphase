"""CAM solver, engine, capacity, postprocessing, and persistence tests."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from qphase.backend.numpy_backend import NumpyBackend
from qphase.data import load_bundle
from qphase_cam.engine import Engine
from qphase_cam.errors import SolutionCapacityError
from qphase_cam.postprocessor.base import CAMAnnotationContext
from qphase_cam.postprocessor.frequency import (
    HamiltonianSpectrum,
    RayleighFrequency,
)
from qphase_cam.postprocessor.jacobian import JacobianSpectrum
from qphase_cam.postprocessor.physicality import Physicality
from qphase_cam.solver.base import (
    SOLVER_OUTPUT_DECLARATIONS,
    CAMProductFragment,
    CAMSolver,
    CAMSolverConfig,
    CAMSolveRequest,
)
from qphase_cam.solver.steady_state import SteadyStateSolver
from qphase_cam.state import CAMSolution, CAMSolverOutput

from models.vdp_2mode import VDP2ModeModel


def _values(product, name):
    return np.asarray(product.handle(name).materialize())


def test_no_jacobian_model_solves_without_jacobian(no_jacobian_model):
    bundle = Engine(
        plugins={
            "backend": NumpyBackend(),
            "model": no_jacobian_model,
            "cam_solver": SteadyStateSolver(method="root"),
        }
    ).run()
    assert isinstance(bundle.products["state"].kind.value, str)
    product = bundle.require("state")
    assert _values(product, "success")[0, 0]
    np.testing.assert_allclose(_values(product, "states")[0, 0], np.eye(2), atol=1e-8)
    # The engine produces typed products, never a result object.
    assert not hasattr(bundle, "postprocess")
    assert not hasattr(bundle, "states")


def test_vdp_engine_postprocess_and_round_trip(tmp_path):
    model = VDP2ModeModel(
        omega_a=0.0,
        omega_b=0.0,
        gamma_a=2.0,
        gamma_b=0.5,
        Gamma=0.0001,
        g=0.5,
    )
    bundle = Engine(
        plugins={
            "backend": NumpyBackend(),
            "model": model,
            "cam_solver": SteadyStateSolver(
                method="cholesky",
                initial_guess=[[20000.0, 20000.0j], [-20000.0j, 20000.0]],
                tolerance=1e-8,
            ),
            "cam_postprocessor": {
                "rayleigh": RayleighFrequency(),
                "spectrum": HamiltonianSpectrum(),
                "jacobian": JacobianSpectrum(),
                "physicality": Physicality(),
            },
        }
    ).run()
    state = bundle.require("state")
    np.testing.assert_array_equal(_values(state, "solution_count"), [1])
    assert _values(state, "success")[0, 0]

    # Every postprocessor produced its own typed annotation product over the
    # solution identity of the state product.
    target = _values(state, "solution_identity")[0, 0]
    assert str(target).startswith("cam:solution:")
    for kind in (
        "rayleigh_frequency",
        "hamiltonian_spectrum",
        "jacobian_spectrum",
        "physicality",
    ):
        annotation = bundle.require(kind)
        assert annotation.attributes["annotation_kind"] == kind
        np.testing.assert_array_equal(_values(annotation, "target_identity"), [target])
        assert list(_values(annotation, "status")) == ["valid"]

    assert _values(bundle.require("jacobian_spectrum"), "jacobian_source")[0] == (
        "analytic"
    )
    assert _values(bundle.require("physicality"), "is_hermitian")[0]

    # Behaviour freeze (Phase 10A): the steady state this model/solver/guess
    # combination converges to, recorded from the implementation rather than
    # derived from the equations. The Cholesky solver has no free parameters
    # left once the guess is fixed, so these values may only change if the
    # solver or the model changes.
    np.testing.assert_allclose(
        _values(state, "states")[0, 0],
        np.asarray(
            [
                [7503.443592415266 + 0.0j, 0.0 + 3750.555313355958j],
                [0.0 - 3750.555313355958j, 7501.610626711914 + 0.0j],
            ]
        ),
        rtol=1e-9,
        atol=1e-9,
    )
    assert float(_values(state, "residuals")[0, 0]) < 1e-10
    # Two-mode spectrum at the frozen state (Hamiltonian postprocessor).
    np.testing.assert_allclose(
        _values(bundle.require("hamiltonian_spectrum"), "mode_frequencies")[0],
        np.asarray([0.43308321934995564, -0.43308321934995586]),
        rtol=1e-9,
    )

    target_path = tmp_path / "cam_bundle"
    bundle.save_dataset(target_path, layout="sharded", shard_target_bytes=4096)
    loaded = load_bundle(target_path)
    assert set(loaded.products) == set(bundle.products)
    np.testing.assert_allclose(
        _values(loaded.require("state"), "states"),
        _values(state, "states"),
        equal_nan=True,
    )


def test_rayleigh_zero_trace_has_no_division_warning(no_jacobian_model):
    from qphase_cam.data import build_state_product

    states = np.zeros((1, 1, 2, 2), dtype=complex)
    payload = {
        "states": states,
        "residuals": np.zeros((1, 1)),
        "success": np.ones((1, 1), dtype=bool),
        "valid_mask": np.ones((1, 1), dtype=bool),
        "solution_count": np.ones(1, dtype=np.int64),
    }
    product = build_state_product(**payload)
    with warnings.catch_warnings(record=True) as caught:
        annotation = RayleighFrequency().process(
            product,
            CAMAnnotationContext(model=no_jacobian_model, backend=NumpyBackend()),
        )
    assert np.isnan(
        np.asarray(annotation.handle("rayleigh_frequency").materialize())[0]
    )
    assert not caught


def test_finite_difference_jacobian_records_source_and_step(no_jacobian_model):
    bundle = Engine(
        plugins={
            "backend": NumpyBackend(),
            "model": no_jacobian_model,
            "cam_solver": SteadyStateSolver(method="root"),
            "cam_postprocessor": JacobianSpectrum(
                allow_finite_difference=True,
                finite_difference_epsilon=2e-6,
            ),
        }
    ).run()
    annotation = bundle.require("jacobian_spectrum")
    assert np.asarray(annotation.handle("jacobian_source").materialize())[0] == (
        "finite_difference"
    )
    metadata = bundle.metadata["annotation_metadata"]["jacobian_spectrum"]
    assert metadata["jacobian_sources"] == ["finite_difference"]
    assert metadata["finite_difference_epsilon"] == 2e-6


def test_capacity_overflow_is_an_error(no_jacobian_model):
    with pytest.raises(SolutionCapacityError):
        Engine(
            plugins={
                "backend": NumpyBackend(),
                "model": no_jacobian_model,
                "cam_solver": _OverflowSolver(),
            }
        ).run()


class _OverflowSolver(CAMSolver[CAMSolverConfig]):
    """A solver that returns two solutions for a capacity-one model."""

    name = "overflow"
    description = "returns more solutions than the model capacity"
    config_schema = CAMSolverConfig
    supported_tasks = ("solve",)
    output_products = (SOLVER_OUTPUT_DECLARATIONS["state"],)

    def solve(self, request: CAMSolveRequest) -> CAMProductFragment:
        from qphase_cam.solver.execution import state_fragment

        return state_fragment(
            request=request,
            run=lambda model, point: self.solve_point(model),
            supports_batch=False,
        )

    def solve_point(self, model, backend=None, point=None) -> CAMSolverOutput:
        del model, backend, point
        return CAMSolverOutput(
            [CAMSolution(np.eye(2) * value, 0.0, True, "test") for value in (1.0, 2.0)]
        )
