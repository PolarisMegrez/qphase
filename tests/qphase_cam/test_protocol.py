"""The declared solver and postprocessor protocol (Phase 10D2, preparation).

These tests freeze the capability tables and the request/fragment types. The
runtime still runs the pre-10D2 path here; the tables are what the engine switch
will read, and they must match what each plugin actually does.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from qphase.core.scan import ScanSpec
from qphase_cam.data import build_state_product, solution_identity
from qphase_cam.data.declarations import (
    CAM_ANNOTATION_DECLARATION,
    SOLUTION_IDENTITY_SEMANTICS,
)
from qphase_cam.engine import validate_solver_task
from qphase_cam.postprocessor import (
    CAMFiniteDelayCarrier,
    CAMPostprocessor,
    CoherencePoleSpectrum,
    HamiltonianSpectrum,
    JacobianSpectrum,
    PetermannSpectrum,
    Physicality,
    RayleighFrequency,
    StochasticValidity,
)
from qphase_cam.postprocessor.base import (
    POSTPROCESSOR_TARGET_ROLES,
    CAMAnnotationContext,
)
from qphase_cam.postprocessor.local_response import LocalResponseValidation
from qphase_cam.solver import (
    BatchedNewtonSolver,
    BifurcationSolver,
    CAMSolver,
    ContinuationSolver,
    DeflationSolver,
    MultistabilitySolver,
    SteadyStateSolver,
)
from qphase_cam.solver.base import (
    CAM_SOLVER_TASKS,
    SOLVER_OUTPUT_DECLARATIONS,
    CAMProductFragment,
    CAMSolverConfig,
    CAMSolveRequest,
)

SOLVER_CLASSES = (
    SteadyStateSolver,
    BatchedNewtonSolver,
    DeflationSolver,
    MultistabilitySolver,
    BifurcationSolver,
    ContinuationSolver,
)

POSTPROCESSOR_CLASSES = (
    RayleighFrequency,
    HamiltonianSpectrum,
    JacobianSpectrum,
    Physicality,
    PetermannSpectrum,
    CoherencePoleSpectrum,
    CAMFiniteDelayCarrier,
    LocalResponseValidation,
    StochasticValidity,
)


# --- solver protocol -------------------------------------------------------


def test_the_solver_protocol_declares_the_four_tasks():
    assert CAM_SOLVER_TASKS == ("solve", "scan", "search", "continue")
    assert set(SOLVER_OUTPUT_DECLARATIONS) == {
        "state",
        "state_scan",
        "candidates",
        "continuation",
    }
    for declaration in SOLVER_OUTPUT_DECLARATIONS.values():
        assert declaration.kind.value == "statistics"


def test_every_solver_declares_a_valid_capability_set():
    for solver_class in SOLVER_CLASSES:
        assert issubclass(solver_class, CAMSolver)
        assert solver_class.supported_tasks, solver_class.name
        assert set(solver_class.supported_tasks) <= set(CAM_SOLVER_TASKS)
        assert solver_class.output_products, solver_class.name
        for declaration in solver_class.output_products:
            assert declaration.name in {
                item.name for item in SOLVER_OUTPUT_DECLARATIONS.values()
            }
        assert solver_class.progress_unit
        assert solver_class.input_product_requirements == ()


@pytest.mark.parametrize(
    "solver_class, tasks, grid, batch, jacobian, symbolic, audit, outputs",
    [
        (SteadyStateSolver, ("solve", "scan"), True, False, False, False, False, 2),
        (BatchedNewtonSolver, ("solve", "scan"), True, True, True, False, False, 2),
        (DeflationSolver, ("solve", "scan"), True, False, True, False, False, 2),
        (MultistabilitySolver, ("solve", "scan"), True, True, False, False, False, 2),
        (BifurcationSolver, ("search",), True, False, True, True, True, 1),
        (ContinuationSolver, ("continue",), False, False, True, False, False, 1),
    ],
)
def test_solver_capabilities_match_their_implementation(
    solver_class, tasks, grid, batch, jacobian, symbolic, audit, outputs
):
    assert solver_class.supported_tasks == tasks
    assert solver_class.supports_parameter_grid is grid
    # Only the continuation, which owns its own path, refuses an external grid.
    if tasks == ("continue",):
        assert grid is False
    assert solver_class.supports_batch is batch
    assert solver_class.requires_jacobian is jacobian
    assert solver_class.requires_symbolic_dynamics is symbolic
    assert solver_class.search_audit_capability is audit
    assert len(solver_class.output_products) == outputs
    # Only a search solver may claim an audit; only a search/continue solver may
    # emit a non-state product.
    if audit:
        assert tasks == ("search",)
    assert "candidates" not in {
        item.name for item in solver_class.output_products
    } or tasks == ("search",)
    assert "continuation" not in {
        item.name for item in solver_class.output_products
    } or tasks == ("continue",)


def test_solve_request_carries_the_grid_and_the_policy():
    grid = ScanSpec.model_validate(
        {"axes": {"gamma": {"target": "model.demo.gamma", "values": [0.0, 1.0]}}}
    ).compile()
    request = CAMSolveRequest(
        task="scan",
        model=object(),
        backend=object(),
        parameter_grid=grid,
        case_failure_policy="record",
    )
    assert request.task == "scan"
    assert request.parameter_grid is grid
    assert request.case_failure_policy == "record"
    assert request.progress is None
    with pytest.raises(dataclasses.FrozenInstanceError):
        request.parameter_grid = None  # frozen


def test_product_fragment_groups_products_audits_and_metadata():
    fragment = CAMProductFragment()
    assert fragment.products == {}
    assert fragment.audits == ()
    assert fragment.metadata == {}


@pytest.mark.parametrize("task", ["solve", "scan", "search"])
def test_a_solver_without_the_grid_capability_is_rejected(task):
    """The capability is checked on every grid path, not only on ``scan``."""

    class NoGridSolver(CAMSolver[CAMSolverConfig]):
        name = "no_grid"
        description = "declares tasks but no grid capability"
        config_schema = CAMSolverConfig
        supported_tasks = (task,)
        output_products = (SOLVER_OUTPUT_DECLARATIONS["state"],)
        supports_parameter_grid = False

        def solve(self, request):  # pragma: no cover - rejected before the call
            raise AssertionError("the grid capability must be checked first")

    grid = ScanSpec.model_validate(
        {"axes": {"gamma": {"target": "model.demo.gamma", "values": [0.0, 1.0]}}}
    ).compile()
    with pytest.raises(ValueError, match="does not support a parameter grid"):
        validate_solver_task(NoGridSolver(), task, grid)
    # Without a grid the same solver is accepted for the tasks that run on one
    # point; ``scan`` needs a grid and is rejected by the task check instead.
    if task != "scan":
        validate_solver_task(NoGridSolver(), task, None)


def test_continuation_rejects_a_grid_through_the_capability_check():
    class Continue(CAMSolver[CAMSolverConfig]):
        name = "continue"
        description = "continuation double"
        config_schema = CAMSolverConfig
        supported_tasks = ("continue",)
        output_products = (SOLVER_OUTPUT_DECLARATIONS["continuation"],)
        supports_parameter_grid = False

        def solve(self, request):  # pragma: no cover - rejected before the call
            raise AssertionError("continuation must reject an external grid")

    grid = ScanSpec.model_validate(
        {"axes": {"gamma": {"target": "model.demo.gamma", "values": [0.0]}}}
    ).compile()
    with pytest.raises(ValueError, match="does not support a parameter grid"):
        validate_solver_task(Continue(), "continue", grid)
    validate_solver_task(Continue(), "continue", None)


def test_the_task_is_explicit_and_matched_against_the_grid():
    """An explicit task never degrades into an inferred scan."""

    class Both(CAMSolver[CAMSolverConfig]):
        name = "both"
        description = "declares solve and scan"
        config_schema = CAMSolverConfig
        supported_tasks = ("solve", "scan")
        output_products = (SOLVER_OUTPUT_DECLARATIONS["state"],)
        supports_parameter_grid = True

        def solve(self, request):  # pragma: no cover - never called here
            raise AssertionError("validation only")

    grid = ScanSpec.model_validate(
        {"axes": {"gamma": {"target": "model.demo.gamma", "values": [0.0]}}}
    ).compile()
    validate_solver_task(Both(), "solve", None)
    validate_solver_task(Both(), "scan", grid)
    with pytest.raises(ValueError, match="does not accept a grid"):
        validate_solver_task(Both(), "solve", grid)
    with pytest.raises(ValueError, match="needs a parameter grid"):
        validate_solver_task(Both(), "scan", None)
    with pytest.raises(ValueError, match="does not implement the 'search' task"):
        validate_solver_task(Both(), "search", None)


# --- state solution identity ----------------------------------------------


def _state_payload(*, states, counts):
    count_array = np.asarray(counts, dtype=np.int64)
    capacity = np.asarray(states).shape[-3]
    residual = np.full(count_array.shape + (capacity,), np.nan)
    success = np.zeros(count_array.shape + (capacity,), dtype=bool)
    valid = np.zeros(count_array.shape + (capacity,), dtype=bool)
    for index in np.ndindex(count_array.shape):
        for slot in range(int(count_array[index])):
            residual[index + (slot,)] = 1e-12
            success[index + (slot,)] = True
            valid[index + (slot,)] = True
    return {
        "states": states,
        "residuals": residual,
        "success": success,
        "valid_mask": valid,
        "solution_count": count_array,
    }


def test_solution_identity_is_content_based_and_skips_empty_slots():
    states = np.asarray([np.eye(2), np.eye(2) * 2], dtype=complex)
    product = build_state_product(
        **_state_payload(states=states, counts=2), parameters={"gamma": 0.5}
    )
    identities = np.asarray(product.handle("solution_identity").materialize())
    assert product.variable("solution_identity").dims == ("scan", "solution")
    assert identities.shape == (1, 2)
    assert all(value.startswith("cam:solution:") for value in identities[0])
    assert identities[0, 0] != identities[0, 1]
    assert product.attributes["solution_identity_variable"] == "solution_identity"
    assert (
        product.attributes["solution_identity_semantics"] == SOLUTION_IDENTITY_SEMANTICS
    )

    partial = build_state_product(
        **_state_payload(states=states, counts=1), parameters={"gamma": 0.5}
    )
    partial_identities = np.asarray(partial.handle("solution_identity").materialize())
    assert partial_identities[0, 0]
    assert partial_identities[0, 1] == ""


def test_solution_identity_tracks_content_not_the_slot():
    left = solution_identity(state_vector=np.eye(2), parameters={"gamma": 0.5})
    again = solution_identity(state_vector=np.eye(2), parameters={"gamma": 0.5})
    other_state = solution_identity(
        state_vector=np.eye(2) * 2, parameters={"gamma": 0.5}
    )
    other_point = solution_identity(state_vector=np.eye(2), parameters={"gamma": 0.6})
    assert left == again
    assert left != other_state
    assert left != other_point
    # Swapping slots is a layout change, so the identity of the same state is
    # unchanged: it never contains a slot index.
    swapped = build_state_product(
        **_state_payload(
            states=np.asarray([np.eye(2) * 2, np.eye(2)], dtype=complex), counts=2
        ),
        parameters={"gamma": 0.5},
    )
    swapped_identities = np.asarray(swapped.handle("solution_identity").materialize())
    assert set(swapped_identities[0]) == {left, other_state}


# --- postprocessor protocol ------------------------------------------------


def test_the_annotation_context_declares_its_inputs():
    context = CAMAnnotationContext(model=object(), backend=object())
    assert context.products == {}
    assert context.metadata == {}
    with pytest.raises(KeyError, match="no product 'state'"):
        context.require_product("state")
    assert POSTPROCESSOR_TARGET_ROLES == (
        "state",
        "state_scan",
        "candidates",
        "continuation",
    )


def test_every_postprocessor_declares_its_annotation_contract():
    assert len({cls.name for cls in POSTPROCESSOR_CLASSES}) == len(
        POSTPROCESSOR_CLASSES
    )
    for processor in POSTPROCESSOR_CLASSES:
        assert issubclass(processor, CAMPostprocessor)
        assert processor.annotation_kind, processor.name
        assert processor.accepted_product_roles, processor.name
        assert set(processor.accepted_product_roles) <= set(POSTPROCESSOR_TARGET_ROLES)
        assert processor.output_products == (CAM_ANNOTATION_DECLARATION,)
        # Every existing annotation is model-aware except the pure physicality
        # check; only the Jacobian one needs a backend.
        if processor is Physicality:
            assert processor.requires_model is False
        else:
            assert processor.requires_model is True
        assert processor.requires_backend is (processor is JacobianSpectrum)


def test_candidate_postprocessors_only_accept_the_candidate_product():
    for processor in (LocalResponseValidation, StochasticValidity):
        assert processor.accepted_product_roles == ("candidates",)
    for processor in POSTPROCESSOR_CLASSES:
        if processor in (LocalResponseValidation, StochasticValidity):
            continue
        assert processor.accepted_product_roles == ("state", "state_scan")
