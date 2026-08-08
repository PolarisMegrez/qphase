from types import SimpleNamespace

import numpy as np
import pytest
from qphase.backend.numpy_backend import NumpyBackend
from qphase.core.dataset import DatasetResultProtocol
from qphase.core.scan import ScanSpec
from qphase_cam.bifurcation_result import CAMBifurcationScanResult
from qphase_cam.engine import Engine, EngineConfig
from qphase_cam.result import CAMResult
from qphase_cam.state import (
    CAMBifurcationCandidate,
    CAMBifurcationOutput,
    CAMSolution,
    CAMSolverOutput,
)

from models.vdp_2mode import VDP2ModeModel


class BatchSolver:
    name = "batch_test"
    supports_batch = True

    def solve(self, model, backend):
        del backend
        values = np.asarray(model.params["omega_a"])
        return CAMSolverOutput(
            [
                [CAMSolution(np.eye(2) * (index + 1), 0.0, True, "test")]
                for index in range(values.size)
            ]
        )


def test_cam_engine_preserves_multidimensional_scan_shape():
    grid = ScanSpec.model_validate(
        {
            "combine": "cartesian",
            "axes": {
                "omega_a": {
                    "target": "model.vdp_2mode.omega_a",
                    "values": [-0.1, 0.1],
                },
                "gamma_b": {
                    "target": "model.vdp_2mode.gamma_b",
                    "values": [0.2, 0.3, 0.4],
                },
            },
        }
    ).compile()
    context = SimpleNamespace(
        parameter_grid=grid,
        progress=None,
    )
    model = VDP2ModeModel(
        omega_a=0.0,
        omega_b=0.0,
        gamma_a=2.0,
        gamma_b=0.5,
        Gamma=0.0001,
        g=0.5,
    )

    result = Engine(
        plugins={
            "backend": NumpyBackend(),
            "model": model,
            "cam_solver": BatchSolver(),
        }
    ).run(context=context)

    assert isinstance(result, DatasetResultProtocol)
    assert result.shape == (2, 3)
    assert result.states.shape == (2, 3, 4, 2, 2)
    assert result.solution_count.shape == (2, 3)
    assert result.params_at((1, 2))["omega_a"] == pytest.approx(0.1)
    assert result.params_at((1, 2))["gamma_b"] == pytest.approx(0.4)


def test_cam_dataset_point_view_and_sharded_storage(tmp_path):
    result = CAMResult(
        states=np.broadcast_to(np.eye(2), (4, 1, 2, 2)).copy(),
        residuals=np.zeros((4, 1)),
        success=np.ones((4, 1), dtype=bool),
        valid_mask=np.ones((4, 1), dtype=bool),
        solution_count=np.ones(4, dtype=int),
        params={"omega": np.arange(4.0)},
        axes={"omega": np.arange(4.0)},
        postprocess={"frequency": np.arange(4.0) + 0.5},
    )

    point = result.point_view((2,))
    report = result.save_dataset(
        tmp_path / "cam",
        layout="sharded",
        shard_target_bytes=1,
    )

    assert point.params["omega"] == 2.0
    assert report.layout == "sharded"
    assert len([path for path in report.files if path.suffix == ".npz"]) == 4

    loaded = CAMResult.load_dataset(tmp_path / "cam")
    np.testing.assert_allclose(loaded.states, result.states)
    np.testing.assert_allclose(loaded.params["omega"], result.params["omega"])
    np.testing.assert_allclose(loaded.axes["omega"], result.axes["omega"])
    np.testing.assert_allclose(
        loaded.postprocess["frequency"], result.postprocess["frequency"]
    )


def test_continuation_rejects_external_scan():
    class Continuation:
        name = "continuation"

    model = VDP2ModeModel(
        omega_a=0.0,
        omega_b=0.0,
        gamma_a=2.0,
        gamma_b=0.5,
        Gamma=0.0001,
        g=0.5,
    )
    context = SimpleNamespace(
        parameter_grid=ScanSpec.model_validate(
            {
                "axes": {
                    "omega": {
                        "target": "model.vdp_2mode.omega_a",
                        "values": [0.0],
                    }
                }
            }
        ).compile(),
        progress=None,
    )

    with pytest.raises(ValueError, match="continuation"):
        Engine(
            plugins={
                "backend": NumpyBackend(),
                "model": model,
                "cam_solver": Continuation(),
            }
        ).run(context=context)


def test_bifurcation_scan_preserves_empty_cases_and_round_trips(tmp_path):
    class Reporter:
        def status(self, *args, **kwargs):
            pass

        def update(self, *args, **kwargs):
            pass

    class Solver:
        name = "bifurcation"
        output_kind = "bifurcation_candidates"
        config = SimpleNamespace(controls={"gamma_a": object()})

        def solve(self, model, backend):
            del backend
            candidates = []
            if model.params["omega_a"] > 0.0:
                candidates.append(
                    CAMBifurcationCandidate(
                        state_vector=np.asarray([1.0, 1.0, 0.0, 0.0]),
                        controls={"gamma_a": 0.5},
                        full_residual_norm=1e-12,
                        search_residual_norm=2e-12,
                        success=True,
                        status="verified",
                        method="test",
                        metadata={"is_physical": True, "is_stable": True},
                    )
                )
            return CAMBifurcationOutput(
                candidates=candidates,
                target="equilibrium_multiplicity",
                order=2,
                metadata={
                    "control_names": ("gamma_a",),
                    "structural_coverage": "test",
                    "numerical_coverage": "test",
                },
            )

    grid = ScanSpec.model_validate(
        {
            "axes": {
                "omega_a": {
                    "target": "model.vdp_2mode.omega_a",
                    "values": [-0.1, 0.1],
                }
            }
        }
    ).compile()
    context = SimpleNamespace(
        parameter_grid=grid,
        progress=Reporter(),
        checkpoints=SimpleNamespace(enabled=False),
        cancellation=SimpleNamespace(raise_if_cancelled=lambda: None),
    )
    model = VDP2ModeModel(
        omega_a=0.0,
        omega_b=0.0,
        gamma_a=2.0,
        gamma_b=0.5,
        Gamma=0.0001,
        g=0.5,
    )
    result = Engine(
        plugins={
            "backend": NumpyBackend(),
            "model": model,
            "cam_solver": Solver(),
        }
    ).run(context=context)

    assert isinstance(result, CAMBifurcationScanResult)
    np.testing.assert_array_equal(result.data, [0, 1])
    np.testing.assert_array_equal(result.candidate_offsets, [0, 0, 1])
    assert len(result.point_view((0,)).states) == 0
    assert len(result.point_view((1,)).states) == 1
    assert result.params_at((1,))["omega_a"] == pytest.approx(0.1)
    target = tmp_path / "bifurcation_scan.npz"
    result.save(target)
    loaded = CAMBifurcationScanResult.load(target)
    np.testing.assert_array_equal(loaded.data, [0, 1])
    assert target.with_name("bifurcation_scan_cases.csv").exists()
    candidate_csv = target.with_name("bifurcation_scan_candidates.csv")
    assert candidate_csv.exists()
    header = candidate_csv.read_text(encoding="utf-8").splitlines()[0]
    assert "r_diag_0" in header
    assert "maximum_jacobian_real_part" in header


def test_bifurcation_scan_rejects_control_axis():
    class Solver:
        name = "bifurcation"
        output_kind = "bifurcation_candidates"
        config = SimpleNamespace(controls={"omega_a": object()})

    model = VDP2ModeModel(
        omega_a=0.0,
        omega_b=0.0,
        gamma_a=2.0,
        gamma_b=0.5,
        Gamma=0.0001,
        g=0.5,
    )
    context = SimpleNamespace(
        parameter_grid=ScanSpec.model_validate(
            {
                "axes": {
                    "omega_a": {
                        "target": "model.vdp_2mode.omega_a",
                        "values": [0.0],
                    }
                }
            }
        ).compile(),
        progress=None,
    )
    with pytest.raises(ValueError, match="not solver controls"):
        Engine(
            plugins={
                "backend": NumpyBackend(),
                "model": model,
                "cam_solver": Solver(),
            }
        ).run(context=context)


def test_bifurcation_scan_records_case_failure(tmp_path):
    class Solver:
        name = "bifurcation"
        output_kind = "bifurcation_candidates"
        config = SimpleNamespace(controls={"gamma_a": object()})
        target = SimpleNamespace(order=3)

        def solve(self, model, backend):
            del backend
            if model.params["omega_a"] < 0.0:
                raise ValueError("outside numerical chart")
            return CAMBifurcationOutput(
                candidates=[],
                target="equilibrium_multiplicity",
                order=3,
                metadata={"control_names": ("gamma_a",)},
            )

    grid = ScanSpec.model_validate(
        {
            "axes": {
                "omega_a": {
                    "target": "model.vdp_2mode.omega_a",
                    "values": [-0.1, 0.1],
                }
            }
        }
    ).compile()
    context = _scan_context(grid)
    result = Engine(
        EngineConfig(case_failure_policy="record"),
        plugins={
            "backend": NumpyBackend(),
            "model": _vdp_model(),
            "cam_solver": Solver(),
        },
    ).run(context=context)

    assert result.case_metadata[0]["case_status"] == "error"
    assert result.case_metadata[0]["case_error_type"] == "ValueError"
    assert result.case_metadata[0]["case_flat_index"] == 0
    assert result.case_metadata[1]["case_status"] == "complete"
    target = tmp_path / "recorded_failure.npz"
    result.save(target)
    rows = target.with_name("recorded_failure_cases.csv").read_text(
        encoding="utf-8"
    )
    assert "case_status,case_error_type,case_error_message" in rows
    assert "error,ValueError,outside numerical chart" in rows


def test_bifurcation_scan_resume_does_not_repeat_completed_case():
    class Checkpoints:
        enabled = True

        def __init__(self):
            self.chunks = {}

        def load_chunk(self, key):
            return self.chunks.get(key)

        def save_chunk(self, key, value):
            self.chunks[key] = value

    class Solver:
        name = "bifurcation"
        output_kind = "bifurcation_candidates"
        config = SimpleNamespace(controls={"gamma_a": object()})
        target = SimpleNamespace(order=3)

        def __init__(self):
            self.calls = []
            self.fail_once = True

        def solve(self, model, backend):
            del backend
            value = float(model.params["omega_a"])
            self.calls.append(value)
            if value == 0.0 and self.fail_once:
                self.fail_once = False
                raise RuntimeError("transient failure")
            return CAMBifurcationOutput(
                candidates=[],
                target="equilibrium_multiplicity",
                order=3,
                metadata={"control_names": ("gamma_a",)},
            )

    grid = ScanSpec.model_validate(
        {
            "axes": {
                "omega_a": {
                    "target": "model.vdp_2mode.omega_a",
                    "values": [-0.1, 0.0, 0.1],
                }
            }
        }
    ).compile()
    checkpoints = Checkpoints()
    context = _scan_context(grid, checkpoints=checkpoints)
    solver = Solver()
    engine = Engine(
        plugins={
            "backend": NumpyBackend(),
            "model": _vdp_model(),
            "cam_solver": solver,
        }
    )

    with pytest.raises(RuntimeError, match="transient failure"):
        engine.run(context=context)
    result = engine.run(context=context)

    assert result.case_shape == (3,)
    assert solver.calls.count(-0.1) == 1
    assert solver.calls.count(0.0) == 2
    assert solver.calls.count(0.1) == 1


def _vdp_model():
    return VDP2ModeModel(
        omega_a=0.0,
        omega_b=0.0,
        gamma_a=2.0,
        gamma_b=0.5,
        Gamma=0.0001,
        g=0.5,
    )


def _scan_context(grid, *, checkpoints=None):
    class Reporter:
        def status(self, *args, **kwargs):
            pass

        def update(self, *args, **kwargs):
            pass

    return SimpleNamespace(
        parameter_grid=grid,
        progress=Reporter(),
        checkpoints=checkpoints or SimpleNamespace(enabled=False),
        cancellation=SimpleNamespace(raise_if_cancelled=lambda: None),
    )
