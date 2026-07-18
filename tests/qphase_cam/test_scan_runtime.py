from types import SimpleNamespace

import numpy as np
import pytest
from qphase.backend.numpy_backend import NumpyBackend
from qphase.core.dataset import DatasetResultProtocol
from qphase.core.scan import ScanSpec
from qphase_cam.engine import Engine
from qphase_cam.result import CAMResult
from qphase_cam.state import CAMSolution, CAMSolverOutput

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
