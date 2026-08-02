from types import SimpleNamespace

import numpy as np
from qphase.backend.numpy_backend import NumpyBackend
from qphase.core.dataset import DatasetResultProtocol
from qphase.core.scan import ScanSpec
from qphase_sde.analyser.lorentz_fitter import _load_input
from qphase_sde.analyser.result import AnalysisResult
from qphase_sde.engine import Engine, EngineConfig
from qphase_sde.integrator.euler_maruyama import EulerMaruyama
from qphase_sde.result import SDEResult
from qphase_sde.scan import SDEScanResult


class ScannedDummyModel:
    name = "scanned_dummy"
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
        return -np.asarray(params["rate"])[:, None] * y

    def diffusion(self, y, t, params):
        del t, params
        return np.zeros(y.shape + (1,))


class MeanAnalyzer:
    name = "mean"
    config = SimpleNamespace(expected_freq_max=None)

    def analyze(self, data, backend):
        del backend
        return AnalysisResult({"mean": float(np.mean(np.asarray(data.data)))})


class StochasticScannedModel(ScannedDummyModel):
    def diffusion(self, y, t, params):
        del t, params
        return np.ones(y.shape + (1,))


def _grid():
    return ScanSpec.model_validate(
        {
            "axes": {
                "rate": {
                    "target": "model.scanned_dummy.rate",
                    "values": [1.0, 2.0],
                }
            }
        }
    ).compile()


def _four_point_grid():
    return ScanSpec.model_validate(
        {
            "axes": {
                "rate": {
                    "target": "model.scanned_dummy.rate",
                    "values": [1.0, 2.0, 3.0, 4.0],
                }
            }
        }
    ).compile()


def test_sde_engine_adapts_parameter_grid_to_existing_fused_path():
    model = ScannedDummyModel()
    engine = Engine(
        config=EngineConfig(
            t0=0.0,
            t1=0.02,
            dt=0.01,
            n_traj=2,
            seed=7,
            ic=[[1.0]],
        ),
        plugins={
            "backend": NumpyBackend(),
            "integrator": EulerMaruyama(),
            "model": model,
        },
    )

    result = engine.run(
        context=SimpleNamespace(parameter_grid=_grid(), progress=None)
    )

    assert isinstance(result, DatasetResultProtocol)
    assert result.shape == (2,)
    assert result.combined.trajectory.data.shape[0] == 4
    assert result.point_view((0,)).trajectory.data.shape[0] == 2
    assert result.point_view((1,)).meta["params"]["rate"] == 2.0
    assert engine.config.n_traj == 2
    assert model.params == {"rate": 1.0}


def test_sde_engine_analyzes_resource_limited_scan_tiles():
    model = ScannedDummyModel()
    engine = Engine(
        config=EngineConfig(
            t0=0.0,
            t1=1.0,
            dt=0.01,
            n_traj=1000,
            seed=7,
            ic=[[1.0]],
            keep_traj=False,
        ),
        plugins={
            "backend": NumpyBackend(),
            "integrator": EulerMaruyama(),
            "model": model,
            "analyser": {"mean": MeanAnalyzer()},
        },
    )
    context = SimpleNamespace(
        parameter_grid=_grid(),
        progress=None,
        resources=SimpleNamespace(
            memory_limit_mib=4,
            gpu_memory_fraction=None,
        ),
        cancellation=None,
    )

    result = engine.run(context=context)

    assert result.combined.trajectory is None
    assert len(result.combined.analysis["mean"]) == 2
    assert result.combined.meta["execution_plan"]["scan_tile_size"] == 1
    assert result.combined.meta["rng_strategy"] == "scan_point_seedsequence_v1"
    assert engine.config.n_traj == 1000
    assert model.params == {"rate": 1.0}


def test_sde_scan_rng_is_independent_of_tile_size():
    def run(memory_limit_mib):
        model = StochasticScannedModel()
        engine = Engine(
            config=EngineConfig(
                t0=0.0,
                t1=0.1,
                dt=0.01,
                n_traj=5000,
                seed=19,
                ic=[[1.0]],
                keep_traj=False,
            ),
            plugins={
                "backend": NumpyBackend(),
                "integrator": EulerMaruyama(),
                "model": model,
                "analyser": {"mean": MeanAnalyzer()},
            },
        )
        context = SimpleNamespace(
            parameter_grid=_four_point_grid(),
            progress=None,
            resources=SimpleNamespace(
                memory_limit_mib=memory_limit_mib,
                gpu_memory_fraction=None,
            ),
            cancellation=None,
        )
        result = engine.run(context=context)
        means = [item["mean"] for item in result.combined.analysis["mean"]]
        return result.combined.meta["execution_plan"]["scan_tile_size"], means

    tile_one, means_one = run(3)
    tile_two, means_two = run(4)

    assert tile_one == 1
    assert tile_two == 2
    np.testing.assert_array_equal(means_one, means_two)


def test_sde_scan_result_saves_finite_shards(tmp_path):
    combined = SDEResult(
        trajectory=SimpleNamespace(
            data=np.zeros((8, 3, 1)),
            t0=0.0,
            dt=0.1,
            meta={},
        ),
        analysis={"psd": [{"value": index} for index in range(4)]},
    )
    grid = ScanSpec.model_validate(
        {
            "axes": {
                "rate": {
                    "target": "model.scanned_dummy.rate",
                    "values": [1.0, 2.0, 3.0, 4.0],
                }
            }
        }
    ).compile()
    result = SDEScanResult(combined, grid, {"rate": 1.0}, 2)

    report = result.save_dataset(
        tmp_path / "sde",
        layout="sharded",
        shard_target_bytes=result.nbytes // 2,
    )

    assert report.layout == "sharded"
    assert len(report.files) == 2
    loaded = SDEScanResult.load_dataset(tmp_path / "sde")
    assert loaded.shape == (4,)
    assert loaded.point_view((3,)).meta["params"]["rate"] == 4.0


def test_lorentz_loader_consumes_sde_dataset_views():
    combined = SDEResult(
        trajectory=None,
        analysis={"psd": [{"frequency": [0.0]}, {"frequency": [0.0]}]},
    )
    dataset = SDEScanResult(combined, _grid(), {"rate": 1.0}, 1)

    loaded = _load_input(dataset, "*.npz")

    assert len(loaded) == 2
    assert [item.meta["params"]["rate"] for item in loaded] == [1.0, 2.0]
