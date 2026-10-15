"""Wave 6E end-to-end acceptance: sampled-scalar PSD/Allan in a real workflow.

Runs the tracked fixture ``configs/workflows/misc/e2e_sampled_scalar_3mode.yaml``
through the real scheduler and validates the Wave 6 migration end to end:

- source trajectories and the transient observable product are not persisted
  by default; terminal typed products are (artifact inventory);
- the shared observable transform executes once per physical batch even with
  two downstream consumers;
- the debug job's published observables match a direct NumPy recomputation
  from same-seed trajectories;
- fixed-seed reruns reproduce identical scalar products;
- restored terminal products carry channels, units, orientation (phase and
  spectral paths only), uncertainty and producer provenance;
- CuPy matches NumPy within float32 tolerance when a GPU is available.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from qphase.backend.numpy_backend import NumpyBackend
from qphase.core.project import ProjectContext
from qphase.core.scan import ScanSpec
from qphase.data import load_products
from qphase_sde.analyser.allan_variance import AllanVarianceAnalyzer
from qphase_sde.analyser.observable_transform import ObservableTransformAnalyzer
from qphase_sde.analyser.psd import PsdAnalyzer
from qphase_sde.engine import Engine, EngineConfig
from qphase_sde.integrator.srk import GenericSRK

_REPO_ROOT = Path(__file__).resolve().parents[2]
_FIXTURE = (
    _REPO_ROOT / "configs" / "workflows" / "misc" / "e2e_sampled_scalar_3mode.yaml"
)

_MODEL_PARAMS = {
    "omega_a": 0.0,
    "omega_b": 0.0,
    "omega_c": 0.0,
    "g_ab": 0.02,
    "g_ac": 0.02,
    "chi": 0.00005,
    "gamma_a": 1.0e-5,
    "gamma_b": 0.1,
    "gamma_c": 0.1,
}
_IC = [["1.0+0.0j", "0.5+0.0j", "0.25+0.0j"]]
_SEED = 20261214

_MAIN_TERMINAL_PRODUCTS = {"raw_psd", "phase_allan", "scalar_psd", "scalar_allan"}


@pytest.fixture(scope="module")
def fixture_runs(tmp_path_factory):
    """Two independent scheduler runs of the fixture workflow (both jobs)."""
    from qphase.core.registry import discovery
    from qphase.core.scheduler import Scheduler
    from qphase.core.system_config import SystemConfig
    from qphase.core.workflow import load_workflow

    discovery.discover_plugins()
    discovery.discover_local_plugins()

    project = ProjectContext.create(
        tmp_path_factory.mktemp("e2e_project"), name="Wave 6E"
    )
    workflow = load_workflow(_FIXTURE)
    runs = []
    for _ in range(2):
        scheduler = Scheduler(system_config=SystemConfig(), project=project)
        results = scheduler.run(workflow)
        assert all(result.success for result in results)
        runs.append({result.job_name: result.job_dir for result in results})
    return runs


def _main_job_config():
    return {
        "t0": 0.0,
        "t1": 25.6,
        "dt": 0.05,
        "n_traj": 16,
        "seed": _SEED,
        "ic": _IC,
        "adaptive": False,
        "save_stride": 1,
        "keep_traj": False,
    }


def _scan_grid():
    return ScanSpec.model_validate(
        {
            "axes": {
                "gamma_a": {
                    "target": "model.kerr_3mode.gamma_a",
                    "values": [1.0e-5, 2.0e-5, 4.0e-5],
                }
            }
        }
    ).compile()


def _kerr_model():
    from qphase.core.registry import discovery, registry

    discovery.discover_plugins()
    discovery.discover_local_plugins()
    return registry.create_plugin_instance(
        "model", {"name": "kerr_3mode", **_MODEL_PARAMS}
    )


def _analyser_plugin_config():
    """Mirror of the fixture's analyser section (YAML-translated)."""
    return {
        "raw_psd": {
            "plugin": "psd",
            "config": {
                "measurement": {
                    "complex_field": {
                        "channels": [0, 1, 2],
                        "representation": "complex",
                    }
                }
            },
        },
        "phase_allan": {
            "plugin": "allan_variance",
            "config": {
                "measurement": {
                    "phase_frequency": {
                        "channels": [0, 1, 2],
                        "min_independent_windows": 1,
                    }
                },
                "points": 8,
                "min_windows": 2,
            },
        },
        "obs": {
            "plugin": "observable_transform",
            "config": {
                "units": "occupation",
                "observables": {
                    "n0": {
                        "matrix": [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
                    },
                    "q_fixed": {
                        "matrix": [[0.5, 0.0, 0.0], [0.0, 0.25, 0.0], [0.0, 0.0, 0.25]]
                    },
                },
            },
        },
        "scalar_psd": {
            "plugin": "psd",
            "config": {
                "measurement": {
                    "sampled_scalar": {
                        "channels": ["n0", "q_fixed"],
                        "detrend": "constant",
                    }
                }
            },
        },
        "scalar_allan": {
            "plugin": "allan_variance",
            "config": {
                "measurement": {
                    "sampled_scalar": {
                        "channels": ["n0", "q_fixed"],
                        "min_nonoverlap_pairs": 2,
                    }
                },
                "points": 8,
                "min_windows": 2,
            },
        },
    }


def _analyser_bindings():
    return {
        "raw_psd": {"signal": "input.trajectories"},
        "phase_allan": {"signal": "input.trajectories"},
        "obs": {"signal": "input.trajectories"},
        "scalar_psd": {"signal": "obs.obs"},
        "scalar_allan": {"signal": "obs.obs"},
    }


def _main_analysers(transform=None):
    return {
        "raw_psd": PsdAnalyzer(
            measurement={
                "complex_field": {"channels": [0, 1, 2], "representation": "complex"}
            }
        ),
        "phase_allan": AllanVarianceAnalyzer(
            measurement={
                "phase_frequency": {"channels": [0, 1, 2], "min_independent_windows": 1}
            },
            points=8,
            min_windows=2,
        ),
        "obs": transform
        or ObservableTransformAnalyzer(
            units="occupation",
            observables={
                "n0": {"matrix": [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]},
                "q_fixed": {
                    "matrix": [[0.5, 0.0, 0.0], [0.0, 0.25, 0.0], [0.0, 0.0, 0.25]]
                },
            },
        ),
        "scalar_psd": PsdAnalyzer(
            measurement={
                "sampled_scalar": {"channels": ["n0", "q_fixed"], "detrend": "constant"}
            }
        ),
        "scalar_allan": AllanVarianceAnalyzer(
            measurement={
                "sampled_scalar": {
                    "channels": ["n0", "q_fixed"],
                    "min_nonoverlap_pairs": 2,
                }
            },
            points=8,
            min_windows=2,
        ),
    }


def test_terminal_products_persisted_and_intermediates_absent(fixture_runs):
    """Artifact inventory: trajectories and the transient observable stay
    off disk; the four terminal typed products persist.
    """
    main_dir = fixture_runs[0]["e2e_main"]
    assert (main_dir / "artifact_manifest.json").exists()
    products = load_products(main_dir)
    assert set(products) == _MAIN_TERMINAL_PRODUCTS
    npz_names = [path.name for path in main_dir.rglob("*.npz")]
    assert not any("trajector" in name or "obs" in name for name in npz_names)

    debug_dir = fixture_runs[0]["e2e_debug_publish"]
    debug_products = load_products(debug_dir)
    assert set(debug_products) == {"obs"}


def test_fixed_seed_reruns_reproduce_scalar_products(fixture_runs):
    first, second = fixture_runs
    for name in ("scalar_psd", "scalar_allan"):
        products_a = load_products(first["e2e_main"])[name].materialize()
        products_b = load_products(second["e2e_main"])[name].materialize()
        for variable in products_a.variables:
            np.testing.assert_allclose(
                products_a.handle(variable.name).materialize(),
                products_b.handle(variable.name).materialize(),
                rtol=1e-12,
                atol=0.0,
            )


def test_restored_products_carry_units_channels_and_provenance(fixture_runs):
    products = load_products(fixture_runs[0]["e2e_main"])

    scalar_psd = products["scalar_psd"].materialize()
    psd_variable = scalar_psd.schema.variable("psd")
    assert psd_variable is not None
    assert psd_variable.units == "occupation^2/inverse_time"
    assert psd_variable.dims == ("scan", "frequency", "channel")
    assert scalar_psd.axis("frequency").units == "inverse_time"
    assert scalar_psd.attributes["measurement_model"] == "sampled_scalar"
    assert scalar_psd.attributes["orientation"] == "unsigned"
    assert scalar_psd.attributes["sidedness"] == "one_sided"
    assert scalar_psd.attributes["channel_names"] == ["n0", "q_fixed"]
    assert scalar_psd.attributes["producer_analyser"] == "scalar_psd"
    assert "source_analyser" not in scalar_psd.attributes
    sem = scalar_psd.schema.variable("psd_sem")
    assert sem is not None and not sem.units

    scalar_allan = products["scalar_allan"].materialize()
    variance = scalar_allan.schema.variable(
        "channel_results.0.allan.sampled_scalar_variance"
    )
    assert variance is not None
    assert variance.units == "occupation^2"
    assert variance.dims == ("scan", "tau")
    assert scalar_allan.axis("tau").units == "time"
    assert scalar_allan.attributes["measurement_model"] == "sampled_scalar"
    assert scalar_allan.attributes["channel_names"] == ["n0", "q_fixed"]
    assert scalar_allan.attributes["producer_analyser"] == "scalar_allan"
    assert "orientation" not in scalar_allan.attributes
    per_trajectory = scalar_allan.schema.variable(
        "channel_results.0.allan.per_trajectory"
    )
    assert per_trajectory is not None
    assert per_trajectory.dims == ("scan", "trajectory", "tau")

    phase_allan = products["phase_allan"].materialize()
    assert phase_allan.attributes["measurement_model"] == "phase_frequency"
    assert phase_allan.attributes["orientation"] in (
        "phase_increasing",
        "phase_decreasing",
    )
    assert phase_allan.attributes["producer_analyser"] == "phase_allan"

    raw_psd = products["raw_psd"].materialize()
    assert raw_psd.attributes["measurement_model"] == "complex_field"
    assert raw_psd.attributes["producer_analyser"] == "raw_psd"
    assert raw_psd.schema.variable("modes") is not None


def test_debug_observable_matches_direct_numpy(fixture_runs):
    """The debug job's published observables equal |a|^2 combinations
    recomputed by hand from same-seed trajectories.
    """
    engine = Engine(
        config=EngineConfig(
            t0=0.0,
            t1=3.2,
            dt=0.05,
            n_traj=8,
            seed=_SEED,
            ic=_IC,
            adaptive=False,
            save_stride=1,
            keep_traj=True,
        ),
        plugins={
            "backend": NumpyBackend(),
            "integrator": GenericSRK(),
            "model": _kerr_model(),
        },
    )
    result = engine.run(context=SimpleNamespace(parameter_grid=None, progress=None))
    alpha = np.asarray(result.products["trajectories"].handle("alpha").materialize())
    if alpha.ndim == 4:  # singleton scan axis
        alpha = alpha[0]
    # alpha: (trajectory, time, mode)
    n0 = np.abs(alpha[:, :, 0]) ** 2
    q_fixed = 0.5 * n0 + 0.25 * np.abs(alpha[:, :, 1]) ** 2 + 0.25 * np.abs(
        alpha[:, :, 2]
    ) ** 2

    obs = load_products(fixture_runs[0]["e2e_debug_publish"])["obs"].materialize()
    value = obs.handle("value").materialize()
    assert obs.attributes["channel_names"] == ["n0", "q_fixed"]
    assert obs.attributes["producer_analyser"] == "obs"
    np.testing.assert_allclose(value[:, :, 0], n0, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(value[:, :, 1], q_fixed, rtol=1e-9, atol=1e-12)


class _CountingTransform(ObservableTransformAnalyzer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.calls = 0
        self.seen_shapes: list[tuple[int, ...]] = []

    def execute(self, inputs, **kwargs):
        self.calls += 1
        signal = inputs.get("signal")
        if signal is not None:
            self.seen_shapes.append(
                tuple(signal.handle("alpha").materialize().shape)
            )
        return super().execute(inputs, **kwargs)


def test_shared_transform_executes_once_per_batch():
    """With two scalar consumers on one transform, the transform still runs
    exactly once per physical batch. Scan points fuse into the batch: the
    physical batch is 8 trajectories of every point, so 16 trajectories over
    3 points execute the transform twice — not 4 (once per consumer) and not
    1 (a single retained full-size pass).
    """
    from qphase.core.registry import discovery, registry
    from qphase_sde.graph import compile_product_graph

    discovery.discover_plugins()
    graph = compile_product_graph(
        plugin_config={"analyser": _analyser_plugin_config()},
        registry=registry,
        input_bindings=_analyser_bindings(),
    )
    assert graph is not None
    transform = _CountingTransform(
        units="occupation",
        observables={
            "n0": {"matrix": [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]},
            "q_fixed": {
                "matrix": [[0.5, 0.0, 0.0], [0.0, 0.25, 0.0], [0.0, 0.0, 0.25]]
            },
        },
    )
    engine = Engine(
        config=EngineConfig(
            **_main_job_config(),
            trajectory_batching="required",
            trajectory_batch_size=8,
            rng_group_size=8,
        ),
        plugins={
            "backend": NumpyBackend(),
            "integrator": GenericSRK(),
            "model": _kerr_model(),
            "analyser": _main_analysers(transform=transform),
        },
        product_graph=graph,
    )
    result = engine.run(
        context=SimpleNamespace(
            parameter_grid=_scan_grid(),
            product_graph=graph,
            progress=None,
            cancellation=None,
            metadata={},
            resources=None,
        )
    )
    assert transform.calls == 2
    assert transform.seen_shapes == [(3, 8, 513, 3)] * 2
    assert set(result.products) == _MAIN_TERMINAL_PRODUCTS


def test_cupy_end_to_end_and_analyser_parity():
    """CuPy: the full graph runs on the GPU backend, and the scalar
    PSD/Allan estimators match NumPy within float64 tolerance on identical
    inputs (engine trajectories diverge between float32/float64 by chaos, so
    the numerical parity check lives at the analyser boundary).
    """
    cupy = pytest.importorskip("cupy")
    try:
        cupy.cuda.Device().use()
    except Exception:
        pytest.skip("No CUDA device available")
    from qphase.backend.cupy_backend import CuPyBackend
    from qphase.core.registry import discovery, registry
    from qphase.data import (
        AxisRole,
        AxisSchema,
        DataKind,
        ProductSchema,
        TimeSeriesDataset,
        VariableSchema,
    )
    from qphase_sde.contracts.quantities import SDEQuantity
    from qphase_sde.graph import compile_product_graph

    discovery.discover_plugins()
    graph = compile_product_graph(
        plugin_config={"analyser": _analyser_plugin_config()},
        registry=registry,
        input_bindings=_analyser_bindings(),
    )
    assert graph is not None

    engine = Engine(
        config=EngineConfig(**_main_job_config()),
        plugins={
            "backend": CuPyBackend(),
            "integrator": GenericSRK(),
            "model": _kerr_model(),
            "analyser": _main_analysers(),
        },
        product_graph=graph,
    )
    result = engine.run(
        context=SimpleNamespace(
            parameter_grid=_scan_grid(),
            product_graph=graph,
            progress=None,
            cancellation=None,
            metadata={},
            resources=None,
        )
    )
    assert set(result.products) == _MAIN_TERMINAL_PRODUCTS
    psd_gpu = np.asarray(result.products["scalar_psd"].handle("psd").materialize())
    assert psd_gpu.shape[0] == 3 and psd_gpu.shape[-1] == 2
    assert bool(np.all(np.isfinite(psd_gpu)))
    assert bool(np.all(psd_gpu >= 0.0))

    # Analyser-level parity on identical sampled-observable inputs.
    rng = np.random.default_rng(20261214)
    values = rng.standard_normal((16, 512, 2))
    schema = ProductSchema(
        kind=DataKind.TIME_SERIES,
        axes=[
            AxisSchema(name="trajectory", role=AxisRole.REALIZATION, size=16),
            AxisSchema(
                name="time",
                role=AxisRole.COORDINATE,
                size=512,
                coordinate="regular",
                start=0.0,
                step=0.05,
                units="time",
            ),
            AxisSchema(name="channel", role=AxisRole.COMPONENT, size=2),
        ],
        variables=[
            VariableSchema(
                name="value",
                dtype="<f8",
                value_domain="real",
                dims=("trajectory", "time", "channel"),
                quantity=SDEQuantity.SAMPLED_OBSERVABLE.value,
                units="occupation",
            ),
            VariableSchema(
                name="valid_length",
                dtype="<i8",
                value_domain="real",
                dims=("trajectory",),
                quantity="valid_length",
            ),
        ],
        attributes={"channel_names": ["n0", "q_fixed"]},
    )

    def observable_product(backend_name):
        if backend_name == "cpu":
            arrays = {"value": values, "valid_length": np.full(16, 512)}
        else:
            arrays = {
                "value": cupy.asarray(values),
                "valid_length": cupy.asarray(np.full(16, 512)),
            }
        return TimeSeriesDataset.from_arrays(
            schema, arrays, owner="test", device=backend_name
        )

    def host(product, leaf):
        array = product.handle(leaf).materialize()
        if isinstance(array, cupy.ndarray):
            return cupy.asnumpy(array)
        return np.asarray(array)

    np_be = NumpyBackend()
    cp_be = CuPyBackend()
    scalar_psd = PsdAnalyzer(
        measurement={
            "sampled_scalar": {"channels": ["n0", "q_fixed"], "detrend": "constant"}
        }
    )
    psd_np = scalar_psd.execute(
        {"signal": observable_product("cpu")}, backend=np_be, label="psd"
    )["psd"]
    psd_cp = scalar_psd.execute(
        {"signal": observable_product("cuda")}, backend=cp_be, label="psd"
    )["psd"]
    np.testing.assert_allclose(host(psd_np, "axis"), host(psd_cp, "axis"))
    for leaf in ("psd", "psd_std", "psd_sem"):
        np.testing.assert_allclose(
            host(psd_np, leaf),
            host(psd_cp, leaf),
            rtol=1e-6,
            atol=1e-30,
        )

    scalar_allan = AllanVarianceAnalyzer(
        measurement={
            "sampled_scalar": {"channels": ["n0", "q_fixed"], "min_nonoverlap_pairs": 2}
        },
        points=8,
        min_windows=2,
    )
    allan_np = scalar_allan.execute(
        {"signal": observable_product("cpu")}, backend=np_be, label="allan"
    )["allan"]
    allan_cp = scalar_allan.execute(
        {"signal": observable_product("cuda")}, backend=cp_be, label="allan"
    )["allan"]
    np.testing.assert_allclose(
        host(allan_np, "channel_results.0.allan.sampled_scalar_variance"),
        host(allan_cp, "channel_results.0.allan.sampled_scalar_variance"),
        rtol=1e-6,
    )
