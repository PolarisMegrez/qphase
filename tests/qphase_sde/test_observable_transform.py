"""qphase_sde: observable_transform analyser contract tests.

The transform maps complex field trajectories to real sampled-observable
time series via the shared Hermitian quadratic evaluator. References are
direct NumPy einsum and the ``quadratic_moments`` analyser.
"""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError
from qphase.backend.numpy_backend import NumpyBackend
from qphase.data import (
    AxisRole,
    AxisSchema,
    DataKind,
    ProductSchema,
    TimeSeriesDataset,
    VariableSchema,
)
from qphase_sde.analyser.observable_transform import (
    ObservableTransformAnalyzer,
    ObservableTransformConfig,
)
from qphase_sde.analyser.quadratic_moments import (
    QuadraticMomentAnalyzer,
    QuadraticMomentConfig,
)
from qphase_sde.contracts.analyser import AnalyserWorkspaceRequest
from qphase_sde.contracts.quantities import SDEQuantity
from qphase_sde.result import _trajectory_product
from qphase_sde.state import TrajectorySet

BACKEND = NumpyBackend()

T0 = 0.2
DT = 0.1


def _cupy_available() -> bool:
    try:
        import cupy

        cupy.cuda.Device().use()
    except Exception:
        return False
    return True


def _field(rng, shape, *, modes=None, scan_size=None, n_traj_per_point=None):
    if modes is None:
        modes = range(shape[-1])
    data = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    return _trajectory_product(
        TrajectorySet(data, t0=T0, dt=DT, meta={"mode_indices": list(modes)}),
        scan_size=scan_size,
        n_traj_per_point=n_traj_per_point,
    )


def _direct_reference(data, columns, matrices, centers):
    """Direct NumPy reference of Re(alpha^dagger Q alpha) - center."""
    selected = data[..., columns]
    matrices = np.asarray(matrices, dtype=complex)
    values = np.einsum("...i,oij,...j->...o", selected.conj(), matrices, selected)
    return values.real - np.asarray(centers, dtype=float)


def _json_matrix(entries):
    """Rebuild a complex matrix from its serialized attribute form."""
    return np.array(
        [[complex(entry) for entry in row] for row in entries], dtype=complex
    )


def _execute(config_kwargs, product, *, label="soft_observable"):
    analyzer = ObservableTransformAnalyzer(ObservableTransformConfig(**config_kwargs))
    outputs = analyzer.execute(
        {"signal": product},
        backend=BACKEND,
        label=label,
        metadata={"source_product": "input.trajectories"},
    )
    return analyzer, outputs[label]


# ---------------------------------------------------------------------------
# numerical baselines
# ---------------------------------------------------------------------------


def test_transform_matches_direct_einsum_multi_channel():
    rng = np.random.default_rng(20261101)
    product = _field(rng, (6, 64, 3), scan_size=2, n_traj_per_point=3)
    indefinite = [[0.5, 0.25j, 0.0], [-0.25j, -0.75, 0.125], [0.0, 0.125, 0.25]]
    projector = [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    _, out = _execute(
        {
            "units": "occupation",
            "observables": {
                "n0": {"matrix": projector, "center": 0.5},
                "mixed": {"matrix": indefinite},
            },
        },
        product,
    )

    value = out.handle("value").materialize()
    data = product.handle("alpha").materialize()
    # Both configured matrices are exactly Hermitian, so used == requested.
    reference = _direct_reference(
        data, [0, 1, 2], [projector, indefinite], [0.5, 0.0]
    )
    np.testing.assert_allclose(value, reference, rtol=1e-12, atol=1e-14)
    # The indefinite matrix produces negative readouts.
    assert np.any(value[..., 1] < 0.0)


def test_transform_matches_quadratic_moments():
    """The shared evaluator keeps both analysers on identical numbers."""
    rng = np.random.default_rng(20261102)
    trajectory = TrajectorySet(
        rng.standard_normal((5, 96, 2)) + 1j * rng.standard_normal((5, 96, 2)),
        t0=T0,
        dt=DT,
        meta={"mode_indices": [0, 1]},
    )
    observables = {
        "n0": {"matrix": [[1.0, 0.0], [0.0, 0.0]]},
        "x1": {"matrix": [[0.0, 0.5], [0.5, 0.0]], "center": 0.125},
    }
    _, out = _execute(
        {"units": "quadrature", "observables": observables},
        _trajectory_product(trajectory, scan_size=None, n_traj_per_point=None),
    )
    moments_product = QuadraticMomentAnalyzer(
        QuadraticMomentConfig(
            observables=observables, max_order=2, time_blocks=2, min_block_samples=2
        )
    ).execute(
        {
            "trajectories": _trajectory_product(
                trajectory, scan_size=None, n_traj_per_point=None
            )
        },
        backend=BACKEND,
        label="quadratic_moments",
    )["quadratic_moments"]

    value = out.handle("value").materialize()  # (scan=1, traj, time, channel)
    per_trajectory = moments_product.handle("per_trajectory_raw_moments").materialize()
    np.testing.assert_allclose(
        per_trajectory[:, :, 0],
        value[0].mean(axis=1),
        rtol=1e-12,
        atol=1e-14,
    )
    np.testing.assert_allclose(
        moments_product.handle("raw_moments").materialize()[:, 0],
        value[0].mean(axis=(0, 1)),
        rtol=1e-12,
        atol=1e-14,
    )


# ---------------------------------------------------------------------------
# product closure and provenance
# ---------------------------------------------------------------------------


def test_transform_product_schema_and_provenance():
    rng = np.random.default_rng(20261103)
    product = _field(rng, (3, 32, 2))
    skew = 1e-12  # below the default 1e-10 Hermitian tolerance
    _, out = _execute(
        {
            "units": "occupation",
            "observables": {
                "soft": {
                    "matrix": [[1.0, skew * 1j], [skew * 1j, 2.0]],
                    "reference_matrix": [[0.25, 0.0], [0.0, 0.5]],
                }
            },
        },
        product,
    )

    axes = {axis.name: axis for axis in out.axes}
    assert [axis.name for axis in out.axes] == [
        "scan",
        "trajectory",
        "time",
        "channel",
    ]
    assert axes["scan"].role is AxisRole.PARAMETER and axes["scan"].size == 1
    assert axes["trajectory"].role is AxisRole.REALIZATION
    assert axes["trajectory"].size == 3
    # The time axis mirrors the source grid, including the canonical unit.
    assert axes["time"].coordinate == "regular"
    assert axes["time"].start == pytest.approx(T0)
    assert axes["time"].step == pytest.approx(DT)
    assert axes["time"].units == "time"
    assert axes["channel"].role is AxisRole.COMPONENT and axes["channel"].size == 1

    value = next(variable for variable in out.variables if variable.name == "value")
    assert value.dims == ("scan", "trajectory", "time", "channel")
    assert value.value_domain == "real"
    assert np.dtype(value.dtype) == np.dtype("float64")
    assert value.quantity == SDEQuantity.SAMPLED_OBSERVABLE.value
    assert value.units == "occupation"

    attributes = out.attributes
    assert attributes["measurement_kind"] == "hermitian_quadratic"
    assert attributes["representation"] == "c_number"
    assert attributes["channel_definition"] == "observable"
    assert attributes["independent_realizations"] == "trajectory"
    assert attributes["channel_names"] == ["soft"]
    assert attributes["mode_labels"] == [0, 1]
    assert attributes["source_product"] == "input.trajectories"
    assert attributes["hermitian_tolerance"] == pytest.approx(1e-10)

    # Requested vs used: the skew is projected out and the residual recorded.
    requested = _json_matrix(attributes["matrices_requested"][0])
    used = _json_matrix(attributes["matrices_used"][0])
    assert requested[0, 1] == pytest.approx(1j * skew)
    np.testing.assert_allclose(used, used.conj().T, rtol=0.0, atol=0.0)
    assert not np.array_equal(requested, used)
    residuals = attributes["hermitian_residuals"]["soft"]
    assert residuals["matrix"] > 0.0
    assert residuals["matrix"] < 1e-10
    assert residuals["reference_matrix"] == 0.0
    # The center comes from Re Tr(Q R_ref) with the projected matrices.
    np.testing.assert_allclose(
        attributes["centers"], [float(np.real(np.trace(used @ np.diag([0.25, 0.5]))))]
    )


def test_transform_mode_reorder():
    rng = np.random.default_rng(20261104)
    product = _field(rng, (4, 48, 3))
    matrix = [[1.0, 0.5j], [-0.5j, -1.0]]
    _, out = _execute(
        {
            "units": "quadrature",
            "modes": [2, 0],
            "observables": {"swap": {"matrix": matrix}},
        },
        product,
    )

    value = out.handle("value").materialize()
    reference = _direct_reference(
        product.handle("alpha").materialize(), [2, 0], [matrix], [0.0]
    )
    np.testing.assert_allclose(value, reference, rtol=1e-12, atol=1e-14)
    assert out.attributes["mode_labels"] == [2, 0]


def test_transform_valid_length_propagates_source_handle():
    rng = np.random.default_rng(20261105)
    data = rng.standard_normal((2, 40, 1)) + 1j * rng.standard_normal((2, 40, 1))
    valid = np.array([40, 25], dtype=np.int64)
    data[1, 25:] = np.nan
    product = _trajectory_product(
        TrajectorySet(data, t0=T0, dt=DT, meta={"valid_length": valid}),
        scan_size=None,
        n_traj_per_point=None,
    )
    _, out = _execute(
        {
            "units": "occupation",
            "observables": {"n0": {"matrix": [[1.0]]}},
        },
        product,
    )

    # The censoring record propagates untouched, on the source's own handle.
    assert out.handle("valid_length") is product.handle("valid_length")
    np.testing.assert_array_equal(
        out.handle("valid_length").materialize(), valid.reshape(1, 2)
    )
    value = out.handle("value").materialize()[0]
    assert np.all(np.isfinite(value[0]))
    assert np.all(np.isnan(value[1, 25:])) and np.all(np.isfinite(value[1, :25]))


def test_transform_mirrors_products_without_scan_axis():
    rng = np.random.default_rng(20261106)
    alpha = rng.standard_normal((2, 16, 1)) + 1j * rng.standard_normal((2, 16, 1))
    schema = ProductSchema(
        kind=DataKind.TIME_SERIES,
        axes=[
            AxisSchema(name="trajectory", role=AxisRole.REALIZATION, size=2),
            AxisSchema(
                name="time",
                role=AxisRole.COORDINATE,
                size=16,
                coordinate="regular",
                start=0.0,
                step=0.5,
                units="time",
            ),
            AxisSchema(name="channel", role=AxisRole.COMPONENT, size=1),
        ],
        variables=[
            VariableSchema(
                name="alpha",
                dtype="<c16",
                value_domain="complex",
                dims=("trajectory", "time", "channel"),
                quantity=SDEQuantity.FIELD_AMPLITUDE.value,
            ),
            VariableSchema(
                name="valid_length",
                dtype="<i8",
                value_domain="real",
                dims=("trajectory",),
                quantity="valid_length",
            ),
        ],
        attributes={},
    )
    product = TimeSeriesDataset.from_arrays(
        schema,
        {"alpha": alpha, "valid_length": np.full(2, 16, dtype=np.int64)},
        owner="test",
    )
    _, out = _execute(
        {"units": "occupation", "observables": {"n0": {"matrix": [[1.0]]}}},
        product,
    )
    assert [axis.name for axis in out.axes] == ["trajectory", "time", "channel"]
    np.testing.assert_allclose(
        out.handle("value").materialize(), np.abs(alpha) ** 2, rtol=1e-12
    )


# ---------------------------------------------------------------------------
# validation gates
# ---------------------------------------------------------------------------


def test_transform_hermitian_residual_gate():
    base = {"units": "occupation"}
    with pytest.raises(ValueError, match="mutually exclusive"):
        ObservableTransformConfig(
            **base,
            observables={
                "bad": {
                    "matrix": [[1.0]],
                    "center": 0.5,
                    "reference_matrix": [[1.0]],
                }
            },
        )
    with pytest.raises(ValidationError):
        ObservableTransformConfig(**base, observables={})
    with pytest.raises(ValidationError):
        ObservableTransformConfig(
            units="",
            observables={"n0": {"matrix": [[1.0]]}},
        )
    with pytest.raises(ValueError, match="duplicates"):
        ObservableTransformConfig(
            **base, modes=[0, 0], observables={"n0": {"matrix": [[1.0]]}}
        )
    with pytest.raises(ValueError, match="non-negative"):
        ObservableTransformConfig(
            **base, modes=[-1], observables={"n0": {"matrix": [[1.0]]}}
        )
    with pytest.raises(ValueError, match="blank"):
        ObservableTransformConfig(
            **base, observables={" ": {"matrix": [[1.0]]}}
        )


def test_transform_compile_gates():
    rng = np.random.default_rng(20261107)
    product = _field(rng, (2, 16, 2))
    with pytest.raises(ValueError, match="not Hermitian"):
        _execute(
            {
                "units": "occupation",
                "observables": {"bad": {"matrix": [[1.0, 1e-3j], [0.0, 1.0]]}},
            },
            product,
        )
    with pytest.raises(ValueError, match="shape"):
        _execute(
            {
                "units": "occupation",
                "observables": {"bad": {"matrix": [[1.0, 0.0]]}},
            },
            product,
        )
    with pytest.raises(ValueError, match="finite"):
        _execute(
            {
                "units": "occupation",
                "observables": {"bad": {"matrix": [[np.nan, 0.0], [0.0, 1.0]]}},
            },
            product,
        )
    with pytest.raises(ValueError, match="not recorded"):
        _execute(
            {
                "units": "occupation",
                "modes": [3],
                "observables": {"n0": {"matrix": [[1.0]]}},
            },
            product,
        )


def test_transform_graph_contract():
    analyzer = ObservableTransformAnalyzer(
        ObservableTransformConfig(
            units="occupation", observables={"n0": {"matrix": [[1.0]]}}
        )
    )
    # Transient transform: batchable but intentionally reducer-free.
    capabilities = analyzer.execution_capabilities()
    assert capabilities.supports_trajectory_batching is True
    assert capabilities.supports_scan_tiling is True
    assert analyzer.reducer() is None

    (requirement,) = analyzer.input_requirements()
    assert requirement.name == "signal"
    assert requirement.kind is DataKind.TIME_SERIES
    assert requirement.quantity == SDEQuantity.FIELD_AMPLITUDE.value
    assert requirement.value_domain == "complex"
    assert requirement.required_fields == ["alpha", "valid_length"]

    (spec,) = analyzer.output_specs(label="soft_observable")
    assert spec.name == "soft_observable"
    assert spec.kind is DataKind.TIME_SERIES
    assert spec.quantity == SDEQuantity.SAMPLED_OBSERVABLE.value
    assert spec.fields == ["value", "valid_length"]


def test_transform_workspace_covers_output():
    analyzer = ObservableTransformAnalyzer(
        ObservableTransformConfig(
            units="occupation",
            modes=[0, 1],
            observables={
                "a": {"matrix": [[1.0, 0.0], [0.0, 0.0]]},
                "b": {"matrix": [[0.0, 0.5], [0.5, 0.0]]},
            },
        )
    )
    request = AnalyserWorkspaceRequest(
        trajectory_bytes=8 * 1000 * 2 * 16,
        n_traj=8,
        saved_samples=1000,
        n_record_modes=2,
        real_itemsize=8,
        backend_name="numpy",
        total_n_traj=64,
    )
    estimate = analyzer.workspace(request)
    assert estimate.host_bytes >= 8 * 1000 * 2 * 8  # invocation output at least
    assert estimate.retained_host_bytes == 64 * 1000 * 2 * 8
    cupy_estimate = analyzer.workspace(
        AnalyserWorkspaceRequest(
            trajectory_bytes=8 * 1000 * 2 * 16,
            n_traj=8,
            saved_samples=1000,
            n_record_modes=2,
            real_itemsize=8,
            backend_name="cupy",
            total_n_traj=64,
        )
    )
    assert cupy_estimate.device_bytes > 0
    assert cupy_estimate.retained_device_bytes == 64 * 1000 * 2 * 8
    assert cupy_estimate.host_bytes == 0


# ---------------------------------------------------------------------------
# persistence roundtrip and device parity
# ---------------------------------------------------------------------------


_ROUNDTRIP_PROBE = """
import json
import sys

from qphase.data import load_bundle

product = load_bundle(sys.argv[1]).products["soft_observable"]
schema = product.schema
print(
    json.dumps(
        {
            "axes": {
                axis.name: [axis.role.value, axis.size, axis.units]
                for axis in schema.axes
            },
            "value_dims": list(schema.variable("value").dims),
            "value_quantity": schema.variable("value").quantity,
            "value_units": schema.variable("value").units,
            "value_sum": float(product.handle("value").materialize().sum()),
            "valid_length": product.handle("valid_length").materialize().tolist(),
            "channel_names": schema.attributes["channel_names"],
            "matrices_used": schema.attributes["matrices_used"],
        }
    )
)
"""


def test_transform_product_clean_process_roundtrip(tmp_path):
    """A concrete transform product roundtrips through a clean process."""
    import json
    import subprocess
    import sys

    from qphase.data import save_products

    rng = np.random.default_rng(20261108)
    product = _field(rng, (4, 32, 2))
    _, out = _execute(
        {
            "units": "occupation",
            "observables": {"n0": {"matrix": [[1.0, 0.0], [0.0, 0.0]]}},
        },
        product,
    )
    save_products(tmp_path, {"soft_observable": out})

    probe = tmp_path / "probe.py"
    probe.write_text(_ROUNDTRIP_PROBE, encoding="utf-8")
    completed = subprocess.run(
        [sys.executable, str(probe), str(tmp_path)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout.strip())
    assert payload["axes"] == {
        "scan": ["parameter", 1, ""],
        "trajectory": ["realization", 4, ""],
        "time": ["coordinate", 32, "time"],
        "channel": ["component", 1, ""],
    }
    assert payload["value_dims"] == ["scan", "trajectory", "time", "channel"]
    assert payload["value_quantity"] == "sampled_observable"
    assert payload["value_units"] == "occupation"
    data = product.handle("alpha").materialize()
    assert payload["value_sum"] == pytest.approx(
        float((np.abs(data[..., 0]) ** 2).sum())
    )
    assert payload["valid_length"] == [[32, 32, 32, 32]]
    assert payload["channel_names"] == ["n0"]
    assert payload["matrices_used"] == [[["(1+0j)", "0j"], ["0j", "0j"]]]


@pytest.mark.gpu
@pytest.mark.skipif(not _cupy_available(), reason="CuPy not available")
def test_transform_cupy_matches_numpy():
    """Device execution keeps value on device and matches the host result."""
    import cupy as cp
    from qphase.backend.cupy_backend import CuPyBackend

    rng = np.random.default_rng(20261109)
    host_product = _field(rng, (4, 64, 2))
    device_product = _trajectory_product(
        TrajectorySet(
            cp.asarray(host_product.handle("alpha").materialize()[0]),
            t0=T0,
            dt=DT,
            meta={"mode_indices": [0, 1]},
        ),
        scan_size=None,
        n_traj_per_point=None,
    )
    config = {
        "units": "occupation",
        "observables": {
            "n0": {"matrix": [[1.0, 0.0], [0.0, 0.0]]},
            "x": {"matrix": [[0.0, 0.5], [0.5, 0.0]], "center": 0.25},
        },
    }
    _, host_out = _execute(config, host_product)
    analyzer = ObservableTransformAnalyzer(ObservableTransformConfig(**config))
    device_out = analyzer.execute(
        {"signal": device_product}, backend=CuPyBackend(), label="soft_observable"
    )["soft_observable"]

    device_value = device_out.handle("value").materialize()
    assert isinstance(device_value, cp.ndarray)
    np.testing.assert_allclose(
        cp.asnumpy(device_value),
        host_out.handle("value").materialize(),
        rtol=1e-12,
        atol=1e-14,
    )
    # valid_length reuses the source (host) handle untouched.
    np.testing.assert_array_equal(
        device_out.handle("valid_length").materialize(),
        host_out.handle("valid_length").materialize(),
    )
