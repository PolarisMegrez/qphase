"""Typed SDE bundle fixtures shared by qphase_viz tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from qphase.backend.numpy_backend import NumpyBackend
from qphase.data import (
    AxisRole,
    AxisSchema,
    DataKind,
    ProductSchema,
    TimeSeriesDataset,
    VariableSchema,
    load_bundle,
)
from qphase_sde.analyser.psd import PsdAnalyzer
from qphase_sde.contracts.bundle import SDEProvenance
from qphase_sde.peak_finder import materialize_candidate_product
from qphase_sde.products import add_scan_parameter_coordinates
from qphase_sde.result import SDEDataBundle


def trajectory_bundle(*, scan_values: tuple[float, ...] = ()) -> SDEDataBundle:
    scan_size = len(scan_values) or 1
    n_traj, n_time, n_channel = 3, 64, 2
    time = np.arange(n_time) * 0.1
    alpha = np.empty((scan_size, n_traj, n_time, n_channel), dtype=np.complex128)
    for scan in range(scan_size):
        for channel in range(n_channel):
            signal = np.exp(1j * (channel + 1 + 0.1 * scan) * time)
            alpha[scan, :, :, channel] = signal
    schema = ProductSchema(
        kind=DataKind.TIME_SERIES,
        axes=[
            AxisSchema(name="scan", role=AxisRole.PARAMETER, size=scan_size),
            AxisSchema(name="trajectory", role=AxisRole.REALIZATION, size=n_traj),
            AxisSchema(
                name="time",
                role=AxisRole.COORDINATE,
                size=n_time,
                coordinate="regular",
                start=0.0,
                step=0.1,
            ),
            AxisSchema(name="channel", role=AxisRole.COMPONENT, size=n_channel),
        ],
        variables=[
            VariableSchema(
                name="alpha",
                dtype="<c16",
                value_domain="complex",
                dims=("scan", "trajectory", "time", "channel"),
                quantity="field_amplitude",
            ),
            VariableSchema(
                name="valid_length",
                dtype="<i8",
                value_domain="real",
                dims=("scan", "trajectory"),
                quantity="valid_length",
            ),
        ],
        attributes={"graph_ready": True},
    )
    trajectory = TimeSeriesDataset.from_arrays(
        schema,
        {
            "alpha": alpha,
            "valid_length": np.full((scan_size, n_traj), n_time, dtype=np.int64),
        },
        owner="test.viz",
    )
    products = {"trajectories": trajectory}
    axes = {"omega": np.asarray(scan_values)} if scan_values else {}
    shape = (scan_size,) if scan_values else ()
    if scan_values:
        products["trajectories"] = add_scan_parameter_coordinates(
            trajectory, {"omega": np.asarray(scan_values)}
        )
    return SDEDataBundle(
        products,
        SDEProvenance(dt=0.1),
        scan_axes=axes,
        scan_shape=shape,
        meta={"scan_combine": "zipped"} if scan_values else {},
        n_traj_per_point=n_traj,
    )


def peak_scan_bundle() -> SDEDataBundle:
    scan = (0.1, 0.2, 0.3)
    base = trajectory_bundle(scan_values=scan)
    points = [
        {
            0: {
                "frequencies": [value, value + 0.5],
                "values": [2.0, 1.0],
            }
        }
        for value in scan
    ]
    peaks = materialize_candidate_product(
        points,
        scan_size=len(scan),
        label="peaks",
        modes=[0],
        source_product="psd",
    )
    peaks = add_scan_parameter_coordinates(peaks, {"omega": np.asarray(scan)})
    return SDEDataBundle(
        {**base.products, "peaks": peaks},
        base.provenance,
        scan_axes={"omega": np.asarray(scan)},
        scan_shape=(len(scan),),
        meta={"scan_combine": "zipped"},
        n_traj_per_point=base.n_traj_per_point,
    )


def spectrum_bundle() -> SDEDataBundle:
    base = trajectory_bundle()
    spectrum = PsdAnalyzer(kind="complex", modes=[0, 1]).execute(
        {"trajectories": base.products["trajectories"]},
        backend=NumpyBackend(),
        label="psd",
    )["psd"]
    return SDEDataBundle(
        {**base.products, "psd": spectrum},
        base.provenance,
        n_traj_per_point=base.n_traj_per_point,
    )


def spectrum_scan_bundle() -> SDEDataBundle:
    scan = (0.1, 0.2, 0.3)
    base = trajectory_bundle(scan_values=scan)
    spectrum = PsdAnalyzer(kind="complex", modes=[0, 1]).execute(
        {"trajectories": base.products["trajectories"]},
        backend=NumpyBackend(),
        label="psd",
    )["psd"]
    spectrum = add_scan_parameter_coordinates(
        spectrum, {"omega": np.asarray(scan)}
    )
    return SDEDataBundle(
        {**base.products, "psd": spectrum},
        base.provenance,
        scan_axes={"omega": np.asarray(scan)},
        scan_shape=(len(scan),),
        meta={"scan_combine": "zipped"},
        n_traj_per_point=base.n_traj_per_point,
    )


def restored(bundle: SDEDataBundle, path: Path) -> SDEDataBundle:
    bundle.save(path)
    result = load_bundle(path)
    assert isinstance(result, SDEDataBundle)
    return result
