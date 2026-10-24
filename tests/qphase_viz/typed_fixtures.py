"""Typed bundle fixtures for ``qphase_viz`` tests.

Products are built directly from the public ``qphase.data`` schema API and
wrapped in :class:`VizTestBundle`, a minimal ``DatasetResultProtocol``
implementation, so the viz tests exercise nothing but qphase contracts.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from qphase.core.dataset import DatasetSaveReport
from qphase.data import (
    ARTIFACT_SCHEMA_VERSION,
    AxisRole,
    AxisSchema,
    CoordinateSchema,
    DataKind,
    Dataset,
    ProductSchema,
    SpectralDataset,
    SpectralQuantity,
    StatisticsDataset,
    TimeSeriesDataset,
    VariableSchema,
    load_products,
    save_products,
)

N_TRAJ = 3
N_TIME = 64
N_CHANNEL = 2
N_FREQUENCY = 32
DT = 0.1


class VizTestBundle:
    """Minimal ``ResultProtocol``/``DatasetResultProtocol`` test bundle.

    The plotters consume exactly ``products`` (named typed datasets), ``axes``
    (scan coordinates) and ``shape`` (scan grid), and call ``point_view`` to
    drop the scan axis. ``data``/``metadata``/``label``/``save`` and
    ``save_dataset`` complete the core result protocols.
    """

    def __init__(
        self,
        products: Mapping[str, Dataset],
        *,
        axes: Mapping[str, Any] | None = None,
        shape: tuple[int, ...] = (),
        meta: Mapping[str, Any] | None = None,
    ) -> None:
        self._products = dict(products)
        self._axes = dict(axes or {})
        self._shape = tuple(shape)
        self._meta = dict(meta or {})

    @property
    def products(self) -> dict[str, Dataset]:
        """Named typed products of the bundle."""
        return dict(self._products)

    @property
    def axes(self) -> dict[str, Any]:
        """Scan-axis coordinates (empty for single-point bundles)."""
        return dict(self._axes)

    @property
    def shape(self) -> tuple[int, ...]:
        """Scan grid shape (empty for single-point bundles)."""
        return self._shape

    @property
    def data(self) -> dict[str, Dataset]:
        """Product mapping exposed through ``ResultProtocol``."""
        return dict(self._products)

    @property
    def metadata(self) -> dict[str, Any]:
        """Bundle metadata (JSON-serializable)."""
        return dict(self._meta)

    @property
    def label(self) -> Any:
        """Optional bundle label."""
        return self._meta.get("label")

    def point_view(self, index: tuple[int, ...]) -> VizTestBundle:
        """Per-scan-point view: products lose their scan axis."""
        if not self._shape:
            if tuple(index) == ():
                return self
            raise IndexError("bundle has no scan axes")
        flat = int(np.ravel_multi_index(tuple(index), self._shape))
        return VizTestBundle(
            {
                name: (
                    product.point_view(scan=flat)
                    if "scan" in {axis.name for axis in product.axes}
                    else product
                )
                for name, product in self._products.items()
            },
            meta={**self._meta, "scan_index": flat},
        )

    def save(self, path: str | Path) -> None:
        """Persist the products through the generic artifact store."""
        save_products(
            Path(path),
            self._products,
            provenance=self._meta,
            layout="single",
        )

    def save_dataset(
        self,
        path: str | Path,
        *,
        layout: str,
        shard_target_bytes: int,
    ) -> DatasetSaveReport:
        """Persist through the generic artifact store (protocol surface)."""
        resolved = "single" if layout == "single" else "sharded"
        save_products(
            Path(path),
            self._products,
            provenance=self._meta,
            shard_target_bytes=shard_target_bytes,
            layout=resolved,
        )
        files = tuple(sorted(item for item in Path(path).rglob("*") if item.is_file()))
        return DatasetSaveReport(
            resolved,
            files,
            loader="npz",
            schema_version=ARTIFACT_SCHEMA_VERSION,
        )


def _scan_axis(scan_size: int) -> AxisSchema:
    return AxisSchema(name="scan", role=AxisRole.PARAMETER, size=scan_size)


def _scan_coordinate(
    scan_values: tuple[float, ...],
) -> tuple[list[VariableSchema], list[CoordinateSchema], dict[str, Any]]:
    """Declare the scan parameter coordinate inside the product schema."""
    if not scan_values:
        return [], [], {}
    values = np.asarray(scan_values, dtype=float)
    variable = VariableSchema(
        name="parameter.omega",
        dtype=values.dtype.str,
        value_domain="real",
        dims=("scan",),
        quantity="scan_parameter",
    )
    coordinate = CoordinateSchema(
        name="omega",
        variable="parameter.omega",
        dims=("scan",),
        role="parameter",
    )
    return [variable], [coordinate], {"parameter.omega": values}


def trajectory_dataset(*, scan_values: tuple[float, ...] = ()) -> Dataset:
    """Trajectory product with ``alpha[scan, trajectory, time, channel]``."""
    scan_size = len(scan_values) or 1
    time = np.arange(N_TIME) * DT
    alpha = np.empty((scan_size, N_TRAJ, N_TIME, N_CHANNEL), dtype=np.complex128)
    for scan in range(scan_size):
        for channel in range(N_CHANNEL):
            signal = np.exp(1j * (channel + 1 + 0.1 * scan) * time)
            alpha[scan, :, :, channel] = signal
    scan_variables, scan_coordinates, arrays = _scan_coordinate(scan_values)
    schema = ProductSchema(
        kind=DataKind.TIME_SERIES,
        axes=[
            _scan_axis(scan_size),
            AxisSchema(name="trajectory", role=AxisRole.REALIZATION, size=N_TRAJ),
            AxisSchema(
                name="time",
                role=AxisRole.COORDINATE,
                size=N_TIME,
                coordinate="regular",
                start=0.0,
                step=DT,
                units="time",
            ),
            AxisSchema(name="channel", role=AxisRole.COMPONENT, size=N_CHANNEL),
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
            *scan_variables,
        ],
        coordinates=scan_coordinates,
        attributes={"graph_ready": True},
    )
    return TimeSeriesDataset.from_arrays(
        schema,
        {
            "alpha": alpha,
            "valid_length": np.full((scan_size, N_TRAJ), N_TIME, dtype=np.int64),
            **arrays,
        },
        owner="test.viz",
    )


def spectrum_dataset(*, scan_values: tuple[float, ...] = ()) -> Dataset:
    """Power-spectral-density product with ``psd[scan, frequency, mode]``."""
    scan_size = len(scan_values) or 1
    frequency = np.linspace(0.0, 8.0, N_FREQUENCY)
    psd = np.empty((scan_size, N_FREQUENCY, N_CHANNEL), dtype=float)
    for scan in range(scan_size):
        for mode in range(N_CHANNEL):
            peak = mode + 1.0 + 0.1 * scan
            psd[scan, :, mode] = 1.0 / (1.0 + (frequency - peak) ** 2)
    scan_variables, scan_coordinates, arrays = _scan_coordinate(scan_values)
    schema = ProductSchema(
        kind=DataKind.SPECTRAL,
        axes=[
            _scan_axis(scan_size),
            AxisSchema(
                name="frequency",
                role=AxisRole.COORDINATE,
                size=N_FREQUENCY,
                coordinate="regular",
                start=float(frequency[0]),
                step=float(frequency[1] - frequency[0]),
                units="Hz",
            ),
            AxisSchema(name="mode", role=AxisRole.COMPONENT, size=N_CHANNEL),
        ],
        variables=[
            VariableSchema(
                name="axis",
                dtype="<f8",
                value_domain="real",
                dims=("scan", "frequency"),
                quantity="frequency",
                units="Hz",
            ),
            VariableSchema(
                name="psd",
                dtype="<f8",
                value_domain="real",
                dims=("scan", "frequency", "mode"),
                quantity=SpectralQuantity.POWER_SPECTRAL_DENSITY.value,
            ),
            VariableSchema(
                name="modes",
                dtype="<i8",
                value_domain="real",
                dims=("mode",),
                quantity="mode_index",
            ),
            *scan_variables,
        ],
        coordinates=scan_coordinates,
        attributes={
            "frequency_units": "Hz",
            "orientation": "phase_decreasing",
            "sidedness": "two_sided",
            "normalization": "power_per_hz",
            "window": "rectangular",
            "estimator": "periodogram",
            "graph_ready": True,
        },
    )
    return SpectralDataset.from_arrays(
        schema,
        {
            "axis": np.broadcast_to(frequency, (scan_size, N_FREQUENCY)).copy(),
            "psd": psd,
            "modes": np.arange(N_CHANNEL, dtype=np.int64),
            **arrays,
        },
        owner="test.viz",
    )


def peak_dataset(*, scan_values: tuple[float, ...]) -> Dataset:
    """Build a grouped peak-candidate product over the scan values."""
    scan_size = len(scan_values)
    group_scan_index: list[int] = []
    group_channel_index: list[int] = []
    candidate_offsets = [0]
    locations: list[float] = []
    intensities: list[float] = []
    for scan_index, value in enumerate(scan_values):
        group_scan_index.append(scan_index)
        group_channel_index.append(0)
        locations.extend([value, value + 0.5])
        intensities.extend([2.0, 1.0])
        candidate_offsets.append(len(locations))
    scan_variables, scan_coordinates, arrays = _scan_coordinate(scan_values)
    schema = ProductSchema(
        kind=DataKind.STATISTICS,
        axes=[
            _scan_axis(scan_size),
            AxisSchema(name="group", role=AxisRole.INDEX, size=len(group_scan_index)),
            AxisSchema(
                name="group_offset", role=AxisRole.INDEX, size=len(candidate_offsets)
            ),
            AxisSchema(name="candidate", role=AxisRole.INDEX, size=len(locations)),
        ],
        variables=[
            VariableSchema(
                name="group_scan_index",
                dtype="<i8",
                value_domain="real",
                dims=("group",),
                quantity="scan_index",
            ),
            VariableSchema(
                name="group_channel_index",
                dtype="<i8",
                value_domain="real",
                dims=("group",),
                quantity="channel_index",
            ),
            VariableSchema(
                name="candidate_offsets",
                dtype="<i8",
                value_domain="real",
                dims=("group_offset",),
                quantity="candidate_offset",
            ),
            VariableSchema(
                name="location",
                dtype="<f8",
                value_domain="real",
                dims=("candidate",),
                quantity="peak_location",
                units="Hz",
            ),
            VariableSchema(
                name="intensity",
                dtype="<f8",
                value_domain="real",
                dims=("candidate",),
                quantity="peak_intensity",
            ),
            *scan_variables,
        ],
        coordinates=scan_coordinates,
        attributes={
            "channel_names": ["mode_0"],
            "mode_labels": ["mode_0"],
            "graph_ready": True,
            "source_finder": "typed",
        },
    )
    return StatisticsDataset.from_arrays(
        schema,
        {
            "group_scan_index": np.asarray(group_scan_index, dtype=np.int64),
            "group_channel_index": np.asarray(group_channel_index, dtype=np.int64),
            "candidate_offsets": np.asarray(candidate_offsets, dtype=np.int64),
            "location": np.asarray(locations, dtype=float),
            "intensity": np.asarray(intensities, dtype=float),
            **arrays,
        },
        owner="test.viz",
    )


def _scan_grid(scan_values: tuple[float, ...]) -> dict[str, Any]:
    if not scan_values:
        return {}
    return {
        "axes": {"omega": np.asarray(scan_values, dtype=float)},
        "shape": (len(scan_values),),
    }


def trajectory_bundle(*, scan_values: tuple[float, ...] = ()) -> VizTestBundle:
    """Bundle carrying only the trajectory product."""
    return VizTestBundle(
        {"trajectories": trajectory_dataset(scan_values=scan_values)},
        **_scan_grid(scan_values),
    )


def spectrum_bundle() -> VizTestBundle:
    """Single-point bundle carrying trajectories plus one PSD product."""
    return VizTestBundle(
        {
            "trajectories": trajectory_dataset(),
            "psd": spectrum_dataset(),
        }
    )


def spectrum_scan_bundle() -> VizTestBundle:
    """Scan bundle carrying trajectories plus a per-scan PSD product."""
    scan = (0.1, 0.2, 0.3)
    return VizTestBundle(
        {
            "trajectories": trajectory_dataset(scan_values=scan),
            "psd": spectrum_dataset(scan_values=scan),
        },
        **_scan_grid(scan),
    )


def peak_scan_bundle() -> VizTestBundle:
    """Scan bundle carrying trajectories plus grouped peak candidates."""
    scan = (0.1, 0.2, 0.3)
    return VizTestBundle(
        {
            "trajectories": trajectory_dataset(scan_values=scan),
            "peaks": peak_dataset(scan_values=scan),
        },
        **_scan_grid(scan),
    )


def restored(bundle: VizTestBundle, path: Path) -> VizTestBundle:
    """Round-trip the bundle through the generic artifact store."""
    bundle.save(path)
    return VizTestBundle(
        load_products(path),
        axes=bundle.axes,
        shape=bundle.shape,
        meta=bundle.metadata,
    )


#: Scan values of the multi-point fixtures below.
SCAN_POINTS = (0.1, 0.2, 0.3, 0.4, 0.5)


def spectral_scan_dataset(
    *, scan_values: tuple[float, ...] = SCAN_POINTS
) -> Dataset:
    """PSD scan product: ``psd[scan, frequency, mode]`` with a scan coordinate.

    Each mode carries a Lorentzian peak that moves with the scan coordinate, so
    a rendered panel visibly changes along the scan.
    """
    scan_size = len(scan_values)
    frequency = np.linspace(-4.0, 4.0, N_FREQUENCY)
    psd = np.empty((scan_size, N_FREQUENCY, N_CHANNEL), dtype=float)
    for scan, value in enumerate(scan_values):
        for mode in range(N_CHANNEL):
            peak = value * (mode + 1)
            psd[scan, :, mode] = 1.0 / (1.0 + (frequency - peak) ** 2)
    scan_variables, scan_coordinates, arrays = _scan_coordinate(scan_values)
    schema = ProductSchema(
        kind=DataKind.SPECTRAL,
        axes=[
            _scan_axis(scan_size),
            AxisSchema(
                name="frequency",
                role=AxisRole.COORDINATE,
                size=N_FREQUENCY,
                coordinate="regular",
                start=float(frequency[0]),
                step=float(frequency[1] - frequency[0]),
                units="Hz",
            ),
            AxisSchema(name="mode", role=AxisRole.COMPONENT, size=N_CHANNEL),
        ],
        variables=[
            VariableSchema(
                name="axis",
                dtype="<f8",
                value_domain="real",
                dims=("scan", "frequency"),
                quantity="frequency",
                units="Hz",
            ),
            VariableSchema(
                name="psd",
                dtype="<f8",
                value_domain="real",
                dims=("scan", "frequency", "mode"),
                quantity=SpectralQuantity.POWER_SPECTRAL_DENSITY.value,
            ),
            VariableSchema(
                name="modes",
                dtype="<i8",
                value_domain="real",
                dims=("mode",),
                quantity="mode_index",
            ),
            *scan_variables,
        ],
        coordinates=scan_coordinates,
        attributes={
            "frequency_units": "Hz",
            "orientation": "phase_decreasing",
            "sidedness": "two_sided",
            "normalization": "power_per_hz",
            "window": "rectangular",
            "estimator": "periodogram",
            "graph_ready": True,
        },
    )
    return SpectralDataset.from_arrays(
        schema,
        {
            "axis": np.broadcast_to(frequency, (scan_size, N_FREQUENCY)).copy(),
            "psd": psd,
            "modes": np.arange(N_CHANNEL, dtype=np.int64),
            **arrays,
        },
        owner="test.viz",
    )


def spectral_scan_bundle() -> VizTestBundle:
    """Multi-point bundle carrying a per-scan PSD product."""
    return VizTestBundle(
        {"psd": spectral_scan_dataset()},
        axes={"omega": np.asarray(SCAN_POINTS, dtype=float)},
        shape=(len(SCAN_POINTS),),
    )


#: Grid of the scalar-heatmap fixture.
HEATMAP_X = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
HEATMAP_Y = (10.0, 11.25, 12.5, 13.75, 15.0)


def heatmap_dataset() -> Dataset:
    """Statistics product carrying a scalar grid plus overlays.

    ``value[row, column]`` is indexed by ``x[rows]`` and ``y[columns]``; the
    mask blanks the upper-right triangle, and the product also carries the
    variables an overlay needs (``height``, ``boundary``, ``marker_x``,
    ``marker_y``).
    """
    x = np.asarray(HEATMAP_X, dtype=float)
    y = np.asarray(HEATMAP_Y, dtype=float)
    value = np.add.outer(x, y)
    valid = ~(np.add.outer(np.arange(x.size), np.arange(y.size)) >= 6)
    height = np.add.outer(x * 2.0, y * 0.5)
    schema = ProductSchema(
        kind=DataKind.STATISTICS,
        axes=[
            AxisSchema(name="row", role=AxisRole.PARAMETER, size=x.size),
            AxisSchema(name="column", role=AxisRole.PARAMETER, size=y.size),
            AxisSchema(name="boundary", role=AxisRole.INDEX, size=3),
            AxisSchema(name="marker", role=AxisRole.INDEX, size=2),
        ],
        variables=[
            VariableSchema(
                name="x",
                dtype="<f8",
                value_domain="real",
                dims=("row",),
                quantity="parameter_value",
            ),
            VariableSchema(
                name="y",
                dtype="<f8",
                value_domain="real",
                dims=("column",),
                quantity="parameter_value",
            ),
            VariableSchema(
                name="value",
                dtype="<f8",
                value_domain="real",
                dims=("row", "column"),
                quantity="scalar_field",
            ),
            VariableSchema(
                name="height",
                dtype="<f8",
                value_domain="real",
                dims=("row", "column"),
                quantity="scalar_field",
            ),
            VariableSchema(
                name="valid",
                dtype="|b1",
                value_domain="real",
                dims=("row", "column"),
                quantity="validity_mask",
            ),
            VariableSchema(
                name="boundary_x",
                dtype="<f8",
                value_domain="real",
                dims=("boundary",),
                quantity="parameter_value",
            ),
            VariableSchema(
                name="boundary_y",
                dtype="<f8",
                value_domain="real",
                dims=("boundary",),
                quantity="parameter_value",
            ),
            VariableSchema(
                name="marker_x",
                dtype="<f8",
                value_domain="real",
                dims=("marker",),
                quantity="parameter_value",
            ),
            VariableSchema(
                name="marker_y",
                dtype="<f8",
                value_domain="real",
                dims=("marker",),
                quantity="parameter_value",
            ),
        ],
        attributes={"graph_ready": True},
    )
    return StatisticsDataset.from_arrays(
        schema,
        {
            "x": x,
            "y": y,
            "value": value,
            "height": height,
            "valid": valid,
            "boundary_x": np.asarray([0.2, 0.5, 0.8]),
            "boundary_y": np.asarray([12.0, 13.0, 14.0]),
            "marker_x": np.asarray([0.0, 1.0]),
            "marker_y": np.asarray([10.0, 15.0]),
        },
        owner="test.viz",
    )


def heatmap_bundle() -> VizTestBundle:
    """Single-point bundle carrying the scalar-heatmap product."""
    return VizTestBundle({"grid": heatmap_dataset()})


#: Realizations per scan point in the distribution fixture.
N_REALIZATION = 400


def distribution_dataset(
    *, scan_values: tuple[float, ...] = SCAN_POINTS
) -> Dataset:
    """Statistics product of per-realization samples: ``samples[scan, realization]``.

    Each scan point observes a normal sample with a mean that shifts with the
    scan coordinate, so a two-point comparison has a non-zero shift. A few
    non-finite samples are planted so the plotter's per-series filtering is
    exercised.
    """
    scan_size = len(scan_values)
    rng = np.random.default_rng(20261021)
    samples = np.empty((scan_size, N_REALIZATION), dtype=float)
    for scan in range(scan_size):
        samples[scan] = rng.normal(
            loc=0.1 * (scan - 1), scale=0.01, size=N_REALIZATION
        )
    samples[0, :3] = np.nan
    scan_variables, scan_coordinates, arrays = _scan_coordinate(scan_values)
    schema = ProductSchema(
        kind=DataKind.STATISTICS,
        axes=[
            _scan_axis(scan_size),
            AxisSchema(
                name="realization", role=AxisRole.REALIZATION, size=N_REALIZATION
            ),
        ],
        variables=[
            VariableSchema(
                name="estimate",
                dtype="<f8",
                value_domain="real",
                dims=("scan", "realization"),
                quantity="frequency_estimate",
            ),
            *scan_variables,
        ],
        coordinates=scan_coordinates,
        attributes={"graph_ready": True},
    )
    return StatisticsDataset.from_arrays(
        schema,
        {"estimate": samples, **arrays},
        owner="test.viz",
    )


def distribution_bundle() -> VizTestBundle:
    """Multi-point bundle carrying a per-realization sample product."""
    return VizTestBundle(
        {"estimates": distribution_dataset()},
        axes={"omega": np.asarray(SCAN_POINTS, dtype=float)},
        shape=(len(SCAN_POINTS),),
    )
