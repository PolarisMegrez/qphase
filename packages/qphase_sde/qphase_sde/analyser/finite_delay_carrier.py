"""Finite-delay coherence carrier reconstructed from saved power spectra."""

from __future__ import annotations

import csv
from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar, Literal, cast

import numpy as np
from pydantic import Field, field_validator, model_validator
from qphase.backend.base import BackendBase
from qphase.core.errors import QPhaseError
from qphase.core.protocols import PluginConfigBase, ResultProtocol
from qphase.data import (
    DataKind,
    Dataset,
    ProductDeclaration,
    ProductRequirement,
    SpectralQuantity,
)

from ..coherence_frequency.math.finite_delay import finite_delay_carrier_from_spectrum
from ..contracts.analyser import (
    AnalyserExecutionCapabilities,
    AnalyserWorkspaceRequest,
    WorkEstimate,
)
from ..products import (
    assemble_row_table_product,
)
from .base import Analyzer
from .frequency_orientation import (
    FrequencyOrientation,
    orientation_metadata,
    resolve_frequency_orientation,
)
from .result import AnalysisResult
from .result_input import load_sde_results
from .typed_input import TypedScanInput, typed_scan_input

__all__ = [
    "FiniteDelayCarrierAnalyzer",
    "FiniteDelayCarrierConfig",
    "finite_delay_carrier_from_spectrum",
]


class FiniteDelayCarrierConfig(PluginConfigBase):
    """Configuration for coherence-weighted finite-delay carriers."""

    scan_param: str = Field(..., description="Parameter used as the scan axis")
    psd_key: str = Field("psd", description="Analysis key containing the PSD")
    readout: int | Literal["trace"] = Field(
        "trace",
        description="Legacy single physical mode or incoherent trace selection",
    )
    readouts: list[int | Literal["trace"]] | None = Field(
        None,
        description=(
            "Physical mode indices and/or incoherent trace selections evaluated in "
            "one dataset pass; overrides readout when provided"
        ),
    )
    detector_rates: list[float] = Field(
        default_factory=lambda: [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0],
        description="Positive exponential detector rates kappa",
    )
    maximum_lag: float = Field(
        4096.0,
        gt=0.0,
        description="Maximum physical lag retained in numerical quadrature",
    )
    tail_time_constants: float = Field(
        12.0,
        gt=1.0,
        description="Detector-weight time constants retained before truncation",
    )
    export: list[str] = Field(default_factory=lambda: ["finite_delay_carrier.csv"])
    output_dir: str | None = Field(None, description="Usually injected by the engine")

    @field_validator("detector_rates")
    @classmethod
    def validate_detector_rates(cls, values: list[float]) -> list[float]:
        if not values or any(
            not np.isfinite(value) or value <= 0.0 for value in values
        ):
            raise ValueError("detector_rates must contain positive finite values")
        if any(right <= left for left, right in zip(values, values[1:], strict=False)):
            raise ValueError("detector_rates must be strictly increasing")
        return values

    @model_validator(mode="after")
    def validate_readout(self) -> FiniteDelayCarrierConfig:
        selections = self.resolved_readouts()
        if not selections:
            raise ValueError("readouts must not be empty")
        if any(isinstance(value, int) and value < 0 for value in selections):
            raise ValueError("readout modes must be non-negative")
        keys = [str(value) for value in selections]
        if len(keys) != len(set(keys)):
            raise ValueError("readouts must contain unique selections")
        return self

    def resolved_readouts(self) -> list[int | Literal["trace"]]:
        """Return the ordered readouts requested for one dataset traversal."""
        return list(self.readouts) if self.readouts is not None else [self.readout]


def _extract_spectrum(
    analysis: dict[str, Any], psd_key: str, readout: int | Literal["trace"]
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if psd_key not in analysis:
        raise QPhaseError(f"analysis is missing PSD payload {psd_key!r}")
    payload = analysis[psd_key]
    axis = np.asarray(payload["axis"], dtype=float).reshape(-1)
    matrix = np.asarray(payload["psd"], dtype=float)
    if matrix.ndim == 1:
        spectrum = matrix
    elif matrix.ndim == 2 and readout == "trace":
        spectrum = np.sum(matrix, axis=1)
    elif matrix.ndim == 2 and isinstance(readout, int):
        modes = [int(mode) for mode in payload.get("modes", [])]
        if readout in modes:
            column = modes.index(readout)
        elif readout < matrix.shape[1]:
            column = readout
        else:
            raise QPhaseError(f"mode {readout} is absent from PSD payload")
        spectrum = matrix[:, column]
    else:
        raise QPhaseError("PSD payload has an unsupported shape")
    return axis, spectrum, payload


class FiniteDelayCarrierAnalyzer(Analyzer):
    """Compute detector-weighted carriers from logical PSD scan datasets."""

    name: ClassVar[str] = "finite_delay_carrier"
    description: ClassVar[str] = "Finite-delay coherence-weighted carrier"
    config_schema: ClassVar[type[FiniteDelayCarrierConfig]] = FiniteDelayCarrierConfig

    def execute(
        self,
        inputs: Mapping[str, Dataset],
        *,
        backend: BackendBase,
        label: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Dataset]:
        metadata = metadata or {}
        config = cast(FiniteDelayCarrierConfig, self.config)
        typed = typed_scan_input(
            inputs, scan_param=config.scan_param, metadata=metadata
        )
        result = self.analyze(typed, backend)
        products = self.build_products(result.data_dict, scan_size=None, label=label)
        if not products:
            raise ValueError(f"analyser {label!r} produced no typed products")
        return products

    def execution_capabilities(self) -> AnalyserExecutionCapabilities:
        return AnalyserExecutionCapabilities(preferred_location="host")

    def work_estimate(self, request: AnalyserWorkspaceRequest) -> WorkEstimate:
        """Work scales with the number of loaded PSD spectra; no total is fabricated."""
        del request
        return WorkEstimate(unit="spectrum", total=None)

    def input_requirements(self) -> list[ProductRequirement]:
        config = cast(FiniteDelayCarrierConfig, self.config)
        required = ["axis", "psd"]
        if any(isinstance(readout, int) for readout in config.resolved_readouts()):
            required.append("modes")
        return [
            ProductRequirement(
                name=config.psd_key,
                kind=DataKind.SPECTRAL,
                quantity=SpectralQuantity.POWER_SPECTRAL_DENSITY.value,
                required_fields=required,
            )
        ]

    def output_specs(self, *, label: str) -> tuple[ProductDeclaration, ...]:
        return (
            ProductDeclaration(
                name=label,
                kind=DataKind.STATISTICS,
                fields=[
                    "detector_rate",
                    "frequency",
                    "instantaneous_frequency",
                    "finite_delay_correction",
                    "coherent_weight",
                    "lag_end",
                    "sample_count",
                    "sample_lag",
                ],
            ),
        )

    def build_products(
        self, payload: Any, *, scan_size: int | None, label: str
    ) -> dict[str, Dataset] | None:
        if not isinstance(payload, dict):
            raise TypeError("finite_delay_carrier payload must be a mapping")
        rows = payload.get("carrier_rows", [])
        if not isinstance(rows, list):
            raise TypeError("finite_delay_carrier carrier_rows must be a list")
        metadata = {
            key: value for key, value in payload.items() if key != "carrier_rows"
        }
        product = assemble_row_table_product(
            label,
            rows,
            scan_size=scan_size,
            fields=[
                "detector_rate",
                "frequency",
                "instantaneous_frequency",
                "finite_delay_correction",
                "coherent_weight",
                "lag_end",
                "sample_count",
                "sample_lag",
            ],
            integer_fields=("sample_count",),
            scan_value_field=(
                cast(FiniteDelayCarrierConfig, self.config).scan_param
                if scan_size is not None
                else None
            ),
            payload_key="carrier_rows",
            payload_meta=metadata,
            attributes={"table": "finite_delay_carrier"},
        )
        return {label: product}

    def analyze(self, data: Any, backend: BackendBase) -> ResultProtocol:
        del backend
        config = cast(FiniteDelayCarrierConfig, self.config)
        loaded = (
            data.points
            if isinstance(data, TypedScanInput)
            else load_sde_results(data)
        )
        if not loaded:
            raise QPhaseError("finite_delay_carrier received no input results")
        rows: list[dict[str, Any]] = []
        reference_orientation: FrequencyOrientation | None = None
        rates = np.asarray(config.detector_rates, dtype=float)
        readouts = config.resolved_readouts()
        for item in loaded:
            params = item.meta.get("params", {})
            if config.scan_param not in params:
                raise QPhaseError(
                    f"input is missing scan parameter {config.scan_param!r}"
                )
            for readout in readouts:
                axis, spectrum, payload = _extract_spectrum(
                    item.analysis, config.psd_key, readout
                )
                orientation = resolve_frequency_orientation(payload)
                if reference_orientation is None:
                    reference_orientation = orientation
                elif orientation != reference_orientation:
                    raise QPhaseError("PSD inputs use different frequency orientations")
                result = finite_delay_carrier_from_spectrum(
                    axis,
                    spectrum,
                    rates,
                    maximum_lag=config.maximum_lag,
                    tail_time_constants=config.tail_time_constants,
                )
                frequencies = np.asarray(result["frequency"], dtype=float)
                coherent_weights = np.asarray(result["coherent_weight"], dtype=float)
                lag_ends = np.asarray(result["lag_end"], dtype=float)
                sample_counts = np.asarray(result["sample_count"], dtype=int)
                instantaneous = float(result["instantaneous_frequency"])
                for rate_index, rate in enumerate(rates):
                    rows.append(
                        {
                            "job_name": item.job_name,
                            config.scan_param: params[config.scan_param],
                            "readout": readout,
                            "measurement_name": (
                                "trace" if readout == "trace" else f"mode_{readout}"
                            ),
                            "measurement_kind": (
                                "incoherent_trace"
                                if readout == "trace"
                                else "bare_mode"
                            ),
                            "detector_rate": float(rate),
                            "frequency": float(frequencies[rate_index]),
                            "instantaneous_frequency": instantaneous,
                            "finite_delay_correction": float(
                                frequencies[rate_index] - instantaneous
                            ),
                            "coherent_weight": float(coherent_weights[rate_index]),
                            "lag_end": float(lag_ends[rate_index]),
                            "sample_count": int(sample_counts[rate_index]),
                            "sample_lag": float(result["sample_lag"]),
                            "orientation": orientation,
                        }
                    )
        readout_order = {str(readout): index for index, readout in enumerate(readouts)}
        rows.sort(
            key=lambda row: (
                float(row[config.scan_param]),
                readout_order[str(row["readout"])],
                row["detector_rate"],
            )
        )
        written: dict[str, str] = {}
        output_dir = getattr(self, "output_dir", None) or config.output_dir
        if output_dir is not None and "finite_delay_carrier.csv" in config.export:
            destination = Path(output_dir)
            destination.mkdir(parents=True, exist_ok=True)
            path = destination / "finite_delay_carrier.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            written["finite_delay_carrier"] = str(path)
        metadata = orientation_metadata(
            reference_orientation or resolve_frequency_orientation(None)
        )
        return AnalysisResult(
            data_dict={
                "carrier_rows": rows,
                "detector_rates": rates,
                "written": written,
                **metadata,
            },
            meta={
                "scan_param": config.scan_param,
                "readout": readouts[0] if len(readouts) == 1 else None,
                "readouts": readouts,
                "definition": (
                    "integral w Im(conj(G) dG/dtau) / integral w |G|^2; "
                    "w=exp(-2*kappa*tau)"
                ),
                "uncertainty": "unavailable from an ensemble-mean PSD alone",
                **metadata,
            },
        )
