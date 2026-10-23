"""Finite-delay coherence carrier reconstructed from saved power spectra."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar, Literal, cast

import numpy as np
from pydantic import Field, field_validator, model_validator
from qphase.backend.base import BackendBase
from qphase.core.errors import QPhaseError
from qphase.core.protocols import PluginConfigBase
from qphase.data import (
    DataKind,
    Dataset,
    ProductDeclaration,
    ProductRequirement,
    SpectralQuantity,
)

from ..contracts.analyser import (
    AnalyserExecutionCapabilities,
    AnalyserWorkspaceRequest,
    WorkEstimate,
)
from ..contracts.quantities import (
    DEFAULT_FREQUENCY_ORIENTATION,
    FrequencyOrientation,
)
from ..numerics.finite_delay import finite_delay_carrier_from_spectrum
from ..products import (
    assemble_row_table_product,
)
from .base import Analyser, check_execution_cancelled, report_execution_progress
from .frequency_orientation import (
    orientation_metadata,
    resolve_frequency_orientation,
)
from .spectral_input import SpectralPoint, SpectralScanInput, spectral_scan_input

__all__ = [
    "FiniteDelayCarrierAnalyser",
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
    source_data: SpectralPoint,
    readout: int | Literal["trace"],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    spectrum, _, _ = source_data.readout(readout)
    return source_data.axis, spectrum, dict(source_data.attributes)


def _build_finite_delay_carrier_products(
    payload: Any,
    *,
    config: FiniteDelayCarrierConfig,
    scan_size: int | None,
    label: str,
) -> dict[str, Dataset]:
    if not isinstance(payload, dict):
        raise TypeError("finite_delay_carrier payload must be a mapping")
    rows = payload.get("carrier_rows", [])
    if not isinstance(rows, list):
        raise TypeError("finite_delay_carrier carrier_rows must be a list")
    metadata = {key: value for key, value in payload.items() if key != "carrier_rows"}
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
        scan_value_field=config.scan_param if scan_size is not None else None,
        payload_key="carrier_rows",
        payload_meta=metadata,
        attributes={"table": "finite_delay_carrier"},
    )
    return {label: product}


class FiniteDelayCarrierAnalyser(Analyser):
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
        context: Any | None = None,
    ) -> Mapping[str, Dataset]:
        metadata = metadata or {}
        config = cast(FiniteDelayCarrierConfig, self.config)
        typed = spectral_scan_input(
            inputs,
            psd_key=config.psd_key,
            scan_param=config.scan_param,
            metadata=metadata,
        )
        payload, _ = self._compute_payload(typed, backend, context=context)
        products = _build_finite_delay_carrier_products(
            payload, config=config, scan_size=None, label=label
        )
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
    def _compute_payload(
        self,
        data: SpectralScanInput,
        backend: BackendBase,
        *,
        context: Any | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        del backend
        config = cast(FiniteDelayCarrierConfig, self.config)
        loaded = data.points
        if not loaded:
            raise QPhaseError("finite_delay_carrier received no input results")
        rows: list[dict[str, Any]] = []
        reference_orientation: FrequencyOrientation | None = None
        rates = np.asarray(config.detector_rates, dtype=float)
        readouts = config.resolved_readouts()
        spectra_done = 0
        for item in loaded:
            check_execution_cancelled(context)
            scan_value = item.scan_value
            job_name = item.job_name
            for readout in readouts:
                axis, spectrum, payload = _extract_spectrum(item, readout)
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
                            "job_name": job_name,
                            config.scan_param: scan_value,
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
            spectra_done += len(readouts)
            report_execution_progress(
                context,
                completed=spectra_done,
                unit="spectrum",
                message=f"finite_delay_carrier: estimated {spectra_done} spectra",
            )
        readout_order = {str(readout): index for index, readout in enumerate(readouts)}
        rows.sort(
            key=lambda row: (
                float(row[config.scan_param]),
                readout_order[str(row["readout"])],
                row["detector_rate"],
            )
        )
        metadata = orientation_metadata(
            reference_orientation or DEFAULT_FREQUENCY_ORIENTATION
        )
        return (
            {
                "carrier_rows": rows,
                "detector_rates": rates,
                **metadata,
            },
            {
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
