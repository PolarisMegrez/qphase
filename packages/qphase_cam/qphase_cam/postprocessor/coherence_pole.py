"""Channel-visible first-order coherence poles."""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
from pydantic import Field, field_serializer, model_validator

from .base import CAMPostprocessor, CAMPostprocessorConfig
from .frequency import _successful_indices


class CoherencePoleConfig(CAMPostprocessorConfig):
    """Configuration for channel-resolved dominant coherence poles."""

    modes: list[int] = Field(
        default_factory=list,
        description="Bare-mode readout indices to include",
    )
    channels: dict[str, list[complex]] = Field(
        default_factory=dict,
        description="Named fixed coherent readout vectors l for c=l^dagger alpha",
    )
    include_trace: bool = Field(
        True,
        description="Include the incoherent equal-weight trace readout",
    )
    relative_residue_floor: float = Field(
        1.0e-3,
        gt=0.0,
        le=1.0,
        description="Minimum pole residue relative to the strongest channel residue",
    )

    @model_validator(mode="after")
    def validate_measurements(self) -> CoherencePoleConfig:
        names = [f"mode_{mode}" for mode in self.modes]
        if self.include_trace:
            names.append("trace")
        names.extend(self.channels)
        if not names:
            raise ValueError("configure at least one mode, channel, or trace")
        if len(names) != len(set(names)):
            raise ValueError("coherence-pole measurement names must be unique")
        if len(self.modes) != len(set(self.modes)) or any(
            mode < 0 for mode in self.modes
        ):
            raise ValueError("modes must contain unique non-negative indices")
        return self

    @field_serializer("channels")
    def serialize_channels(
        self, channels: dict[str, list[complex]]
    ) -> dict[str, list[str]]:
        return {
            name: [str(value) for value in values] for name, values in channels.items()
        }


def _measurement_matrices(
    config: CoherencePoleConfig, n_modes: int
) -> tuple[list[str], np.ndarray]:
    names: list[str] = []
    matrices: list[np.ndarray] = []
    for mode in config.modes:
        if mode >= n_modes:
            raise ValueError(f"mode {mode} is outside a {n_modes}-mode model")
        matrix = np.zeros((n_modes, n_modes), dtype=complex)
        matrix[mode, mode] = 1.0
        names.append(f"mode_{mode}")
        matrices.append(matrix)
    if config.include_trace:
        names.append("trace")
        matrices.append(np.eye(n_modes, dtype=complex))
    for name, values in config.channels.items():
        channel = np.asarray(values, dtype=complex)
        if channel.shape != (n_modes,):
            raise ValueError(
                f"channel {name!r} must contain {n_modes} complex amplitudes"
            )
        norm = float(np.linalg.norm(channel))
        if not np.isfinite(norm) or norm <= np.finfo(float).tiny:
            raise ValueError(f"channel {name!r} must have a finite non-zero norm")
        channel = channel / norm
        names.append(name)
        matrices.append(np.outer(channel, channel.conj()))
    return names, np.asarray(matrices)


def channel_coherence_poles(
    hamiltonian: np.ndarray,
    state: np.ndarray,
    measurements: np.ndarray,
    *,
    relative_residue_floor: float,
) -> dict[str, np.ndarray | float]:
    """Select the slowest visible pole of ``Tr[W exp(-i H tau) R]``."""
    eigenvalues, right = np.linalg.eig(np.asarray(hamiltonian, dtype=complex))
    condition_number = float(np.linalg.cond(right))
    try:
        left_adjoint = np.linalg.inv(right)
    except np.linalg.LinAlgError:
        count = int(np.asarray(measurements).shape[0])
        return {
            "eigenvalue": np.full(count, np.nan + 1j * np.nan),
            "frequency": np.full(count, np.nan),
            "decay_rate": np.full(count, np.nan),
            "residue": np.full(count, np.nan + 1j * np.nan),
            "relative_residue": np.full(count, np.nan),
            "valid": np.zeros(count, dtype=bool),
            "eigenvector_condition_number": condition_number,
        }

    decay_rates = -np.imag(eigenvalues)
    count = int(np.asarray(measurements).shape[0])
    selected_eigenvalue = np.full(count, np.nan + 1j * np.nan)
    selected_residue = np.full(count, np.nan + 1j * np.nan)
    selected_visibility = np.full(count, np.nan)
    valid = np.zeros(count, dtype=bool)
    for measurement_index, measurement in enumerate(measurements):
        residues = np.asarray(
            [
                left_adjoint[pole] @ state @ measurement @ right[:, pole]
                for pole in range(eigenvalues.size)
            ]
        )
        scale = float(np.max(np.abs(residues)))
        if not np.isfinite(scale) or scale <= np.finfo(float).tiny:
            continue
        relative = np.abs(residues) / scale
        visible = np.flatnonzero(relative >= relative_residue_floor)
        if visible.size == 0:
            continue
        selected = int(visible[np.argmin(decay_rates[visible])])
        selected_eigenvalue[measurement_index] = eigenvalues[selected]
        selected_residue[measurement_index] = residues[selected]
        selected_visibility[measurement_index] = relative[selected]
        valid[measurement_index] = True

    return {
        "eigenvalue": selected_eigenvalue,
        "frequency": np.real(selected_eigenvalue),
        "decay_rate": -np.imag(selected_eigenvalue),
        "residue": selected_residue,
        "relative_residue": selected_visibility,
        "valid": valid,
        "eigenvector_condition_number": condition_number,
    }


class CoherencePoleSpectrum(CAMPostprocessor[CoherencePoleConfig]):
    """Select channel-visible long-time poles of the closed CAM propagator."""

    name: ClassVar[str] = "coherence_pole_spectrum"
    description: ClassVar[str] = "Channel-visible poles of Tr[W exp(-i H tau) R]"
    config_schema: ClassVar[type[CoherencePoleConfig]] = CoherencePoleConfig

    def process(self, result: Any, model: Any, backend: Any) -> dict[str, Any]:
        del backend
        names, measurements = _measurement_matrices(self.config, int(model.n_modes))
        base_shape = np.asarray(result.valid_mask).shape
        output_shape = base_shape + (len(names),)
        eigenvalue = np.full(output_shape, np.nan + 1j * np.nan)
        frequency = np.full(output_shape, np.nan)
        decay_rate = np.full(output_shape, np.nan)
        residue = np.full(output_shape, np.nan + 1j * np.nan)
        relative_residue = np.full(output_shape, np.nan)
        valid = np.zeros(output_shape, dtype=bool)
        condition_number = np.full(base_shape, np.nan)

        for index in _successful_indices(result):
            state = np.asarray(result.states[index])
            hamiltonian = np.asarray(
                model.cam_hamiltonian(state, result.params_at(index))
            )
            selected = channel_coherence_poles(
                hamiltonian,
                state,
                measurements,
                relative_residue_floor=self.config.relative_residue_floor,
            )
            eigenvalue[index] = selected["eigenvalue"]
            frequency[index] = selected["frequency"]
            decay_rate[index] = selected["decay_rate"]
            residue[index] = selected["residue"]
            relative_residue[index] = selected["relative_residue"]
            valid[index] = selected["valid"]
            condition_number[index] = selected["eigenvector_condition_number"]

        self.result_metadata = {
            "measurement_names": names,
            "frequency_orientation": "phase_decreasing",
            "correlation_definition": "Tr[W exp(-i H tau) R]",
            "selection": "minimum_decay_rate_among_visible_residues",
            "relative_residue_floor": self.config.relative_residue_floor,
        }
        return {
            "coherence_pole_eigenvalue": eigenvalue,
            "coherence_pole_frequency": frequency,
            "coherence_pole_decay_rate": decay_rate,
            "coherence_pole_residue": residue,
            "coherence_pole_relative_residue": relative_residue,
            "coherence_pole_valid": valid,
            "coherence_pole_eigenvector_condition_number": condition_number,
        }


__all__ = [
    "CoherencePoleConfig",
    "CoherencePoleSpectrum",
    "channel_coherence_poles",
]
