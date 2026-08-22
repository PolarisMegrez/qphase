"""Finite-delay carriers of closed-CAM first-order coherence channels."""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
from pydantic import Field, field_validator

from .base import CAMPostprocessor
from .coherence_pole import CoherencePoleConfig, _measurement_matrices
from .frequency import _successful_indices

__all__ = [
    "CAMFiniteDelayCarrier",
    "CAMFiniteDelayCarrierConfig",
    "finite_delay_carrier_from_poles",
]


class CAMFiniteDelayCarrierConfig(CoherencePoleConfig):
    """Channel and detector settings for finite-delay CAM carriers."""

    detector_rates: list[float] = Field(
        default_factory=lambda: [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0],
        description="Positive exponential detector rates kappa",
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


def finite_delay_carrier_from_poles(
    eigenvalues: np.ndarray,
    residues: np.ndarray,
    detector_rates: np.ndarray,
) -> dict[str, np.ndarray | float]:
    """Evaluate the detector-weighted carrier from all visible CAM poles."""
    values = np.asarray(eigenvalues, dtype=complex).reshape(-1)
    residue = np.asarray(residues, dtype=complex).reshape(-1)
    rates = np.asarray(detector_rates, dtype=float).reshape(-1)
    if values.size != residue.size or values.size == 0:
        raise ValueError("eigenvalues and residues must have equal non-zero length")
    frequency = np.real(values)
    decay = -np.imag(values)
    amplitude = np.conj(residue)
    exponent = -decay + 1j * frequency
    initial = np.sum(amplitude)
    derivative = np.sum(amplitude * exponent)
    instantaneous = float(
        np.imag(np.conj(initial) * derivative) / max(abs(initial) ** 2, 1e-300)
    )
    carrier = np.full(rates.shape, np.nan)
    coherent_weight = np.full(rates.shape, np.nan)
    for index, rate in enumerate(rates):
        integral = 1.0 / (
            2.0 * rate
            + decay[:, None]
            + decay[None, :]
            - 1j * (frequency[None, :] - frequency[:, None])
        )
        products = np.conj(amplitude)[:, None] * amplitude[None, :]
        denominator = float(np.real(np.sum(products * integral)))
        numerator = float(np.imag(np.sum(products * exponent[None, :] * integral)))
        if denominator > 1e-300:
            carrier[index] = numerator / denominator
            coherent_weight[index] = denominator
    return {
        "instantaneous_frequency": instantaneous,
        "frequency": carrier,
        "coherent_weight": coherent_weight,
    }


class CAMFiniteDelayCarrier(CAMPostprocessor[CAMFiniteDelayCarrierConfig]):
    """Compute finite-delay carriers for fixed CAM readout channels."""

    name: ClassVar[str] = "finite_delay_carrier"
    description: ClassVar[str] = "Finite-delay carriers from closed-CAM poles"
    config_schema: ClassVar[type[CAMFiniteDelayCarrierConfig]] = (
        CAMFiniteDelayCarrierConfig
    )

    def process(self, result: Any, model: Any, backend: Any) -> dict[str, Any]:
        del backend
        names, measurements = _measurement_matrices(self.config, int(model.n_modes))
        rates = np.asarray(self.config.detector_rates, dtype=float)
        base_shape = np.asarray(result.valid_mask).shape
        carrier = np.full(base_shape + (len(names), rates.size), np.nan)
        coherent_weight = np.full_like(carrier, np.nan)
        instantaneous = np.full(base_shape + (len(names),), np.nan)
        rayleigh = np.full_like(instantaneous, np.nan)
        condition_number = np.full(base_shape, np.nan)

        for index in _successful_indices(result):
            state = np.asarray(result.states[index], dtype=complex)
            hamiltonian = np.asarray(
                model.cam_hamiltonian(state, result.params_at(index)),
                dtype=complex,
            )
            eigenvalues, right = np.linalg.eig(hamiltonian)
            condition_number[index] = float(np.linalg.cond(right))
            try:
                left_adjoint = np.linalg.inv(right)
            except np.linalg.LinAlgError:
                continue
            for measurement_index, measurement in enumerate(measurements):
                residues = np.asarray(
                    [
                        left_adjoint[pole] @ state @ measurement @ right[:, pole]
                        for pole in range(eigenvalues.size)
                    ]
                )
                evaluated = finite_delay_carrier_from_poles(
                    eigenvalues, residues, rates
                )
                carrier[index + (measurement_index,)] = evaluated["frequency"]
                coherent_weight[index + (measurement_index,)] = evaluated[
                    "coherent_weight"
                ]
                instantaneous[index + (measurement_index,)] = evaluated[
                    "instantaneous_frequency"
                ]
                denominator = np.trace(measurement @ state)
                if abs(denominator) > 1e-300:
                    rayleigh[index + (measurement_index,)] = float(
                        np.real(
                            np.trace(measurement @ hamiltonian @ state) / denominator
                        )
                    )
        self.result_metadata = {
            "measurement_names": names,
            "detector_rates": rates.tolist(),
            "frequency_orientation": "phase_decreasing",
            "definition": (
                "integral exp(-2*kappa*tau) Im(conj(G) dG/dtau) / "
                "integral exp(-2*kappa*tau) |G|^2"
            ),
            "instantaneous_limit": "generalized Rayleigh quotient",
        }
        return {
            "finite_delay_carrier_frequency": carrier,
            "finite_delay_carrier_coherent_weight": coherent_weight,
            "finite_delay_carrier_instantaneous_frequency": instantaneous,
            "finite_delay_carrier_rayleigh_frequency": rayleigh,
            "finite_delay_carrier_instantaneous_rayleigh_residual": (
                instantaneous - rayleigh
            ),
            "finite_delay_carrier_eigenvector_condition_number": condition_number,
        }
