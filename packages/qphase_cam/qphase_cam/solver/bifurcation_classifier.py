"""Local scaling classification for CAM equilibrium bifurcations."""

from __future__ import annotations

from fractions import Fraction
from typing import Any, ClassVar, Protocol, runtime_checkable

import numpy as np
from pydantic import ConfigDict, Field
from qphase.core.protocols import PluginConfigBase


@runtime_checkable
class BifurcationClassifier(Protocol):
    name: ClassVar[str]

    def classify(
        self,
        reduction: Any,
        value: np.ndarray,
        state_vector: np.ndarray,
        params: dict[str, Any],
        adapter: Any,
        *,
        perturbation: str,
        scale: float,
        side: str,
        verification_digits: int,
    ) -> dict[str, Any]: ...


class ScalingSignatureConfig(PluginConfigBase):
    model_config = ConfigDict(extra="forbid")
    max_total_order: int = Field(6, ge=2, le=8)
    coefficient_tolerance: float = Field(1e-9, gt=0.0)
    max_exponent: float | None = Field(None, gt=0.0)


class ScalingSignatureClassifier:
    """Classify lower Newton edges as named ``(n,k,m)`` signatures."""

    name: ClassVar[str] = "scaling_signature"
    description: ClassVar[str] = "Single-parameter CAM scaling signatures"
    config_schema: ClassVar[type[ScalingSignatureConfig]] = ScalingSignatureConfig

    def __init__(self, config: ScalingSignatureConfig) -> None:
        self.config = config

    def classify(
        self,
        reduction: Any,
        value: np.ndarray,
        state_vector: np.ndarray,
        params: dict[str, Any],
        adapter: Any,
        *,
        perturbation: str,
        scale: float,
        side: str,
        verification_digits: int,
    ) -> dict[str, Any]:
        try:
            series = reduction.local_scaling_series(
                value,
                perturbation=perturbation,
                scale=scale,
                max_total_order=self.config.max_total_order,
                digits=max(30, verification_digits),
            )
        except Exception as exc:
            return {
                "classification_status": "series_unavailable",
                "classification_error": str(exc),
                "scaling_signatures": (),
                "classification_accepted": False,
            }

        finite = [
            abs(item) for item in series.coefficients.values() if np.isfinite(item)
        ]
        reference = max(finite, default=1.0)
        threshold = self.config.coefficient_tolerance * max(1.0, reference)
        active = {
            key: item
            for key, item in series.coefficients.items()
            if np.isfinite(item) and abs(item) > threshold
        }
        pure_state = sorted(
            state_order
            for (state_order, parameter_order) in active
            if parameter_order == 0 and state_order > 0
        )
        if not pure_state:
            return {
                "classification_status": "state_order_unresolved",
                "coefficient_threshold": threshold,
                "scaling_signatures": (),
                "classification_accepted": False,
            }
        n = pure_state[0]
        state_coefficient = active[(n, 0)]
        tangent_matrix = self._state_tangent_matrix(
            adapter, state_vector, params, series.state_tangent
        )
        signatures = []
        seen: set[tuple[int, int, int]] = set()
        for (m, k), perturbation_coefficient in sorted(active.items()):
            if k < 1 or m >= n:
                continue
            signature_key = (n, k, m)
            if signature_key in seen:
                continue
            exponent = Fraction(k, n - m)
            target_weight = Fraction(n) * exponent
            if any(
                Fraction(state_order) * exponent + parameter_order < target_weight
                for state_order, parameter_order in active
                if (state_order, parameter_order) != (0, 0)
            ):
                continue
            seen.add(signature_key)
            edge_terms = tuple(
                sorted(
                    key
                    for key in active
                    if Fraction(key[0]) * exponent + key[1] == target_weight
                )
            )
            branches = self._real_branches(
                n=n,
                k=k,
                m=m,
                ratio=-perturbation_coefficient / state_coefficient,
                side=side,
                tangent_matrix=tangent_matrix,
            )
            signatures.append(
                {
                    "state_order": n,
                    "perturbation_order": k,
                    "coupling_state_order": m,
                    "exponent_numerator": exponent.numerator,
                    "exponent_denominator": exponent.denominator,
                    "exponent": float(exponent),
                    "sublinear": bool(exponent < 1),
                    "state_coefficient": state_coefficient,
                    "perturbation_coefficient": perturbation_coefficient,
                    "state_coefficient_decimal": (
                        series.coefficient_decimals.get((n, 0), "")
                    ),
                    "perturbation_coefficient_decimal": (
                        series.coefficient_decimals.get((m, k), "")
                    ),
                    "edge_terms": edge_terms,
                    "compound_edge": len(edge_terms) > 2,
                    "branches": branches,
                }
            )
        accepted = bool(signatures)
        if self.config.max_exponent is not None:
            accepted = any(
                item["exponent"] <= self.config.max_exponent for item in signatures
            )
        return {
            "classification_status": "classified" if signatures else "no_edge",
            "coefficient_threshold": threshold,
            "state_tangent_vector": series.state_tangent,
            "state_tangent_matrix": tangent_matrix,
            "scaling_signatures": tuple(signatures),
            "classification_accepted": accepted,
        }

    @staticmethod
    def _state_tangent_matrix(
        adapter: Any,
        state_vector: np.ndarray,
        params: dict[str, Any],
        tangent: np.ndarray,
    ) -> np.ndarray:
        norm = float(np.linalg.norm(tangent))
        step = 1e-6 / max(1.0, norm)
        plus = adapter.state_matrix(state_vector + step * tangent, params)
        minus = adapter.state_matrix(state_vector - step * tangent, params)
        return np.asarray((plus - minus) / (2.0 * step))

    @staticmethod
    def _real_branches(
        *,
        n: int,
        k: int,
        m: int,
        ratio: float,
        side: str,
        tangent_matrix: np.ndarray,
    ) -> tuple[dict[str, Any], ...]:
        degree = n - m
        signs = (-1, 1) if side == "both" else ((1,) if side == "positive" else (-1,))
        output = []
        for epsilon_sign in signs:
            value = ratio * epsilon_sign**k
            amplitudes: tuple[float, ...]
            if degree % 2:
                amplitudes = (float(np.copysign(abs(value) ** (1.0 / degree), value)),)
            elif value > 0.0:
                root = float(value ** (1.0 / degree))
                amplitudes = (-root, root)
            else:
                amplitudes = ()
            for amplitude in amplitudes:
                output.append(
                    {
                        "epsilon_side": epsilon_sign,
                        "amplitude": amplitude,
                        "leading_state_coefficient": amplitude * tangent_matrix,
                    }
                )
        return tuple(output)
