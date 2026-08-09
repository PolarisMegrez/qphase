"""Ensemble first-order coherence matrices and modal-purity diagnostics."""

from __future__ import annotations

import math
from statistics import NormalDist
from typing import Any, ClassVar, cast

import numpy as np
from pydantic import Field, model_validator
from qphase.backend.base import BackendBase
from qphase.backend.xputil import convert_to_numpy
from qphase.core.protocols import PluginConfigBase

from ..utils import resolve_mode_columns
from .base import (
    Analyzer,
    AnalyzerExecutionCapabilities,
    AnalyzerWorkspaceEstimate,
    AnalyzerWorkspaceRequest,
)
from .result import AnalysisResult

__all__ = ["CoherenceMatrixAnalyzer", "CoherenceMatrixConfig"]


class CoherenceMatrixConfig(PluginConfigBase):
    """Configuration for ensemble first-order coherence statistics."""

    modes: list[int] | None = Field(
        None,
        min_length=1,
        description="Physical mode indices; None analyzes every recorded mode",
    )
    time_blocks: int = Field(
        8,
        ge=1,
        le=256,
        description="Contiguous blocks retained for stationarity diagnostics",
    )
    min_block_samples: int = Field(
        32,
        ge=2,
        description="Minimum saved samples in each requested time block",
    )
    confidence_level: float = Field(
        0.95,
        gt=0.0,
        lt=1.0,
        description="Normal-approximation confidence level around jackknife SEM",
    )
    trace_tolerance: float = Field(
        1e-14,
        ge=0.0,
        description="Minimum positive trace required for normalized diagnostics",
    )

    @model_validator(mode="after")
    def validate_modes(self) -> CoherenceMatrixConfig:
        if self.modes is not None and len(set(self.modes)) != len(self.modes):
            raise ValueError("modes must not contain duplicates")
        return self


class CoherenceMatrixAnalyzer(Analyzer):
    """Estimate ``R_ij = <alpha_i alpha_j*>`` without retaining trajectories."""

    name: ClassVar[str] = "coherence_matrix"
    description: ClassVar[str] = (
        "Ensemble first-order coherence matrix, modal purity, and convergence"
    )
    config_schema: ClassVar[type[CoherenceMatrixConfig]] = CoherenceMatrixConfig

    def capabilities(self) -> AnalyzerExecutionCapabilities:
        return AnalyzerExecutionCapabilities(
            execution_location="backend",
            requires_full_trajectory=True,
            supports_trajectory_batching=True,
            supports_time_streaming=False,
        )

    def estimate_workspace(
        self, request: AnalyzerWorkspaceRequest
    ) -> AnalyzerWorkspaceEstimate:
        config = cast(CoherenceMatrixConfig, self.config)
        matrix_bytes = (
            request.n_traj
            * request.n_record_modes
            * request.n_record_modes
            * 2
            * request.real_itemsize
        )
        block_bytes = (
            config.time_blocks
            * request.n_record_modes
            * request.n_record_modes
            * 2
            * request.real_itemsize
        )
        retained = 2 * matrix_bytes + block_bytes
        if request.backend_name == "cupy":
            return AnalyzerWorkspaceEstimate(
                device_bytes=retained,
                host_bytes=matrix_bytes + block_bytes,
            )
        return AnalyzerWorkspaceEstimate(host_bytes=retained)

    def analyze(self, data: Any, backend: BackendBase) -> AnalysisResult:
        config = cast(CoherenceMatrixConfig, self.config)
        values = getattr(data, "data", data)
        if (
            not hasattr(values, "ndim")
            or values.ndim != 3
            or not np.issubdtype(values.dtype, np.complexfloating)
        ):
            raise ValueError(
                "coherence_matrix expects complex shape (n_traj, n_time, n_modes)"
            )
        n_traj, n_samples, _ = map(int, values.shape)
        if n_traj < 1 or n_samples < 1:
            raise ValueError("coherence_matrix requires non-empty trajectories")

        modes = _resolve_modes(data, config.modes, int(values.shape[2]))
        columns = resolve_mode_columns(data, modes)
        selected = values[:, :, columns]
        per_trajectory = backend.einsum(
            "rti,rtj->rij", selected, selected.conj()
        ) / float(n_samples)
        mean_amplitude = backend.mean(selected, axis=1)

        boundaries = _block_boundaries(
            n_samples, config.time_blocks, config.min_block_samples
        )
        block_matrices = []
        for start, stop in zip(boundaries[:-1], boundaries[1:], strict=True):
            block = selected[:, start:stop]
            matrix = backend.einsum(
                "rti,rtj->ij", block, block.conj()
            ) / float(n_traj * (stop - start))
            block_matrices.append(convert_to_numpy(matrix))

        payload = self._summarize(
            per_trajectory=np.asarray(convert_to_numpy(per_trajectory)),
            per_trajectory_mean=np.asarray(convert_to_numpy(mean_amplitude)),
            block_matrices=np.asarray(block_matrices),
            modes=modes,
            n_samples=n_samples,
            t0=float(getattr(data, "t0", 0.0)),
            dt=float(getattr(data, "dt", 1.0)),
            boundaries=boundaries,
        )
        meta = {
            "quantity": payload["quantity"],
            "modes": payload["modes"],
            "n_traj": payload["n_traj"],
            "n_samples": payload["n_samples"],
            "purity_definition": payload["purity_definition"],
            "uncertainty_unit": "trajectory",
        }
        return AnalysisResult(data_dict=payload, meta=meta)

    def create_result_accumulator(self) -> CoherenceMatrixResultAccumulator:
        return CoherenceMatrixResultAccumulator(self)

    def _summarize(
        self,
        *,
        per_trajectory: np.ndarray,
        per_trajectory_mean: np.ndarray,
        block_matrices: np.ndarray,
        modes: list[int],
        n_samples: int,
        t0: float,
        dt: float,
        boundaries: np.ndarray,
    ) -> dict[str, Any]:
        config = cast(CoherenceMatrixConfig, self.config)
        n_traj, n_modes, _ = per_trajectory.shape
        matrix = _hermitian(np.mean(per_trajectory, axis=0))
        mean_amplitude = np.mean(per_trajectory_mean, axis=0)
        connected = _hermitian(matrix - np.outer(mean_amplitude, mean_amplitude.conj()))
        normalized = _normalized_matrix(matrix, config.trace_tolerance)
        metrics = _matrix_metrics(matrix, config.trace_tolerance)
        per_purity = np.asarray(
            [
                _matrix_metrics(item, config.trace_tolerance)["purity"]
                for item in per_trajectory
            ],
            dtype=float,
        )

        if n_traj > 1:
            delta = per_trajectory - matrix[None, :, :]
            sem_real = np.std(per_trajectory.real, axis=0, ddof=1) / math.sqrt(n_traj)
            sem_imag = np.std(per_trajectory.imag, axis=0, ddof=1) / math.sqrt(n_traj)
            matrix_sem = np.sqrt(
                np.sum(np.abs(delta) ** 2, axis=0) / (n_traj - 1) / n_traj
            )
            jackknife = _purity_jackknife(
                per_trajectory,
                matrix,
                metrics["purity"],
                config.trace_tolerance,
                config.confidence_level,
            )
        else:
            sem_real = np.full((n_modes, n_modes), np.nan)
            sem_imag = np.full((n_modes, n_modes), np.nan)
            matrix_sem = np.full((n_modes, n_modes), np.nan)
            jackknife = {
                "purity_sem": math.nan,
                "purity_ci": np.asarray([math.nan, math.nan]),
                "purity_jackknife_bias_corrected": math.nan,
            }

        block_matrices = np.asarray([_hermitian(item) for item in block_matrices])
        block_metrics = [
            _matrix_metrics(item, config.trace_tolerance) for item in block_matrices
        ]
        matrix_norm = max(float(np.linalg.norm(matrix)), config.trace_tolerance)
        block_distance = np.asarray(
            [
                float(np.linalg.norm(item - matrix) / matrix_norm)
                for item in block_matrices
            ]
        )
        block_purity = np.asarray([item["purity"] for item in block_metrics])
        block_trace = np.asarray([item["trace"] for item in block_metrics])
        starts = boundaries[:-1]
        stops = boundaries[1:]

        diagonal = np.real(np.diag(matrix))
        denominator = np.sqrt(np.maximum(diagonal[:, None] * diagonal[None, :], 0.0))
        first_order_coherence = np.full(matrix.shape, np.nan + 0.0j)
        np.divide(
            matrix,
            denominator,
            out=first_order_coherence,
            where=denominator > config.trace_tolerance,
        )
        purity_status = "ok" if normalized is not None else "nonpositive_trace"
        normalized_payload = (
            normalized
            if normalized is not None
            else np.full(matrix.shape, np.nan + 0.0j)
        )
        return {
            "quantity": "ensemble_first_order_coherence_matrix",
            "definition": "R_ij = mean_trajectory,time(alpha_i * conj(alpha_j))",
            "ordering_correction": "none",
            "modes": list(modes),
            "n_traj": int(n_traj),
            "n_samples": int(n_samples),
            "t0": float(t0),
            "dt": float(dt),
            "observation_duration": float(max(0, n_samples - 1) * dt),
            "matrix": matrix,
            "matrix_sem": matrix_sem,
            "matrix_sem_real": sem_real,
            "matrix_sem_imag": sem_imag,
            "normalized_matrix": normalized_payload,
            "mean_amplitude": mean_amplitude,
            "connected_matrix": connected,
            "first_order_coherence": first_order_coherence,
            "eigenvalues": metrics["eigenvalues"],
            "normalized_eigenvalues": metrics["normalized_eigenvalues"],
            "trace": metrics["trace"],
            "purity": metrics["purity"],
            "purity_sem": jackknife["purity_sem"],
            "purity_ci": jackknife["purity_ci"],
            "purity_jackknife_bias_corrected": jackknife[
                "purity_jackknife_bias_corrected"
            ],
            "purity_definition": "Tr[(R / Tr(R))^2]",
            "purity_status": purity_status,
            "effective_rank": metrics["effective_rank"],
            "spectral_entropy": metrics["spectral_entropy"],
            "principal_fraction": metrics["principal_fraction"],
            "minimum_eigenvalue": metrics["minimum_eigenvalue"],
            "hermiticity_residual": metrics["hermiticity_residual"],
            "per_trajectory_matrix": per_trajectory,
            "per_trajectory_mean_amplitude": per_trajectory_mean,
            "per_trajectory_purity": per_purity,
            "time_blocks": {
                "count": int(block_matrices.shape[0]),
                "start_index": starts,
                "stop_index": stops,
                "start_time": t0 + starts * dt,
                "stop_time": t0 + np.maximum(stops - 1, starts) * dt,
                "matrix": block_matrices,
                "purity": block_purity,
                "trace": block_trace,
                "relative_matrix_distance": block_distance,
                "first_last_matrix_distance": float(
                    np.linalg.norm(block_matrices[-1] - block_matrices[0])
                    / matrix_norm
                ),
                "purity_range": float(
                    np.nanmax(block_purity) - np.nanmin(block_purity)
                ),
            },
            "uncertainty": {
                "available": n_traj > 1,
                "independent_unit": "trajectory",
                "n_independent": int(n_traj),
                "matrix_method": "sample_sem_across_trajectory_time_means",
                "purity_method": "leave_one_trajectory_out_jackknife",
                "confidence_level": config.confidence_level,
                "time_blocks_are_independent": False,
            },
        }


class CoherenceMatrixResultAccumulator:
    """Merge independent trajectory batches and recompute nonlinear metrics."""

    def __init__(self, analyzer: CoherenceMatrixAnalyzer) -> None:
        self.analyzer = analyzer
        self.payloads: list[dict[str, Any]] = []

    def update(self, payload: dict[str, Any]) -> None:
        if self.payloads:
            first = self.payloads[0]
            if first["modes"] != payload["modes"]:
                raise ValueError("coherence-matrix batches used different modes")
            if first["n_samples"] != payload["n_samples"]:
                raise ValueError("coherence-matrix batches used different time grids")
            for key in ("start_index", "stop_index"):
                if not np.array_equal(
                    first["time_blocks"][key], payload["time_blocks"][key]
                ):
                    raise ValueError(
                        "coherence-matrix batches used different time blocks"
                    )
        self.payloads.append(payload)

    def finalize(self) -> dict[str, Any]:
        if not self.payloads:
            raise RuntimeError("cannot finalize an empty coherence-matrix accumulator")
        first = self.payloads[0]
        counts = np.asarray([int(item["n_traj"]) for item in self.payloads])
        total = int(np.sum(counts))
        block_matrices = sum(
            np.asarray(item["time_blocks"]["matrix"]) * count
            for item, count in zip(self.payloads, counts, strict=True)
        ) / float(total)
        return self.analyzer._summarize(
            per_trajectory=np.concatenate(
                [np.asarray(item["per_trajectory_matrix"]) for item in self.payloads],
                axis=0,
            ),
            per_trajectory_mean=np.concatenate(
                [
                    np.asarray(item["per_trajectory_mean_amplitude"])
                    for item in self.payloads
                ],
                axis=0,
            ),
            block_matrices=block_matrices,
            modes=list(first["modes"]),
            n_samples=int(first["n_samples"]),
            t0=float(first["t0"]),
            dt=float(first["dt"]),
            boundaries=np.concatenate(
                (
                    np.asarray(first["time_blocks"]["start_index"]),
                    np.asarray(first["time_blocks"]["stop_index"])[-1:],
                )
            ),
        )


def _resolve_modes(data: Any, configured: list[int] | None, stored: int) -> list[int]:
    if configured is not None:
        return list(configured)
    meta = getattr(data, "meta", None)
    mode_indices = meta.get("mode_indices") if isinstance(meta, dict) else None
    if mode_indices is None:
        return list(range(stored))
    if len(mode_indices) != stored:
        raise ValueError("trajectory mode_indices do not match the stored mode count")
    return [int(mode) for mode in mode_indices]


def _block_boundaries(n_samples: int, requested: int, minimum: int) -> np.ndarray:
    maximum = max(1, n_samples // minimum)
    count = min(requested, maximum)
    return np.linspace(0, n_samples, count + 1, dtype=int)


def _hermitian(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=complex)
    return (matrix + matrix.conj().T) / 2.0


def _normalized_matrix(matrix: np.ndarray, tolerance: float) -> np.ndarray | None:
    trace = float(np.real(np.trace(matrix)))
    if not np.isfinite(trace) or trace <= tolerance:
        return None
    return matrix / trace


def _matrix_metrics(matrix: np.ndarray, tolerance: float) -> dict[str, Any]:
    raw = np.asarray(matrix, dtype=complex)
    scale = max(float(np.linalg.norm(raw)), tolerance)
    residual = float(np.linalg.norm(raw - raw.conj().T) / scale)
    matrix = _hermitian(raw)
    eigenvalues = np.linalg.eigvalsh(matrix)
    trace = float(np.real(np.sum(eigenvalues)))
    if not np.isfinite(trace) or trace <= tolerance:
        normalized_eigenvalues = np.full(eigenvalues.shape, np.nan)
        purity = effective_rank = entropy = principal = math.nan
    else:
        normalized_eigenvalues = np.clip(eigenvalues / trace, 0.0, None)
        total = float(np.sum(normalized_eigenvalues))
        if total > 0.0:
            normalized_eigenvalues /= total
        purity = float(np.sum(normalized_eigenvalues**2))
        effective_rank = 1.0 / purity if purity > 0.0 else math.nan
        positive = normalized_eigenvalues > 0.0
        entropy = float(
            -np.sum(
                normalized_eigenvalues[positive]
                * np.log(normalized_eigenvalues[positive])
            )
        )
        principal = float(np.max(normalized_eigenvalues))
    return {
        "eigenvalues": eigenvalues,
        "normalized_eigenvalues": normalized_eigenvalues,
        "trace": trace,
        "purity": purity,
        "effective_rank": effective_rank,
        "spectral_entropy": entropy,
        "principal_fraction": principal,
        "minimum_eigenvalue": float(np.min(eigenvalues)),
        "hermiticity_residual": residual,
    }


def _purity_jackknife(
    per_trajectory: np.ndarray,
    matrix: np.ndarray,
    estimate: float,
    tolerance: float,
    confidence_level: float,
) -> dict[str, Any]:
    count = int(per_trajectory.shape[0])
    leave_one_out = (count * matrix[None, :, :] - per_trajectory) / (count - 1)
    values = np.asarray(
        [_matrix_metrics(item, tolerance)["purity"] for item in leave_one_out]
    )
    mean = float(np.mean(values))
    sem = float(math.sqrt((count - 1) / count * np.sum((values - mean) ** 2)))
    corrected = float(count * estimate - (count - 1) * mean)
    z = NormalDist().inv_cdf((1.0 + confidence_level) / 2.0)
    lower_bound = 1.0 / per_trajectory.shape[1]
    interval = np.asarray(
        [max(lower_bound, estimate - z * sem), min(1.0, estimate + z * sem)]
    )
    return {
        "purity_sem": sem,
        "purity_ci": interval,
        "purity_jackknife_bias_corrected": corrected,
    }
