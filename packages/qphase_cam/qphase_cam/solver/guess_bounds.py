"""Model-aware initial-guess bounds for CAM multi-root searches."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, field_validator
from scipy.optimize import root


class GuessBoundsConfig(BaseModel):
    """Strict schema for explicitly configured Hermitian guess bounds."""

    model_config = ConfigDict(extra="forbid")

    diag_lower: list[float]
    diag_upper: list[float]
    offdiag_scale: float | list[list[float]] = 1.0

    @field_validator("offdiag_scale")
    @classmethod
    def validate_offdiag_scale(
        cls, value: float | list[list[float]]
    ) -> float | list[list[float]]:
        if np.any(np.asarray(value, dtype=float) <= 0.0):
            raise ValueError("offdiag_scale values must be positive")
        return value


@dataclass(frozen=True)
class GuessBounds:
    """Validated per-element sampling bounds."""

    diag_lower: np.ndarray
    diag_upper: np.ndarray
    offdiag_scale: np.ndarray

    @classmethod
    def from_config(cls, config: GuessBoundsConfig, n_modes: int) -> GuessBounds:
        lower = np.asarray(config.diag_lower, dtype=float)
        upper = np.asarray(config.diag_upper, dtype=float)
        if lower.shape != (n_modes,) or upper.shape != (n_modes,):
            raise ValueError(
                f"guess-bound diagonals must each contain {n_modes} values"
            )
        if np.any(lower >= upper):
            raise ValueError("each diag_lower value must be less than diag_upper")
        scale = np.asarray(config.offdiag_scale, dtype=float)
        if scale.ndim == 0:
            scale = np.full((n_modes, n_modes), float(scale))
        if scale.shape != (n_modes, n_modes) or np.any(scale <= 0.0):
            raise ValueError(
                "offdiag_scale must be positive and scalar or an n_modes square matrix"
            )
        return cls(lower, upper, scale)

    def sample(
        self,
        count: int,
        seed: int | None,
        tail_fraction: float,
        tail_orders: float,
    ) -> list[np.ndarray]:
        """Sample ordinary and heavy-tailed Hermitian initial matrices."""
        rng = np.random.default_rng(seed)
        typical = np.maximum(np.abs(self.diag_lower), np.abs(self.diag_upper))
        typical = np.maximum(typical, 1e-8)
        guesses: list[np.ndarray] = []
        for _ in range(count):
            state = np.zeros((len(typical), len(typical)), dtype=complex)
            tail = rng.random() < tail_fraction
            if tail:
                magnitude = typical * 10.0 ** rng.uniform(
                    -tail_orders, tail_orders, size=len(typical)
                )
                diagonal = magnitude * rng.choice((-1.0, 1.0), size=len(typical))
            else:
                diagonal = rng.uniform(self.diag_lower, self.diag_upper)
            np.fill_diagonal(state, diagonal)
            for i in range(len(typical)):
                for j in range(i + 1, len(typical)):
                    scale = self.offdiag_scale[i, j]
                    if tail:
                        scale *= 10.0 ** rng.uniform(-tail_orders, tail_orders)
                    value = rng.normal(0.0, scale) + 1j * rng.normal(0.0, scale)
                    state[i, j] = value
                    state[j, i] = value.conjugate()
            guesses.append(state)
        return guesses


def infer_guess_bounds(
    model: Any,
    params: dict[str, Any],
    *,
    seed: int | None,
    starts: int = 32,
    fallback_scale: float = 100.0,
    margin: float = 2.0,
) -> GuessBounds:
    """Infer inexpensive bounds from diagonal Liouvillian balance roots."""
    n_modes = int(model.n_modes)

    def diagonal_residual(diagonal: np.ndarray) -> np.ndarray:
        state = np.diag(np.asarray(diagonal, dtype=float)).astype(complex)
        hamiltonian = np.asarray(model.cam_hamiltonian(state, params))
        diffusion = np.asarray(model.cam_diffusion(state, params))
        return 2.0 * diagonal * np.imag(np.diag(hamiltonian)) + np.real(
            np.diag(diffusion)
        )

    rng = np.random.default_rng(seed)
    candidates = [np.zeros(n_modes)]
    for _ in range(max(starts - 1, 0)):
        magnitudes = 10.0 ** rng.uniform(-4.0, 6.0, size=n_modes)
        candidates.append(magnitudes * rng.choice((-1.0, 1.0), size=n_modes))
    roots: list[np.ndarray] = []
    for candidate in candidates:
        solved = root(diagonal_residual, candidate, method="hybr", tol=1e-8)
        if (
            solved.success
            and np.linalg.norm(diagonal_residual(solved.x), ord=np.inf) < 1e-5
        ):
            if all(np.linalg.norm(solved.x - known) > 1e-3 for known in roots):
                roots.append(np.asarray(solved.x, dtype=float))
    if roots:
        values = np.vstack((np.zeros((1, n_modes)), np.asarray(roots)))
        lower = np.minimum(np.min(values, axis=0), 0.0)
        upper = np.maximum(np.max(values, axis=0), 0.0)
        scale = max(float(np.max(np.abs(values))), 1.0)
        lower = np.where(lower < 0.0, margin * lower, -0.1 * scale)
        upper = np.where(upper > 0.0, margin * upper, 0.1 * scale)
    else:
        lower = np.full(n_modes, -fallback_scale)
        upper = np.full(n_modes, fallback_scale)
    widths = np.maximum(upper - lower, 1e-8)
    offdiag = 0.5 * np.sqrt(widths[:, None] * widths[None, :])
    return GuessBounds(lower, upper, offdiag)
